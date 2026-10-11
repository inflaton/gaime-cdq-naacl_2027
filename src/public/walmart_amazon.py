#!/usr/bin/env python3
"""Reproducible public, listwise product-label curation; see docs/walmart-amazon.md.

Preparation and inference use only product attributes. Reference matches are
loaded exclusively by the explicit evaluate command. The pilot uses development
queries; the default run uses the separately reserved 1,000 test queries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import unicodedata
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = ('https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/'
              'Structured/Walmart-Amazon/walmart_amazon_raw_data.zip')
SOURCE_SHA256 = 'c49a755801896b828ec68438cb39fc9fa1f127bcc716a26aaebbb05308a71c85'
MEMBER_HASHES = {
    'tableA.csv': 'e9d829d49788e8315a690112a98bcee639f55a8e4340be7d7342834c79732a1c',
    'tableB.csv': '15e0b0df4e2fafb9a34e8612f8968a7ed8f8835e065001729daff92057cfcf8d',
    'matches.csv': '0124598fa3a3b204f43eebe0eea394fb2d15a2e340d5d07b84130f1c01d6596e',
}
MODELS = {'gemma': ('gemma4:31b-it-qat-text-only', False),
          'muse': ('muse-glimmer:30b-nvfp4-dflash', 'low')}
# The paper's runs recorded the same local q4_K_M build under an earlier tag.
LEGACY_TAGS = {'gemma': ('gemma4:31b-it-qat-mlx',)}
FIELDS = ['query_id', 'split', 'query', 'candidates']
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'row_id': {'type': ['string', 'null']},
                   'match_confidence': {'type': ['number', 'null']},
                   'evidence': {'type': 'string'}},
    'required': ['row_id', 'match_confidence', 'evidence'],
}


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(payload if isinstance(payload, bytes) else payload.encode())
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen configuration changed: {path}; use a new directory.')
    else:
        save_json(path, value)


@contextmanager
def lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process is using {directory}') from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_csv_bytes(payload):
    return list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def clean(text, limit):
    text = re.sub(r'<[^>]*>', ' ', html.unescape(text or ''))
    text = ' '.join(unicodedata.normalize('NFKC', text).split())
    return text[:limit]


def project_product(row, side):
    """Exclude UPCs, ASINs, URLs, source IDs and retriever scores from model facts."""
    descriptions = ('shortdescr', 'shelfdescr', 'longdescr') if side == 'A' else (
        'proddescrshort', 'prodfeatures', 'proddescrlong')
    description = next((row.get(k) for k in descriptions if row.get(k)), '')
    return {'row_id': row['custom_id'], 'title': clean(row.get('title'), 512),
            'brand': clean(row.get('brand'), 128), 'modelno': clean(row.get('modelno'), 128),
            'price': clean(row.get('price'), 64), 'description': clean(description, 512)}


def tokens(record):
    return re.findall(r'\w+', ' '.join(record[k] for k in ('title', 'brand', 'modelno')).casefold())


class BM25:
    """Okapi BM25, k1=1.2, b=.75; unique query terms, deterministic ID ties."""
    def __init__(self, records):
        self.records = records
        self.postings = defaultdict(list)
        self.lengths = []
        for i, record in enumerate(records):
            counts = Counter(tokens(record))
            self.lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
        self.average = sum(self.lengths) / len(records) or 1

    def retrieve(self, query, k):
        scores = defaultdict(float)
        n = len(self.records)
        for term in sorted(set(tokens(query))):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (n - len(postings) + .5) / (len(postings) + .5))
            for i, freq in postings:
                norm = 1.2 * (.25 + .75 * self.lengths[i] / self.average)
                scores[i] += idf * freq * 2.2 / (freq + norm)
        order = sorted(range(n), key=lambda i: (-scores.get(i, 0), int(self.records[i]['row_id'])))[:k]
        return [(self.records[i], scores.get(i, 0)) for i in order]


def reserve_ids(ids, seed, test_size, dev_size):
    if len(set(ids)) != len(ids) or test_size + dev_size > len(ids):
        raise ValueError('Duplicate IDs or insufficient query records')
    ranked = sorted(ids, key=lambda i: (sha(f'{seed}:{i}'), int(i)))
    return {'test': ranked[:test_size], 'dev': ranked[test_size:test_size + dev_size],
            'unused': ranked[test_size + dev_size:]}


def preparation_config(args):
    return {'protocol': 'walmart-amazon-listwise-v1', 'source_url': SOURCE_URL,
            'archive_sha256': SOURCE_SHA256, 'member_sha256': MEMBER_HASHES,
            'seed': args.seed, 'test_size': args.test_size, 'dev_size': args.dev_size,
            'top_k': args.top_k, 'reservation': 'sha256(seed:custom_id), test then dev',
            'retriever': 'BM25 k1=1.2 b=.75; unique query terms; descending score, numeric ID ties',
            'retrieval_fields': ['title', 'brand', 'modelno'], 'candidate_order': 'ranked',
            'projection': 'NFKC HTML-cleaned; title512 brand128 model128 price64 description512',
            'implementation_sha256': sha(Path(__file__).read_bytes())}


def prepare(args):
    destination = args.dataset_dir
    cfg = preparation_config(args)
    with lock(destination):
        if (destination / 'manifest.json').exists():
            manifest = json.loads((destination / 'manifest.json').read_text())
            if manifest['config'] != cfg:
                raise ValueError('Preparation settings/code changed; use a new --dataset-dir.')
            load_prepared(destination, 'test')
            load_prepared(destination, 'dev')
            for filename, checksum in manifest['artifact_sha256'].items():
                if sha((destination / filename).read_bytes()) != checksum:
                    raise ValueError(f'Prepared artifact checksum mismatch: {filename}')
            for filename, checksum in MEMBER_HASHES.items():
                if sha((destination / 'source' / filename).read_bytes()) != checksum:
                    raise ValueError(f'Source member checksum mismatch: {filename}')
            print(f'Existing reservation verified: {destination}')
            return
        archive = destination / 'source/walmart_amazon_raw_data.zip'
        if not archive.exists():
            payload = (args.source_zip.read_bytes() if args.source_zip else
                       urllib.request.urlopen(SOURCE_URL, timeout=120).read())
            if sha(payload) != SOURCE_SHA256:
                raise ValueError('Unexpected source archive checksum; no reservation created.')
            atomic_write(archive, payload)
        if sha(archive.read_bytes()) != SOURCE_SHA256:
            raise ValueError('Source archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            for name, checksum in MEMBER_HASHES.items():
                payload = z.read(name)
                if sha(payload) != checksum:
                    raise ValueError(f'Unexpected source member: {name}')
                atomic_write(destination / 'source' / name, payload)
        # The reference CSV is preserved above, but never parsed during preparation.
        products = {}
        for side, expected in [('A', 2554), ('B', 22074)]:
            source = read_csv_bytes((destination / 'source' / f'table{side}.csv').read_bytes())
            records = [project_product(r, side) for r in source]
            ids = [r['row_id'] for r in records]
            if len(records) != expected or len(set(ids)) != expected or any(
                    not re.fullmatch(r'[1-9][0-9]*', x) for x in ids):
                raise ValueError(f'Invalid original table {side}')
            products[side] = sorted(records, key=lambda r: int(r['row_id']))
        membership = reserve_ids([r['row_id'] for r in products['A']], args.seed,
                                 args.test_size, args.dev_size)
        queries = {r['row_id']: r for r in products['A']}
        retriever = BM25(products['B'])
        artifacts = {}
        retrieval_rows = []
        for split in ('test', 'dev'):
            rows = []
            for rid in membership[split]:
                query = queries[rid]
                retrieved = retriever.retrieve(query, args.top_k)
                rows.append({'query_id': rid, 'split': split, 'query': dumps(query),
                             'candidates': dumps([r for r, _ in retrieved])})
                retrieval_rows.extend({'query_id': rid, 'split': split, 'rank': rank,
                                       'candidate_id': record['row_id'], 'bm25': format(score, '.12g')}
                                      for rank, (record, score) in enumerate(retrieved, 1))
            payload = csv_bytes(rows, FIELDS)
            artifacts[f'{split}.csv'] = sha(payload)
            atomic_write(destination / f'{split}.csv', payload)
        payload = csv_bytes(retrieval_rows, ['query_id', 'split', 'rank', 'candidate_id', 'bm25'])
        artifacts['retrieval.csv'] = sha(payload)
        atomic_write(destination / 'retrieval.csv', payload)
        save_json(destination / 'manifest.json', {
            'created_at_utc': now(), 'config': cfg, 'query_count': len(products['A']),
            'catalog_count': len(products['B']), 'membership': membership,
            'artifact_sha256': artifacts, 'labels_used_for_selection_or_retrieval': False,
            'model_outputs_used': False, 'separation': 'query-record IDs, not product entities'})
        print(f'Reserved {args.test_size} test + {args.dev_size} development queries; '
              f'{args.top_k} candidates each. No inference or reference scoring.')
        print(destination)


def load_prepared(directory, split):
    manifest = json.loads((directory / 'manifest.json').read_text())
    payload = (directory / f'{split}.csv').read_bytes()
    if sha(payload) != manifest['artifact_sha256'][f'{split}.csv']:
        raise ValueError(f'Prepared {split} checksum mismatch')
    rows = read_csv_bytes(payload)
    if [r['query_id'] for r in rows] != manifest['membership'][split]:
        raise ValueError('Reservation membership/order mismatch')
    for row in rows:
        if row['split'] != split:
            raise ValueError('Split mismatch')
        row['query'] = json.loads(row['query'])
        row['candidates'] = json.loads(row['candidates'])
        ids = [c['row_id'] for c in row['candidates']]
        if len(ids) != manifest['config']['top_k'] or len(ids) != len(set(ids)):
            raise ValueError('Invalid candidate pool')
    return rows, manifest


def render_prompt(template, row):
    # One-pass substitution: text embedded inside records is never a template.
    values = {'query': dumps(row['query']), 'candidates': dumps(row['candidates'])}
    return re.sub(r'\{\{(query|candidates)\}\}', lambda m: values[m[1]], template)


def parse_verdict(raw, candidate_ids):
    text = raw.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        text = re.sub(r'\s*```$', '', text)
    def invalid_constant(value):
        raise ValueError(f'Invalid number {value}')
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    verdict = json.loads(text, parse_constant=invalid_constant, object_pairs_hook=unique_object)
    if not isinstance(verdict, dict) or set(verdict) != set(SCHEMA['required']):
        raise ValueError('Response fields differ from schema')
    rid, confidence = verdict['row_id'], verdict['match_confidence']
    if not isinstance(verdict['evidence'], str):
        raise ValueError('Evidence must be a string')
    if rid is None:
        if confidence is not None:
            raise ValueError('Abstention must have null selection confidence')
    elif not isinstance(rid, str) or rid not in candidate_ids:
        raise ValueError('Selection must be an exact candidate string ID')
    elif type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid selection confidence')
    return verdict


def gate(verdict, threshold):
    if verdict['row_id'] is not None and verdict['match_confidence'] >= threshold:
        return verdict['row_id']
    return None


def pair_decision(a, b, positive=.86, negative=.80, mode='gated'):
    if not 0 <= negative <= positive <= 1:
        raise ValueError('Require 0 <= negative threshold <= positive threshold <= 1')
    if a is None or b is None:
        return 'incomplete', None
    pa, pb = gate(a, positive), gate(b, positive)
    if pa is not None and pa == pb:
        return 'positive', pa
    if mode == 'positive_only':
        return 'defer', None
    nulls = (a['row_id'] is None and b['row_id'] is None) if mode == 'native_null' else (
        gate(a, negative) is None and gate(b, negative) is None)
    return ('negative', None) if nulls else ('defer', None)


def journal_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'Incomplete journal {path}; inspect before resuming') from exc
    return events


def latest_results(directory, rows, prompt_hashes):
    lookup = {r['query_id']: r for r in rows}
    events = journal_events(directory / 'responses.jsonl')
    latest = {}
    for event in events:
        rid = event['query_id']
        if rid not in lookup or event['prompt_sha256'] != prompt_hashes[rid]:
            raise ValueError('Journal does not match reserved inputs/prompts')
        if event['status'] == 'ok':
            parsed = parse_verdict(event['raw_output'], {c['row_id'] for c in lookup[rid]['candidates']})
            if parsed != event['parsed']:
                raise ValueError('Journal parsed output differs from raw response')
        elif event['status'] != 'error':
            raise ValueError('Unknown journal status')
        latest[rid] = event
    return latest


def append_event(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(dumps(value) + '\n')
        f.flush()
        os.fsync(f.fileno())


def api(host, route, body=None, timeout=30):
    request = urllib.request.Request(host.rstrip('/') + route,
        data=None if body is None else dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_model(name, args, directory, rows, prompts, params):
    hashes = {rid: sha(prompt) for rid, prompt in prompts.items()}
    latest = latest_results(directory, rows, hashes)
    selected = rows[:args.limit] if args.limit is not None else rows
    todo = [r for r in selected if r['query_id'] not in latest or
            (args.retry_errors and latest[r['query_id']]['status'] == 'error')]
    if not todo:
        print(f'{name}: no new requests.')
        return not any(latest[r['query_id']]['status'] == 'error' for r in selected)
    inventory = api(args.ollama_host, '/api/tags')['models']
    installed = next((m for m in inventory if m.get('model', m.get('name')) == params['model']), None)
    if not installed or not installed.get('digest'):
        raise ValueError(f'Model is not installed: {params["model"]}')
    freeze(directory / 'model_identity.json', {'model': params['model'], 'digest': installed['digest']})
    save_json(directory / f'invocation-{uuid.uuid4().hex}.json', {
        'started_at_utc': now(), 'host': platform.node(), 'platform': platform.platform(),
        'ollama_host': args.ollama_host, 'server': api(args.ollama_host, '/api/version'),
        'model': installed, 'retry_errors': args.retry_errors, 'limit': args.limit})
    errors = consecutive = 0
    try:
        for i, row in enumerate(todo, 1):
            rid = row['query_id']
            event = {'event_id': uuid.uuid4().hex, 'query_id': rid, 'prompt_sha256': hashes[rid],
                     'started_at_utc': now(), 'status': 'error', 'raw_output': '', 'error': ''}
            start = time.monotonic()
            request_failed = False
            try:
                response = api(args.ollama_host, '/api/chat',
                    {**params, 'messages': [{'role': 'user', 'content': prompts[rid]}]}, args.timeout)
                event['response'] = response
                event['raw_output'] = response.get('message', {}).get('content', '')
                try:
                    if not response.get('done') or response.get('done_reason') != 'stop':
                        raise ValueError('Incomplete/truncated completion')
                    event['parsed'] = parse_verdict(event['raw_output'], {c['row_id'] for c in row['candidates']})
                    event['status'] = 'ok'
                except (ValueError, TypeError) as exc:
                    event['error'] = f'{type(exc).__name__}: {exc}'
            except Exception as exc:
                event['error'] = f'{type(exc).__name__}: {exc}'
                request_failed = True
            event['latency_ms'] = round((time.monotonic() - start) * 1000)
            append_event(directory / 'responses.jsonl', event)
            print(f'{name}: {i}/{len(todo)} query {rid} {event["status"]}', flush=True)
            if event['status'] == 'error':
                errors += 1
                consecutive += 1
                print(event['error'], flush=True)
                if request_failed or consecutive >= args.max_consecutive_errors:
                    return False
            else:
                consecutive = 0
    finally:
        try:
            api(args.ollama_host, '/api/generate', {'model': params['model'], 'keep_alive': 0}, 30)
        except Exception as exc:
            print(f'{name}: model release failed: {exc}')
    final = latest_results(directory, rows, hashes)
    return errors == 0 and all(final[r['query_id']]['status'] == 'ok' for r in selected)


def policy_outputs(rows, results):
    policies = defaultdict(list)
    for row in rows:
        rid = row['query_id']
        a, b = [results.get(name, {}).get(rid, {}).get('parsed') if
                results.get(name, {}).get(rid, {}).get('status') == 'ok' else None for name in MODELS]
        for name, ta, tn, mode in [
                ('dual_086_080', .86, .80, 'gated'), ('dual_086_086', .86, .86, 'gated'),
                ('dual_080_080', .80, .80, 'gated'), ('dual_native_null', .86, .80, 'native_null'),
                ('dual_positive_only', .86, .80, 'positive_only')]:
            policies[name].append(pair_decision(a, b, ta, tn, mode))
        for name, verdict in [('gemma', a), ('muse', b)]:
            if verdict is None:
                decision = 'incomplete', None
            elif gate(verdict, .86) is not None:
                decision = 'positive', gate(verdict, .86)
            elif gate(verdict, .80) is None:
                decision = 'negative', None
            else:
                decision = 'defer', None
            policies[name + '_086_080'].append(decision)
    return dict(policies)


def export_curation(output, rows, results):
    policies = policy_outputs(rows, results)
    records = []
    for row, (decision, selected) in zip(rows, policies['dual_086_080']):
        rid = row['query_id']
        item = {'query_id': rid, 'split': row['split'], 'decision': decision,
                'selected_id': selected or '', 'query': dumps(row['query']),
                'candidates': dumps(row['candidates'])}
        for name in MODELS:
            event = results.get(name, {}).get(rid, {})
            item[name + '_status'] = event.get('status', 'not_run')
            item[name + '_verdict'] = dumps(event.get('parsed'))
        records.append(item)
    fields = list(records[0])
    atomic_write(output / 'curation.csv', csv_bytes(records, fields))
    atomic_write(output / 'accepted.csv', csv_bytes([r for r in records if r['decision'] in ('positive', 'negative')], fields))
    return policies


def proportion(correct, total):
    if not total:
        return {'correct': correct, 'n': total, 'rate': None, 'wilson95': None}
    p, z = correct / total, 1.959963984540054
    scale = 1 + z*z/total
    center = (p + z*z/(2*total)) / scale
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / scale
    return {'correct': correct, 'n': total, 'rate': p, 'wilson95': [center-radius, center+radius]}


def score_policy(decisions, gold):
    counts, correct = Counter(), Counter()
    for (decision, selected), accepted in zip(decisions, gold):
        counts[decision] += 1
        if decision == 'positive':
            correct['positive'] += selected in accepted
        elif decision == 'negative':
            correct['negative'] += not accepted
    n = len(decisions)
    retained = counts['positive'] + counts['negative']
    return {'positive': proportion(correct['positive'], counts['positive']),
            'negative': proportion(correct['negative'], counts['negative']),
            'total': proportion(sum(correct.values()), retained), 'counts': dict(counts),
            'coverage': retained/n, 'positive_yield': correct['positive']/n,
            'negative_yield': correct['negative']/n}


def judge_metrics(verdicts, gold, threshold=.86):
    counts = Counter({'CM': 0, 'WC': 0, 'SM': 0, 'MM': 0, 'TN': 0})
    for verdict, accepted in zip(verdicts, gold):
        selected = gate(verdict, threshold)
        key = ('CM' if selected in accepted else 'WC') if selected is not None and accepted else (
            'SM' if selected is not None else ('MM' if accepted else 'TN'))
        counts[key] += 1
    cm, wc, sm, mm = [counts[k] for k in ('CM', 'WC', 'SM', 'MM')]
    # A wrong candidate on a matchable query is both a false positive and a false negative.
    denom = 1.25*cm + 1.25*wc + sm + .25*mm
    return {**counts, 'threshold': threshold,
            'precision': cm/(cm+wc+sm) if cm+wc+sm else None,
            'recall': cm/(cm+wc+mm) if cm+wc+mm else None,
            'f0_5': 1.25*cm/denom if denom else None,
            'no_match_false_match_rate': sm/(sm+counts['TN']) if sm+counts['TN'] else None}


def load_gold(directory, rows):
    payload = (directory / 'source/matches.csv').read_bytes()
    if sha(payload) != MEMBER_HASHES['matches.csv']:
        raise ValueError('Reference match checksum mismatch')
    matches = read_csv_bytes(payload)
    if len(matches) != 1154 or len({(r['id1'], r['id2']) for r in matches}) != 1154:
        raise ValueError('Unexpected reference match mapping')
    mapping = defaultdict(set)
    for match in matches:
        if not 1 <= int(match['id1']) <= 2554 or not 1 <= int(match['id2']) <= 22074:
            raise ValueError('Reference ID outside source tables')
        mapping[match['id1']].add(match['id2'])
    catalog_gold = [mapping[r['query_id']] for r in rows]
    pool_gold = [g & {c['row_id'] for c in r['candidates']} for g, r in zip(catalog_gold, rows)]
    return catalog_gold, pool_gold


def evaluate(args, output, rows, results, policies):
    # Require complete curator results before consulting any test reference labels.
    for name in MODELS:
        if any(results.get(name, {}).get(r['query_id'], {}).get('status') != 'ok' for r in rows):
            raise ValueError('Evaluation requires valid outputs from both curators for every query in this split.')
    catalog, gold = load_gold(args.dataset_dir, rows)
    matchable = sum(bool(g) for g in catalog)
    retrieved = sum(bool(g) for g in gold)
    report = {'split': args.split, 'n': len(rows), 'evaluated_at_utc': now(),
              'reference': 'released matches.csv; absence interpreted under benchmark closed-world convention',
              'retrieval': {'catalog_matchable_queries': matchable, 'pool_matchable_queries': retrieved,
                            'no_catalog_match_queries': len(rows)-matchable,
                            'retrieval_miss_queries': matchable-retrieved,
                            'query_recall_at_k': retrieved/matchable if matchable else None,
                            'pair_recall_at_k': sum(map(len, gold))/sum(map(len, catalog)) if matchable else None},
              'policies': {name: score_policy(decisions, gold) for name, decisions in policies.items()},
              'judges': {}, 'journal_sha256': {}, 'operational': {}}
    errors = []
    for name in MODELS:
        verdicts = [results[name][r['query_id']]['parsed'] for r in rows]
        report['judges'][name] = judge_metrics(verdicts, gold)
        report['journal_sha256'][name] = sha((output / name / 'responses.jsonl').read_bytes())
        events = journal_events(output / name / 'responses.jsonl')
        latencies = sorted(e['latency_ms'] for e in results[name].values())
        report['operational'][name] = {
            'attempts': len(events), 'failed_attempts': sum(e['status'] == 'error' for e in events),
            'median_final_call_latency_ms': (latencies[(len(latencies)-1)//2] + latencies[len(latencies)//2])/2,
            'all_attempt_wall_ms': sum(e.get('latency_ms', 0) for e in events),
            'input_tokens': sum(e.get('response', {}).get('prompt_eval_count', 0) or 0 for e in events),
            'output_tokens': sum(e.get('response', {}).get('eval_count', 0) or 0 for e in events)}
        errors.append([gate(v, .80) not in g if gate(v, .80) is not None else bool(g)
                       for v, g in zip(verdicts, gold)])
    report['shared_errors_at_080'] = {'both_wrong': sum(a and b for a, b in zip(*errors)),
        'gemma_wrong': sum(errors[0]), 'muse_wrong': sum(errors[1])}
    # Query-paired bootstrap: recompute selected denominators for each policy.
    import numpy as np
    indices = np.random.default_rng(42).integers(0, len(rows), size=(2000, len(rows)))
    estimates = {}
    for name, decisions in policies.items():
        estimates[name] = {}
        for polarity in ('positive', 'negative', 'total'):
            included = np.array([d == polarity if polarity != 'total' else d in ('positive', 'negative')
                                 for d, _ in decisions])
            correct = np.array([(s in g if d == 'positive' else not g if d == 'negative' else False)
                                for (d, s), g in zip(decisions, gold)]) & included
            denominator = included[indices].sum(axis=1)
            estimates[name][polarity] = np.divide(correct[indices].sum(axis=1), denominator,
                out=np.full(2000, np.nan), where=denominator != 0)
    differences = {}
    for baseline in policies:
        if baseline == 'dual_086_080':
            continue
        differences[baseline] = {}
        for polarity in ('positive', 'negative', 'total'):
            delta = estimates['dual_086_080'][polarity] - estimates[baseline][polarity]
            finite = delta[np.isfinite(delta)]
            a = report['policies']['dual_086_080'][polarity]['rate']
            b = report['policies'][baseline][polarity]['rate']
            differences[baseline][polarity] = {
                'difference': a-b if a is not None and b is not None else None,
                'paired_bootstrap95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                'defined_resamples': len(finite)}
    report['primary_minus_baseline_purity'] = differences
    report['bootstrap'] = {'n': 2000, 'seed': 42, 'unit': 'query', 'intervals': 'pointwise percentile'}
    scored = []
    for row, cg, pg, (decision, selected) in zip(rows, catalog, gold, policies['dual_086_080']):
        scored.append({'query_id': row['query_id'], 'catalog_gold_ids': dumps(sorted(cg, key=int)),
                       'pool_gold_ids': dumps(sorted(pg, key=int)), 'decision': decision,
                       'selected_id': selected or '', 'correct': (selected in pg if decision == 'positive' else
                            not pg if decision == 'negative' else '')})
    save_json(output / 'evaluation.json', report)
    atomic_write(output / 'evaluation_rows.csv', csv_bytes(scored, list(scored[0])))
    print(f'Evaluation written to {output / "evaluation.json"}; rates are benchmark-relative, not fresh human truth.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'pilot', 'run', 'collect', 'status', 'evaluate'])
    p.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/public/walmart-amazon')
    p.add_argument('--output-dir', type=Path, help='Default: DATASET/runs/product-v1; split subdirectories are added')
    p.add_argument('--source-zip', type=Path, help='Optional already-downloaded pinned original ZIP')
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--test-size', type=int, default=1000)
    p.add_argument('--dev-size', type=int, default=100)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--split', choices=['test', 'dev'], help='Default dev for pilot, test otherwise')
    p.add_argument('--models', default='gemma,muse')
    p.add_argument('--gemma-model', default=MODELS['gemma'][0])
    p.add_argument('--muse-model', default=MODELS['muse'][0])
    p.add_argument('--prompt-file', type=Path, default=ROOT / 'data/prompts/product_curation_v1.txt')
    p.add_argument('--limit', type=int, help='Pilot defaults to 25 development queries; run defaults to the full split')
    p.add_argument('--local-max-tokens', type=int, default=4096)
    p.add_argument('--num-ctx', type=int, default=32768)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--ollama-host', default=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
    p.add_argument('--retry-errors', action='store_true', help='Retry technical failures, never valid abstentions')
    p.add_argument('--max-consecutive-errors', type=int, default=3)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.test_size or not 25 <= args.dev_size or args.test_size + args.dev_size > 2554:
        raise ValueError('Require positive test size, at least 25 development rows, and at most 2,554 total')
    if not 1 <= args.top_k <= 22074 or (args.limit is not None and args.limit < 1):
        raise ValueError('Invalid top-k or limit')
    if args.local_max_tokens < 1 or args.num_ctx <= args.local_max_tokens or args.timeout <= 0 or args.max_consecutive_errors < 1:
        raise ValueError('Invalid inference limits')
    if args.stage == 'prepare':
        prepare(args)
        return
    if args.stage == 'pilot' and args.split == 'test':
        raise ValueError('Pilot is development-only; use run for reserved test queries.')
    args.split = args.split or ('dev' if args.stage == 'pilot' else 'test')
    if args.stage == 'pilot' and args.limit is None:
        args.limit = 25
    names = args.models.split(',')
    if len(names) != len(set(names)) or not set(names) <= set(MODELS):
        raise ValueError('--models must be distinct names from gemma,muse')
    rows, manifest = load_prepared(args.dataset_dir, args.split)
    if args.limit is not None and args.limit > len(rows):
        raise ValueError('Limit exceeds split size')
    output = (args.output_dir or args.dataset_dir / 'runs/product-v1') / args.split
    template = args.prompt_file.read_text()
    if template.count('{{query}}') != 1 or template.count('{{candidates}}') != 1:
        raise ValueError('Prompt must contain query and candidates placeholders exactly once')
    prompts = {r['query_id']: render_prompt(template, r) for r in rows}
    hashes = {rid: sha(text) for rid, text in prompts.items()}
    if args.stage == 'status':
        for name in MODELS:
            latest = latest_results(output / name, rows, hashes)
            counts = Counter(e['status'] for e in latest.values())
            print(f'{name}: {counts["ok"]} valid, {counts["error"]} errors, {len(rows)-len(latest)} not run ({args.split})')
        return
    with lock(output):
        protocol = {'protocol': 'product-curation-v1', 'split': args.split,
                    'dataset_manifest_sha256': sha((args.dataset_dir / 'manifest.json').read_bytes()),
                    'prompt_template_sha256': sha(template), 'prompt_sha256': hashes,
                    'implementation_sha256': sha(Path(__file__).read_bytes()),
                    'primary_thresholds': {'selection': .86, 'abstention': .80}}
        freeze(output / 'protocol.json', protocol)
        failures = []
        for name in names:
            params = {'model': getattr(args, name + '_model'), 'think': MODELS[name][1],
                      'format': SCHEMA, 'stream': False, 'keep_alive': '30m',
                      'options': {'temperature': 0, 'seed': 42, 'repeat_penalty': 1.12,
                                  'num_ctx': args.num_ctx, 'num_predict': args.local_max_tokens}}
            freeze(output / name / 'config.json', params)
            atomic_write(output / name / 'first-prompt.txt', prompts[rows[0]['query_id']])
            if args.stage in ('pilot', 'run'):
                if not run_model(name, args, output / name, rows, prompts, params):
                    failures.append(name)
                    break
        results = {name: latest_results(output / name, rows, hashes) for name in MODELS}
        policies = export_curation(output, rows, results)
        for name in MODELS:
            counts = Counter(e['status'] for e in results[name].values())
            print(f'{name}: {counts["ok"]} valid, {counts["error"]} errors, {len(rows)-len(results[name])} not run ({args.split})')
        if args.stage == 'evaluate':
            evaluate(args, output, rows, results, policies)
        if failures:
            raise ValueError('Inference needs attention: ' + ', '.join(failures))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
