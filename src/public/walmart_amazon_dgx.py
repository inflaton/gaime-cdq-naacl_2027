#!/usr/bin/env python3
"""Walmart-Amazon public curation on the DGX Spark: the bf16 curator pair over vLLM.

The reservation, retrieval, prompt rendering, verdict parser, policies and scoring are
the released public pipeline, vendored unchanged as src/public/walmart_amazon.py
(byte-identical to quantized-local-judge/src/walmart_amazon.py).  This module adds
only an OpenAI-compatible vLLM client with the curators' DGX merchant-curation
settings, concurrent requests, and a frozen, development-only adjustment ladder.

    python src/public/walmart_amazon_dgx.py prepare            # download (13.8 MB) + reserve; no models
    python src/public/walmart_amazon_dgx.py pilot  --model gemma   # 25 development queries
    python src/public/walmart_amazon_dgx.py run    --model gemma   # 1,000 reserved test queries
    python src/public/walmart_amazon_dgx.py status
    python src/public/walmart_amazon_dgx.py evaluate           # reads test references; both curators done

RULES (fixed 9 Oct 2026, before any DGX output; PLAN.md "Public protocol"):
  * Curators: Gemma 4 31B bf16 thinking off, JSON grammar (response_format, the
    product schema); Muse Glimmer 30B bf16 reasoning_strength=low, structured output
    off -- the settings of their merchant curation cells (scripts/run-spark-pool.sh).
    temperature 0, seed 42, repetition_penalty 1.12, 16 concurrent requests.
  * Output cap 4,096 tokens (as in the released study).  Adjustment ladder, decided on
    the 25 development queries only and recorded in the run directory:
      rung 1  cap 4,096, structured setting as above
      rung 2  cap 8,192                          (if any pilot response is truncated)
      rung 3  cap 8,192 + response_format schema (Muse only; if pilot verdicts fail the
                                                   released parser for any other reason)
      rung 4/5/6  as 1/2/3 with the schema written as two explicit shapes (anyOf), only if
              the server rejects the released schema (HTTP 400 -- a grammar-compile issue)
    The test run uses the first rung whose pilot returns 25/25 valid verdicts; no rung
    passing means the public run is abandoned.  No prompt, threshold or reference-label
    information is used to choose a rung.
  * Validation is the released parser (exact schema, exact candidate string IDs,
    confidence in [0, 1], null confidence on abstention), applied to both curators.
    A truncated or unparseable response is a technical failure, never a negative
    label.  The test pass retries failed rows once (logged as new journal events).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import sys
import time
import types
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walmart_amazon as wa  # noqa: E402  (vendored released pipeline)

ROOT = HERE.parents[1]
DATASET = ROOT / "data/public/walmart-amazon"
RUNS = Path(os.environ.get("PUBLIC_RUNS_DIR") or DATASET / "runs/product-v1-dgx")   # override: tests only
PROMPT = ROOT / "data/prompts/product_curation_v1.txt"
RESERVATION = HERE / "walmart-amazon-reservation.json"   # released aggregate, for the check

CURATORS = {
    "gemma": {"served": "gemma4:31b-bf16-vllm", "hf": "google/gemma-4-31B-it",
              "chat_template_kwargs": {"enable_thinking": False}, "structured": "response_format"},
    "muse":  {"served": "muse-glimmer:30b-bf16-vllm", "hf": "meta-models/Muse-Glimmer-30B",
              "chat_template_kwargs": {"reasoning_strength": "low"}, "structured": "off"},
}
LADDER = {1: {"max_tokens": 4096, "force_schema": False, "schema": "released"},
          2: {"max_tokens": 8192, "force_schema": False, "schema": "released"},
          3: {"max_tokens": 8192, "force_schema": True,  "schema": "released"},
          # only if the server rejects the released schema (HTTP 400: grammar compile)
          4: {"max_tokens": 4096, "force_schema": False, "schema": "anyof"},
          5: {"max_tokens": 8192, "force_schema": False, "schema": "anyof"},
          6: {"max_tokens": 8192, "force_schema": True,  "schema": "anyof"}}
# The released schema as two explicit shapes; it admits exactly the same verdicts.
SCHEMA_ANYOF = {"anyOf": [
    {"type": "object", "additionalProperties": False, "required": wa.SCHEMA["required"],
     "properties": {"row_id": {"type": "null"}, "match_confidence": {"type": "null"},
                    "evidence": {"type": "string"}}},
    {"type": "object", "additionalProperties": False, "required": wa.SCHEMA["required"],
     "properties": {"row_id": {"type": "string"}, "match_confidence": {"type": "number"},
                    "evidence": {"type": "string"}}}]}
STALL_STOP = ["\n \n \n \n", "\n\n\n\n\n"]          # src/eval.py VLLM_STALL_STOP
PILOT_N = 25


# ------------------------------------------------------------------ requests ----
def request_body(name: str, prompt: str, rung: int) -> dict:
    c, r = CURATORS[name], LADDER[rung]
    body = {"model": c["served"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": r["max_tokens"], "temperature": 0, "seed": 42, "stop": STALL_STOP,
            "repetition_penalty": 1.12, "chat_template_kwargs": c["chat_template_kwargs"]}
    if c["structured"] == "response_format" or r["force_schema"]:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "product_verdict", "strict": True,
                                                   "schema": SCHEMA_ANYOF if r["schema"] == "anyof"
                                                   else wa.SCHEMA}}
    return body


def http(base: str, route: str, body: dict | None = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(base.rstrip("/") + route,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def one_call(base: str, name: str, row: dict, prompt: str, phash: str, rung: int, timeout: float) -> dict:
    """One request -> one journal event in the released shape (wa.latest_results reads it)."""
    event = {"event_id": uuid.uuid4().hex, "query_id": row["query_id"], "prompt_sha256": phash,
             "started_at_utc": wa.now(), "status": "error", "raw_output": "", "error": "",
             "rung": rung}
    t0 = time.monotonic()
    try:
        out = http(base, "/chat/completions", request_body(name, prompt, rung), timeout)
        choice = out["choices"][0]
        msg = choice.get("message") or {}
        usage = out.get("usage") or {}
        event["raw_output"] = msg.get("content") or ""
        event["finish_reason"] = choice.get("finish_reason")
        # Keys the released evaluator sums for its operational table.
        event["response"] = {"model": out.get("model"), "finish_reason": choice.get("finish_reason"),
                             "prompt_eval_count": usage.get("prompt_tokens", 0),
                             "eval_count": usage.get("completion_tokens", 0),
                             "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}
        try:
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"Incomplete/truncated completion (finish_reason={choice.get('finish_reason')})")
            event["parsed"] = wa.parse_verdict(event["raw_output"], {c["row_id"] for c in row["candidates"]})
            event["status"] = "ok"
        except (ValueError, TypeError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:                       # transport / server failure
        event["error"] = f"REQUEST {type(exc).__name__}: {exc}"
    event["latency_ms"] = round((time.monotonic() - t0) * 1000)
    return event


def run_batch(base, name, directory, rows, prompts, hashes, rung, workers, timeout,
              max_request_failures=3, retry_errors=False) -> Counter:
    """Send every row not yet answered (or errored, with retry_errors) at `workers` concurrency."""
    latest = wa.latest_results(directory, rows, hashes)
    todo = [r for r in rows if r["query_id"] not in latest or
            (retry_errors and latest[r["query_id"]]["status"] == "error")]
    stats = Counter()
    if not todo:
        return stats
    request_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one_call, base, name, r, prompts[r["query_id"]], hashes[r["query_id"]],
                            rung, timeout): r for r in todo}
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                ev = fut.result()
                wa.append_event(directory / "responses.jsonl", ev)   # single writer: this thread
                stats[ev["status"]] += 1
                if ev["status"] == "error":
                    stats["truncated"] += "Incomplete/truncated" in ev["error"]
                    print(f"{name}: query {ev['query_id']} ERROR {ev['error'][:160]}", flush=True)
                    if ev["error"].startswith("REQUEST"):
                        request_failures += 1
                        if request_failures >= max_request_failures:
                            raise RuntimeError(f"{request_failures} request failures; server unhealthy")
                if i % 50 == 0 or i == len(todo):
                    print(f"{name}: {i}/{len(todo)} done ({dict(stats)})", flush=True)
        except BaseException:
            for f in futs:
                f.cancel()
            raise
    return stats


# ------------------------------------------------------------------ helpers ----
def load_split(split):
    rows, manifest = wa.load_prepared(DATASET, split)
    template = PROMPT.read_text()
    prompts = {r["query_id"]: wa.render_prompt(template, r) for r in rows}
    return rows, manifest, template, prompts, {k: wa.sha(v) for k, v in prompts.items()}


def server_identity(base, name):
    served = [m.get("id") for m in http(base, "/models")["data"]]
    want = CURATORS[name]["served"]
    if want not in served:
        raise ValueError(f"server at {base} serves {served}, not {want}")
    return {"served_models": served, "base_url": base}


def freeze_protocol(out, split, template, hashes, extra):
    wa.freeze(out / "protocol.json", {
        "protocol": "product-curation-v1-dgx", "split": split,
        "dataset_manifest_sha256": wa.sha((DATASET / "manifest.json").read_bytes()),
        "prompt_template_sha256": wa.sha(template), "prompt_sha256": hashes,
        "released_pipeline_sha256": wa.sha((HERE / "walmart_amazon.py").read_bytes()),
        "dgx_runner_sha256": wa.sha(Path(__file__).read_bytes()),
        "primary_thresholds": {"selection": .86, "abstention": .80},
        "curators": CURATORS, "ladder": LADDER, "pilot_n": PILOT_N, **extra})


def invocation(directory, base, name, rung, args):
    wa.save_json(directory / f"invocation-{uuid.uuid4().hex}.json", {
        "started_at_utc": wa.now(), "host": platform.node(), "platform": platform.platform(),
        "server": server_identity(base, name), "rung": rung, "workers": args.workers,
        "vllm_image": os.environ.get("VLLM_IMAGE", ""), "request_example": request_body(name, "<prompt>", rung)})


def chosen_rung(name):
    f = RUNS / "dev" / name / "chosen_rung.json"
    return json.loads(f.read_text())["rung"] if f.exists() else None


# ------------------------------------------------------------------ stages ----
def stage_prepare(args):
    a = wa.parser().parse_args(["prepare", "--dataset-dir", str(DATASET)] +
                               (["--source-zip", str(args.source_zip)] if args.source_zip else []))
    wa.prepare(a)
    man = json.loads((DATASET / "manifest.json").read_text())
    if RESERVATION.exists():
        ref = json.loads(RESERVATION.read_text())
        if man["membership"] != ref["membership"] or man["artifact_sha256"] != ref["artifact_sha256"]:
            raise ValueError("Reservation differs from the released study's; stop.")
        print("Reservation, retrieval and projected artifacts equal the released study's.")


def stage_pilot(args):
    """Climb the ladder on the first 25 development queries; record the chosen rung."""
    name, base = args.model, args.base_url
    rows, _, template, prompts, hashes = load_split("dev")
    rows = rows[:PILOT_N]
    done = chosen_rung(name)
    if done:
        print(f"{name}: pilot already chose rung {done} -- skip")
        return 0
    tried, rung = [], 1
    while rung is not None:
        tried.append(rung)
        out = RUNS / "dev" / name / f"rung-{rung}"
        rejected = False
        with wa.lock(out):
            freeze_protocol(out, "dev", template, hashes, {"model": name, "rung": rung})
            invocation(out, base, name, rung, args)
            try:
                run_batch(base, name, out, rows, prompts, hashes, rung, args.workers, args.timeout)
            except RuntimeError as exc:
                print(f"{name}: pilot rung {rung} stopped: {exc}", flush=True)
            latest = wa.latest_results(out, rows, hashes)
            rejected = any("HTTP Error 400" in e.get("error", "") for e in latest.values())
        errs = [e for e in latest.values() if e["status"] != "ok"]
        trunc = sum("Incomplete/truncated" in e["error"] for e in errs)
        verdict = {"rung": rung, "valid": len(latest) - len(errs), "n": len(rows),
                   "truncated": trunc, "other_errors": len(errs) - trunc, "schema_rejected": rejected,
                   "decided_at_utc": wa.now(),
                   "basis": "development pilot validity only; no reference labels read"}
        wa.save_json(out / "pilot_result.json", verdict)
        print(f"{name}: pilot rung {rung}: {verdict}", flush=True)
        if not errs and len(latest) == len(rows):
            wa.save_json(RUNS / "dev" / name / "chosen_rung.json", verdict)
            return 0
        # The ladder (module docstring): truncation -> longer cap; other parser failures
        # -> the product schema as a grammar (Muse only; Gemma already has one).
        if rejected:
            rung = {1: 4, 2: 5, 3: 6}.get(rung)
        elif trunc and rung in (1, 4):
            rung += 1
        elif verdict["other_errors"] and name == "muse" and rung in (1, 2):
            rung = 3
        else:
            rung = None
    rungs = tried
    wa.save_json(RUNS / "dev" / name / "pilot_failed.json", {"at_utc": wa.now(), "rungs_tried": rungs})
    print(f"{name}: no rung passed the pilot -- public run abandoned for this curator", flush=True)
    return 3


def stage_run(args):
    name, base = args.model, args.base_url
    rung = chosen_rung(name)
    if rung is None:
        raise ValueError(f"{name}: no passing pilot rung recorded; run the pilot first")
    rows, _, template, prompts, hashes = load_split("test")
    out = RUNS / "test" / name
    with wa.lock(out):
        freeze_protocol(out, "test", template, hashes, {"model": name, "rung": rung})
        invocation(out, base, name, rung, args)
        s1 = run_batch(base, name, out, rows, prompts, hashes, rung, args.workers, args.timeout)
        s2 = run_batch(base, name, out, rows, prompts, hashes, rung, args.workers, args.timeout,
                       retry_errors=True)            # one retry of technical failures
        latest = wa.latest_results(out, rows, hashes)
    c = Counter(e["status"] for e in latest.values())
    print(f"{name}: test {c['ok']} valid, {c['error']} errors, {len(rows) - len(latest)} not run "
          f"(first pass {dict(s1)}, retry {dict(s2)})", flush=True)
    return 0 if c["ok"] == len(rows) else (4 if c["ok"] >= 0.98 * len(rows) else 5)


def stage_status(args):
    for split in ("dev", "test"):
        rows, _, _, _, hashes = load_split(split)
        for name in CURATORS:
            dirs = ([RUNS / "dev" / name / f"rung-{k}" for k in LADDER] if split == "dev"
                    else [RUNS / "test" / name])
            for d in dirs:
                if (d / "responses.jsonl").exists():
                    sub = rows[:PILOT_N] if split == "dev" else rows
                    latest = wa.latest_results(d, sub, hashes)
                    c = Counter(e["status"] for e in latest.values())
                    print(f"{split:4} {name:5} {d.name:7} {c['ok']} valid, {c['error']} errors, "
                          f"{len(sub) - len(latest)} not run")
        if split == "dev":
            for name in CURATORS:
                print(f"dev  {name:5} chosen rung: {chosen_rung(name)}")


def stage_evaluate(args):
    """Released scoring on the test split.  Rows a curator never answered validly stay
    'incomplete' (deferred, never negative); if any exist the strict released evaluator
    is not used and an incomplete-aware report is written instead."""
    rows, _, _, _, hashes = load_split("test")
    out = RUNS / "test"
    results = {n: wa.latest_results(out / n, rows, hashes) for n in wa.MODELS}
    policies = wa.export_curation(out, rows, results)
    incomplete = {n: sum(results[n].get(r["query_id"], {}).get("status") != "ok" for r in rows)
                  for n in wa.MODELS}
    ns = types.SimpleNamespace(dataset_dir=DATASET, split="test")
    if not any(incomplete.values()):
        wa.evaluate(ns, out, rows, results, policies)
    else:
        _, gold = wa.load_gold(DATASET, rows)
        rep = {"split": "test", "n": len(rows), "incomplete": incomplete, "evaluated_at_utc": wa.now(),
               "note": "strict released evaluator not used: some rows have no valid verdict "
                       "(policies treat them as incomplete/deferred)",
               "policies": {k: wa.score_policy(v, gold) for k, v in policies.items()}}
        wa.save_json(out / "evaluation.json", rep)
        print(f"Incomplete-aware evaluation written; incomplete rows {incomplete}")
    ev = json.loads((out / "evaluation.json").read_text())
    for k in ("dual_086_080", "gemma_086_080", "muse_086_080"):
        p = ev["policies"][k]
        print(f"{k:14} retained {p['counts'].get('positive', 0) + p['counts'].get('negative', 0):4}  "
              f"pos {p['positive']['rate']}  neg {p['negative']['rate']}  total {p['total']['rate']}  "
              f"coverage {p['coverage']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prepare", "pilot", "run", "status", "evaluate"])
    ap.add_argument("--model", choices=list(CURATORS))
    ap.add_argument("--base-url", default=os.environ.get("PUBLIC_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--source-zip", type=Path)
    a = ap.parse_args(argv)
    if a.stage in ("pilot", "run") and not a.model:
        ap.error("--model is required for pilot and run")
    return {"prepare": stage_prepare, "pilot": stage_pilot, "run": stage_run,
            "status": stage_status, "evaluate": stage_evaluate}[a.stage](a) or 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
