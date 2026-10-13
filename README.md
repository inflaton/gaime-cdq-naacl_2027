# Public validation of agreement-based curation on Walmart–Amazon

Code, protocol records and results for the **Public Validation** section of the paper
*(under review, NAACL 2027 Industry Track; title and authors withheld during review)*. The paper's
main experiments use merchant data that cannot be released. This repository repeats the
*curation step* — two local LLM judges labelling retrieval candidates, with labels kept only
where both judges agree above a confidence gate — on the public Walmart–Amazon
entity-resolution release, with the same bf16 models, settings and gates, on the same
workstation (NVIDIA DGX Spark). Everything needed to re-run it and every output of the run
reported in the paper is here, except the models themselves (downloaded from Hugging Face)
and the dataset archive (downloaded from its original host and verified by checksum).

## Result

1,000 reserved test queries, scored against the release's own match list. Cells are retained
labels with purity (%); coverage is retained queries / 1,000. Policies retain different
subsets, so this is not a matched-coverage comparison.

| Policy | Positives | Negatives | Purity (%) | 95% Wilson | Cov. (%) |
|---|---|---|---|---|---|
| Pair (0.86, 0.80) — paper's gates | 279 (98.92) | 588 (97.79) | **98.15** | [97.0, 98.9] | 86.7 |
| Gemma 4 31B alone | 409 (91.69) | 588 (97.79) | 95.29 | [93.8, 96.4] | 99.7 |
| Muse Glimmer 30B alone | 280 (98.93) | 671 (89.87) | 92.53 | [90.7, 94.0] | 95.1 |
| Pair (0.80, 0.80) | 328 (98.48) | 588 (97.79) | 98.03 | [96.9, 98.8] | 91.6 |
| Pair (0.86, 0.86) | 279 (98.92) | 591 (97.46) | 97.93 | [96.8, 98.7] | 87.0 |

Paired-bootstrap purity gain of the pair over each judge alone (2,000 resamples over queries,
seed 42): **+2.87 [1.80, 3.99]** over Gemma and **+5.62 [4.25, 7.08]** over Muse. The judges
fail on opposite polarities (Gemma's errors are false positives, Muse's false negatives) and
each vetoes the other's: at a 0.80 gate 48 Gemma and 73 Muse verdicts are wrong, 20 of them
shared. Retrieval recovers an acceptable candidate for all 392 matchable queries (0 misses);
absence from the released matches is read as *no match* for the other 608.

Operational: both judges answered 1,000/1,000 test queries with 0 failed requests at the
first rung of the adjustment ladder (4,096-token cap, default structured-output setting).
Gemma took 68 min of wall time (median final-call latency 62.6 s at 16 concurrent;
3.07 M input / 82 k output tokens); Muse took 94 min (84.9 s; 2.85 M / 254 k). All numbers
are in `results/product-v1-dgx/test/evaluation.json`; `paper/public_values.py` renders the
paper's table and macros from it (`python3 paper/public_values.py --from-json --check`).

## Protocol (fixed before any inference)

* **Data.** Walmart–Amazon structured release (DeepMatcher data), archive pinned by SHA-256
  `c49a7558…a71c85` with per-member hashes; see `SOURCE_URL` in `src/public/walmart_amazon.py`.
  The 2,554 Walmart rows are queries, the 22,074 Amazon rows the catalog. 1,000 test and
  100 development queries are reserved by `sha256(seed:query_id)` (seed 20260914),
  independently of labels; the reservation summary with the reserved IDs is
  `src/public/walmart-amazon-reservation.json` (written 14 Sep 2026 by an earlier revision of
  the same pipeline; the runner checks that `prepare` reproduces it exactly).
* **Retrieval.** BM25 (k1 = 1.2, b = 0.75) over title, brand and model number, top-20
  candidates per query in ranked order, catalog identifiers hidden from the judges. Labels are
  never used for selection or retrieval.
* **Prompt.** `data/prompts/product_curation_v1.txt` (SHA-256 `4cee9041…392960`), a
  product-specific rubric replacing the merchant prompt; the judge returns
  `{row_id, match_confidence, evidence}` (schema in `walmart_amazon.py`, `SCHEMA`). `row_id` is
  the selected candidate's identifier; the merchant judge's verdict carries the same field, which
  the paper writes as `merchant_id` (the product team's name for it).
* **Judges and settings** — identical to the paper's merchant curation cells:

  | | Gemma 4 31B | Muse Glimmer 30B |
  |---|---|---|
  | Hugging Face model | `google/gemma-4-31B-it` | `meta-models/Muse-Glimmer-30B` |
  | Precision | bf16 | bf16 |
  | Reasoning | thinking **off** (`enable_thinking: false`) | `reasoning_strength: low`, `--reasoning-parser muse_glimmer` |
  | Structured output | `response_format` = JSON schema (strict) | off (parser validates) |
  | Sampling | temperature 0, seed 42, repetition_penalty 1.12 | same |
  | Output cap | 4,096 tokens (ladder rung 1) | same |
  | Concurrency | 16 requests | 16 |
  | vLLM | `vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174`, `--dtype bfloat16 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 --enable-prefix-caching --trust-remote-code` | same |

* **Gates.** Selection gate 0.86, abstention gate 0.80 (the paper's production gates). A query
  is *retained* when both judges select the same candidate with confidence ≥ 0.86 (positive)
  or both abstain with confidence ≥ 0.80 (negative); otherwise it is deferred.
* **Adjustment ladder (development queries only).** 25 development queries are run first.
  Rung 1 is the cap/structured setting above; rung 2 raises the cap to 8,192 if any pilot
  response is truncated; rung 3 forces the schema for Muse if verdicts fail the parser for
  another reason; rungs 4–6 repeat 1–3 with the schema written as an explicit `anyOf` if the
  server rejects it. The test run uses the first rung with 25/25 valid verdicts; no rung
  passing abandons the run. Reference labels are never read while choosing (see
  `results/product-v1-dgx/dev/*/chosen_rung.json`). Both judges passed at rung 1.
* **Validation and scoring.** The released parser (exact schema, exact candidate IDs,
  confidence in [0, 1], null confidence on abstention) is applied to both judges. A truncated
  or unparseable response is a technical failure (retried once), never a negative label.
  `evaluate` is the only stage that opens `matches.csv`; it writes `evaluation.json`,
  `curation.csv`, `accepted.csv` and `evaluation_rows.csv`. Intervals: Wilson 95% for
  purities, paired percentile bootstrap (2,000 resamples over queries, seed 42) for
  differences.

The exact protocol, including every prompt's hash and the pinned hashes of the dataset
manifest, prompt template, released pipeline and DGX runner, is frozen in
`results/product-v1-dgx/test/{gemma,muse}/protocol.json` before the first request.

## Reproduce

Requirements: Linux with Docker and an NVIDIA GPU with enough memory for a 31B model in bf16
plus a 32k-token KV cache at 16 concurrent requests (the paper used a DGX Spark, 128 GB unified
memory; models are served one at a time), Python ≥ 3.10 (standard
library only; no third-party packages), `curl`, and a Hugging Face token with access to both
gated models (`export HF_TOKEN=…`; it is passed to the container only as an environment
variable and never written to disk by these scripts).

From the repository root, run:

```bash
python3 src/public/walmart_amazon_dgx.py prepare   # download the 13.8 MB archive, verify, reserve, retrieve
bash scripts/selftest-public-dgx.sh                # optional: runner + evaluator against stand-in servers, no GPU (~10 s)
bash scripts/run_public.sh                         # Gemma pilot+run -> Muse pilot+run -> evaluate (~3 h)
```

`run_public.sh` is a thin orchestration of the Python runner; the stages can be run by hand:

```bash
python3 src/public/walmart_amazon_dgx.py prepare               # download (13.8 MB), verify, reserve, retrieve
bash scripts/serve_vllm.sh gemma &                             # vLLM, port 8000 (serves gemma4:31b-bf16-vllm)
python3 src/public/walmart_amazon_dgx.py pilot --model gemma   # 25 dev queries -> dev/gemma/chosen_rung.json
python3 src/public/walmart_amazon_dgx.py run   --model gemma   # 1,000 test queries -> test/gemma/responses.jsonl
docker stop vllm-public-gemma
bash scripts/serve_vllm.sh muse &  # ... then pilot and run with --model muse, docker stop vllm-public-muse
python3 src/public/walmart_amazon_dgx.py status
python3 src/public/walmart_amazon_dgx.py evaluate              # -> test/evaluation.json (reads matches.csv)
python3 paper/public_values.py                                 # paper table/macros from the new evaluation.json
```

Outputs go to `data/public/walmart-amazon/runs/product-v1-dgx/` (set `PUBLIC_RUNS_DIR` to
write elsewhere, e.g. to compare against `results/`). Stages are resumable: rows with a valid
answer are skipped, and a recorded pilot rung is reused. Greedy decoding at temperature 0 is
deterministic for a fixed vLLM build and batch composition, but concurrent batching can change
floating-point reduction order, so individual verdicts near the gates may differ between
re-runs; the aggregate result should reproduce within the reported intervals.

## Layout

```
src/public/walmart_amazon.py           released public pipeline (reservation, BM25, prompt, parser,
                                       policies, scorer) — vendored byte-identical; its own CLI
                                       defaults target an Ollama-style local server and are not
                                       used here (the DGX runner passes explicit paths)
src/public/walmart_amazon_dgx.py       vLLM client + concurrency + adjustment ladder (this run)
src/public/walmart-amazon-reservation.json  reserved test/dev query IDs summary (hash-based)
data/prompts/product_curation_v1.txt   the judge prompt template
scripts/serve_vllm.sh                  vLLM launcher with the exact flags, one judge at a time
scripts/run_public.sh                  end-to-end orchestration (resumable)
scripts/selftest-public-dgx.sh         offline self-test with tests/fake_vllm_public.py (no GPU)
tests/fake_vllm_public.py              stand-in OpenAI-compatible server for the self-test
results/product-v1-dgx/                the paper's run, complete:
  dev/{gemma,muse}/chosen_rung.json      ladder decision (rung 1 for both)
  dev/{gemma,muse}/rung-1/               pilot responses.jsonl, pilot_result.json, protocol/invocation
  test/{gemma,muse}/responses.jsonl      every request/response event (raw outputs, latencies, tokens)
  test/{gemma,muse}/protocol.json        frozen protocol and hashes; invocation-*.json: host, server, image
  test/evaluation.json                   all reported numbers; curation.csv / accepted.csv /
                                         evaluation_rows.csv: per-query decisions and scoring
paper/public_values.py                 renders the paper's table (table_public.tex) and macros
                                       (data/public_values.tex); --from-json --check verifies them
paper/data/public_evaluation.json      copy of test/evaluation.json used by the paper build
```

`responses.jsonl` is an append-only journal (one JSON event per request, including the
single retry pass, which added no events here since nothing failed); `latest_results` in the
released pipeline resolves the final verdict per query.

## What is not included

* The merchant benchmark, candidate pools, merchant prompt, production labels, the distilled
  student and its training data — the paper's main experiments. Those contain merchant records
  that can identify individuals and cannot be released. This repository transfers the curation
  step only; no student is trained on the public data.
* Model weights (gated on Hugging Face) and the Walmart–Amazon archive (downloaded by
  `prepare` and verified against the pinned checksums).
* `docs/walmart-amazon.md`, referenced in the vendored pipeline's docstring, belongs to the
  pipeline's original repository; this README covers the protocol as run here.

## License

Original code, scripts, the prompt template, documentation and author-created analysis
artifacts are released under the [BSD 3-Clause License](LICENSE), the same software license
used by [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/LICENSE).

This license does not relicense Walmart–Amazon source records, reference labels or other
third-party content, including source content reproduced in the released results. The raw
dataset archive contains no separate license notice; DeepMatcher's software license is not
treated here as a license grant for that data. Model weights retain their providers' terms.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and license scope.
