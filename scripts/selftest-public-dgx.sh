#!/usr/bin/env bash
# Offline self-test of src/public/walmart_amazon_dgx.py against stand-in servers
# (tests/fake_vllm_public.py): exercises the pilot ladder (truncation -> 8,192 cap; parser
# failure -> schema rung for Muse), the concurrent test pass and the released
# evaluator.  No GPU, no models; writes only under a temporary runs directory.
#   bash scripts/selftest-public-dgx.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${GAIME_PYTHON:-$(command -v python || command -v python3)}"
D=src/public/walmart_amazon_dgx.py
TMP="$(mktemp -d)"; export PUBLIC_RUNS_DIR="$TMP/runs"
PG=$(( 18100 + RANDOM % 400 )); PM=$(( PG + 1 ))
FAKE_MODEL=gemma4:31b-bf16-vllm FAKE_MODE=truncate-small "$PY" tests/fake_vllm_public.py "$PG" & G=$!
FAKE_MODEL=muse-glimmer:30b-bf16-vllm FAKE_MODE=garbage-unless-schema "$PY" tests/fake_vllm_public.py "$PM" & M=$!
trap 'kill $G $M 2>/dev/null; rm -rf "$TMP"' EXIT
sleep 2
"$PY" "$D" pilot --model gemma --base-url "http://127.0.0.1:$PG/v1" > "$TMP/g.log"
"$PY" "$D" run   --model gemma --base-url "http://127.0.0.1:$PG/v1" >> "$TMP/g.log"
"$PY" "$D" pilot --model muse  --base-url "http://127.0.0.1:$PM/v1" > "$TMP/m.log" 2>&1
"$PY" "$D" run   --model muse  --base-url "http://127.0.0.1:$PM/v1" >> "$TMP/m.log"
"$PY" "$D" evaluate > "$TMP/e.log"
"$PY" - "$PUBLIC_RUNS_DIR" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
g = json.loads((r / "dev/gemma/chosen_rung.json").read_text())["rung"]
m = json.loads((r / "dev/muse/chosen_rung.json").read_text())["rung"]
ev = json.loads((r / "test/evaluation.json").read_text())
assert g == 2 and m == 3, (g, m)
assert ev["n"] == 1000 and "dual_086_080" in ev["policies"], ev.keys()
print(f"PASS  public DGX runner self-test (ladder gemma->{g}, muse->{m}; 2 x 1,000 test rows; released evaluator)")
PY
