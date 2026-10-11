#!/usr/bin/env bash
# End-to-end public validation (paper Section 7) on one GPU workstation:
#   prepare -> [serve gemma -> pilot -> run -> stop] -> [serve muse -> pilot -> run -> stop] -> evaluate
#
#   bash scripts/run_public.sh                 # everything; resumes where a previous run stopped
#   bash scripts/run_public.sh gemma           # one curator only (prepare first if needed)
#   bash scripts/run_public.sh evaluate        # score once both curators have 1,000/1,000 rows
#
# Each stage is idempotent: the runner skips rows that already have a valid answer, and the
# pilot's chosen rung is reused once recorded.  Output lands in data/public/walmart-amazon/runs/
# product-v1-dgx/ (override with PUBLIC_RUNS_DIR).  Reference matches are read only by evaluate.
# Expect roughly 70 min (Gemma) and 95 min (Muse) for the 1,000 test queries on a DGX Spark.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$(command -v python3 || command -v python)}"
RUNNER=src/public/walmart_amazon_dgx.py
PORT="${PORT:-8000}"
export PUBLIC_VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174}"
LOGS="${LOGS:-$ROOT/logs}"; mkdir -p "$LOGS"

wait_for_server() {   # until /v1/models answers (model load takes a few minutes), 40 min max
    local i
    for i in $(seq 1 480); do
        curl -fsS "$PUBLIC_VLLM_BASE_URL/models" > /dev/null 2>&1 && return 0
        kill -0 "$1" 2>/dev/null || { echo "vLLM exited early; see $LOGS/vllm-$2.log" >&2; return 1; }
        sleep 5
    done
    echo "vLLM did not come up in 40 min" >&2; return 1
}

curator() {           # curator gemma|muse
    local name="$1" pid
    echo "=== $name: starting vLLM ($(date '+%F %T'))"
    PORT="$PORT" CONTAINER_NAME="vllm-public-$name" bash scripts/serve_vllm.sh "$name" > "$LOGS/vllm-$name.log" 2>&1 & pid=$!
    trap 'docker stop "vllm-public-'"$name"'" > /dev/null 2>&1 || true' RETURN
    wait_for_server "$pid" "$name"
    echo "=== $name: pilot (25 development queries, adjustment ladder)"
    "$PY" "$RUNNER" pilot --model "$name" 2>&1 | tee -a "$LOGS/pilot-$name.log"
    echo "=== $name: test run (1,000 reserved queries, 16 concurrent) ($(date '+%F %T'))"
    "$PY" "$RUNNER" run --model "$name" 2>&1 | tee -a "$LOGS/run-$name.log"
    echo "=== $name: done ($(date '+%F %T')); stopping vLLM"
    docker stop "vllm-public-$name" > /dev/null 2>&1 || true
    wait "$pid" 2>/dev/null || true
}

case "${1:-all}" in
    all)      "$PY" "$RUNNER" prepare; curator gemma; curator muse; "$PY" "$RUNNER" evaluate ;;
    gemma|muse) [[ -f data/public/walmart-amazon/manifest.json ]] || "$PY" "$RUNNER" prepare; curator "$1" ;;
    prepare|status|evaluate) "$PY" "$RUNNER" "$1" ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac
