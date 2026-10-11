#!/usr/bin/env bash
# Serve one curator with vLLM exactly as in the paper's DGX Spark run (Section 7).
#
#   bash scripts/serve_vllm.sh gemma      # Gemma 4 31B, bf16, no reasoning parser (thinking off)
#   bash scripts/serve_vllm.sh muse       # Muse Glimmer 30B, bf16, --reasoning-parser muse_glimmer
#
# Runs in the foreground (nohup it, or use scripts/run_public.sh which manages it).
# Served names are what src/public/walmart_amazon_dgx.py expects.  Both models are
# gated on Hugging Face: accept their licences and export HF_TOKEN before the first start.
#
# Settings (identical to the merchant curation cells): dtype bfloat16, --max-model-len 32768,
# --max-num-seqs 16, --gpu-memory-utilization 0.85, prefix caching on, pinned image
# vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174 (vLLM main after
# 2026-08-14, when Muse Glimmer support merged).  Muse's chain of thought is a channel
# message, not <think> tags, so its parser is required; with that parser vLLM's JSON grammar
# is never applied to the answer, which is why the client sends Muse requests with
# structured output off (the pilot ladder's rung 3 forces a schema only if parsing fails).
set -euo pipefail
NAME="${1:-}"
case "$NAME" in
    gemma) HF_MODEL="${HF_MODEL:-google/gemma-4-31B-it}";   SERVED=gemma4:31b-bf16-vllm;       PARSER="" ;;
    muse)  HF_MODEL="${HF_MODEL:-meta-models/Muse-Glimmer-30B}"; SERVED=muse-glimmer:30b-bf16-vllm; PARSER=muse_glimmer ;;
    *) echo "usage: $0 gemma|muse" >&2; exit 2 ;;
esac
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly-e962733e08d10f7ca65dac4df99e116460b8b174}"
PORT="${PORT:-8000}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER="${CONTAINER_NAME:-vllm-public-$NAME}"

ARGS=("$HF_MODEL" --served-model-name "$SERVED" --dtype bfloat16 --max-model-len 32768
      --max-num-seqs 16 --gpu-memory-utilization 0.85 --trust-remote-code
      --enable-prefix-caching --host 0.0.0.0 --port "$PORT")
[[ -n "$PARSER" ]] && ARGS+=(--reasoning-parser "$PARSER")

echo "=== vLLM $SERVED  ($HF_MODEL, $VLLM_IMAGE, port $PORT) ==="
exec $DOCKER_CMD run --rm --name "$CONTAINER" --ipc=host --gpus all -p "${PORT}:${PORT}" \
    -v "$HF_CACHE:/root/.cache/huggingface" ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    "$VLLM_IMAGE" "${ARGS[@]}"
