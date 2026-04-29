#!/usr/bin/env bash
# Launch vllm-mlx serving the default opus-mxfp4 model with continuous
# batching + Prometheus metrics enabled. Designed to be paired with
# scripts/benchmark-vllm-mlx-client.py running in another terminal.
#
# Usage:
#   scripts/run-vllm-mlx-server.sh                 # use defaults
#   scripts/run-vllm-mlx-server.sh --port 8001     # override flag (passed to vllm-mlx)
#   MODEL=~/models/<other> scripts/run-vllm-mlx-server.sh
set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen3.6-35b-opus-abl-mxfp4-mlx}"
PORT="${PORT:-8000}"

if ! command -v vllm-mlx >/dev/null 2>&1; then
  echo "vllm-mlx not on PATH (try: pip install vllm-mlx)" >&2
  exit 127
fi

echo "[vllm-mlx] serving $MODEL on port $PORT (continuous-batching + metrics)"
exec vllm-mlx serve "$MODEL" \
  --port "$PORT" \
  --continuous-batching \
  --metrics \
  --served-model-name opus-mxfp4 \
  "$@"
