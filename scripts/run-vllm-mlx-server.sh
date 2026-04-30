#!/usr/bin/env bash
# Launch vllm-mlx serving the default opus-mxfp4 model with continuous
# batching + Prometheus metrics enabled. Designed to be paired with
# scripts/benchmark-vllm-mlx-client.py running in another terminal.
#
# Usage:
#   scripts/run-vllm-mlx-server.sh                 # use defaults
#   scripts/run-vllm-mlx-server.sh --port 8001     # override flag (passed to vllm-mlx)
#   MODEL=~/models/<other> scripts/run-vllm-mlx-server.sh
#
# Note on --scheduling-policy: upstream vllm-mlx v0.2.9 from PyPI does
# NOT accept this flag (the priority feature is unwired). The local fork
# at /Users/edward/workspace/forks/vllm-mlx (branch feat/priority-scheduling)
# does accept it. Do NOT add the flag to this launcher unconditionally
# — a fresh PyPI reinstall would make the launcher exit non-zero. Pass
# the flag at the CLI when invoking this script if you want priority:
#   scripts/run-vllm-mlx-server.sh --scheduling-policy priority
# See research/vllm-mlx-playbook.md "Client prioritization" for details.
set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen3.6-35b-opus-abl-mxfp4-mlx}"
PORT="${PORT:-8000}"

if ! command -v vllm-mlx >/dev/null 2>&1; then
  echo "vllm-mlx not on PATH (try: pip install vllm-mlx)" >&2
  exit 127
fi

# --max-num-seqs is pinned at 32 (the throughput knee on M3 Max 48 GB).
# Tried 64 briefly; OOMed at C=32 during repeated 512-token sweeps and
# advice from another model agreed on dropping back to 32. Raise again
# only with a corresponding bump to --max-cache-blocks and after
# verifying KV memory headroom at 512+ tokens.
echo "[vllm-mlx] serving $MODEL on port $PORT (continuous-batching + metrics)"
exec vllm-mlx serve "$MODEL" \
  --port "$PORT" \
  --continuous-batching \
  --enable-metrics \
  --served-model-name opus-mxfp4 \
  --chunked-prefill-tokens 1024 \
  --max-num-seqs 32 \
  --use-paged-cache \
  "$@"
