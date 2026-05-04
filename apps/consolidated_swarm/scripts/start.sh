#!/usr/bin/env bash
# start.sh — Launch the Consolidated Agent Swarm (M5 Max 128GB)
#
# Usage:
#   ./apps/consolidated_swarm/scripts/start.sh                 # starts server + interactive swarm
#   ./apps/consolidated_swarm/scripts/start.sh --server-only   # starts vllm-mlx only
#   ./apps/consolidated_swarm/scripts/start.sh --query "..."   # runs a single query
#
# Prerequisites:
#   pip install vllm-mlx
#   export VLLM_MLX_MODEL_PATH=/path/to/qwen3.6-35b-opus-abl-mxfp4-mlx
#   (or set MODEL_PATH below)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MCP_CONFIG_DIR="${REPO_ROOT}/mcp-configs"
MODEL_PATH="${VLLM_MLX_MODEL_PATH:-${HOME}/models/qwen3.6-35b-opus-abl-mxfp4-mlx}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_PID_FILE="/tmp/vllm-mlx-consolidated.pid"
LOG_DIR="${REPO_ROOT}/runtime/logs"

mkdir -p "${LOG_DIR}" \
         "${REPO_ROOT}/runtime/faiss_index" \
         "${REPO_ROOT}/runtime/workspace" \
         "${REPO_ROOT}/runtime"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
  local max_wait=120
  local waited=0
  log "Waiting for vllm-mlx to be ready on port ${VLLM_PORT}..."
  while ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge $max_wait ]; then
      log "ERROR: vllm-mlx did not become ready within ${max_wait}s — check ${LOG_DIR}/vllm-mlx.log"
      exit 1
    fi
  done
  log "vllm-mlx ready (${waited}s)."
}

stop_server() {
  if [ -f "${VLLM_PID_FILE}" ]; then
    local pid
    pid="$(cat "${VLLM_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "Stopping vllm-mlx (PID ${pid})..."
      kill "${pid}"
    fi
    rm -f "${VLLM_PID_FILE}"
  fi
}

# ---------------------------------------------------------------------------
# Check model path
# ---------------------------------------------------------------------------

if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: Model not found at ${MODEL_PATH}"
  echo "Set VLLM_MLX_MODEL_PATH to the directory containing your"
  echo "qwen3.6-35b-opus-abl-mxfp4-mlx model weights."
  exit 1
fi

# ---------------------------------------------------------------------------
# Launch vllm-mlx
# ---------------------------------------------------------------------------

log "Starting vllm-mlx server..."
log "  Model:  ${MODEL_PATH}"
log "  Port:   ${VLLM_PORT}"
log "  MCP:    ${MCP_CONFIG_DIR}/mcp-core.json"

vllm-mlx serve "${MODEL_PATH}" \
  --port "${VLLM_PORT}" \
  --continuous-batching \
  --use-paged-cache \
  --cache-memory-percent 0.15 \
  --embedding-model mlx-community/all-MiniLM-L6-v2-4bit \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --mcp-config "${MCP_CONFIG_DIR}/mcp-core.json" \
  > "${LOG_DIR}/vllm-mlx.log" 2>&1 &

echo $! > "${VLLM_PID_FILE}"
log "vllm-mlx PID: $(cat ${VLLM_PID_FILE})"

trap stop_server EXIT INT TERM

wait_for_server

# ---------------------------------------------------------------------------
# Launch swarm
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--server-only" ]; then
  log "Server-only mode. vllm-mlx running on port ${VLLM_PORT}."
  log "Press Ctrl-C to stop."
  wait "$(cat ${VLLM_PID_FILE})"
elif [ "${1:-}" = "--query" ] && [ -n "${2:-}" ]; then
  log "Running query: ${2}"
  PYTHONPATH="${REPO_ROOT}" python -m apps.consolidated_swarm.main "${2}"
else
  log "Starting interactive swarm..."
  PYTHONPATH="${REPO_ROOT}" python -m apps.consolidated_swarm.main --interactive
fi
