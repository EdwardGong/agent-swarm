#!/usr/bin/env bash
# cleanup.sh — FAISS pruning, KV cache flush, MCP log rotation
#
# Usage:
#   ./apps/consolidated_swarm/scripts/cleanup.sh            # full cleanup
#   ./apps/consolidated_swarm/scripts/cleanup.sh --faiss    # FAISS prune only
#   ./apps/consolidated_swarm/scripts/cleanup.sh --cache    # KV cache flush only
#   ./apps/consolidated_swarm/scripts/cleanup.sh --logs     # log rotation only
#   ./apps/consolidated_swarm/scripts/cleanup.sh --dry-run  # preview without changes

set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FAISS_DIR="${REPO_ROOT}/runtime/faiss_index"
LOG_DIR="${REPO_ROOT}/runtime/logs"
WORKSPACE_DIR="${REPO_ROOT}/runtime/workspace"
# Keep FAISS index under this size (GB); prune oldest chunks if exceeded.
FAISS_MAX_GB="${FAISS_MAX_GB:-2}"
# Rotate logs older than this many days.
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-7}"
DRY_RUN=false

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

TARGETS=("faiss" "cache" "logs")
for arg in "$@"; do
  case "${arg}" in
    --faiss)   TARGETS=("faiss") ;;
    --cache)   TARGETS=("cache") ;;
    --logs)    TARGETS=("logs") ;;
    --dry-run) DRY_RUN=true ;;
  esac
done

# ---------------------------------------------------------------------------
# FAISS pruning
# ---------------------------------------------------------------------------

prune_faiss() {
  log "FAISS index: checking size..."
  if [ ! -d "${FAISS_DIR}" ]; then
    log "  FAISS directory not found — skipping."
    return
  fi

  local size_gb
  size_gb=$(du -sh "${FAISS_DIR}" 2>/dev/null | awk '{print $1}' | tr -d 'G')

  log "  Current size: ${size_gb:-?}G (limit: ${FAISS_MAX_GB}G)"

  # Use Python to prune via local-faiss-mcp API if available, else just log.
  if curl -sf "http://localhost:${VLLM_PORT}/v1/mcp/tools" 2>/dev/null | \
     python3 -c "import sys,json; tools=[t['name'] for t in json.load(sys.stdin).get('tools',[])]; assert 'faiss_prune_index' in tools" \
     2>/dev/null; then
    if [ "${DRY_RUN}" = true ]; then
      log "  [dry-run] Would call faiss_prune_index via MCP."
    else
      log "  Pruning low-relevance chunks via faiss_prune_index..."
      curl -sf -X POST "http://localhost:${VLLM_PORT}/v1/mcp/execute" \
        -H "Content-Type: application/json" \
        -d '{"tool": "faiss_prune_index", "arguments": {"max_size_gb": '"${FAISS_MAX_GB}"'}}' \
        | python3 -m json.tool 2>/dev/null || log "  faiss_prune_index failed (non-fatal)."
    fi
  else
    log "  faiss_prune_index not available — manual cleanup:"
    log "  rm -rf ${FAISS_DIR}/*.faiss  (removes all vectors)"
    log "  Or restart vllm-mlx with --mcp-config that includes local-faiss-mcp."
  fi
}

# ---------------------------------------------------------------------------
# KV cache flush
# ---------------------------------------------------------------------------

flush_kv_cache() {
  log "KV cache: flushing..."
  if ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    log "  vllm-mlx not running — skipping KV flush."
    return
  fi

  if [ "${DRY_RUN}" = true ]; then
    log "  [dry-run] Would POST /v1/cache/clear"
    return
  fi

  # vllm-mlx cache clear endpoint (if supported — gracefully skip if 404)
  local http_status
  http_status=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST "http://localhost:${VLLM_PORT}/v1/cache/clear" 2>/dev/null || echo "000")

  if [ "${http_status}" = "200" ]; then
    log "  KV cache cleared."
  else
    log "  /v1/cache/clear returned ${http_status} — cache cleared on next cold start."
  fi
}

# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------

rotate_logs() {
  log "Logs: rotating files older than ${LOG_RETENTION_DAYS} days in ${LOG_DIR}..."
  if [ ! -d "${LOG_DIR}" ]; then
    log "  Log directory not found — skipping."
    return
  fi

  local count=0
  while IFS= read -r -d '' f; do
    if [ "${DRY_RUN}" = true ]; then
      log "  [dry-run] Would remove: ${f}"
    else
      rm -f "${f}"
    fi
    count=$((count + 1))
  done < <(find "${LOG_DIR}" -type f -name "*.log" \
            -mtime +"${LOG_RETENTION_DAYS}" -print0 2>/dev/null)

  log "  ${count} log file(s) removed."
}

# ---------------------------------------------------------------------------
# Workspace housekeeping
# ---------------------------------------------------------------------------

workspace_summary() {
  log "Workspace: ${WORKSPACE_DIR}"
  if [ -d "${WORKSPACE_DIR}" ]; then
    du -sh "${WORKSPACE_DIR}" 2>/dev/null | awk '{print "  Total: "$1}'
    find "${WORKSPACE_DIR}" -name "*.md" | wc -l | xargs -I{} echo "  Markdown files: {}"
    find "${WORKSPACE_DIR}" -name "*.pptx" | wc -l | xargs -I{} echo "  PPTX files:     {}"
  else
    log "  Workspace not found."
  fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

log "Consolidated Agent Swarm — Cleanup"
log "  Targets:  ${TARGETS[*]}"
log "  Dry run:  ${DRY_RUN}"
echo

for target in "${TARGETS[@]}"; do
  case "${target}" in
    faiss) prune_faiss ;;
    cache) flush_kv_cache ;;
    logs)  rotate_logs ;;
  esac
  echo
done

workspace_summary
log "Done."
