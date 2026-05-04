#!/usr/bin/env bash
# monitor.sh — Real-time monitoring for the Consolidated Agent Swarm
#
# Usage:
#   ./apps/consolidated_swarm/scripts/monitor.sh          # one-shot snapshot
#   ./apps/consolidated_swarm/scripts/monitor.sh --watch  # refresh every 5s
#   ./apps/consolidated_swarm/scripts/monitor.sh --mcp    # MCP tool status only

VLLM_PORT="${VLLM_PORT:-8000}"
REFRESH_INTERVAL=5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

hr() { printf '%0.s─' {1..60}; echo; }

vllm_status() {
  echo "── vllm-mlx ────────────────────────────────────────────"
  if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "  Status  : ✅  Running on port ${VLLM_PORT}"
    # Model list
    curl -s "http://localhost:${VLLM_PORT}/v1/models" | \
      python3 -c "import sys,json; d=json.load(sys.stdin); [print('  Model   :', m['id']) for m in d.get('data',[])]" \
      2>/dev/null || true
    # MCP status
    if curl -sf "http://localhost:${VLLM_PORT}/v1/mcp/status" > /dev/null 2>&1; then
      echo "  MCP     : ✅  Active"
      curl -s "http://localhost:${VLLM_PORT}/v1/mcp/status" | \
        python3 -m json.tool 2>/dev/null | grep -E '(server|tools|status)' | head -20 || true
    else
      echo "  MCP     : ⚠️  /v1/mcp/status not responding"
    fi
  else
    echo "  Status  : ❌  Not running on port ${VLLM_PORT}"
    echo "  Start   : ./apps/consolidated_swarm/scripts/start.sh"
  fi
}

mcp_tools() {
  echo "── MCP Tools ───────────────────────────────────────────"
  for profile in core research code crm content; do
    count=$(curl -s "http://localhost:${VLLM_PORT}/v1/mcp/tools?profile=${profile}" 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('tools',[])))" 2>/dev/null || echo "?")
    printf "  %-10s : %s tools\n" "${profile}" "${count}"
  done
}

memory_status() {
  echo "── Memory ──────────────────────────────────────────────"
  # Unified memory pressure via vm_stat
  vm_stat 2>/dev/null | awk '
    /Pages free/          { free=$NF+0 }
    /Pages wired down/    { wired=$NF+0 }
    /Pages active/        { active=$NF+0 }
    /Pages inactive/      { inactive=$NF+0 }
    /Pages occupied/      { compressed=$NF+0 }
    END {
      page=16384
      total_gb=(free+wired+active+inactive+compressed)*page/1024/1024/1024
      used_gb=(wired+active)*page/1024/1024/1024
      printf "  Used    : %.1f GB / %.1f GB\n", used_gb, total_gb
    }
  ' || true

  # Swap
  local swap
  swap=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}')
  if [ -n "${swap}" ]; then
    echo "  Swap in : ${swap}"
  fi

  # Memory pressure colour (green = ok, yellow = compressing, red = swapping)
  local pressure
  pressure=$(memory_pressure 2>/dev/null | grep "System-wide" | head -1 || echo "unknown")
  echo "  Pressure: ${pressure}"
}

gpu_status() {
  echo "── GPU / Neural Engine ─────────────────────────────────"
  # ioreg GPU mapped memory
  local gpu_mem
  gpu_mem=$(ioreg -r -d 1 -c AGXAccelerator 2>/dev/null | \
    awk -F'"' '/PerformanceStatistics/{p=1} p && /GPU Mapped Memory/{gsub(/[^0-9]/,"",$0); printf "%.1f GB\n", $0/1024/1024/1024; p=0}' | head -1)
  [ -n "${gpu_mem}" ] && echo "  GPU mem : ${gpu_mem}" || echo "  GPU mem : (run as sudo for ioreg stats)"

  # powermetrics if available (requires sudo normally)
  echo "  Tip: sudo powermetrics --samplers gpu_power -i 1000 -n 1 | grep -E '(GPU|ANE)'"
}

process_status() {
  echo "── Processes ───────────────────────────────────────────"
  ps aux | awk '/vllm.mlx|vllm-mlx/ && !/awk/' | \
    awk '{printf "  %-8s %5s%%CPU %5s%%MEM  %s\n", $1, $3, $4, substr($0,index($0,$11),60)}' || true
}

snapshot() {
  echo
  echo "Consolidated Agent Swarm — Monitor  $(date '+%Y-%m-%d %H:%M:%S')"
  hr
  vllm_status
  echo
  mcp_tools
  echo
  memory_status
  echo
  gpu_status
  echo
  process_status
  echo
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if [ "${1:-}" = "--mcp" ]; then
  mcp_tools
elif [ "${1:-}" = "--watch" ]; then
  while true; do
    clear
    snapshot
    sleep "${REFRESH_INTERVAL}"
  done
else
  snapshot
fi
