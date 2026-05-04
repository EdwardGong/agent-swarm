# Resource Optimization — Apple Silicon

Running a multi-agent swarm on local hardware requires careful resource
management.  This guide documents the optimized configuration for this
project on macOS with Apple Silicon, covering memory budgeting, model
selection, inference tuning, and operational tooling.

All benchmarks were taken on a **MacBook Pro M3 Max / 48 GB**.

---

## Inference Engine Comparison

MLX is the fastest inference engine on Apple Silicon. Benchmarked on M3 Max 48GB:

- **MLX** (`mlx_vlm`): ~1.8× faster than llama.cpp, ~2.7× faster than Ollama
- **llama.cpp** (`llama-server`): Good for quantized KV cache, direct GGUF support
- **Ollama**: Convenient but slowest; bundled llama.cpp doesn't support `qwen35` arch for imported GGUFs with vision

**Recommendation**: Use `mlx_vlm` for all inference. Serve via `mlx_vlm.server` for OpenAI-compatible API.

---

## MLX Model Benchmarks (M3 Max 48GB, thinking OFF, 1024 tokens)

All models are Qwen3.6 architecture. MoE = 35B total / 3B active per token.

**MoE Models (35B-A3B)**:
- `opus-abl-4bit-vlm` (affine, 4.649 BPW): **66.8 tok/s**, 20.6GB ← BEST all-rounder
- `opus-abl-mxfp4-vlm` (MXFP4, 4.402 BPW): 65.2 tok/s, 19.5GB ← slower than affine (delete candidate)
- `abl-4.4bit-msq` (MSQ, 4.4 BPW): 70.5 tok/s, 20.9GB ← fast but degrades at long context
- `optiq-4bit` (OptIQ, ~4 BPW): 69.3 tok/s, 21.3GB ← no vision
- `mxfp4` (official, ~4.25 BPW): 88.3 tok/s, 18.6GB ← fastest but not abliterated/distilled

**Dense Models (27B)**:
- `27b-4bit` (official): 20.2 tok/s, 16.3GB
- `27b-abl-4.5bit-msq` (abliterated MSQ): 19.1 tok/s, 17.3GB

**Key findings**:
- MoE is 3-4× faster than dense on Apple Silicon (67-88 tok/s vs 15-20 tok/s)
- MXFP4 on Opus-abliterated weights is slower than affine — distillation/abliteration shifts weight distributions unfavorably for power-of-two scaling
- Official MXFP4 is 25% faster due to less data per token, but has 3.3 PPL quality gap
- MSQ degrades at long context (55 tok/s at 2048) while MXFP4 stays flat (72 tok/s)
- Thinking mode slows generation 10-17% — disable with `enable_thinking=False`
- `quant-predicate` only works with affine mode, not MXFP4
- `mlx_vlm.convert` preserves vision; `mlx_lm.convert` strips it

**Conversion command** (affine 4-bit VLM, the winner):
```bash
python3 -m mlx_vlm convert \
  --hf-path huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated \
  --mlx-path ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx \
  -q --q-bits 4 --q-group-size 32 --dtype bfloat16
```

**Benchmark script**: `scripts/benchmark-mlx.py`
**Model inventory**: `scripts/model-inventory.sh`

---

## 128GB M5 Architecture Plan

> **Implementation:** `apps/consolidated_swarm/` — runs the full consolidated swarm
> with native MCP support via vllm-mlx.

### Runtime: vllm-mlx (replaces Ollama for this config)

| Model | Role | Size | Port | Speed |
|-------|------|------|------|-------|
| `qwen3.6-35b-opus-abl-mxfp4` | All tiers (workers + orchestrator) | ~25 GB | 8000 | ~65-90 tok/s |
| (optional) `qwen3.6-27b-instruct-Q4_K_M` dense | Orchestrator upgrade | ~16 GB | 8001 | ~20 tok/s |

On 128 GB, both models can run simultaneously. The MoE 35B model serves all
worker tiers by default. Swap the orchestrator tier to the dense 27B model
(port 8001) for tasks requiring stronger single-turn reasoning.

### Memory Budget

| Component | Size |
|-----------|------|
| `qwen3.6-35b-opus-abl-mxfp4` weights | ~25 GB |
| KV cache (`--cache-memory-percent 0.15` × 128 GB) | ~19 GB |
| `mlx-community/all-MiniLM-L6-v2-4bit` embedding model | ~25 MB |
| MCP servers + Node.js processes | ~1-2 GB |
| Firecrawl Docker stack (Redis + PostgreSQL + Playwright) | ~2-4 GB |
| FAISS index + OS + system overhead | ~5-8 GB |
| **Free headroom for spike load / long contexts** | **~60-75 GB** |

### Launch Command

```bash
export VLLM_MLX_MODEL_PATH=~/models/qwen3.6-35b-opus-abl-mxfp4-mlx

vllm-mlx serve "$VLLM_MLX_MODEL_PATH" \
  --port 8000 \
  --continuous-batching \
  --use-paged-cache \
  --cache-memory-percent 0.15 \
  --embedding-model mlx-community/all-MiniLM-L6-v2-4bit \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --mcp-config mcp-configs/mcp-core.json
```

> **Note:** Do not use `taskpolicy -b` to launch the inference server.
> It restricts the process to efficiency cores and background I/O priority,
> which halves generation throughput (~35 tok/s → ~65 tok/s).  Use
> `taskpolicy -b` only for non-latency-critical background jobs like the
> sweep scheduler.

Or use the wrapper script:

```bash
./apps/consolidated_swarm/scripts/start.sh
```

### Key vllm-mlx Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--continuous-batching` | — | 2-3.4× throughput at 5 concurrent requests |
| `--use-paged-cache` | — | Efficient KV memory reuse for prefix caching |
| `--cache-memory-percent` | `0.15` | Reserves 19 GB for KV cache; auto-evicts stale entries. Raise to `0.20` if you see cache misses under load. |
| `--embedding-model` | `mlx-community/all-MiniLM-L6-v2-4bit` | Powers FAISS ingestion without a separate embedding server |
| `--reasoning-parser` | `qwen3` | Strips `<think>` blocks from qwen3.6 model output |
| `--enable-auto-tool-choice` | — | Allows model to call tools automatically |
| `--tool-call-parser` | `qwen` | Parses tool calls in Qwen format |
| `--mcp-config` | `mcp-configs/mcp-core.json` | Loads MCP servers natively; switch profile per agent |

### Native MCP Endpoints

vllm-mlx exposes MCP management via three endpoints:

```bash
# Check loaded MCP servers
curl -s http://localhost:8000/v1/mcp/status | python3 -m json.tool

# List all available tools (optionally filter by profile)
curl -s http://localhost:8000/v1/mcp/tools
curl -s "http://localhost:8000/v1/mcp/tools?profile=research"

# Execute a tool directly
curl -s -X POST http://localhost:8000/v1/mcp/execute \
  -H 'Content-Type: application/json' \
  -d '{"tool": "filesystem_read_file", "arguments": {"path": "/path/to/file"}}'
```

### Context Window Sizing (128 GB)

| Model | `num_ctx` / `max_tokens` | Notes |
|-------|--------------------------|-------|
| MoE 35B-A3B (MXFP4) | 131072 | Full 128K context fits comfortably in 128 GB |
| Dense 27B (Q4_K_M) | 131072 | Leaves ~87 GB free for KV cache and system |

### Efficiency Rules (128 GB)

1. **Prefix cache is the biggest lever.** Shared system prompts across agents
   = 80%+ KV cache hit rate. Keep agent system prompts stable between turns.
2. **`--cache-memory-percent 0.15` is conservative.** 19 GB of KV cache
   covers most multi-turn sessions. Increase to `0.20` if you have long research
   loops; decrease to `0.10` if running Docker + Firecrawl + heavy background tasks.
3. **4-8 parallel workers max.** vllm-mlx continuous batching batches requests
   efficiently up to ~8 concurrent. Beyond that, per-request latency climbs.
4. **Kill idle MCP servers.** LMCP + Filesystem stay always-on; Firecrawl
   (Docker) and Safari MCP spin up on demand.
5. **FAISS index < 2 GB.** Run `./apps/consolidated_swarm/scripts/cleanup.sh --faiss`
   after long research sessions.
6. **Context compaction every N steps.** Summarise accumulated results → FAISS
   → prune context window. The research worker does this automatically at session end.

---

## Hardware Constraints

Apple Silicon uses **unified memory** — CPU, GPU, and Neural Engine share
the same pool.  There is no separate VRAM.

| Parameter               | Value                              |
|-------------------------|------------------------------------|
| Total unified memory    | 48 GB                              |
| OS + system overhead    | ~8 GB                              |
| 20% safety margin       | ~8 GB                              |
| **Usable model budget** | **~32 GB**                         |
| Memory bandwidth        | 400 GB/s (M3 Max)                  |

When the model footprint exceeds physical memory, macOS swaps to SSD
silently.  This drops inference speed by ~100×.  The **Memory Pressure
graph** in Activity Monitor (green/yellow/red) is the only reliable
indicator — raw "memory used" numbers are misleading.

### The 60% Rule

Model weights should never exceed 60% of unified memory.  The remaining
40% is consumed by the KV cache (grows with context length), the OS, and
other applications.

---

## Ollama Environment Variables

These are set via `launchctl` (not `.zshrc` — Ollama runs as a macOS app,
not a shell process) and persisted in `~/.zprofile` so they survive
reboots.

```bash
launchctl setenv OLLAMA_MAX_LOADED_MODELS 3
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KEEP_ALIVE 10m
launchctl setenv OLLAMA_NUM_PARALLEL 1
```

| Variable                   | Value  | Purpose                                                   |
|----------------------------|--------|-----------------------------------------------------------|
| `OLLAMA_MAX_LOADED_MODELS` | `3`    | Auto-evicts the least-recently-used model when a 4th is requested. Prevents OOM.  |
| `OLLAMA_FLASH_ATTENTION`   | `1`    | Reduces memory during attention and speeds up prompt processing.  |
| `OLLAMA_KEEP_ALIVE`        | `10m`  | Warm models stay loaded 10 min after last use.  No cold starts for active work, but frees memory when idle.  |
| `OLLAMA_NUM_PARALLEL`      | `1`    | Single-user mode.  More slots split your memory budget.   |

After changing any variable, **restart Ollama** (quit from menu bar, reopen)
for it to take effect.

---

## Model Tiers — Optimized

The default `config.yaml` ships with dense models sized for large-memory
machines.  The optimized tiers below are tuned for 48 GB.

### `apps/research_swarm/config.yaml`

```yaml
model_tiers:
  orchestrator: "qwen3:30b-a3b"          # MoE — 30B quality, 3B active
  worker: "qwen2.5:7b-instruct-q8_0"     # General-purpose, warm
  worker_fast: "qwen2.5:7b-instruct-q8_0"
  worker_light: "qwen2.5:1.5b"           # Fast classification / formatting
  worker_micro: "qwen2.5:0.5b"           # Ultralight — draft / trivial tasks
  coder: "qwen3:30b-a3b"                 # Reuse MoE for coding (shared memory)
  reasoner: "qwen3:30b-a3b"              # Reuse MoE (no separate 72B needed)
```

### Why MoE (Mixture-of-Experts)

`qwen3:30b-a3b` is a 30 B parameter model that activates only ~3 B per
token.  It occupies 18–19 GB of memory but runs at the speed of a small
model.

Benchmark results (warm, 256-token generation):

| Model                             | Memory   | GPU     | tok/s | Prompt tok/s | TTFT    |
|-----------------------------------|----------|---------|-------|--------------|---------|
| qwen2.5:7b-instruct-q8_0 (dense) | 11 GB    | 100%    | 43.1  | 234.6        | 319 ms  |
| **qwen3:30b-a3b (MoE)**          | **19 GB**| **100%**|**73.0**| 92.1        | 467 ms  |

The MoE model generates tokens **70% faster** and provides 30 B-class
reasoning quality.  Prompt processing is slower (92 vs 235 tok/s) because
all 30 B parameters participate during prefill, but for agent workloads
where output dominates, the net effect is a large speedup.

> **Critical**: Pass `num_ctx=8192` (or up to 16384) for the MoE model on
> 48 GB.  The default 262K context inflates the footprint to 45 GB and
> spills to CPU (18%/82% CPU/GPU split), destroying prompt speed.

### Warm Fleet Pattern

Keep 2–3 small models permanently warm and load large models on demand.

| Tier      | Models                            | Memory   | Role                        |
|-----------|-----------------------------------|----------|-----------------------------|
| Warm      | qwen3:30b-a3b                     | 19 GB    | Orchestrator + reasoning    |
| Warm      | qwen2.5:7b-instruct-q8_0          | 8 GB     | General worker              |
| Warm      | qwen2.5:1.5b                      | 1 GB     | Fast subtasks               |
| **Total** |                                   | **28 GB**| 4 GB headroom within budget |
| On-demand | qwen2.5:14b                       | 9 GB     | Evicts a warm model         |

`OLLAMA_MAX_LOADED_MODELS=3` enforces this automatically.

---

## Flash Attention Impact

Enabling `OLLAMA_FLASH_ATTENTION=1` improved **prompt processing by 83%** on
the qwen2.5:7b model:

| Metric           | Before (off) | After (on) | Change  |
|------------------|-------------|------------|---------|
| Prompt tok/s     | 128.2       | 234.6      | **+83%** |
| TTFT (prompt)    | 421 ms      | 230 ms     | **-45%** |
| Generation tok/s | 43.1        | 43.1       | same     |

Generation speed is unchanged because flash attention optimizes the
attention computation, which is most impactful during prompt processing
and long-context inference.  For short outputs (< 256 tokens), the
benefit is primarily in TTFT.

---

## Context Window Sizing

The KV cache grows linearly with context length and can exceed model
weight memory.  Choose context size based on your memory tier:

| Unified Memory | Max `num_ctx` for MoE 30B | Max `num_ctx` for 7B Q8 |
|----------------|---------------------------|-------------------------|
| 48 GB          | 8192–16384                | 32768                   |
| 64 GB          | 32768                     | 65536                   |
| 128 GB         | 131072                    | 131072                  |

For long-context work on constrained memory, use `llama.cpp` with
**quantized KV cache** (installed via `brew install llama.cpp`):

```bash
llama-server -m /path/to/model.gguf \
  -ngl 99 \
  -c 32768 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 127.0.0.1
```

The `--cache-type-k q4_0 --cache-type-v q4_0` flags cut KV cache memory
by **75%** vs the default FP16.

---

## Speculative Decoding (Future)

A draft model (`qwen2.5:0.5b`, 397 MB) is pre-pulled and ready.  When
Ollama adds native speculative decoding support, pair it with the
qwen2.5 family models for +25–40% throughput at zero quality cost.

For same-family pairings, the draft model must run ≥ 2.5× faster than
the target.  On M3 Max:

| Draft          | Target                      | Speed Ratio | Expected Gain |
|----------------|-----------------------------|-------------|---------------|
| qwen2.5:0.5b  | qwen2.5:7b-instruct-q8_0   | ~3×         | +25%          |
| qwen2.5:0.5b  | qwen2.5:14b                 | ~4×         | +35%          |

---

## Operational Tooling

### Monitoring

Aliases are defined in `~/.zshrc`:

```bash
memcheck    # Memory pressure + loaded models + swap
gpucheck    # GPU utilization and mapped memory (IOKit)
agentbg     # Run a command at lowest priority + efficiency cores
            # Usage: agentbg python -m apps.research_swarm.scheduler
```

Quick check:

```bash
# See what's loaded and how much memory each uses
ollama ps

# Watch memory pressure (green = OK, yellow = compressing, red = swapping)
# System Settings → Activity Monitor → Memory tab
```

### Orphan Process Cleanup

Agent sessions can leak MCP server and language server processes.  A
LaunchAgent at `~/Library/LaunchAgents/com.edward.agent-cleanup.plist`
runs `~/bin/agent-cleanup.sh` every 90 minutes to kill orphaned
processes (PPID = 1, running > 10 min).

Logs: `/tmp/agent-cleanup.log`

Manual run:

```bash
~/bin/agent-cleanup.sh
```

### Process Priority

When running the sweep scheduler or long background tasks, use the
`agentbg` alias to lower CPU and I/O priority so interactive work is not
starved:

```bash
# Lowest priority + efficiency cores only
agentbg python -m apps.research_swarm.scheduler

# Or renice an already-running process
renice +15 -p $(pgrep -f "research_swarm")
```

On Apple Silicon, `taskpolicy -b` restricts the process to efficiency
cores and sets background I/O and network priority via the
`PRIO_DARWIN_BG` mechanism.

---

## Model Loading Best Practices

### Never load models in parallel

Simultaneous model loads spike memory and can cause OOM.  If writing a
warmup script, load sequentially with breathing room:

```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen3:30b-a3b","prompt":"","keep_alive":"10m","options":{"num_ctx":8192}}' \
  && sleep 2

curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b-instruct-q8_0","prompt":"","keep_alive":"10m"}' \
  && sleep 2

curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:1.5b","prompt":"","keep_alive":"10m"}'
```

### Stagger sweep jobs

The research sweep scheduler fires multiple jobs.  Ensure sweeps do not
overlap — each sweep can load different model tiers and compete for
memory.  The scheduler already handles this sequentially, but if running
manual sweeps, wait for one to finish before starting the next.

---

## Disk Cleanup

Dense models that the MoE model replaces can be removed to free ~40 GB
of disk:

```bash
# Only if you've confirmed the MoE model covers your needs
ollama rm qwen2.5:32b           # 19 GB — replaced by qwen3:30b-a3b
ollama rm qwen2.5-coder:32b     # 19 GB — replaced by qwen3:30b-a3b
```

Keep `qwen2.5:14b` as an on-demand fallback and the smaller models for
the worker tiers.

---

## Troubleshooting — Consolidated Swarm

### Queries hang or time out (504 Gateway Timeout)

**Symptom:** The swarm routes a query to a worker agent, but the worker
never returns.  vllm-mlx logs show tokens being generated for 300+s
before a 504 cancellation.

**Root cause:** `max_tokens` not reaching vllm-mlx.  `langchain-openai`
≥1.2 silently maps the `max_tokens` constructor parameter to
`max_completion_tokens` in the API request body.  vllm-mlx (and other
OpenAI-compatible servers) only reads the `max_tokens` field, so the
limit is ignored and the server defaults to its own maximum (32,768).

**Fix (in `orchestrator/models.py`):** Pass `max_tokens` via `extra_body`
so the raw field reaches the server:

```python
params["extra_body"] = {"max_tokens": max_tok}
```

Also set `request_timeout` (default 180s) on the `ChatOpenAI` instance
as a hard backstop.

### Model not found (404)

**Symptom:** `NotFoundError: The model 'qwen3.6-35b-opus-abl-mxfp4'
does not exist.  Available model: '/Users/.../qwen3.6-35b-opus-abl-mxfp4-mlx'`

**Root cause:** vllm-mlx registers models by their full filesystem path.
The model name in `config.yaml` must match exactly.

**Fix:** Use the full path as the model name in `config.yaml`:

```yaml
model_tiers:
  orchestrator: "/Users/edward/models/qwen3.6-35b-opus-abl-mxfp4-mlx"
```

### Slow generation (~35 tok/s instead of ~65 tok/s)

**Symptom:** Token generation is roughly half the expected speed.

**Root cause:** `taskpolicy -b` in the launch command restricts the
process to efficiency cores and background QoS.

**Fix:** Remove `taskpolicy -b` from `start.sh` / manual launch.
Reserve it for the sweep scheduler and other non-latency-critical jobs.

---

## Installed Tools

| Tool            | Install method       | Purpose                                 |
|-----------------|----------------------|-----------------------------------------|
| Ollama 0.19.0   | macOS app            | Primary inference server (MLX backend)  |
| llama.cpp        | `brew install`      | Low-latency TTFT, quantized KV cache    |
| agent-cleanup.sh | `~/bin/`            | Orphan process cleanup (LaunchAgent)    |

---

## Quick Reference

```bash
# Check system health
memcheck

# Start the swarm at background priority
agentbg python -m apps.research_swarm.main --interactive

# Run a one-shot sweep
python -m apps.research_swarm.scheduler --run-now --sweep crypto-market-pulse

# See loaded models + memory
ollama ps

# Force-unload a model to free memory
curl -s http://localhost:11434/api/generate \
  -d '{"model":"MODEL_NAME","keep_alive":"0"}'

# Kill all loaded models (emergency memory recovery)
for m in $(ollama ps | tail -n +2 | awk '{print $1}'); do
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":\"0\"}"
done
```
