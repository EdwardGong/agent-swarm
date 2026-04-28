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

Run both models simultaneously:
- Dense 27B for orchestrator (higher quality on hard coding, 77.2% vs 73.4% SWE-bench)
- MoE 35B for fast workers (3-4× throughput)
- Serve via `mlx_vlm.server` on separate ports

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
