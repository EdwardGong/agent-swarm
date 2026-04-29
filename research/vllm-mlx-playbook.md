# vllm-mlx playbook for the agent swarm

Practical reference for serving and tuning `vllm-mlx` (v0.2.9, Apr 2026)
in the swarm, captured from inspecting the installed package and CLI on
this machine. Confidence flags follow each non-obvious claim.

## TL;DR for the swarm
- Use `scripts/run-vllm-mlx-server.sh` to serve `qwen3.6-35b-opus-abl-mxfp4-mlx`
  with continuous batching and Prometheus metrics.
- Drive it with our `scripts/benchmark-vllm-mlx-client.py` for apples-to-apples
  comparison against the in-process `mlx_lm.batch_generate` baseline. Use
  `vllm-mlx bench-serve` only for sanity / cross-checks (different prompt set,
  no DRAM bandwidth sampling, different metric shapes).
- **Priority queue: implemented in code but NOT exposed in v0.2.9.** See
  "Client prioritization" below for current options and the upstream gap.

## Built-in benchmarks vs our wrappers
The CLI ships four bench subcommands:

| Command | Scope | What it measures | Run while server is | Comparable to ours? |
|---|---|---|---|---|
| `vllm-mlx bench` | offline / in-process | tok/s with various cache configs | not running | yes; covers similar ground to `benchmark-concurrency.py` but in-process |
| `vllm-mlx bench-serve` | client → live server | tok/s, latency at concurrency levels | running | yes, **closest to our** `benchmark-vllm-mlx-client.py` |
| `vllm-mlx bench-kv-cache` | KV-quant memory | memory savings at given shape | not running | no — synthetic shape, not a model |
| `vllm-mlx bench-detok` | streaming detok cost | detokenizer-only throughput | not running | no — micro-bench |

### `vllm-mlx bench` (offline)
Useful for trying prefix-cache / paged-cache / KV-quant flags fast without
spinning up a server. Key flags worth knowing about:

- `--max-num-seqs N` — cap on concurrent in-flight sequences. Direct analogue
  to our concurrency-sweep B. Default appears to be auto.
- `--prefill-batch-size`, `--completion-batch-size` — stage-specific batch
  sizes. Setting completion higher than prefill can be a small win.
- `--enable-prefix-cache` (default on), `--prefix-cache-size`,
  `--cache-memory-mb`, `--cache-memory-percent` — prefix sharing.
- `--kv-cache-quantization {4,8}-bit` with `--kv-cache-quantization-group-size`
  and `--kv-cache-min-quantize-tokens` — same approach as the K/V-quant
  story we discussed earlier. Default group_size=64, min_tokens=256
  (skip quant for short prompts).
- `--use-paged-cache` (experimental) with `--paged-cache-block-size 64`,
  `--max-cache-blocks 1000` — paged KV; needed at high concurrency.

### `vllm-mlx bench-serve` (client → live server)
This is the closest peer to our `benchmark-vllm-mlx-client.py`. **Don't replace
ours** — keep `bench-serve` as a cross-check tool. Reasons our wrapper stays:

1. Same `DEFAULT_PROMPTS` as `benchmark-concurrency.py` → diff-able runs.
2. Continuous mactop bandwidth sampler with per-window slicing → captures
   `dram_total_peak_GBs`, `bw_util_peak_pct`, `gpu_pwr_peak_W`. `bench-serve`
   has `--scrape-metrics` (Prometheus) but no DRAM bandwidth.
3. JSON shape that diffs cleanly against the in-process baseline (same
   `concurrency`, `generation_tps_aggregate`, `ttft_p50_s`, `wall_p95_s`).

`vllm-mlx bench-serve --concurrency 1,4 --max-tokens 256 --scrape-metrics`
defaults are cohesive; useful when sanity-checking server behavior in
isolation.

### `vllm-mlx bench-kv-cache`
Pure-shape memory math: `--layers --seq-len --heads --head-dim --group-size`.
Defaults (32/512/32/128/64) are Llama-style. For our 35B-A3B we'd use
`--layers 40 --heads 16 --head-dim 256 --group-size 64`. Verifies the KV
memory budget at a given quant config without loading weights. One-shot
sanity tool; not part of routine bench runs.

### `vllm-mlx bench-detok`
Measures the streaming detokenizer cost. Default model
`mlx-community/Qwen3-0.6B-8bit`. Useful only if you suspect Python-side
detokenize is bottlenecking decode at very high tok/s. Skip for now.

## Server flags for the swarm orchestration tier
From `vllm-mlx serve --help` on v0.2.9. Not exhaustive — only what's relevant.

### Almost-always-on
- `--continuous-batching` — mid-flight insertion of new requests.
- `--enable-metrics` (or `--metrics`, both appear) — Prometheus on the same
  port at `/metrics`. Use for monitoring without scraping client side.
  *Note*: our launcher passes `--metrics`; if the server rejects, switch to
  `--enable-metrics` (the help line shows `--enable-metrics` as the flag).
- `--enable-prefix-cache` — default on. **High value for the swarm**: if all
  workers share a system prompt + tool schema, those tokens prefill once.
- `--served-model-name opus-mxfp4` — short alias for client requests.

### Throughput knobs
- `--max-num-seqs N` — concurrency cap. Set near the knee of our scaling
  curve (per the in-process probe, B=16-32 is the throughput sweet spot
  on M3 Max). Start at 16, raise if memory permits.
- `--prefill-batch-size` and `--completion-batch-size` — see offline bench
  notes; non-default values can buy a few percent.
- `--chunked-prefill-tokens N` — caps prefill tokens per scheduler step;
  `0` disables. Helps when long prompts collide with short-ttft streams.
- `--prefill-step-size N` — granularity of prefill chunks (default 2048
  in upstream mlx_lm).

### Memory / KV
- `--use-paged-cache --paged-cache-block-size 64 --max-cache-blocks 1000` —
  needed if you push beyond ~32 concurrent sequences on 48 GB.
- `--kv-cache-quantization --kv-cache-quantization-bits 8` — keeps quality
  high. **Asymmetric K8/V4** (which we discussed earlier) does **not appear
  exposed in v0.2.9** — you only get a single bit-width for both K and V.
  Worth filing upstream.
- `--kv-cache-min-quantize-tokens 256` — only quantize the cache once the
  conversation is long enough that the savings matter.
- `--cache-memory-mb` / `--cache-memory-percent` — explicit budget for the
  prefix cache vs auto.
- `--gpu-memory-utilization 0.9` — wired-memory budget. Combine with the
  `sudo sysctl iogpu.wired_limit_mb=N` recipe for >27 GB models.

### Speculative decoding (latency, modest tok/s gain)
- `--enable-mtp --mtp-num-draft-tokens 3` — multi-token prediction (Qwen3-Next
  has it baked in, usable here per the README). Good if first-token latency
  matters.
- `--specprefill --specprefill-draft-model <small-model> --specprefill-keep-pct 0.5
   --specprefill-threshold 0.5` — speculative prefill for big prompts. Heavier
  setup; only worth it for long-context worker tier.

### Long-running / mixed workload
- `--ssd-cache-dir ~/.cache/vllm-mlx-ssd --ssd-cache-max-gb 50` — overflow
  prefix cache to SSD. Useful if many distinct system prompts.
- `--warm-prompts <path-or-text>` — preload prefixes at startup so the first
  request from each worker class doesn't pay prefill cost.
- `--rate-limit N` — request-rate cap. Useful as a defense against runaway
  worker loops.

### Multimodal / extras (not relevant to text swarm yet)
- `--mllm`, `--mllm-prefill-step-size`, `--max-audio-upload-mb`, `--rerank-model`,
  `--embedding-model`, `--reasoning-parser`, `--tool-call-parser`,
  `--enable-auto-tool-choice`, `--mcp-config`. Worth a second pass when
  the swarm gets vision / TTS / embeddings tiers.

## Client prioritization — the actual story
**Goal**: orchestrator agent's calls preempt or jump the queue ahead of
worker agents' calls.

### Status in v0.2.9 (verified, HIGH confidence)
- `vllm_mlx/request.py:100` — `Request.priority: int = 0  # Lower is higher priority`
- `vllm_mlx/request.py:181-184` — `Request.__lt__` uses `(priority, arrival_time)`
  for priority-queue ordering.
- `vllm_mlx/scheduler.py:45-49` — `class SchedulingPolicy(Enum): FCFS = "fcfs"; PRIORITY = "priority"`.
- `vllm_mlx/scheduler.py:61` — `SchedulerConfig.policy: SchedulingPolicy = SchedulingPolicy.FCFS` (default FCFS).

So the data model and scheduler comparator already exist. **However**:
- No CLI flag (`vllm-mlx serve --help` shows no `--scheduling-policy`,
  `--policy`, `--priority`, or similar).
- No API surface (`grep priority vllm_mlx/api/` returns only an unrelated
  tool-calling comment). The OpenAI request body has no priority field;
  HTTP headers aren't read for priority.

The priority queue is **plumbed but not user-reachable**. This is a real
limitation of v0.2.9.

### Options (with confidence)
1. **Wait for upstream** [HIGH confidence in feasibility, MEDIUM in timing].
   File a feature request on the [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)
   repo asking for either `--scheduling-policy priority` plus an `X-Priority`
   request header or a `priority` field in the request body. The work is
   small (existing comparator + a header-reading shim). Track in
   research/TODOs.
2. **Two server instances on different ports** [HIGH confidence; safest now].
   - `:8000` for orchestrator (smaller `--max-num-seqs`, e.g. 4 → low queueing).
   - `:8001` for workers (larger `--max-num-seqs`, e.g. 24).
   - Cost: each instance loads a separate copy of weights (~17 GB for
     opus-mxfp4 → 34 GB total). On 48 GB this leaves ~10 GB headroom — workable
     but tight; on 128 GB (M5) it's comfortable.
3. **Monkey-patch at server boot** [MEDIUM confidence; brittle].
   Wrap `vllm-mlx serve` with a small Python shim that:
   - Imports `vllm_mlx.scheduler` and overrides `SchedulerConfig.policy =
     SchedulingPolicy.PRIORITY` before the engine starts.
   - Subclasses the OpenAI-compatible request handler to read an
     `X-Request-Priority` header (or a `metadata.priority` field) and pass
     it through to `Request(priority=...)`.
   We get a single-instance solution but accept upstream-coupling risk.
   Worth doing if (1) drags > 1 month and (2) is too memory-tight on M3 Max.
4. **Application-layer priority via separate queues** [HIGH confidence;
   no server change]. The orchestrator submits to vllm-mlx directly while
   workers go through an asyncio.PriorityQueue your client owns. Workers
   wait for tokens budget to be available. Loses continuous-batching
   intermixing benefits but works without server modification.

### Recommendation
- **Now (M3 Max 48 GB):** option 4 (client-side queue). Simple and
  immediately useful.
- **On M5 Max 128 GB arrival:** option 2 (two instances). Cleanest
  isolation; orchestrator never blocks on workers.
- **In parallel:** open the upstream feature request (option 1) so option
  3 / 4 can be retired later.

## Other things worth documenting
- **Model name in API requests** = `--served-model-name`, not the file
  path. The launcher sets it to `opus-mxfp4`.
- **Continuous batching is opt-in** in this build (must pass
  `--continuous-batching`); the older static-batching path is the default.
- **Prometheus endpoint** at `/metrics` when `--enable-metrics` is set.
  Worth adding the Prometheus + Grafana stack for the swarm production
  story; for now `bench-serve --scrape-metrics` and our client's snapshot
  are enough.
- **OpenAI + Anthropic compatibility**: vllm-mlx exposes both
  `/v1/chat/completions` and `/v1/messages`. Use the OpenAI path for
  worker calls; if Claude Code is in the loop the Anthropic path is
  drop-in.
- **MoE expert reduction (`--moe-top-k`)**: the upstream README claims
  +7-16% on Qwen3-30B-A3B by reducing top-k below the trained value.
  **Did not appear in v0.2.9 `serve --help`** in this install — may be
  in `bench` or under a different name. Verify before relying on it.
- **Wired memory bump**: for >0.9× of recommended working set, use
  `sudo sysctl iogpu.wired_limit_mb=<N>` on macOS 15+ before launching the
  server. mlx_lm's `wired_limit` context already auto-bumps but only up
  to `max_recommended_working_set_size`.

## Open TODOs
- File upstream FR for `--scheduling-policy priority` + `X-Priority` header.
- Verify whether `--moe-top-k` exists in this install (or a flag with a
  similar name) and benchmark its quality impact on Opus-distilled weights.
- Confirm `--metrics` vs `--enable-metrics` flag spelling on first server
  launch; patch the launcher if needed.
- Asymmetric K/V quantization (K8/V4) is not exposed in v0.2.9 — file an FR.
- Once the comparison run is done, capture a `vllm-mlx vs in-process` diff
  in `reports/benchmarks/` so future agents have ground truth, not lore.
