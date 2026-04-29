# vllm-mlx playbook for the agent swarm

Practical reference for serving and tuning `vllm-mlx` (v0.2.9, Apr 2026)
in the swarm, captured from inspecting the installed package and CLI on
this machine. Confidence flags follow each non-obvious claim.

## TL;DR for the swarm
- Use `scripts/run-vllm-mlx-server.sh` to serve `qwen3.6-35b-opus-abl-mxfp4-mlx`
  with continuous batching, paged KV cache, and Prometheus metrics enabled.
- Drive it with our `scripts/benchmark-vllm-mlx-client.py` for apples-to-apples
  comparison against the in-process `mlx_lm.batch_generate` baseline. Use
  `vllm-mlx bench-serve` only for sanity / cross-checks (different prompt set,
  no DRAM bandwidth sampling, different metric shapes).
- **Priority scheduling**: not in upstream v0.2.9 from PyPI, but a
  patched fork at `/Users/edward/workspace/forks/vllm-mlx`
  (branch `feat/priority-scheduling`) is installed editable into
  `/Users/edward/miniforge3` and adds the missing wiring. With the fork
  active, `--scheduling-policy priority` on the server + per-request
  `priority=N` (or `extra_body={"priority": N}`) work end-to-end.
  `scripts/run-vllm-mlx-server.sh` does NOT pass the flag by default
  (so a fresh PyPI reinstall of vllm-mlx won't break the launcher);
  pass `--scheduling-policy priority` explicitly to enable it. See
  `Client prioritization` below for full setup and conventions.

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
- `--enable-metrics` — Prometheus on the same port at `/metrics`. The
  short `--metrics` form does **not** exist in v0.2.9 (`cli.py:1019` is
  the sole arg); ignore older notes that suggested it.
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

### Speculative decoding (see dedicated section below)
Two flags exist; their semantics are easy to confuse and only one is
actually decode-time speculation:
- `--enable-mtp --mtp-num-draft-tokens N` — multi-token prediction. Real
  decode-time spec; requires the model to ship the MTP / next-N
  predictor head **weights** (not just the config field). The architecture
  for `qwen3.6-35b-opus-abl-mxfp4-mlx` declares one head
  (`text_config.mtp_num_hidden_layers=1`) but its quant pipeline
  dropped the weights, so `--enable-mtp` no-ops on our current target.
  See the dedicated section for the path to re-enable.
- `--specprefill --specprefill-draft-model <small-mlx-model>` — prefill-only
  TTFT optimization (arxiv.org/abs/2502.02789). Does **not** improve decode
  tok/s. Only worth it for long-context worker tier.

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

## Client prioritization — UNWIRED upstream, available via local fork
**Status**: v0.2.9 from PyPI ships the priority feature as inert
dataclass machinery (no CLI flag, no API plumbing, no scheduler
ordering). A local patch closing the five wiring gaps lives at
`/Users/edward/workspace/forks/vllm-mlx` on branch
`feat/priority-scheduling`, installed editable into
`/Users/edward/miniforge3` (verified `vllm-mlx serve --help | grep
schedul` shows `--scheduling-policy {fcfs,priority}`).

With the fork installed, both server-side `--scheduling-policy priority`
and per-request `priority=N` (or OpenAI `extra_body={"priority": N}`)
work end-to-end. The evidence below documents the v0.2.9 baseline gaps
that the fork closes — keep it for context until the FR is upstreamed,
at which point delete this section's evidence subsection.

### Evidence (verified against the installed v0.2.9 in this venv)
1. **No CLI flag**: `vllm-mlx serve --help | grep -i schedul` returns only
   the unrelated `--chunked-prefill-tokens` help blurb. Exhaustive grep of
   `vllm_mlx/**/*.py` for `--schedul*`, `scheduling_policy`,
   `SchedulingPolicy`, `policy=` yields hits in only two files —
   `scheduler.py:45-61` (the enum + dataclass field definition) and
   nowhere in `cli.py`/`server.py`/`engine_core.py`/`api/*`.
2. **CLI never passes `policy=`**: `cli.py:183-214` constructs
   `SchedulerConfig(...)` for the serve command from argparse args; the
   kwarg list does not include `policy`. Same for the offline-bench path
   at `cli.py:314-333`. So `SchedulerConfig.policy` is always
   `SchedulingPolicy.FCFS` (the dataclass default at `scheduler.py:61`).
3. **Scheduler never reads `policy`**: `Scheduler` (`scheduler.py:1100`)
   stores `self.config = config or SchedulerConfig()` but no codepath
   branches on `self.config.policy` or compares against
   `SchedulingPolicy.PRIORITY`. The waiting queue is
   `self.waiting: deque[Request]` (line 1139) used as FIFO. The only
   `priority`-adjacent mention in the file is a comment on line 2416
   about the cache-corruption retry path that uses `deque.appendleft`.
4. **API path never sets `Request.priority`**: every `Request(...)`
   construction site (`engine_core.py:300`, `engine_core.py:517`) omits
   `priority`. `AsyncEngineCore.add_request` (line 271) takes no
   `priority` parameter, so even if the OpenAI / Anthropic adapter
   parsed `extra_body["priority"]`, there is no kwarg to forward it on.
   Grep of `api/*.py` confirms no `priority=` assignment.
5. **`Request.__lt__` is dead code**: `request.py:180-184` orders by
   `(priority, arrival_time)`, but no `bisect`, `heapq`, or `sorted()`
   call in the package consumes it.

Original (now-corrected) claim cited `cli.py:700,706` as evidence — those
lines are actually the `extra_body` regex split inside
`bench_serve_command`, not a scheduling-policy flag. The regex parses the
bench-client's `--extra-body '{"a":1},{"b":2}'` argument; it has no
relationship to per-request priority on the serve path.

### Using priority on the patched fork
Launcher: pass `--scheduling-policy priority` (the fork accepts it).
```bash
vllm-mlx serve <model> --scheduling-policy priority \
    --continuous-batching --use-paged-cache --enable-metrics \
    --served-model-name opus-mxfp4
```
Client: lower `priority` value = scheduled sooner.
```python
client.chat.completions.create(
    model="opus-mxfp4",
    messages=[...],
    extra_body={"priority": -10},  # orchestrator tier
)
```
Conventions for the swarm:
- Orchestrator: `priority = -10`
- Default worker: `priority = 0` (omit field)
- Background / low-priority worker: `priority = 10`
- Reserve `-5` for user-interactive requests if any share the model

Behavior caveats (see fork commit `feat/priority-scheduling`):
- Priority affects **queueing order**, not preemption — a request
  already in the running batch isn't kicked out by a higher-priority
  arrival.
- Within a single priority class, ordering is FCFS by `arrival_time`.
- The fork keeps `Scheduler.waiting` as a `deque`; `_pop_next_waiting`
  scans for the min element under PRIORITY. O(N) per scheduling step
  where N is bounded by `max_num_seqs` plus accumulated waiting load.
  Replace with `heapq` upstream if N grows large in production.

### When two-server isolation is still useful
For *hard* isolation, spin up two server instances on different ports
(e.g. orchestrator with `--max-num-seqs 4`, workers with
`--max-num-seqs 24`). Costs ~34 GB of weights on M3 Max 48 GB (tight);
comfortable on 128 GB. Use this *in addition* to priority within each
instance once the M5 box is online.

### Upstream FR
The fork's commit (`feat/priority-scheduling`) is small and
self-contained — ready to PR upstream against `waybarrios/vllm-mlx`
when we have time. Until that ships, install from the local fork:
```bash
pip install -e /Users/edward/workspace/forks/vllm-mlx
```

## Speculative decoding
Verified against installed v0.2.9 + the model's `config.json` on Apr 29
2026. Two flags exist; only one is decode-time speculation, and the
decode-time one does not apply to our current target model.

### MTP — `--enable-mtp --mtp-num-draft-tokens N` (decode-time)
MTP installs a multi-token-prediction wrapper around the BatchGenerator
(`scheduler.py:1046-1097`). For a model to benefit, the *target* model
must ship a next-N predictor head; the wrapper draws draft tokens from
that head and verifies them in the next decode step.

**Our target model `qwen3.6-35b-opus-abl-mxfp4-mlx` declares an MTP head
in its config but does not ship the MTP weights** (verified Apr 29 2026
by starting the server and reading `INFO:vllm_mlx.utils.tokenizer:[MTP]
Config has num_nextn_predict_layers=1 but MTP weights not found,
skipping MTP.`).

From `~/models/qwen3.6-35b-opus-abl-mxfp4-mlx/config.json`:
- `architectures = ["Qwen3_5MoeForConditionalGeneration"]`,
  `model_type = "qwen3_5_moe"` — Qwen3.5 MoE family.
- `text_config.mtp_num_hidden_layers = 1` and
  `text_config.mtp_use_dedicated_embeddings = false` — i.e. the
  architecture allocates a single MTP head with shared embeddings.
  vllm-mlx's loader recognizes both this and the older
  `num_nextn_predict_layers` field (`engine/batched.py:323-328`).
- Earlier playbook text claimed no MTP fields were present — that was
  wrong; the recursive search now finds them under `text_config.*`.
  The original top-level grep missed the nested keys.
- Implication: the **abliterated/mxfp4 quantization pipeline dropped
  the MTP head weights from the safetensors**, so `--enable-mtp` is a
  no-op (the loader logs the skip and continues). This is fixable by
  re-quantizing from a base model that retains the MTP weights, *not*
  a fundamental limitation of the architecture or of vllm-mlx.

Path to enable MTP on this target:
1. Find or produce a Qwen3.5 MoE quant that keeps the MTP head
   (`mtp.*` weight tensors in the safetensors index). Many
   community quants strip these because they're rarely used and add
   ~few hundred MB.
2. Verify the loader picks them up: log line should change from
   "MTP weights not found, skipping MTP" to
   `[MTP] installed with num_draft_tokens=N, ...`
   (`scheduler.py:1095-1097`).
3. Bench with `--enable-mtp --mtp-num-draft-tokens {1,3}`. The
   `--mtp-optimistic` flag skips the verify step (5-10% wrong
   tokens; trade quality for ~1.5x decode tok/s).

### `--specprefill` — TTFT optimization, not decode-time speculation
Name is misleading. `specprefill.py:1-39` cites arxiv.org/abs/2502.02789
("SpecPrefill: Speculative Prefilling"). The mechanism:
1. Run a small draft model over the prompt to score per-token attention
   importance.
2. Keep the top `--specprefill-keep-pct` fraction of tokens.
3. Sparse-prefill the target model with only those tokens, using
   `_OffsetAdjustedRoPE` to preserve original positional encoding.
4. Decode normally on the resulting (smaller) prefix cache.

This reduces **prefill time / TTFT on long prompts** (`--specprefill-threshold`
defaults to 8192 prompt tokens). It does not change steady-state decode
tok/s — there is no draft-verify loop during decode. Treat it as a
latency lever for the long-context worker tier, not a throughput lever
for short-prompt swarm calls.

Draft model requirements (from `cli.py:985-991`): same tokenizer as the
target, fast to score with, mlx_lm-loadable. For Qwen3.5 MoE, candidates
include `mlx-community/Qwen3-0.6B-*` family (same Qwen3 BPE tokenizer);
`Qwen2.5` variants will not match. Verify token id parity with
`tokenizer.json` before enabling.

### What's missing in v0.2.9
There is no `--draft-model` style classic decode-time speculative
decoding (Leviathan et al. 2022 / vLLM's `speculative_config`). For our
current quant of `qwen3.6-35b-opus-abl-mxfp4-mlx`, neither MTP nor
SpecPrefill will move the decode-tok/s needle right now — MTP is
architecturally available but the weights are absent in the safetensors,
and SpecPrefill is a TTFT lever, not a decode lever. The realistic
speculative-decoding paths are:
1. **Re-quantize the current model with MTP weights retained** (best
   path — architecture already supports it; see the dedicated section).
2. Switch to a different Qwen3.5 MoE quant that kept MTP weights.
3. File an upstream FR for a `--draft-model` hook into
   `Scheduler.batch_generator` that runs draft-then-verify around the
   target's `_step`, for cases when the target lacks an MTP head.

Documented as `Open TODO` below.

## Other things worth documenting
- **Model name in API requests** = `--served-model-name`, not the file
  path. The launcher sets it to `opus-mxfp4`.
- **Continuous batching is opt-in** in this build (must pass
  `--continuous-batching`); the older static-batching path is the default.
- **Restart between repeat bench sweeps**. Memory accumulates across
  consecutive client runs against a long-lived server (paged-cache pool
  growth, prefix-cache fill, possibly Metal allocator fragmentation).
  Observed Apr 29 2026: a 512-token sweep run twice in a row hit a
  failure at C=32 on the second attempt and OOMed at C=32 on the third;
  a fresh server restart cleared it and the sweep completed cleanly.
  Operational guidance: restart `scripts/run-vllm-mlx-server.sh` between
  back-to-back full sweeps when chasing apples-to-apples numbers,
  especially at higher max_tokens. The cleaner long-term fix is
  upstream — exposing a `/reset_cache` admin endpoint or an opt-in
  per-request `evict_after` flag — but neither exists in v0.2.9.
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

## Head-to-head bench results
Bench unblocked Apr 29 2026 by upgrading vllm-mlx to master HEAD (the
thread-binding bug in `mlx_lm/generate.py:1369` that emitted
`RuntimeError: There is no Stream(gpu, 3) in current thread.` from the
executor thread is now fixed there). Full numbers and findings in
`reports/benchmarks/qwen3.6-35b-parallel/vllm-mlx-vs-in-process.md`.
Raw JSON is split by `max_tokens`:
- `256-tokens/mlx-lm-baseline.json` — in-process B=1..32, max_tokens=256.
- `256-tokens/vllm-mlx-baseline-v2.json` — server with
  `--use-paged-cache`.
- `256-tokens/vllm-mlx-baseline.json` — server *without*
  `--use-paged-cache` (memory_aware_cache), kept for the cache-backend
  A/B.
- `512-tokens/mlx-lm-baseline.json` — in-process B=1..32, max_tokens=512
  (added Apr 29 2026).
- `512-tokens/vllm-mlx-baseline.json` — server C=1..32 with
  `--use-paged-cache` at max_tokens=512 (re-collected Apr 29 2026 after
  the directory reorg). Operator note: this run required a fresh server
  restart — see “Restart between repeat bench sweeps” above.
- **Cache backend matters a lot**. Switching the server from
  `memory_aware_cache` to `paged_cache` (v1 → v2) lifted aggregate
  decode tok/s by 9–19% across C=1..32 with 10–46% lower mean TTFT.
  The launcher already passes `--use-paged-cache`; v1 is from before
  that change.
- **v2 server is at near-parity with in-process up to C=16** (within
  ±2% at C=1, 4, 8, 16; ~10% faster at C=2). At C=32 the server is
  ~9% behind in-process (424.7 vs 468.6 tok/s) — a roughly constant
  per-step overhead (HTTP + SSE + detokenizer + scheduler), not
  per-token. Earlier notes citing a ~20% gap were against v1; that's
  no longer the operating point.
- **Bandwidth and power**: in-process peaks DRAM 162 GB/s (40.6% of
  spec) at GPU peak 51.8 W; v2 peaks DRAM 193 GB/s (48.3% of spec) at
  GPU peak 51.5 W. Neither configuration is bandwidth-bound; both are
  compute / kernel-launch bound around the same DRAM ceiling.
- **Sweet spots (server v2)**: C=2 for best TTFT (122 ms, 158 tok/s);
  C=16 for throughput-with-bounded-tail (359 tok/s, TTFT p50 ~800 ms);
  C=32 only if you accept TTFT p95 ≈ 2.4 s.
- **max_tokens scaling (in-process)**: aggregate decode tok/s is
  essentially flat between max_tokens=256 and 512 across B=2..32 (≤3%
  drift; B=1 is N=1 noise). Peak Metal climbs only 22.9 GB → 23.2 GB
  at B=32. The 512-token server pairing has been re-collected
  (`512-tokens/vllm-mlx-baseline.json`).
- **Headroom**: v2 reports `cache_utilization_ratio = 0.178` at C=32 —
  the paged pool is barely populated, so concurrency can rise well
  past 32 before memory becomes the bottleneck. A previous 512-token
  server run at C=64 collapsed to 9.7 tok/s aggregate, flagging a knee
  somewhere between C=32 and C=64 worth bisecting separately.
- **Stale fix history**: an earlier installation against the v0.2.9
  PyPI wheel hit 100% request failure with the stream-binding error.
  The fix shipped on master post-v0.2.9. If the launcher starts
  failing again, check whether `pip install vllm-mlx` reinstalled the
  pinned 0.2.9 wheel and dropped the master fix — reinstall from
  master in that case.

## Open TODOs
- Verify whether `--moe-top-k` exists in this install (or a flag with a
  similar name) and benchmark its quality impact on Opus-distilled weights.
- Asymmetric K/V quantization (K8/V4) is not exposed in v0.2.9 — file an FR.
- Re-bench with `--chunked-prefill-tokens` ∈ {512, 1024, 2048} to see if a
  smaller chunk further flattens TTFT at high C. The launcher already runs
  with 2048, and v2 still hits TTFT p95 ~2.4 s at C=32.
- ~~Re-bench with `--use-paged-cache` against master HEAD now that the
  thread-binding bug is fixed.~~ **DONE** (Apr 29 2026): paged cache lifts
  decode tok/s 9–19% across C=1..32 vs `memory_aware_cache`. See
  `reports/benchmarks/qwen3.6-35b-parallel/256-tokens/vllm-mlx-baseline-v2.json`
  and `../vllm-mlx-vs-in-process.md`.
- Sweep `--max-num-seqs` ∈ {32, 48, 64} at max_tokens=256 to locate the
  new throughput knee (cache utilization is only 17.8% at C=32 with paged
  cache; lots of room to grow).
- ~~Re-collect the server-side 512-token sweep into
  `reports/benchmarks/qwen3.6-35b-parallel/512-tokens/vllm-mlx-baseline.json`.~~
  **DONE** (Apr 29 2026); paired in-proc reference also in place at
  `512-tokens/mlx-lm-baseline.json`.
- Bisect the C=64 collapse seen at max_tokens=512 — likely paged-cache
  exhaustion or scheduler thrash; re-run with a higher
  `--max-cache-blocks` and explicit `--max-num-seqs 64`.
- **Priority scheduling FR**: file upstream issue covering the 5 wiring
  gaps documented in `Client prioritization`. Once it ships, restore the
  priority-stress test (24 workers at `priority=0` + 4 orchestrator
  requests at `priority=-10`, verify orchestrator wall_p95 ≪ worker mean).
- **Speculative decoding**: MTP is architecturally available on this
  Qwen3.5 MoE target but the current quant dropped the head weights.
  Highest-leverage follow-up: re-quantize from a base that retains the
  MTP head (look for `mtp.*` tensors in the safetensors index), then
  bench `--enable-mtp --mtp-num-draft-tokens 3`. As a fallback, file an
  upstream FR for a `--draft-model` decode-time speculator and document
  quality vs tok/s tradeoff.
- **Tokenizer regex warning**: server startup logs `[transformers] ...
  incorrect regex pattern ... fix_mistral_regex=True ...`. Verified
  Apr 29 2026: this is a transformers heuristic that fires on any
  GPT-style BPE pre-tokenizer (`(?i:'s|'t|'re...)|[\p{L}\p{M}]+|...`),
  not a real bug in the Qwen3.5 tokenizer. Setting `fix_mistral_regex=
  True` would deviate from the tokenization the model was trained on,
  so leave the flag off. The warning is identical between the in-process
  baseline and the vllm-mlx server, so throughput diffs are still valid.
  Re-evaluate if/when we run a quality benchmark.
