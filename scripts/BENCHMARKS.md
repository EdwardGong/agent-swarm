# Benchmark recipes

Canonical invocations for `scripts/benchmark-mlx.py` (single-model) and
`scripts/benchmark-compare.py` (multi-model leaderboard). Reports land in
`reports/benchmarks/` as `<stem>.{md,json}` plus per-model `<model>-<ts>.{json,txt}`.

Both scripts:
- Cycle through a default set of varied prompts across runs (`--prompt`/`--prompts-from` to override).
- Run 3 silent warmups before measured runs.
- Try `mlx_vlm` first and fall back to `mlx_lm` for vision-stripped models.
- Write generated text to disk; only compact metrics print to stdout.

## Metrics surfaced
- `mean / median / min / max gen tok/s` — decode throughput.
- `cv%` — stdev/mean × 100. Single-number stability score across runs.
- `ttft ms` — prefill seconds × 1000 (time-to-first-token proxy at batch=1).
- `wall s` — wall-clock seconds for the full `max_tokens` response.
- `peak GB` — max resident model + KV-cache memory.
- `disk GB` — on-disk archive size.
- `vs best` — gen tok/s mean as a fraction of the leader.
- `load s` — model load time.

JSON also retains per-run prompt index, prompt/gen tok counts, and prompt_tps.

## Single-model bench
```bash
python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
    --runs 3 --max-tokens 1024
```

## Two-model apples-to-apples (used to settle the mxfp4 mystery)
```bash
python3 scripts/benchmark-compare.py \
    ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
    ~/models/qwen3.6-35b-mxfp4-mlx \
    --runs 5 --max-tokens 1024 \
    --report-name mxfp4-pair
```
Use n=5 when investigating a suspected gap; n=3 is the routine default.

## Full 35B leaderboard (all variants, mixed VLM + LM-only backends)
```bash
python3 scripts/benchmark-compare.py \
    ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx \
    ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
    ~/models/qwen3.6-35b-opus-abl-mxfp4-mlx \
    ~/models/qwen3.6-35b-mxfp4-mlx \
    ~/models/qwen3.6-35b-abl-4.4bit-msq-mlx \
    ~/models/qwen3.6-35b-optiq-4bit-mlx \
    --runs 3 --max-tokens 1024 \
    --report-name 35b-all-variants
```

## Dense 27B comparison
```bash
python3 scripts/benchmark-compare.py \
    ~/models/qwen3.6-27b-4bit-mlx \
    ~/models/qwen3.6-27b-abl-4.5bit-msq-mlx \
    --runs 3 --max-tokens 1024 \
    --report-name 27b-dense
```

## Long-context decode (when you suspect KV-cache regressions)
```bash
python3 scripts/benchmark-compare.py \
    ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
    --runs 3 --max-tokens 4096 \
    --report-name long-context-mxfp4
```

## Custom prompt set from file
```bash
# prompts.txt: one prompt per line; lines starting with # are ignored.
python3 scripts/benchmark-compare.py \
    ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
    --prompts-from research/prompts/coding.txt \
    --runs 3 --max-tokens 1024 \
    --report-name coding-prompts
```

## Models from a file (handy for repeat sweeps)
```bash
# 35b_models.txt: one absolute path per line.
python3 scripts/benchmark-compare.py \
    --models-from configs/35b_models.txt \
    --runs 3 --max-tokens 1024 \
    --report-name 35b-sweep
```

## vllm-mlx server benchmark
For measuring continuous-batching throughput against a real OpenAI-compatible
server (rather than in-process `mlx_lm.batch_generate`):
```bash
# Terminal A: launch the server (default port 8000, served-model-name=opus-mxfp4)
scripts/run-vllm-mlx-server.sh

# Terminal B: benchmark with the same prompts/concurrencies as benchmark-concurrency.py
python3 scripts/benchmark-vllm-mlx-client.py \
    --concurrencies 1,2,4,8,16,32 --max-tokens 256 \
    --report-name vllm-mlx-baseline
```
The client driver matches the in-process probe's metric shape (aggregate /
per-stream tok/s, TTFT p50/p95, wall p95, mactop bandwidth + GPU power) so the
two reports can be diffed directly. JSON output also includes a Prometheus
snapshot from the server's `/metrics` endpoint for cross-validation.

## Conventions
- **Naming**: `--report-name` should describe the cohort and any non-default
  setting. Examples: `35b-all-variants`, `mxfp4-pair`, `long-context-mxfp4`,
  `coding-prompts`, `n5-stability-check`.
- **`--runs`**: 3 for routine sweeps; 5 when investigating a suspected gap;
  larger only when CV% is itself the question.
- **`--max-tokens`**: 1024 for routine; 256 for quick sanity; 4096+ to expose
  long-context decode regressions.
- **Compare under one process invocation**: model load order can matter for
  thermal/wired-memory state. Pass all models to a single `benchmark-compare.py`
  call rather than running one model at a time.
