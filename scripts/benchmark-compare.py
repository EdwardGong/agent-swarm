#!/usr/bin/env python3
"""Benchmark multiple MLX VLM models with identical settings and emit a
leaderboard.

Generated text + per-run JSON go to disk; only the comparison table prints to
stdout. Use this to get clean apples-to-apples tok/s numbers across variants.

Usage:
    python3 scripts/benchmark-compare.py \
        ~/models/qwen3.6-35b-opus-abl-mxfp4-vlm-mlx \
        ~/models/qwen3.6-35b-mxfp4-mlx \
        ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx \
        --runs 5 --max-tokens 1024

    # From a file (one path per line; lines starting with # ignored):
    python3 scripts/benchmark-compare.py --models-from configs/35b_models.txt --runs 5
"""

import argparse
import datetime as dt
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "~/models/qwen3.6-35b-opus-abl-mxfp4-mlx"

# Varied prompts across domains/lengths so any per-prompt caching effect
# (input token cache, kernel cache for fixed input length, etc.) is averaged
# out across runs rather than benefiting only the first measured pass.
DEFAULT_PROMPTS = [
    "Design a distributed message queue system with partitioning, "
    "replication, and exactly-once delivery.",
    "Explain the differences between cooperative and preemptive multitasking, "
    "with examples from real operating systems.",
    "Write a step-by-step plan to migrate a monolithic Postgres database to a "
    "sharded architecture without downtime.",
    "Describe how Raft achieves consensus and what failure modes it handles "
    "versus Paxos.",
    "Compare HNSW, IVF, and ScaNN as vector indexing strategies for "
    "billion-scale similarity search.",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "benchmarks"


def _result_to_dict(result) -> dict:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "total_tokens": result.total_tokens,
        "prompt_tps": result.prompt_tps,
        "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory,
    }


def _disk_size_gb(model_path: str) -> float:
    """Return on-disk model size in GB. Uses safetensors index when available."""
    p = Path(os.path.expanduser(model_path))
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        try:
            meta = json.loads(idx.read_text()).get("metadata", {})
            if "total_size" in meta:
                return int(meta["total_size"]) / (1024 ** 3)
        except Exception:
            pass
    # Fallback: sum all files in the directory.
    total = 0
    if p.is_dir():
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    return total / (1024 ** 3)


def _thermal_prewarm(seconds: int) -> None:
    """Run sustained GPU matmul to bring the device to a steady-state thermal
    regime before measured benches. No-op if seconds <= 0.
    """
    if seconds <= 0:
        return
    import mlx.core as mx
    print(
        f"[prewarm] sustaining ~{seconds}s of GPU matmul to settle thermal state",
        flush=True,
    )
    a = mx.random.normal((4096, 4096))
    b = mx.random.normal((4096, 4096))
    deadline = time.perf_counter() + seconds
    iters = 0
    while time.perf_counter() < deadline:
        c = a @ b
        mx.eval(c)
        iters += 1
    print(f"[prewarm] done ({iters} matmul iters)", flush=True)


def _free_memory():
    """Release as much memory as possible between models."""
    gc.collect()
    try:
        import mlx.core as mx
        # mx.clear_cache replaces the deprecated mx.metal.clear_cache.
        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:
        pass


def _load_model(model_path: str):
    """Load a model via mlx_vlm; fall back to mlx_lm for vision-stripped models.

    Returns (model, processor_or_tokenizer, generate_fn, backend_label).
    """
    try:
        from mlx_vlm import generate as vlm_generate, load as vlm_load
        model, processor = vlm_load(model_path)
        return model, processor, vlm_generate, "mlx_vlm"
    except Exception as vlm_err:
        try:
            from mlx_lm import load as lm_load, stream_generate
            model, tokenizer = lm_load(model_path)
            return model, tokenizer, stream_generate, "mlx_lm"
        except Exception as lm_err:
            raise RuntimeError(
                f"failed via mlx_vlm ({vlm_err!r}) and mlx_lm ({lm_err!r})"
            )


def _generate_and_measure(
    model, processor_or_tokenizer, formatted_prompt: str, max_tokens: int,
    backend: str,
) -> dict:
    """Run one generation and return a metrics dict."""
    if backend == "mlx_vlm":
        from mlx_vlm import generate as vlm_generate
        result = vlm_generate(
            model, processor_or_tokenizer, formatted_prompt,
            max_tokens=max_tokens, verbose=False,
        )
        return {
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "total_tokens": result.total_tokens,
            "prompt_tps": result.prompt_tps,
            "generation_tps": result.generation_tps,
            "peak_memory_gb": result.peak_memory,
            "text": result.text,
        }
    # mlx_lm fallback: stream_generate yields GenerationResponse objects whose
    # last value carries final cumulative metrics in the same shape as mlx_vlm.
    from mlx_lm import stream_generate
    tokenizer = processor_or_tokenizer
    text_parts: list[str] = []
    last = None
    for resp in stream_generate(
        model, tokenizer, formatted_prompt, max_tokens=max_tokens
    ):
        if resp.text:
            text_parts.append(resp.text)
        last = resp
    if last is None:
        return {
            "prompt_tokens": 0, "generation_tokens": 0, "total_tokens": 0,
            "prompt_tps": 0.0, "generation_tps": 0.0, "peak_memory_gb": 0.0,
            "text": "",
        }
    return {
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "total_tokens": last.prompt_tokens + last.generation_tokens,
        "prompt_tps": last.prompt_tps,
        "generation_tps": last.generation_tps,
        "peak_memory_gb": last.peak_memory,
        "text": "".join(text_parts),
    }


def _format_chat(processor_or_tokenizer, prompt: str, thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    return processor_or_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def _bench_one(
    model_path: str,
    max_tokens: int,
    thinking: bool,
    prompts: list[str],
    runs: int,
    output_dir: Path,
) -> dict:
    model_path = os.path.expanduser(model_path)
    model_name = os.path.basename(model_path.rstrip("/"))

    print(f"\n[load] {model_name}", flush=True)
    t0 = time.perf_counter()
    model, proc, _generate_fn, backend = _load_model(model_path)
    load_s = time.perf_counter() - t0
    print(f"        backend={backend}, load={load_s:.1f}s", flush=True)

    formatted_prompts = [_format_chat(proc, p, thinking) for p in prompts]

    print("[warmup] 3 silent passes (cycling prompts)", flush=True)
    _generate_and_measure(model, proc, formatted_prompts[0], 16, backend)
    _generate_and_measure(
        model, proc, formatted_prompts[1 % len(formatted_prompts)], 256, backend
    )
    _generate_and_measure(
        model, proc, formatted_prompts[2 % len(formatted_prompts)], max_tokens, backend
    )

    runs_data = []
    last_text = ""
    for i in range(1, runs + 1):
        prompt_idx = (i - 1) % len(formatted_prompts)
        result = _generate_and_measure(
            model, proc, formatted_prompts[prompt_idx], max_tokens, backend
        )
        m = {k: v for k, v in result.items() if k != "text"}
        m["prompt_index"] = prompt_idx
        runs_data.append(m)
        last_text = result["text"]
        print(
            f"  run {i}/{runs} (p{prompt_idx}): gen={m['generation_tokens']} tok @ "
            f"{m['generation_tps']:.1f} tok/s | "
            f"prompt={m['prompt_tokens']} @ {m['prompt_tps']:.1f} tok/s | "
            f"peak={m['peak_memory_gb']:.2f} GB",
            flush=True,
        )

    gen_tps = [m["generation_tps"] for m in runs_data]
    prompt_tps = [m["prompt_tps"] for m in runs_data]
    peak_mem = [m["peak_memory_gb"] for m in runs_data]
    gen_tokens = [m["generation_tokens"] for m in runs_data]
    prompt_tokens = [m["prompt_tokens"] for m in runs_data]

    gen_mean = statistics.fmean(gen_tps)
    gen_stdev = statistics.stdev(gen_tps) if len(gen_tps) > 1 else 0.0
    prompt_mean = statistics.fmean(prompt_tps)
    # Per-run TTFT and wall-clock; aggregate as means.
    ttft_s_per_run = [
        (pt / ptps) if ptps > 0 else 0.0
        for pt, ptps in zip(prompt_tokens, prompt_tps)
    ]
    wall_s_per_run = [
        (gt / gts) if gts > 0 else 0.0
        for gt, gts in zip(gen_tokens, gen_tps)
    ]

    summary = {
        "model": model_name,
        "model_path": model_path,
        "backend": backend,
        "load_seconds": load_s,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "prompts": prompts,
        "runs": runs_data,
        "generation_tps_mean": gen_mean,
        "generation_tps_median": statistics.median(gen_tps),
        "generation_tps_min": min(gen_tps),
        "generation_tps_max": max(gen_tps),
        "generation_tps_stdev": gen_stdev,
        "generation_tps_cv": (gen_stdev / gen_mean) if gen_mean > 0 else 0.0,
        "prompt_tps_mean": prompt_mean,
        "ttft_seconds_mean": statistics.fmean(ttft_s_per_run),
        "wall_seconds_mean": statistics.fmean(wall_s_per_run),
        "peak_memory_gb_max": max(peak_mem),
        "on_disk_size_gb": _disk_size_gb(model_path),
    }

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    # String concat: Path.with_suffix mangles names with dots (e.g. "qwen3.6").
    base_stem = str(output_dir / f"{model_name}-{timestamp}")
    json_path = Path(base_stem + ".json")
    txt_path = Path(base_stem + ".txt")
    json_path.write_text(json.dumps(summary, indent=2))
    txt_path.write_text(last_text)
    summary["json_path"] = str(json_path.relative_to(REPO_ROOT))

    # Drop refs before next model
    del model, proc
    _free_memory()

    return summary


def _format_table(summaries: list[dict]) -> str:
    rows = sorted(summaries, key=lambda s: -s["generation_tps_mean"])
    fastest = rows[0]["generation_tps_mean"]
    name_w = max(len("model"), max(len(r["model"]) for r in rows))
    header = (
        f"{'model':<{name_w}}  "
        f"{'mean':>6}  {'med':>6}  {'min':>6}  {'max':>6}  "
        f"{'cv%':>5}  {'ttft ms':>8}  {'wall s':>7}  "
        f"{'peak GB':>8}  {'disk GB':>8}  "
        f"{'vs best':>8}  {'load s':>7}"
    )
    sep = "-" * len(header)
    out = [header, sep]
    for r in rows:
        rel = r["generation_tps_mean"] / fastest if fastest else 1.0
        out.append(
            f"{r['model']:<{name_w}}  "
            f"{r['generation_tps_mean']:>6.1f}  "
            f"{r['generation_tps_median']:>6.1f}  "
            f"{r['generation_tps_min']:>6.1f}  "
            f"{r['generation_tps_max']:>6.1f}  "
            f"{r['generation_tps_cv']*100:>5.2f}  "
            f"{r['ttft_seconds_mean']*1000:>8.1f}  "
            f"{r['wall_seconds_mean']:>7.2f}  "
            f"{r['peak_memory_gb_max']:>8.2f}  "
            f"{r['on_disk_size_gb']:>8.2f}  "
            f"{rel*100:>7.1f}%  "
            f"{r['load_seconds']:>7.1f}"
        )
    return "\n".join(out)


def _format_markdown(summaries: list[dict], settings: dict) -> str:
    rows = sorted(summaries, key=lambda s: -s["generation_tps_mean"])
    fastest = rows[0]["generation_tps_mean"]
    lines = [
        f"# Benchmark comparison ({settings['timestamp']})",
        "",
        f"- prompts ({len(settings['prompts'])} cycled): "
        + "; ".join(
            f"`{p[:60]}{'...' if len(p) > 60 else ''}`"
            for p in settings['prompts']
        ),
        f"- max_tokens: {settings['max_tokens']}",
        f"- runs per model: {settings['runs']}",
        f"- thinking: {settings['thinking']}",
        "",
        "| rank | model | gen mean | median | min | max | CV% | TTFT ms | wall s | peak GB | disk GB | vs best | load s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        rel = r["generation_tps_mean"] / fastest if fastest else 1.0
        lines.append(
            f"| {i} | `{r['model']}` | "
            f"{r['generation_tps_mean']:.1f} | "
            f"{r['generation_tps_median']:.1f} | "
            f"{r['generation_tps_min']:.1f} | "
            f"{r['generation_tps_max']:.1f} | "
            f"{r['generation_tps_cv']*100:.2f} | "
            f"{r['ttft_seconds_mean']*1000:.1f} | "
            f"{r['wall_seconds_mean']:.2f} | "
            f"{r['peak_memory_gb_max']:.2f} | "
            f"{r['on_disk_size_gb']:.2f} | "
            f"{rel*100:.1f}% | "
            f"{r['load_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple MLX VLM models")
    parser.add_argument("model_paths", nargs="*", help="Paths to MLX model directories")
    parser.add_argument(
        "--models-from",
        type=Path,
        help="Optional file with model paths (one per line, # for comments)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking mode"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt(s) to cycle through across runs (repeatable). "
             "Defaults to a curated set of varied prompts.",
    )
    parser.add_argument(
        "--prompts-from",
        type=Path,
        help="File with prompts (one per line, # for comments). Combined with --prompt.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs per model after warmup",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-model JSON + text artifacts",
    )
    parser.add_argument(
        "--report-name",
        default=None,
        help="Stem for the comparison report file (default: timestamped)",
    )
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=True,
        help="Shuffle model load order to mitigate thermal/state ordering bias "
             "(default: on for multi-model runs).",
    )
    parser.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false",
        help="Disable shuffle; load models in argument order.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="PRNG seed for shuffle ordering (default: time-based, non-reproducible).",
    )
    parser.add_argument(
        "--prewarm-seconds", type=int, default=0,
        help="Run sustained GPU matmul for N seconds before the first model to "
             "bring the device to thermal steady-state. Default 0 (off). "
             "Only effective for multi-model runs.",
    )
    args = parser.parse_args()

    paths: list[str] = list(args.model_paths)
    if args.models_from:
        for line in args.models_from.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    if not paths:
        paths = [DEFAULT_MODEL]
        print(f"[info] no models specified; defaulting to {DEFAULT_MODEL}", flush=True)

    is_multi = len(paths) > 1
    if is_multi and args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        print(
            f"[shuffle] order (seed={args.seed!r}): "
            + ", ".join(os.path.basename(p) for p in paths),
            flush=True,
        )
    if is_multi and args.prewarm_seconds > 0:
        _thermal_prewarm(args.prewarm_seconds)
    elif not is_multi and args.prewarm_seconds > 0:
        print(
            "[warn] --prewarm-seconds ignored for single-model runs",
            file=sys.stderr,
        )

    prompts: list[str] = list(args.prompt) if args.prompt else []
    if args.prompts_from:
        for line in args.prompts_from.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = {
        "timestamp": timestamp,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "thinking": args.thinking,
        "prompts": prompts,
    }

    summaries: list[dict] = []
    for p in paths:
        try:
            s = _bench_one(
                p, args.max_tokens, args.thinking, prompts,
                args.runs, args.output_dir,
            )
            summaries.append(s)
        except KeyboardInterrupt:
            print("\n[abort] interrupted", file=sys.stderr)
            break
        except Exception as e:
            print(f"[error] {p}: {e}", file=sys.stderr)

    if not summaries:
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.report_name or f"compare-{timestamp}"
    md_path = args.output_dir / f"{stem}.md"
    json_path = args.output_dir / f"{stem}.json"
    md_path.write_text(_format_markdown(summaries, settings))
    json_path.write_text(json.dumps({"settings": settings, "results": summaries}, indent=2))

    print("\n" + _format_table(summaries))
    print(
        f"\n[done] report={md_path.relative_to(REPO_ROOT)} "
        f"json={json_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
