#!/usr/bin/env python3
"""Benchmark an MLX VLM model with consistent methodology.

Generated text is written to disk; only compact metrics print to stdout.

Usage:
    python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx
    python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx --max-tokens 2048
    python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx --thinking
    python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx --prompt "Explain quicksort."
    python3 scripts/benchmark-mlx.py ~/models/qwen3.6-35b-opus-abl-4bit-vlm-mlx --runs 3
"""

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from pathlib import Path

DEFAULT_PROMPT = (
    "Design a distributed message queue system with partitioning, "
    "replication, and exactly-once delivery."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "benchmarks"
DEFAULT_MODEL = "~/models/qwen3.6-35b-opus-abl-mxfp4-mlx"


def _load_model(model_path: str):
    """Load via mlx_vlm; fall back to mlx_lm for vision-stripped models."""
    try:
        from mlx_vlm import load as vlm_load
        model, processor = vlm_load(model_path)
        return model, processor, "mlx_vlm"
    except Exception as vlm_err:
        try:
            from mlx_lm import load as lm_load
            model, tokenizer = lm_load(model_path)
            return model, tokenizer, "mlx_lm"
        except Exception as lm_err:
            raise RuntimeError(
                f"failed via mlx_vlm ({vlm_err!r}) and mlx_lm ({lm_err!r})"
            )


def _generate_one(model, proc, prompt: str, max_tokens: int, backend: str):
    """Run one generation; return a metrics dict + text."""
    if backend == "mlx_vlm":
        from mlx_vlm import generate as vlm_generate
        r = vlm_generate(model, proc, prompt, max_tokens=max_tokens, verbose=False)
        return {
            "prompt_tokens": r.prompt_tokens,
            "generation_tokens": r.generation_tokens,
            "total_tokens": r.total_tokens,
            "prompt_tps": r.prompt_tps,
            "generation_tps": r.generation_tps,
            "peak_memory_gb": r.peak_memory,
            "text": r.text,
        }
    from mlx_lm import stream_generate
    text_parts: list[str] = []
    last = None
    for resp in stream_generate(model, proc, prompt, max_tokens=max_tokens):
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


def _result_to_dict(result) -> dict:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "total_tokens": result.total_tokens,
        "prompt_tps": result.prompt_tps,
        "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory,
    }


def _fmt_metrics(label: str, m: dict) -> str:
    return (
        f"{label}: "
        f"gen={m['generation_tokens']} tok @ {m['generation_tps']:.1f} tok/s | "
        f"prompt={m['prompt_tokens']} tok @ {m['prompt_tps']:.1f} tok/s | "
        f"peak={m['peak_memory_gb']:.2f} GB"
    )


def run_benchmark(
    model_path: str,
    max_tokens: int,
    thinking: bool,
    prompt: str,
    runs: int,
    output_dir: Path,
):
    model_path = os.path.expanduser(model_path)
    model_name = os.path.basename(model_path.rstrip("/"))

    print(f"[load] {model_name}", flush=True)
    model, processor, backend = _load_model(model_path)
    print(f"        backend={backend}", flush=True)

    messages = [{"role": "user", "content": prompt}]
    formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    # Warmup: 3 silent passes of increasing length
    print("[warmup] 3 passes (silent)", flush=True)
    _generate_one(model, processor, formatted, 16, backend)
    _generate_one(model, processor, formatted, 256, backend)
    _generate_one(model, processor, formatted, max_tokens, backend)

    thinking_label = "thinking=ON" if thinking else "thinking=OFF"
    print(
        f"[run]  {model_name} | max_tokens={max_tokens} | {thinking_label} | "
        f"runs={runs}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Build paths via string concat: Path.with_suffix mangles names with dots
    # (e.g. "qwen3.6-foo" -> ".with_suffix('.json')" yields "qwen3.json").
    base_stem = str(output_dir / f"{model_name}-{timestamp}")

    all_metrics: list[dict] = []
    last_text = ""
    for i in range(1, runs + 1):
        result = _generate_one(model, processor, formatted, max_tokens, backend)
        metrics = {k: v for k, v in result.items() if k != "text"}
        all_metrics.append(metrics)
        last_text = result["text"]
        print(_fmt_metrics(f"  run {i}/{runs}", metrics), flush=True)

    gen_tps = [m["generation_tps"] for m in all_metrics]
    summary = {
        "model": model_name,
        "model_path": model_path,
        "timestamp": timestamp,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "prompt": prompt,
        "runs": all_metrics,
        "generation_tps_mean": statistics.fmean(gen_tps),
        "generation_tps_min": min(gen_tps),
        "generation_tps_max": max(gen_tps),
        "generation_tps_stdev": statistics.stdev(gen_tps) if len(gen_tps) > 1 else 0.0,
    }

    json_path = Path(base_stem + ".json")
    txt_path = Path(base_stem + ".txt")
    json_path.write_text(json.dumps(summary, indent=2))
    txt_path.write_text(last_text)

    print(
        f"[done] mean={summary['generation_tps_mean']:.1f} tok/s "
        f"(min={summary['generation_tps_min']:.1f}, "
        f"max={summary['generation_tps_max']:.1f}, "
        f"stdev={summary['generation_tps_stdev']:.2f}) | "
        f"output={json_path.relative_to(REPO_ROOT)}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark an MLX VLM model")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to the MLX model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--thinking", action="store_true", help="Enable thinking mode"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measured runs after warmup (mean/min/max reported)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write benchmark JSON + generated text",
    )
    args = parser.parse_args()

    try:
        run_benchmark(
            args.model_path,
            args.max_tokens,
            args.thinking,
            args.prompt,
            args.runs,
            args.output_dir,
        )
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
