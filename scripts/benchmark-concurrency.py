#!/usr/bin/env python3
"""Concurrency / batch throughput probe for a loaded MLX-LM model.

Three phases:
  1. Compatibility probe — does ``mlx_lm.batch_generate`` actually run on the
     loaded architecture? Many models (e.g. linear-attention / Mamba / sliding
     window variants) are NOT yet supported by mlx_lm's BatchKVCache as of the
     v0.31.x line. Source: https://github.com/ml-explore/mlx-lm/pull/443
  2. Batch sweep (only if compat=PASS) — sweep batch sizes and report
     aggregate gen tok/s, per-stream gen tok/s, and the scaling factor vs B=1.
  3. Thread control (negative result) — fire N OS threads each running
     ``stream_generate`` against the same loaded model. Expectation: total
     throughput ~= single-thread, because all threads serialize onto the same
     Metal stream. Confirms that thread-level concurrency is not the lever.

Outputs JSON to ``reports/benchmarks/<report-name>.json``.

Usage:
    python3 scripts/benchmark-concurrency.py
    python3 scripts/benchmark-concurrency.py --model ~/models/<name>
    python3 scripts/benchmark-concurrency.py --batches 1,2,4,8 --max-tokens 256
    python3 scripts/benchmark-concurrency.py --threads-control 4
"""

import argparse
import datetime as dt
import gc
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

# Make sibling helper module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_lib import BandwidthSampler, detect_machine_specs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "~/models/qwen3.6-35b-opus-abl-mxfp4-mlx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "benchmarks"

# Same prompt set as benchmark-compare.py for consistency.
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


def _free_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx
        clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
        if clear is not None:
            clear()
    except Exception:
        pass


def _format_chat_ids(tokenizer, prompt: str) -> list[int]:
    """Return token ids (mlx_lm.batch_generate expects List[int] per prompt)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        # tokenize=True is the default; returns a list of ints.
    )


def _format_chat_str(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def probe_batch_compat(model, tokenizer, prompt_ids: list[int]) -> tuple[bool, str]:
    """Returns (ok, message). Smoke-tests mlx_lm.batch_generate at B=2."""
    try:
        from mlx_lm import batch_generate
    except ImportError as e:
        return False, f"batch_generate not exported by mlx_lm ({e})"
    try:
        result = batch_generate(
            model, tokenizer, [prompt_ids, prompt_ids],
            max_tokens=8, verbose=False,
        )
        sample = (result.texts[0] if result.texts else "")[:40].replace("\n", " ")
        return True, f"compatible (sample text head: {sample!r})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def measure_batch(model, tokenizer, prompt_ids_list: list[list[int]], max_tokens: int) -> dict:
    """Run a single batch_generate and return aggregated metrics.

    The mlx_lm BatchStats object reports ``generation_tps`` as **aggregate**
    throughput (total tokens / wall seconds), not per-stream. We surface both
    aggregate and per-stream so callers can reason about scaling cleanly.
    """
    from mlx_lm import batch_generate
    t0 = time.perf_counter()
    result = batch_generate(
        model, tokenizer, prompt_ids_list,
        max_tokens=max_tokens, verbose=False,
    )
    elapsed = time.perf_counter() - t0
    stats = getattr(result, "stats", None)
    b = len(prompt_ids_list)
    agg_tps = getattr(stats, "generation_tps", 0.0) if stats else 0.0
    return {
        "batch_size": b,
        "wall_seconds": elapsed,
        "prompt_tokens": getattr(stats, "prompt_tokens", 0) if stats else 0,
        "prompt_tps": getattr(stats, "prompt_tps", 0.0) if stats else 0.0,
        "generation_tokens": getattr(stats, "generation_tokens", 0) if stats else 0,
        "generation_tps_aggregate": agg_tps,
        "generation_tps_per_stream": (agg_tps / b) if b > 0 else 0.0,
        "peak_memory_gb": getattr(stats, "peak_memory", 0.0) if stats else 0.0,
    }


def measure_thread_control(
    model, tokenizer, prompt_str: str, max_tokens: int, n_threads: int
) -> dict:
    """Fire N threads, each running stream_generate. Returns aggregated metrics.

    Threads are gated by a Barrier so they all start together; total wall is
    measured around the join. Aggregate tok/s = sum(generated tokens) / wall.
    """
    from mlx_lm import stream_generate
    results: list[dict] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        t_start = time.perf_counter()
        last = None
        for resp in stream_generate(model, tokenizer, prompt_str, max_tokens=max_tokens):
            last = resp
        elapsed = t_start_local = time.perf_counter() - t_start
        gen = last.generation_tokens if last else 0
        with results_lock:
            results.append({
                "elapsed": elapsed,
                "gen_tokens": gen,
                "tps": (gen / elapsed) if elapsed > 0 else 0.0,
            })

    t_wall_start = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_wall = time.perf_counter() - t_wall_start

    total_gen = sum(r["gen_tokens"] for r in results)
    elapsed_list = [r["elapsed"] for r in results]
    tps_list = [r["tps"] for r in results]

    def _pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        # Nearest-rank (simple, fine for small N).
        k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[k]

    return {
        "n_threads": n_threads,
        "total_wall_seconds": total_wall,
        "aggregate_tps": (total_gen / total_wall) if total_wall > 0 else 0.0,
        "per_stream_tps": tps_list,
        "per_stream_elapsed": elapsed_list,
        "per_stream_elapsed_p50": _pct(elapsed_list, 0.50),
        "per_stream_elapsed_p95": _pct(elapsed_list, 0.95),
        "per_stream_elapsed_max": max(elapsed_list) if elapsed_list else 0.0,
        "per_stream_tps_stdev": (
            statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0
        ),
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    base = rows[0]["generation_tps_aggregate"]
    header = (
        f"{'B':>3}  "
        f"{'agg tok/s':>10}  "
        f"{'per-stream':>10}  "
        f"{'agg vs B=1':>11}  "
        f"{'per-stream eff':>14}  "
        f"{'wall s':>7}  "
        f"{'peak GB':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        b = r["batch_size"]
        agg = r["generation_tps_aggregate"]
        per = r["generation_tps_per_stream"]
        scale = (agg / base) if base > 0 else 0.0
        eff = (per / (base / 1)) if base > 0 else 0.0  # vs single-stream baseline
        print(
            f"{b:>3}  "
            f"{agg:>10.1f}  "
            f"{per:>10.1f}  "
            f"{scale:>10.2f}x  "
            f"{eff*100:>13.1f}%  "
            f"{r['wall_seconds']:>7.1f}  "
            f"{r['peak_memory_gb']:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Probe concurrency / batch throughput for an MLX model"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Path to MLX model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Generation length per stream (default: 256)",
    )
    parser.add_argument(
        "--batches", default="1,2,4,8,16,32",
        help="Comma-separated batch sizes to sweep (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--threads-control", type=int, default=4,
        help="Threads for the negative-control test. 0 = skip. (default: 4)",
    )
    parser.add_argument(
        "--bandwidth-sampling", action="store_true", default=True,
        help="Sample DRAM bandwidth + GPU power via mactop during measurements (default: on).",
    )
    parser.add_argument(
        "--no-bandwidth-sampling", dest="bandwidth_sampling", action="store_false",
        help="Disable mactop bandwidth sampling.",
    )
    parser.add_argument(
        "--bandwidth-interval-ms", type=int, default=1000,
        help="mactop sampling interval in ms (default: 1000).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--report-name", default=None,
        help="Stem for the JSON report (default: timestamped concurrency-...)",
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    model_name = os.path.basename(model_path.rstrip("/"))

    specs = detect_machine_specs()
    print(
        f"[machine] {specs.chip} | gpu_cores={specs.gpu_core_count} | "
        f"ram={specs.ram_gb:.1f} GB | dram_bw_spec={specs.dram_bandwidth_gbs} GB/s "
        f"({specs.bandwidth_confidence})",
        flush=True,
    )

    print(f"[load] {model_name} (via mlx_lm)", flush=True)
    from mlx_lm import load as lm_load
    t0 = time.perf_counter()
    model, tokenizer = lm_load(model_path)
    load_s = time.perf_counter() - t0
    print(f"        load={load_s:.1f}s", flush=True)

    prompt_ids = _format_chat_ids(tokenizer, DEFAULT_PROMPTS[0])
    prompt_str = _format_chat_str(tokenizer, DEFAULT_PROMPTS[0])

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out: dict = {
        "settings": {
            "model": model_name,
            "model_path": model_path,
            "max_tokens": args.max_tokens,
            "batches": args.batches,
            "threads_control": args.threads_control,
            "timestamp": timestamp,
            "load_seconds": load_s,
            "machine": specs.as_dict(),
        },
    }

    # Phase 1: compatibility probe.
    print("[probe] mlx_lm.batch_generate compatibility ...", flush=True)
    compat_ok, compat_msg = probe_batch_compat(model, tokenizer, prompt_ids)
    print(f"        {'PASS' if compat_ok else 'FAIL'}: {compat_msg}", flush=True)
    out["compat"] = {"batch_generate": compat_ok, "message": compat_msg}

    # Phase 2: batch sweep.
    if compat_ok:
        try:
            batch_sizes = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
        except ValueError:
            print(f"[error] invalid --batches: {args.batches!r}", file=sys.stderr)
            sys.exit(2)
        print(f"[batch sweep] B={batch_sizes}, max_tokens={args.max_tokens}", flush=True)
        # Light warmup at B=1 to compile kernels.
        _ = measure_batch(model, tokenizer, [prompt_ids], max_tokens=32)

        # Continuous sampler across the entire sweep. Pre-warm so the first
        # batch window has data (mactop's first sample lags ~5s).
        sampler: BandwidthSampler | None = None
        if args.bandwidth_sampling:
            sampler = BandwidthSampler(
                spec_gbs=specs.dram_bandwidth_gbs,
                interval_ms=args.bandwidth_interval_ms,
            )
            sampler.start()
            if sampler.unavailable_reason:
                print(
                    f"  [bandwidth] disabled: {sampler.unavailable_reason}",
                    flush=True,
                )
                sampler = None
            else:
                print(
                    "  [bandwidth] warming mactop (~5s for first sample)...",
                    flush=True,
                )
                if not sampler.wait_until_warm(timeout_s=10.0):
                    print("  [bandwidth] warm timeout; will record any later samples", flush=True)

        sweep_rows: list[dict] = []
        try:
            for b in batch_sizes:
                prompts_b = [
                    _format_chat_ids(tokenizer, DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)])
                    for i in range(b)
                ]
                print(f"  B={b} ...", end=" ", flush=True)
                t_win_start = time.perf_counter()
                try:
                    m = measure_batch(model, tokenizer, prompts_b, args.max_tokens)
                except Exception as e:
                    print(f"FAIL ({type(e).__name__}: {e})", flush=True)
                    sweep_rows.append({"batch_size": b, "error": f"{type(e).__name__}: {e}"})
                    continue
                t_win_end = time.perf_counter()

                bw_str = ""
                if sampler is not None:
                    win_summary = sampler.summary_window(t_win_start, t_win_end)
                    m["bandwidth"] = win_summary.as_dict()
                    if win_summary.enabled:
                        util = (
                            f", bw_util_peak={win_summary.bw_util_peak_pct:.1f}%"
                            if win_summary.bw_util_peak_pct is not None else ""
                        )
                        bw_str = (
                            f" | gpu_pwr_peak={win_summary.gpu_power_peak_w:.1f} W"
                            f" | dram_pwr_mean={win_summary.dram_power_mean_w:.1f} W"
                            f" | dram_total_peak={win_summary.dram_total_peak_gbs:.1f} GB/s"
                            f"{util}"
                        )
                sweep_rows.append(m)
                print(
                    f"agg={m['generation_tps_aggregate']:.1f} tok/s, "
                    f"per-stream={m['generation_tps_per_stream']:.1f}, "
                    f"wall={m['wall_seconds']:.1f}s, "
                    f"peak={m['peak_memory_gb']:.2f} GB" + bw_str,
                    flush=True,
                )
        finally:
            if sampler is not None:
                sampler.stop()
                out["bandwidth_global"] = sampler.summary.as_dict()

        out["batch_sweep"] = sweep_rows
        valid_rows = [r for r in sweep_rows if "error" not in r]
        if valid_rows:
            print()
            _print_table(valid_rows)
    else:
        print("[batch sweep] skipped (compat=FAIL)", flush=True)

    # Phase 3: thread control.
    if args.threads_control > 0:
        n = args.threads_control
        print(
            f"\n[control] {n} OS threads each running stream_generate "
            f"(expected: serialize on Metal stream, ~no aggregate uplift)",
            flush=True,
        )
        # Warm baseline so kernel cache is hot.
        _ = measure_thread_control(model, tokenizer, prompt_str, max_tokens=64, n_threads=1)

        single = measure_thread_control(model, tokenizer, prompt_str, args.max_tokens, n_threads=1)
        multi = measure_thread_control(model, tokenizer, prompt_str, args.max_tokens, n_threads=n)
        speedup = (multi["aggregate_tps"] / single["aggregate_tps"]) if single["aggregate_tps"] > 0 else 0.0
        print(
            f"  single thread: {single['aggregate_tps']:.1f} tok/s aggregate "
            f"({single['total_wall_seconds']:.1f}s wall)",
            flush=True,
        )
        print(
            f"  {n}x threads:   {multi['aggregate_tps']:.1f} tok/s aggregate "
            f"({multi['total_wall_seconds']:.1f}s wall, speedup={speedup:.2f}x)",
            flush=True,
        )
        out["threads_control"] = {
            "single_thread": single,
            "multi_thread": multi,
            "aggregate_speedup": speedup,
        }

    # Save.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.report_name or f"concurrency-{timestamp}"
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] report={json_path.relative_to(REPO_ROOT)}", flush=True)

    _free_memory()


if __name__ == "__main__":
    main()
