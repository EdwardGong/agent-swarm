#!/usr/bin/env python3
"""Concurrency benchmark for a running vllm-mlx server.

Mirrors the shape of ``benchmark-concurrency.py`` so results compare cleanly:
  - Same default prompts (cycled).
  - Same concurrency sweep [1, 2, 4, 8, 16, 32].
  - Same continuous-mactop bandwidth sampler with per-window slicing.
  - Per-request TTFT (time-to-first-token), per-request total time.
  - Aggregate gen tok/s = total completion tokens / wall window seconds.

Pre-req: a running vllm-mlx server on the configured URL.
  Start with: ``scripts/run-vllm-mlx-server.sh``  (default port 8000)

Outputs JSON to ``reports/benchmarks/<report-name>.json``.

Usage:
    python3 scripts/benchmark-vllm-mlx-client.py
    python3 scripts/benchmark-vllm-mlx-client.py --concurrencies 1,2,4,8
    python3 scripts/benchmark-vllm-mlx-client.py --max-tokens 512
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

# Make sibling helper module importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_lib import BandwidthSampler, detect_machine_specs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "benchmarks"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_SERVED_MODEL = "opus-mxfp4"

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


async def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> bool:
    """Poll the server's health endpoint until it responds or we time out."""
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.perf_counter() < deadline:
            for path in ("/health", "/v1/models"):
                try:
                    r = await client.get(f"{base_url}{path}")
                    if r.status_code < 500:
                        return True
                except (httpx.ConnectError, httpx.ReadTimeout):
                    pass
            await asyncio.sleep(0.5)
    return False


async def _scrape_metrics(base_url: str) -> Optional[str]:
    """Best-effort fetch of /metrics (Prometheus exposition). Returns text or None."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{base_url}/metrics")
            if r.status_code == 200:
                return r.text
    except Exception:
        pass
    return None


async def _one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
) -> dict:
    """Drive one OpenAI-style chat-completions streaming request.

    If ``priority`` is given, it is included in the request body as the
    ``priority`` field (lower = higher priority on a vllm-mlx server
    started with ``--scheduling-policy priority``). The official OpenAI
    SDK forwards this via ``extra_body={"priority": N}``; we just put it
    directly in the JSON.

    Returns metrics dict with TTFT, total wall, completion_tokens.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    if priority is not None:
        payload["priority"] = priority
    t_send = time.perf_counter()
    t_first_token: Optional[float] = None
    n_completion = 0
    n_prompt = 0
    last_usage: Optional[dict] = None
    error: Optional[str] = None
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions",
            json=payload, timeout=httpx.Timeout(120.0, connect=10.0),
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # Most OpenAI-compatible servers emit `choices[].delta.content`;
                # vllm-mlx may also include a final `usage` block.
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        n_completion += 1
                if obj.get("usage"):
                    last_usage = obj["usage"]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    t_done = time.perf_counter()

    if last_usage:
        # Authoritative token counts when the server reports them.
        n_prompt = int(last_usage.get("prompt_tokens", 0)) or 0
        n_completion = int(last_usage.get("completion_tokens", n_completion)) or n_completion

    ttft_s = (t_first_token - t_send) if t_first_token else None
    wall_s = t_done - t_send
    return {
        "ok": error is None,
        "error": error,
        "ttft_s": ttft_s,
        "wall_s": wall_s,
        "prompt_tokens": n_prompt,
        "completion_tokens": n_completion,
        "tps": (n_completion / (wall_s - (ttft_s or 0)))
        if (n_completion and wall_s and ttft_s and wall_s > ttft_s)
        else (n_completion / wall_s if wall_s > 0 else 0.0),
    }


async def _run_concurrency(
    base_url: str, model: str, prompts: list[str],
    concurrency: int, max_tokens: int,
    priority: Optional[int] = None,
) -> dict:
    """Fire `concurrency` requests in parallel; return per-window metrics."""
    selected = [prompts[i % len(prompts)] for i in range(concurrency)]
    t_window_start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                _one_request(client, base_url, model, p, max_tokens, priority)
                for p in selected
            ],
            return_exceptions=False,
        )
    t_window_end = time.perf_counter()
    wall = t_window_end - t_window_start

    ok_results = [r for r in results if r["ok"]]
    n_completion_total = sum(r["completion_tokens"] for r in ok_results)
    ttfts = [r["ttft_s"] for r in ok_results if r["ttft_s"] is not None]
    walls = [r["wall_s"] for r in ok_results]

    def _pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[k]

    return {
        "concurrency": concurrency,
        "wall_seconds": wall,
        "successful": len(ok_results),
        "failed": len(results) - len(ok_results),
        "completion_tokens_total": n_completion_total,
        "generation_tps_aggregate": (n_completion_total / wall) if wall > 0 else 0.0,
        "generation_tps_per_stream": (
            (n_completion_total / wall) / concurrency if (wall > 0 and concurrency > 0) else 0.0
        ),
        "ttft_mean_s": (sum(ttfts) / len(ttfts)) if ttfts else 0.0,
        "ttft_p50_s": _pct(ttfts, 0.50),
        "ttft_p95_s": _pct(ttfts, 0.95),
        "wall_p50_s": _pct(walls, 0.50),
        "wall_p95_s": _pct(walls, 0.95),
        "wall_max_s": max(walls) if walls else 0.0,
        "errors": [r["error"] for r in results if not r["ok"]],
        "_t_window_start": t_window_start,
        "_t_window_end": t_window_end,
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    base = rows[0]["generation_tps_aggregate"]
    header = (
        f"{'C':>3}  "
        f"{'agg tok/s':>10}  "
        f"{'per-stream':>10}  "
        f"{'agg vs C=1':>10}  "
        f"{'TTFT p50 ms':>11}  "
        f"{'TTFT p95 ms':>11}  "
        f"{'wall p95 s':>10}  "
        f"{'ok':>4}/{'fail':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        c = r["concurrency"]
        agg = r["generation_tps_aggregate"]
        per = r["generation_tps_per_stream"]
        scale = (agg / base) if base > 0 else 0.0
        print(
            f"{c:>3}  "
            f"{agg:>10.1f}  "
            f"{per:>10.1f}  "
            f"{scale:>9.2f}x  "
            f"{r['ttft_p50_s']*1000:>11.1f}  "
            f"{r['ttft_p95_s']*1000:>11.1f}  "
            f"{r['wall_p95_s']:>10.2f}  "
            f"{r['successful']:>4}/{r['failed']:>4}"
        )


async def amain():
    parser = argparse.ArgumentParser(
        description="Benchmark a running vllm-mlx server via OpenAI-compatible API"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_SERVED_MODEL,
                        help="served-model-name as known to the server (default: opus-mxfp4)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--concurrencies", default="1,2,4,8,16,32",
        help="Comma-separated concurrency levels to sweep.",
    )
    parser.add_argument(
        "--bandwidth-sampling", action="store_true", default=True,
        help="Sample DRAM bandwidth + GPU power via mactop (default on).",
    )
    parser.add_argument(
        "--no-bandwidth-sampling", dest="bandwidth_sampling", action="store_false",
    )
    parser.add_argument("--bandwidth-interval-ms", type=int, default=1000)
    parser.add_argument(
        "--priority", type=int, default=None,
        help="Per-request priority sent in the body as 'priority'. Lower = "
             "higher priority. Requires the server to be running with "
             "--scheduling-policy priority. Conventions: orchestrator=-10, "
             "default worker=0 (omit), background worker=10.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()

    specs = detect_machine_specs()
    print(
        f"[machine] {specs.chip} | gpu_cores={specs.gpu_core_count} | "
        f"ram={specs.ram_gb:.1f} GB | dram_bw_spec={specs.dram_bandwidth_gbs} GB/s "
        f"({specs.bandwidth_confidence})",
        flush=True,
    )

    print(f"[server] waiting for {args.base_url} to become healthy ...", flush=True)
    if not await _wait_for_health(args.base_url, timeout_s=30.0):
        print(
            f"[error] no response from {args.base_url} within 30s. "
            f"Start the server: scripts/run-vllm-mlx-server.sh",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[server] healthy", flush=True)

    try:
        concurrencies = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    except ValueError:
        print(f"[error] invalid --concurrencies: {args.concurrencies!r}", file=sys.stderr)
        sys.exit(2)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out: dict = {
        "settings": {
            "base_url": args.base_url,
            "served_model": args.model,
            "max_tokens": args.max_tokens,
            "concurrencies": concurrencies,
            "priority": args.priority,
            "timestamp": timestamp,
            "machine": specs.as_dict(),
        },
    }

    # Warmup: one request to trigger any first-time loads.
    print("[warmup] single request ...", flush=True)
    async with httpx.AsyncClient() as client:
        await _one_request(
            client, args.base_url, args.model, DEFAULT_PROMPTS[0], 32,
            priority=args.priority,
        )

    # Continuous bandwidth sampler around the whole sweep.
    sampler: BandwidthSampler | None = None
    if args.bandwidth_sampling:
        sampler = BandwidthSampler(
            spec_gbs=specs.dram_bandwidth_gbs,
            interval_ms=args.bandwidth_interval_ms,
        )
        sampler.start()
        if sampler.unavailable_reason:
            print(f"[bandwidth] disabled: {sampler.unavailable_reason}", flush=True)
            sampler = None
        else:
            print("[bandwidth] warming mactop (~5s for first sample)...", flush=True)
            sampler.wait_until_warm(timeout_s=10.0)

    sweep_rows: list[dict] = []
    print(f"[sweep] C={concurrencies}, max_tokens={args.max_tokens}", flush=True)
    try:
        for c in concurrencies:
            tag = f" prio={args.priority}" if args.priority is not None else ""
            print(f"  C={c}{tag} ...", end=" ", flush=True)
            r = await _run_concurrency(
                args.base_url, args.model, DEFAULT_PROMPTS, c, args.max_tokens,
                priority=args.priority,
            )
            bw_str = ""
            if sampler is not None:
                win = sampler.summary_window(r["_t_window_start"], r["_t_window_end"])
                r["bandwidth"] = win.as_dict()
                if win.enabled:
                    util = (
                        f", bw_util_peak={win.bw_util_peak_pct:.1f}%"
                        if win.bw_util_peak_pct is not None else ""
                    )
                    bw_str = (
                        f" | gpu_pwr_peak={win.gpu_power_peak_w:.1f} W"
                        f" | dram_total_peak={win.dram_total_peak_gbs:.1f} GB/s"
                        f"{util}"
                    )
            sweep_rows.append(r)
            print(
                f"agg={r['generation_tps_aggregate']:.1f} tok/s, "
                f"per-stream={r['generation_tps_per_stream']:.1f}, "
                f"TTFT_p50={r['ttft_p50_s']*1000:.0f} ms, "
                f"wall_p95={r['wall_p95_s']:.1f}s, "
                f"ok={r['successful']}/{r['successful']+r['failed']}" + bw_str,
                flush=True,
            )
    finally:
        if sampler is not None:
            sampler.stop()
            out["bandwidth_global"] = sampler.summary.as_dict()

    out["sweep"] = sweep_rows

    # Optional Prometheus snapshot at end (cheap, useful for cross-checks).
    metrics_text = await _scrape_metrics(args.base_url)
    if metrics_text:
        out["prometheus_snapshot"] = metrics_text

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.report_name or f"vllm-mlx-{timestamp}"
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps({k: v for k, v in out.items() if not k.startswith("_")},
                                    indent=2, default=str))

    valid = [r for r in sweep_rows if r.get("successful", 0) > 0]
    if valid:
        print()
        _print_table(valid)
    print(f"\n[done] report={json_path.relative_to(REPO_ROOT)}", flush=True)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
