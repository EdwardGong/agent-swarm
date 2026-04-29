"""Shared utilities for the bench scripts.

- ``detect_machine_specs`` infers chip identity, GPU core count, RAM, and the
  theoretical DRAM bandwidth ceiling for the host. We use these to compute
  ``bw_util %`` (peak observed DRAM bandwidth divided by the spec maximum).

- ``BandwidthSampler`` is a context manager that runs ``mactop --headless`` in
  the background while a measurement runs, then exposes mean/peak DRAM
  read/write bandwidth, GPU power, and thermal observations. Sampling only
  happens during the ``with`` block so warmup runs are excluded.

The DRAM bandwidth values mactop reports on M5+ are auto-calibrated power-based
estimates per the mactop README, not direct DCS-counter reads (Apple removed
those from powermetrics in macOS 13). On M3-class hardware mactop currently
reports zeros for ``dram_read_bw_gbs`` / ``dram_write_bw_gbs`` in light testing,
so we always surface ``dram_power_W`` (a strict ground truth from IOReport) as a
fallback proxy. Treat ``bw_util %`` as a relative comparator across runs on the
same machine, not as ground truth.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# --- machine specs ------------------------------------------------------------

# Apple-published peak DRAM bandwidth (GB/s) by chip identifier. For chips with
# multiple SKUs (M3 Max 30-core vs 40-core GPU, M4 Max binned), entries below
# are keyed on (chip_name, gpu_core_count). HIGH confidence for M1-M4 entries
# from Apple's own product pages; M5* entries are best-effort and flagged.
#
# Reference: each chip's Apple announcement page; cross-referenced with
# https://en.wikipedia.org/wiki/Apple_silicon
KNOWN_BANDWIDTH_GBS: dict[tuple[str, Optional[int]], tuple[float, str]] = {
    # (chip, gpu_core_count or None) -> (GB/s, confidence)
    ("Apple M1", None):                (68.25, "HIGH"),
    ("Apple M1 Pro", None):           (200.0, "HIGH"),
    ("Apple M1 Max", None):           (400.0, "HIGH"),
    ("Apple M1 Ultra", None):         (800.0, "HIGH"),
    ("Apple M2", None):               (100.0, "HIGH"),
    ("Apple M2 Pro", None):           (200.0, "HIGH"),
    ("Apple M2 Max", None):           (400.0, "HIGH"),
    ("Apple M2 Ultra", None):         (800.0, "HIGH"),
    ("Apple M3", None):               (100.0, "HIGH"),
    ("Apple M3 Pro", None):           (150.0, "HIGH"),
    ("Apple M3 Max", 30):             (300.0, "HIGH"),
    ("Apple M3 Max", 40):             (400.0, "HIGH"),
    ("Apple M3 Ultra", None):         (800.0, "HIGH"),
    ("Apple M4", None):               (120.0, "HIGH"),
    ("Apple M4 Pro", None):           (273.0, "HIGH"),
    ("Apple M4 Max", 32):             (410.0, "HIGH"),
    ("Apple M4 Max", 40):             (546.0, "HIGH"),
    # M5 family: Apple has not published a unified table at time of writing;
    # values below are best-effort estimates and should be verified.
    ("Apple M5", None):               (153.0, "MEDIUM"),
    ("Apple M5 Pro", None):           (273.0, "MEDIUM"),
    ("Apple M5 Max", None):           (546.0, "MEDIUM"),
    ("Apple M5 Ultra", None):         (1092.0, "LOW"),
}


@dataclass
class MachineSpecs:
    chip: str
    gpu_core_count: Optional[int]
    ram_gb: float
    dram_bandwidth_gbs: Optional[float]
    bandwidth_confidence: str  # "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"

    def as_dict(self) -> dict:
        return {
            "chip": self.chip,
            "gpu_core_count": self.gpu_core_count,
            "ram_gb": round(self.ram_gb, 2),
            "dram_bandwidth_gbs": self.dram_bandwidth_gbs,
            "bandwidth_confidence": self.bandwidth_confidence,
        }


def _run(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_machine_specs() -> MachineSpecs:
    """Detect chip name, GPU core count, RAM, and known bandwidth spec."""
    chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip()

    # GPU core count from system_profiler. Slow (~1s) but only called once.
    gpu_cores: Optional[int] = None
    sp = _run(["system_profiler", "SPDisplaysDataType"], timeout=10.0)
    if sp:
        for line in sp.splitlines():
            line = line.strip()
            if line.startswith("Total Number of Cores:"):
                try:
                    gpu_cores = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

    ram_bytes_str = (_run(["sysctl", "-n", "hw.memsize"]) or "0").strip()
    try:
        ram_gb = int(ram_bytes_str) / (1024 ** 3)
    except ValueError:
        ram_gb = 0.0

    bw, conf = _lookup_bandwidth(chip, gpu_cores)
    return MachineSpecs(
        chip=chip or "unknown",
        gpu_core_count=gpu_cores,
        ram_gb=ram_gb,
        dram_bandwidth_gbs=bw,
        bandwidth_confidence=conf,
    )


def _lookup_bandwidth(chip: str, gpu_cores: Optional[int]) -> tuple[Optional[float], str]:
    # Try (chip, gpu_cores) first, then (chip, None) fallback.
    if (chip, gpu_cores) in KNOWN_BANDWIDTH_GBS:
        bw, conf = KNOWN_BANDWIDTH_GBS[(chip, gpu_cores)]
        return bw, conf
    if (chip, None) in KNOWN_BANDWIDTH_GBS:
        bw, conf = KNOWN_BANDWIDTH_GBS[(chip, None)]
        return bw, conf
    return None, "UNKNOWN"


# --- bandwidth sampler --------------------------------------------------------

@dataclass
class BandwidthSummary:
    """Aggregated metrics from a measurement window."""
    enabled: bool
    samples: int = 0
    duration_s: float = 0.0
    # DRAM bandwidth (GB/s)
    dram_rd_mean_gbs: float = 0.0
    dram_rd_peak_gbs: float = 0.0
    dram_wr_mean_gbs: float = 0.0
    dram_wr_peak_gbs: float = 0.0
    dram_total_mean_gbs: float = 0.0
    dram_total_peak_gbs: float = 0.0
    # Power (W)
    gpu_power_mean_w: float = 0.0
    gpu_power_peak_w: float = 0.0
    dram_power_mean_w: float = 0.0
    total_power_mean_w: float = 0.0
    # GPU activity / freq
    gpu_active_mean_pct: float = 0.0
    gpu_freq_mean_mhz: float = 0.0
    # Thermals
    cpu_temp_peak_c: float = 0.0
    gpu_temp_peak_c: float = 0.0
    # Bandwidth-utilization vs spec
    bw_util_peak_pct: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class BandwidthSampler:
    """Context manager that samples mactop JSON in a background thread.

    Usage:
        sampler = BandwidthSampler(spec_gbs=400.0, interval_ms=1000)
        with sampler:
            run_measured_workload()
        summary = sampler.summary  # BandwidthSummary
    """

    def __init__(
        self,
        spec_gbs: Optional[float] = None,
        interval_ms: int = 1000,
        mactop_path: Optional[str] = None,
    ):
        self.spec_gbs = spec_gbs
        self.interval_ms = interval_ms
        self.mactop_path = mactop_path or shutil.which("mactop")
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self.summary: BandwidthSummary = BandwidthSummary(enabled=False)
        self.unavailable_reason: Optional[str] = None

        if not self.mactop_path:
            self.unavailable_reason = "mactop not found on PATH (brew install mactop)"

    def __enter__(self) -> "BandwidthSampler":
        return self.start()

    def start(self) -> "BandwidthSampler":
        """Spawn mactop and begin collecting samples in the background.

        Note: mactop has a ~5s first-sample latency. For short measurement
        windows, prefer the continuous-sampling pattern: start() once around
        the whole sweep, mark window timestamps with mark_window(), then
        summary_window(t_start, t_end) per slice.
        """
        if self.unavailable_reason:
            return self
        try:
            self._proc = subprocess.Popen(
                [
                    self.mactop_path,
                    "--headless",
                    "--format", "json",
                    "--interval", str(self.interval_ms),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.unavailable_reason = f"failed to spawn mactop: {e}"
            return self
        self._stop.clear()
        self._samples = []
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def stop(self) -> None:
        """Terminate mactop and finalize the global summary."""
        if self._proc is None and self._thread is None:
            return
        self._t_end = time.perf_counter()
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._build_summary()
        self._proc = None
        self._thread = None

    def is_warm(self) -> bool:
        """True once at least one sample has been received."""
        return len(self._samples) > 0

    def wait_until_warm(self, timeout_s: float = 8.0, poll_s: float = 0.25) -> bool:
        """Block until first sample arrives or timeout. Returns True if warm."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.is_warm():
                return True
            time.sleep(poll_s)
        return self.is_warm()

    def summary_window(self, t_window_start: float, t_window_end: float) -> BandwidthSummary:
        """Build a summary from samples whose arrival time falls in [start, end).

        Both bounds are perf_counter() values (same clock as start()/stop()).
        Used for the continuous-sampling pattern: one sampler around a sweep,
        per-window summaries computed afterwards.
        """
        in_window = [s for s in self._samples
                     if t_window_start <= s.get("_t_recv", 0) < t_window_end]
        return self._build_summary_from(in_window, t_window_end - t_window_start)

    def _reader_loop(self) -> None:
        # mactop --headless --format json emits a top-level JSON array of one
        # object per sample, but in streaming mode it appears to emit one
        # standalone JSON object per line. We accept either shape.
        if self._proc is None or self._proc.stdout is None:
            return
        buf: list[str] = []
        depth = 0
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            stripped = line.strip()
            if not stripped:
                continue
            # Track brace depth so we accumulate complete top-level objects
            # even when mactop pretty-prints across multiple lines.
            for ch in stripped:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            buf.append(stripped)
            if depth == 0:
                blob = "".join(buf)
                buf = []
                # Trim leading/trailing array bracketing or commas.
                blob = blob.lstrip("[, \t").rstrip(",] \t")
                if not blob:
                    continue
                try:
                    obj = json.loads(blob)
                except json.JSONDecodeError:
                    continue
                # Stamp arrival time for window-slicing.
                t_recv = time.perf_counter()
                if isinstance(obj, list):
                    for o in obj:
                        if isinstance(o, dict):
                            o["_t_recv"] = t_recv
                            self._samples.append(o)
                elif isinstance(obj, dict):
                    obj["_t_recv"] = t_recv
                    self._samples.append(obj)

    def _build_summary(self) -> None:
        self.summary = self._build_summary_from(
            self._samples, self._t_end - self._t_start
        )

    def _build_summary_from(
        self, samples: list[dict], duration_s: float
    ) -> BandwidthSummary:
        if self.unavailable_reason:
            return BandwidthSummary(enabled=False, notes=[self.unavailable_reason])
        if not samples:
            return BandwidthSummary(
                enabled=False,
                duration_s=duration_s,
                notes=[
                    "mactop produced no samples in this window (mactop has a "
                    "~5s first-sample latency; consider continuous sampling)"
                ],
            )

        def _pull(metric: str, default: float = 0.0) -> list[float]:
            out = []
            for s in samples:
                soc = s.get("soc_metrics") or {}
                v = soc.get(metric)
                out.append(float(v) if isinstance(v, (int, float)) else default)
            return out

        rd = _pull("dram_read_bw_gbs")
        wr = _pull("dram_write_bw_gbs")
        total = _pull("dram_bw_combined_gbs")
        if not any(total):
            total = [r + w for r, w in zip(rd, wr)]

        gpu_p = _pull("gpu_power")
        dram_p = _pull("dram_power")
        total_p = _pull("total_power")
        gpu_act = _pull("gpu_active")
        gpu_freq = _pull("gpu_freq_mhz")
        cpu_temp = _pull("cpu_temp")
        gpu_temp = _pull("gpu_temp")

        def _mean(xs: list[float]) -> float:
            return (sum(xs) / len(xs)) if xs else 0.0

        notes: list[str] = []
        if rd and all(v == 0 for v in rd) and all(v == 0 for v in wr):
            notes.append(
                "DRAM bandwidth (rd/wr) is zero across all samples — likely a "
                "mactop limitation on this chip/macOS version. Treat "
                "dram_power_W as the proxy."
            )

        peak_total = max(total) if total else 0.0
        bw_util = None
        if self.spec_gbs and self.spec_gbs > 0:
            bw_util = (peak_total / self.spec_gbs) * 100.0

        return BandwidthSummary(
            enabled=True,
            samples=len(samples),
            duration_s=duration_s,
            dram_rd_mean_gbs=_mean(rd),
            dram_rd_peak_gbs=max(rd) if rd else 0.0,
            dram_wr_mean_gbs=_mean(wr),
            dram_wr_peak_gbs=max(wr) if wr else 0.0,
            dram_total_mean_gbs=_mean(total),
            dram_total_peak_gbs=peak_total,
            gpu_power_mean_w=_mean(gpu_p),
            gpu_power_peak_w=max(gpu_p) if gpu_p else 0.0,
            dram_power_mean_w=_mean(dram_p),
            total_power_mean_w=_mean(total_p),
            gpu_active_mean_pct=_mean(gpu_act),
            gpu_freq_mean_mhz=_mean(gpu_freq),
            cpu_temp_peak_c=max(cpu_temp) if cpu_temp else 0.0,
            gpu_temp_peak_c=max(gpu_temp) if gpu_temp else 0.0,
            bw_util_peak_pct=bw_util,
            notes=notes,
        )


__all__ = [
    "MachineSpecs",
    "detect_machine_specs",
    "BandwidthSummary",
    "BandwidthSampler",
]
