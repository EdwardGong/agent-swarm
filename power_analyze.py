#!/usr/bin/env python3
"""Parse macpow JSON logs and visualize power consumption trends.

Usage:
    # Capture data first (run for as long as you want):
    macpow --json --interval 1000 > power_log.json 2>/dev/null &
    kill %1  # when done

    # Then analyze:
    python power_analyze.py power_log.json
    python power_analyze.py power_log.json --output power_report.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


def parse_macpow_json(path: str) -> list[dict]:
    """Parse concatenated pretty-printed JSON objects from macpow output.

    macpow emits multiple pretty-printed JSON objects back-to-back.
    We split them by tracking brace depth, then parse each individually.
    """
    text = Path(path).read_text()
    samples = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and "soc" in obj:
                        samples.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return samples


def extract_series(samples: list[dict], interval_ms: float) -> dict:
    """Extract time-series data from parsed samples."""
    n = len(samples)
    t = [i * interval_ms / 1000 for i in range(n)]

    def safe(sample, *keys, default=0.0):
        val = sample
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val if val is not None else default

    return {
        "time_s": t,
        # System-level power
        "sys_power_w": [safe(s, "sys_power_w") for s in samples],
        "adapter_power_w": [safe(s, "adapter_power_w") for s in samples],
        "backlight_power_w": [safe(s, "backlight_power_w") for s in samples],
        # SoC breakdown
        "cpu_w": [safe(s, "soc", "cpu_w") for s in samples],
        "gpu_w": [safe(s, "soc", "gpu_w") for s in samples],
        "ane_w": [safe(s, "soc", "ane_w") for s in samples],
        "dram_w": [safe(s, "soc", "dram_w") for s in samples],
        "soc_total_w": [safe(s, "soc", "total_w") for s in samples],
        # Frequencies
        "ecpu_freq_mhz": [safe(s, "soc", "ecpu_freq_mhz") for s in samples],
        "pcpu_freq_mhz": [safe(s, "soc", "pcpu_freq_mhz") for s in samples],
        "gpu_freq_mhz": [safe(s, "soc", "gpu_freq_mhz") for s in samples],
        # Battery
        "battery_pct": [safe(s, "battery", "percent") for s in samples],
        "battery_drain_w": [safe(s, "battery", "drain_w") for s in samples],
        "battery_temp_c": [safe(s, "battery", "temperature_c") for s in samples],
        "battery_charging": [safe(s, "battery", "charging", default=False) for s in samples],
        # Fans
        "fan_rpms": [
            [f.get("actual_rpm", 0) for f in safe(s, "fans", default=[])]
            for s in samples
        ],
        # Memory
        "mem_used_gb": [safe(s, "mem_used_gb") for s in samples],
    }


def print_summary(samples: list[dict], series: dict):
    """Print a text summary of key stats."""
    n = len(samples)
    duration_s = series["time_s"][-1] if n > 1 else 0

    def avg(vals):
        valid = [v for v in vals if v is not None and v != 0]
        return sum(valid) / len(valid) if valid else 0

    bat = samples[-1].get("battery", {})
    adapter = samples[-1].get("adapter", {})

    print(f"{'─' * 52}")
    print(f"  macpow Power Analysis  —  {n} samples, {duration_s:.0f}s")
    print(f"{'─' * 52}")
    print(f"  System power : avg {avg(series['sys_power_w']):6.1f} W  "
          f"max {max(series['sys_power_w']):6.1f} W")
    print(f"  Adapter power: avg {avg(series['adapter_power_w']):6.1f} W  "
          f"max {max(series['adapter_power_w']):6.1f} W")
    print(f"  SoC total    : avg {avg(series['soc_total_w']):6.1f} W  "
          f"max {max(series['soc_total_w']):6.1f} W")
    print(f"  CPU          : avg {avg(series['cpu_w']):6.1f} W  "
          f"max {max(series['cpu_w']):6.1f} W")
    print(f"  GPU          : avg {avg(series['gpu_w']):6.1f} W  "
          f"max {max(series['gpu_w']):6.1f} W")
    print(f"  DRAM         : avg {avg(series['dram_w']):6.1f} W  "
          f"max {max(series['dram_w']):6.1f} W")
    print(f"  Display      : avg {avg(series['backlight_power_w']):6.1f} W")
    print()
    print(f"  Battery      : {bat.get('percent', '?')}%  "
          f"{'⚡ Charging' if bat.get('charging') else '🔋 Discharging'}  "
          f"{bat.get('temperature_c', '?')}°C  "
          f"{bat.get('cycle_count', '?')} cycles  "
          f"health {bat.get('health_pct', 0):.1f}%")
    if adapter.get("connected"):
        print(f"  Adapter      : {adapter.get('watts', '?')}W  "
              f"{adapter.get('voltage_mv', 0) / 1000:.1f}V  "
              f"{adapter.get('current_ma', 0) / 1000:.2f}A")
    print(f"  Memory       : {series['mem_used_gb'][-1]:.1f} / "
          f"{samples[-1].get('dram_gb', '?')} GB")
    print(f"{'─' * 52}")


def plot(series: dict, output: str | None = None):
    """Generate a multi-panel power consumption dashboard."""
    t = series["time_s"]
    duration = t[-1]

    # Use minutes for x-axis if duration > 120s
    if duration > 120:
        t_plot = [v / 60 for v in t]
        x_label = "Time (min)"
    else:
        t_plot = t
        x_label = "Time (s)"

    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a2e")
    fig.suptitle("macpow Power Consumption Dashboard", color="white",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28,
                  left=0.07, right=0.97, top=0.93, bottom=0.06)

    dark = "#1a1a2e"
    grid_color = "#2a2a4a"
    colors = {
        "sys": "#ff6b6b", "adapter": "#ffd93d", "cpu": "#6bcb77",
        "gpu": "#4d96ff", "dram": "#ff922b", "ane": "#cc5de8",
        "soc": "#20c997", "bat_pct": "#74c0fc", "bat_temp": "#ff8787",
        "bat_drain": "#a9e34b", "fan": "#da77f2",
    }

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(dark)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#aaa", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    # ── Panel 1: System & Adapter Power ──
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "System & Adapter Power", "Watts")
    ax1.fill_between(t_plot, series["sys_power_w"], alpha=0.25, color=colors["sys"])
    ax1.plot(t_plot, series["sys_power_w"], color=colors["sys"], lw=1.2, label="System")
    ax1.plot(t_plot, series["adapter_power_w"], color=colors["adapter"], lw=1.2,
             label="Adapter", linestyle="--")
    ax1.legend(fontsize=8, facecolor=dark, edgecolor=grid_color, labelcolor="white")

    # ── Panel 2: SoC Breakdown (stacked) ──
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "SoC Power Breakdown", "Watts")
    ax2.stackplot(
        t_plot,
        series["cpu_w"], series["gpu_w"], series["dram_w"], series["ane_w"],
        labels=["CPU", "GPU", "DRAM", "ANE"],
        colors=[colors["cpu"], colors["gpu"], colors["dram"], colors["ane"]],
        alpha=0.75,
    )
    ax2.plot(t_plot, series["soc_total_w"], color=colors["soc"], lw=1, ls="--",
             label="SoC Total", alpha=0.8)
    ax2.legend(fontsize=7, facecolor=dark, edgecolor=grid_color, labelcolor="white",
               loc="upper left", ncol=3)

    # ── Panel 3: Battery State ──
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Battery", "Charge %")
    ax3.plot(t_plot, series["battery_pct"], color=colors["bat_pct"], lw=1.5, label="Charge %")
    ax3.set_ylim(0, 105)
    ax3_r = ax3.twinx()
    ax3_r.plot(t_plot, series["battery_drain_w"], color=colors["bat_drain"], lw=1,
               label="Drain (W)", alpha=0.8)
    ax3_r.set_ylabel("Drain (W)", color="#aaa", fontsize=9)
    ax3_r.tick_params(colors="#888", labelsize=8)
    ax3_r.spines["right"].set_color(grid_color)
    # Merge legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, facecolor=dark,
               edgecolor=grid_color, labelcolor="white")

    # ── Panel 4: Battery Temperature ──
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Battery Temperature", "°C")
    ax4.plot(t_plot, series["battery_temp_c"], color=colors["bat_temp"], lw=1.2)
    ax4.axhspan(35, max(series["battery_temp_c"] or [40]) + 5, alpha=0.1, color="red")
    ax4.axhline(35, color="#ff4444", ls=":", lw=0.8, alpha=0.6)
    ax4.annotate("35°C threshold", xy=(t_plot[0], 35), fontsize=7, color="#ff4444", alpha=0.7)

    # ── Panel 5: CPU/GPU Frequencies ──
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, "CPU / GPU Frequency", "MHz")
    ax5.plot(t_plot, series["ecpu_freq_mhz"], color="#69db7c", lw=1, label="E-CPU", alpha=0.9)
    ax5.plot(t_plot, series["pcpu_freq_mhz"], color="#4dabf7", lw=1, label="P-CPU", alpha=0.9)
    ax5.plot(t_plot, series["gpu_freq_mhz"], color="#da77f2", lw=1, label="GPU", alpha=0.9)
    ax5.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.1f}G" if x >= 1000 else f"{x:.0f}"))
    ax5.legend(fontsize=8, facecolor=dark, edgecolor=grid_color, labelcolor="white", ncol=3)

    # ── Panel 6: Fan RPM ──
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, "Fan Speed", "RPM")
    if series["fan_rpms"] and series["fan_rpms"][0]:
        num_fans = len(series["fan_rpms"][0])
        fan_colors = ["#da77f2", "#ffd43b", "#63e6be"]
        for i in range(num_fans):
            rpms = [s[i] if i < len(s) else 0 for s in series["fan_rpms"]]
            ax6.plot(t_plot, rpms, color=fan_colors[i % len(fan_colors)], lw=1.2,
                     label=f"Fan {i}")
        ax6.legend(fontsize=8, facecolor=dark, edgecolor=grid_color, labelcolor="white")
    else:
        ax6.text(0.5, 0.5, "No fan data", transform=ax6.transAxes, ha="center",
                 color="#666", fontsize=12)

    # Shared x-label
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel(x_label, color="#888", fontsize=8)

    if output:
        fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="power_analyze",
        description="Analyze and visualize macpow JSON power logs.",
        epilog=(
            "examples:\n"
            "  # Capture data with macpow (run as long as you want):\n"
            "  macpow --json --interval 1000 > power_log.json 2>/dev/null &\n"
            "  kill %%1  # stop when done\n"
            "\n"
            "  # Interactive plot (opens a matplotlib window):\n"
            "  %(prog)s power_log.json\n"
            "\n"
            "  # Save dashboard to a PNG file:\n"
            "  %(prog)s power_log.json -o report.png\n"
            "\n"
            "  # Text summary only (no GUI required):\n"
            "  %(prog)s power_log.json --no-plot\n"
            "\n"
            "  # Specify a custom sampling interval:\n"
            "  %(prog)s power_log.json --interval 1000\n"
            "\n"
            "supported output formats (via --output):\n"
            "  .png, .jpg, .pdf, .svg, .eps\n"
            "\n"
            "dashboard panels:\n"
            "  1. System & Adapter Power  — total draw vs adapter supply\n"
            "  2. SoC Power Breakdown     — stacked CPU / GPU / DRAM / ANE\n"
            "  3. Battery                  — charge %% and drain/charge rate\n"
            "  4. Battery Temperature      — with 35°C warning threshold\n"
            "  5. CPU / GPU Frequency      — E-CPU, P-CPU, GPU clocks\n"
            "  6. Fan Speed                — RPM per fan\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "logfile",
        help="path to a macpow JSON log file (captured with `macpow --json`)",
    )

    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-o", "--output",
        metavar="FILE",
        help="save the dashboard to FILE (png/pdf/svg) instead of showing it",
    )
    output_group.add_argument(
        "--no-plot",
        action="store_true",
        help="print the text summary only; skip the plot entirely",
    )

    data_group = parser.add_argument_group("data options")
    data_group.add_argument(
        "--interval",
        type=float,
        default=None,
        metavar="MS",
        help="sampling interval in milliseconds (default: 500, matching macpow's default)",
    )

    args = parser.parse_args()

    samples = parse_macpow_json(args.logfile)
    if not samples:
        print("No valid JSON samples found.", file=sys.stderr)
        sys.exit(1)

    # Default interval: macpow default is 250ms for TUI, but --json often uses custom
    interval_ms = args.interval or 500
    series = extract_series(samples, interval_ms)
    print_summary(samples, series)

    if not args.no_plot:
        plot(series, args.output)


if __name__ == "__main__":
    main()
