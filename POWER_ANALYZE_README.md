# power_analyze

Analyze and visualize power consumption data captured from [macpow](https://github.com/k06a/macpow) on Apple Silicon Macs.

Produces a 6-panel dark-themed dashboard and a text summary covering system power, SoC breakdown, battery state, temperatures, CPU/GPU frequencies, and fan speeds.

![Dashboard screenshot](power_report.png)

## Prerequisites

- **macOS** with Apple Silicon (M1–M5)
- **Python 3.12+**
- **macpow** — the data source
- **matplotlib** — for plotting

### Install dependencies

```bash
# Install macpow
brew tap k06a/tap
brew install macpow

# Install matplotlib
pip3 install matplotlib
```

## Quick start

```bash
# 1. Capture power data (Ctrl-C or kill %1 to stop)
macpow --json --interval 1000 > power_log.json 2>/dev/null &

# 2. Let it run for a while, then stop
kill %1

# 3. Visualize
python3 power_analyze.py power_log.json
```

## Capturing data

macpow's `--json` mode emits one pretty-printed JSON object per sampling interval. Redirect to a file to collect a log:

```bash
# Sample every 500ms (default)
macpow --json > power_log.json 2>/dev/null &

# Sample every 2 seconds
macpow --json --interval 2000 > power_log.json 2>/dev/null &
```

Stop the capture with `kill %1` (or Ctrl-C if running in the foreground). Partial trailing objects from a killed process are handled gracefully by the parser.

> **Tip:** The first sample will have zeroed-out SoC data while IOReport initializes. This is normal — it settles after one interval.

## Usage

```
usage: power_analyze [-h] [-o FILE] [--no-plot] [--interval MS] logfile
```

### Arguments

| Argument | Description |
|---|---|
| `logfile` | Path to a macpow JSON log file |
| `-o FILE`, `--output FILE` | Save the dashboard to a file (PNG/PDF/SVG/EPS/JPG) instead of opening an interactive window |
| `--no-plot` | Print the text summary only — no GUI or file output |
| `--interval MS` | Sampling interval in milliseconds. Must match the `--interval` used during capture (default: 500) |
| `-h`, `--help` | Show help and exit |

### Examples

```bash
# Interactive plot (opens a matplotlib window)
python3 power_analyze.py power_log.json

# Save to PNG
python3 power_analyze.py power_log.json -o report.png

# Save to PDF (vector, good for reports)
python3 power_analyze.py power_log.json -o report.pdf

# Text summary only (no GUI needed, works over SSH)
python3 power_analyze.py power_log.json --no-plot

# Custom interval (must match what macpow used)
python3 power_analyze.py power_log.json --interval 2000 -o report.png
```

## Dashboard panels

| # | Panel | Description |
|---|---|---|
| 1 | **System & Adapter Power** | Total system draw (filled area) vs adapter power supply (dashed) in watts |
| 2 | **SoC Power Breakdown** | Stacked area chart of CPU, GPU, DRAM, and ANE power with a SoC total overlay |
| 3 | **Battery** | Charge percentage (left axis) and drain/charge rate in watts (right axis) |
| 4 | **Battery Temperature** | Temperature in °C with a 35°C warning threshold line |
| 5 | **CPU / GPU Frequency** | E-CPU, P-CPU, and GPU clock speeds in MHz (auto-formats to GHz) |
| 6 | **Fan Speed** | RPM per fan |

## Text summary

When run, the script always prints a summary to stdout:

```
────────────────────────────────────────────────────
  macpow Power Analysis  —  58 samples, 28s
────────────────────────────────────────────────────
  System power : avg   60.7 W  max   62.5 W
  Adapter power: avg   62.9 W  max   62.9 W
  SoC total    : avg   42.0 W  max   45.1 W
  CPU          : avg    1.3 W  max    3.8 W
  GPU          : avg   16.2 W  max   18.5 W
  DRAM         : avg   13.3 W  max   14.2 W
  Display      : avg    4.1 W

  Battery      : 7.0%  ⚡ Charging  30.4°C  146 cycles  health 88.7%
  Adapter      : 65W  20.0V  3.24A
  Memory       : 46.5 / 48 GB
────────────────────────────────────────────────────
```

## Data format

macpow outputs concatenated pretty-printed JSON objects (not newline-delimited single-line JSON). Each object contains:

| Key | Contents |
|---|---|
| `soc` | CPU/GPU/ANE/DRAM power (W), frequencies (MHz), utilization |
| `battery` | Voltage, amperage, charge %, drain rate, temperature, cycle count, health |
| `adapter` | Wattage, voltage, current, wired/wireless |
| `display` | Brightness %, nits, estimated power |
| `fans` | RPM per fan, estimated power |
| `sys_power_w` | Total system power from SMC |
| `adapter_power_w` | Total adapter delivery from SMC |
| `top_processes` | Per-process energy, disk I/O, memory |
| `wifi` / `bluetooth` | Connection info, signal, estimated power |

The parser uses brace-depth tracking to reliably split concatenated objects, even when a capture is killed mid-write.

## Troubleshooting

**"No valid JSON samples found"**
The log file is empty or contains no complete macpow JSON objects. Make sure macpow ran long enough to emit at least one full sample (~1 second).

**X-axis times don't match wall-clock duration**
The `--interval` flag must match the value used during `macpow --json --interval`. macpow doesn't embed timestamps, so the script reconstructs time from the sample index × interval.

**Plot window doesn't appear**
matplotlib needs a GUI backend. If running over SSH or in a headless environment, use `--output` to save to a file, or `--no-plot` for text only.

**First data point shows zero for SoC metrics**
This is expected — macpow's IOReport source needs one full interval to initialize. The first sample is a baseline with zeroed energy counters.
