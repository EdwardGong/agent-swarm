#!/usr/bin/env python3
"""Automated research sweep scheduler.

Reads sweep definitions from sweeps.yaml and runs them on a schedule.
Each sweep is a full research-mode query through the swarm, with results
saved to the sweep's designated memory namespace and report directory.

Usage:
    # Run the scheduler daemon (stays alive, executes sweeps on schedule):
    python -m apps.research_swarm.scheduler

    # Run all enabled sweeps once immediately (good for testing / cron):
    python -m apps.research_swarm.scheduler --run-now

    # Run a specific sweep by name:
    python -m apps.research_swarm.scheduler --run-now --sweep crypto-market-pulse

    # List configured sweeps:
    python -m apps.research_swarm.scheduler --list
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SWEEPS_PATH = Path(__file__).parent / "sweeps.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")


@dataclass
class SweepJob:
    """A single sweep definition parsed from YAML."""
    name: str
    query: str
    schedule: str          # "daily", "weekly", "hourly", "every 30m", "every 4h"
    namespace: str
    report_dir: str
    enabled: bool = True


def load_sweeps(path: Path = SWEEPS_PATH) -> list[SweepJob]:
    """Parse sweep definitions from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    jobs = []
    for entry in data.get("sweeps", []):
        jobs.append(SweepJob(
            name=entry["name"],
            query=entry["query"].strip(),
            schedule=entry.get("schedule", "daily"),
            namespace=entry.get("namespace", "general"),
            report_dir=entry.get("report_dir", "~/workspace/ai/agent-swarm/reports"),
            enabled=entry.get("enabled", True),
        ))
    return jobs


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def run_sweep(job: SweepJob) -> str:
    """Execute a single sweep job through the research swarm."""
    # Lazy import to avoid loading the full stack at parse time
    from apps.research_swarm.main import _build_app
    from langchain_core.messages import HumanMessage

    report_dir = Path(job.report_dir).expanduser()
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H%M")
    report_path = report_dir / f"{job.name}_{timestamp}.md"

    query = (
        f"RESEARCH TASK: {job.query}\n\n"
        f"Instructions:\n"
        f"- Use the appropriate namespace '{job.namespace}' when calling "
        f"save_to_memory and recall_research.\n"
        f"- After gathering information, use the writer_agent to produce "
        f"a report saved to {report_path}\n"
    )

    log.info(f"Starting sweep '{job.name}' (namespace={job.namespace})")

    app = _build_app()
    final_answer = ""

    try:
        for event in app.stream(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 50},
        ):
            for node_name, node_output in event.items():
                if node_name == "orchestrator":
                    next_ag = node_output.get("next_agent", "")
                    sub = node_output.get("sub_task", "")
                    if next_ag and next_ag != "done":
                        log.info(f"  [{job.name}] -> {next_ag}: {sub[:120]}")
                    if node_output.get("done"):
                        msgs = node_output.get("messages", [])
                        if msgs:
                            final_answer = msgs[-1].content
                else:
                    results = node_output.get("results", [])
                    for r in results:
                        log.info(f"  [{job.name}] {node_name}: {r[:150]}...")

        log.info(f"Sweep '{job.name}' complete. Report: {report_path}")
    except Exception as e:
        log.error(f"Sweep '{job.name}' failed: {e}")
        final_answer = f"Error: {e}"

    return final_answer


# ---------------------------------------------------------------------------
# Schedule loop
# ---------------------------------------------------------------------------

def _parse_interval_seconds(schedule: str) -> int:
    """Convert a schedule string to seconds between runs."""
    s = schedule.strip().lower()
    if s == "hourly":
        return 3600
    if s == "daily":
        return 86400
    if s == "weekly":
        return 604800

    # "every Xm" or "every Xh"
    m = re.match(r"every\s+(\d+)\s*(m|h)", s)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        return val * 60 if unit == "m" else val * 3600

    log.warning(f"Unknown schedule '{schedule}', defaulting to daily")
    return 86400


def run_daemon(sweeps: list[SweepJob]):
    """Run sweeps on their configured schedules in a blocking loop."""
    last_run: dict[str, float] = {}

    log.info(f"Scheduler daemon started with {len(sweeps)} sweep(s)")
    for job in sweeps:
        interval = _parse_interval_seconds(job.schedule)
        log.info(f"  {job.name}: every {interval}s ({job.schedule})")

    try:
        while True:
            now = time.time()
            for job in sweeps:
                interval = _parse_interval_seconds(job.schedule)
                elapsed = now - last_run.get(job.name, 0)

                if elapsed >= interval:
                    try:
                        run_sweep(job)
                    except Exception as e:
                        log.error(f"Sweep '{job.name}' crashed: {e}")
                    last_run[job.name] = time.time()

            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research Sweep Scheduler")
    parser.add_argument("--run-now", action="store_true",
                        help="Run all enabled sweeps immediately, then exit")
    parser.add_argument("--sweep", type=str, metavar="NAME",
                        help="Run only the named sweep (use with --run-now)")
    parser.add_argument("--list", action="store_true",
                        help="List configured sweeps and exit")
    parser.add_argument("--config", type=str, default=str(SWEEPS_PATH),
                        help="Path to sweeps YAML config")
    args = parser.parse_args()

    sweeps = load_sweeps(Path(args.config))
    enabled = [s for s in sweeps if s.enabled]

    if args.list:
        print(f"{'Name':<30} {'Schedule':<12} {'Namespace':<15} {'Enabled'}")
        print("-" * 70)
        for s in sweeps:
            print(f"{s.name:<30} {s.schedule:<12} {s.namespace:<15} {s.enabled}")
        return

    if args.run_now:
        targets = enabled
        if args.sweep:
            targets = [s for s in enabled if s.name == args.sweep]
            if not targets:
                print(f"Sweep '{args.sweep}' not found or not enabled.")
                sys.exit(1)

        log.info(f"Running {len(targets)} sweep(s) now...")
        for job in targets:
            run_sweep(job)
        log.info("All sweeps complete.")
        return

    # Default: run as daemon
    if not enabled:
        print("No enabled sweeps found. Add sweeps to sweeps.yaml.")
        sys.exit(1)

    run_daemon(enabled)


if __name__ == "__main__":
    main()
