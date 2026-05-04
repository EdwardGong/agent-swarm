"""Live integration tests for the Consolidated Agent Swarm.

These tests require a running vllm-mlx server on port 8000 and are skipped
automatically when the server is unreachable.  All four query types
(research, code, content, CRM) are run in parallel via ThreadPoolExecutor.

Start the server first::

    ./apps/consolidated_swarm/scripts/start.sh --server-only

Then run::

    PYTHONPATH=. pytest tests/test_integration_live.py -v -s

Or run only the parallel smoke test::

    PYTHONPATH=. pytest tests/test_integration_live.py -v -s -k test_all_agents_parallel
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest
import requests

from apps.consolidated_swarm.main import _build_app, run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server liveness check — skip all tests if vllm-mlx is not running
# ---------------------------------------------------------------------------

_VLLM_URL = "http://localhost:8000"


def _server_is_up() -> bool:
    try:
        r = requests.get(f"{_VLLM_URL}/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_is_up(),
    reason="vllm-mlx server not running on localhost:8000",
)


# ---------------------------------------------------------------------------
# Query definitions — one per agent type
# ---------------------------------------------------------------------------

QUERIES = {
    "research": "What is MLX and how does it compare to PyTorch for Apple Silicon inference? Be brief.",
    "code": "List the Python files in the orchestrator/ directory and briefly describe what each does.",
    "content": "Create a 3-slide outline for a presentation about multi-agent architectures.",
    "crm": "Initialize the CRM database and describe the contacts table schema.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_timed(label: str, query: str) -> dict[str, Any]:
    """Run a single query and return timing + result metadata."""
    t0 = time.monotonic()
    logger.info("[%s] START: %s", label, query[:80])
    try:
        answer = run_query(query)
        elapsed = time.monotonic() - t0
        logger.info("[%s] DONE in %.1fs (%d chars)", label, elapsed, len(answer))
        return {
            "label": label,
            "answer": answer,
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error("[%s] FAILED in %.1fs: %s", label, elapsed, exc)
        return {
            "label": label,
            "answer": "",
            "elapsed": elapsed,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveServerHealth:
    """Basic sanity checks against the running vllm-mlx server."""

    def test_models_endpoint(self):
        r = requests.get(f"{_VLLM_URL}/v1/models", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) >= 1

    def test_mcp_status(self):
        r = requests.get(f"{_VLLM_URL}/v1/mcp/status", timeout=5)
        # 200 even if 0 MCP servers are loaded — just verifies the endpoint exists
        assert r.status_code == 200

    def test_app_builds(self):
        app = _build_app()
        assert app is not None


class TestLiveQueriesParallel:
    """Run all four agent queries in parallel and verify they complete."""

    @pytest.fixture(autouse=True)
    def _log_separator(self):
        logger.info("=" * 60)
        yield
        logger.info("=" * 60)

    def test_all_agents_parallel(self):
        """Fire all 4 queries concurrently, assert each produces a non-empty answer."""
        results: dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_run_timed, label, query): label
                for label, query in QUERIES.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                results[label] = future.result()

        # Print summary table
        print("\n" + "=" * 62)
        print(f"  {'Agent':<12} {'Time':>8} {'Chars':>8}  Status")
        print("-" * 62)
        for label in ("research", "code", "content", "crm"):
            r = results[label]
            status = "✅ OK" if not r["error"] else f"❌ {r['error'][:40]}"
            print(f"  {label:<12} {r['elapsed']:>7.1f}s {len(r['answer']):>8}  {status}")
        print("=" * 62 + "\n")

        # Assertions
        for label, r in results.items():
            assert r["error"] is None, f"{label} failed: {r['error']}"
            assert len(r["answer"]) > 0, f"{label} returned empty answer"

    def test_research_query(self):
        result = _run_timed("research", QUERIES["research"])
        assert result["error"] is None, result["error"]
        assert len(result["answer"]) > 20

    def test_code_query(self):
        result = _run_timed("code", QUERIES["code"])
        assert result["error"] is None, result["error"]
        assert len(result["answer"]) > 20

    def test_content_query(self):
        result = _run_timed("content", QUERIES["content"])
        assert result["error"] is None, result["error"]
        assert len(result["answer"]) > 20

    def test_crm_query(self):
        result = _run_timed("crm", QUERIES["crm"])
        assert result["error"] is None, result["error"]
        assert len(result["answer"]) > 20
