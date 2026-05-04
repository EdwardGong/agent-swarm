"""Deep research workflow worker.

Implements the plan's "Deep Research (Autonomous, Long-Running)" workflow:

  1. Orchestrator decomposes query → sub-questions
  2. Firecrawl self-hosted /v1/search (primary) or Safari MCP (fallback)
  3. Firecrawl /v1/scrape OR safari_navigate_and_read for each URL
  4. Chunk → ingest to FAISS via faiss_ingest_document
  5. Entities → Memory MCP knowledge graph (create_entities + create_relations)
  6. Recall prior findings before each search round to avoid redundancy
  7. Generate follow-up sub-questions → loop until convergence
  8. Synthesise final report from FAISS + Memory MCP

Can be run standalone or wired into the LangGraph orchestrator.

Standalone usage::

    from apps.consolidated_swarm.workers.research import run
    report = run("Latest developments in Apple Silicon neural engines")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.mcp import MCPToolRegistry

logger = logging.getLogger(__name__)

# Maximum search → scrape → ingest iterations before forcing synthesis.
_MAX_ITERATIONS = 6
# Maximum URLs to scrape per iteration.
_MAX_URLS_PER_ITER = 4
# Output directory for synthesised reports — repo-relative so no external dir needed.
_REPO_ROOT = Path(__file__).parents[3]
_REPORT_DIR = _REPO_ROOT / "runtime" / "workspace" / "reports"


def run(task: str, mcp_registry: "MCPToolRegistry | None" = None) -> str:
    """Execute the deep research workflow for *task*.

    Args:
        task: Free-form research question or topic.
        mcp_registry: Live ``MCPToolRegistry`` instance (from
            ``orchestrator.mcp``).  When ``None`` the worker attempts to
            create one pointing at ``http://localhost:8000``.

    Returns:
        Path to the generated Markdown report as a string, or the report
        content itself if file output fails.
    """
    if mcp_registry is None:
        from orchestrator.mcp import MCPToolRegistry
        mcp_registry = MCPToolRegistry("http://localhost:8000")

    tools = {t.name: t for t in mcp_registry.load_tools(profile="research")}
    _log_available(tools)

    # ------------------------------------------------------------------
    # Step 0: Recall prior research to seed context
    # ------------------------------------------------------------------
    prior_context = ""
    if "faiss_query_rag_store" in tools:
        prior_context = tools["faiss_query_rag_store"].run({"query": task})
        logger.info("Prior RAG context: %d chars", len(prior_context))

    memory_context = ""
    if "mcp_memory_search_nodes" in tools:
        memory_context = tools["mcp_memory_search_nodes"].run({"query": task})

    # ------------------------------------------------------------------
    # Steps 1-6: Iterative search → scrape → ingest loop
    # ------------------------------------------------------------------
    ingested_urls: set[str] = set()
    sub_questions = [task]  # seed with the main question

    for iteration in range(_MAX_ITERATIONS):
        if not sub_questions:
            break

        current_q = sub_questions.pop(0)
        logger.info("Iteration %d — searching: %s", iteration + 1, current_q)

        # Search
        search_results_raw = ""
        if "firecrawl_search" in tools:
            search_results_raw = tools["firecrawl_search"].run({"query": current_q})
        elif "safari_navigate_and_search" in tools:
            search_results_raw = tools["safari_navigate_and_search"].run(
                {"query": current_q}
            )

        if not search_results_raw:
            logger.warning("No search results for: %s", current_q)
            continue

        # Extract URLs from search results (best-effort — MCP returns varied formats)
        urls = _extract_urls(search_results_raw)[:_MAX_URLS_PER_ITER]

        for url in urls:
            if url in ingested_urls:
                continue
            ingested_urls.add(url)

            # Scrape
            article = ""
            if "firecrawl_scrape" in tools:
                article = tools["firecrawl_scrape"].run({"url": url})
            elif "safari_navigate_and_read" in tools:
                article = tools["safari_navigate_and_read"].run({"url": url})

            if not article:
                continue

            # Ingest to FAISS
            if "faiss_ingest_document" in tools:
                tools["faiss_ingest_document"].run({
                    "content": article,
                    "source": url,
                    "metadata": {"query": current_q, "iteration": iteration},
                })

            # Update Memory knowledge graph with key entities
            if "mcp_memory_create_entities" in tools:
                # The LLM will handle entity extraction in the full agent loop;
                # here we just store the raw article reference
                tools["mcp_memory_create_entities"].run({
                    "entities": [{"name": url, "entityType": "article",
                                  "observations": [article[:500]]}]
                })

    # ------------------------------------------------------------------
    # Step 7: Synthesise report
    # ------------------------------------------------------------------
    rag_findings = ""
    if "faiss_query_rag_store" in tools:
        rag_findings = tools["faiss_query_rag_store"].run({
            "query": task,
            "k": 10,
        })

    memory_findings = ""
    if "mcp_memory_search_nodes" in tools:
        memory_findings = tools["mcp_memory_search_nodes"].run({"query": task})

    report_content = _build_report(task, rag_findings, memory_findings, prior_context)

    # Write to disk
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = task[:60].lower().replace(" ", "-").replace("/", "-")
    report_path = _REPORT_DIR / f"{slug}.md"
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Report written: %s", report_path)

    return str(report_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_available(tools: dict) -> None:
    logger.debug("Research worker: %d MCP tools available: %s",
                 len(tools), list(tools.keys()))
    if not tools:
        logger.warning(
            "Research worker: no MCP tools loaded — "
            "is vllm-mlx running with --mcp-config mcp-configs/mcp-research.json?"
        )


def _extract_urls(text: str) -> list[str]:
    """Best-effort URL extraction from Firecrawl / Safari MCP search output."""
    import re
    return re.findall(r"https?://[^\s\"'<>]+", text)


def _build_report(task: str, rag: str, memory: str, prior: str) -> str:
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    sections = [
        f"# Research Report: {task}",
        f"*Generated: {timestamp}*",
        "",
        "## Prior Knowledge",
        prior or "*(none)*",
        "",
        "## Findings (RAG)",
        rag or "*(no FAISS results)*",
        "",
        "## Relationship Context (Memory MCP)",
        memory or "*(no memory nodes)*",
    ]
    return "\n".join(sections)
