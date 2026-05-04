"""Code workflow worker.

Implements the plan's "Code / Website Building" workflow:

  1. Read project files via Filesystem MCP
  2. Generate / edit files via Filesystem MCP tool calls
  3. Git MCP for staging and commits
  4. Safari MCP or Playwright for visual preview / testing (optional)

Prefix cache gives 10-30× TTFT improvement on multi-turn coding tasks when
the system prompt stays stable across calls — keep the CodeAgent's system
prompt unchanged between turns.

Standalone usage::

    from apps.consolidated_swarm.workers.coder import run
    result = run("Add type hints to orchestrator/router.py")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.mcp import MCPToolRegistry

logger = logging.getLogger(__name__)


def run(task: str, mcp_registry: "MCPToolRegistry | None" = None) -> str:
    """Execute the code workflow for *task*.

    The worker loads the MCP 'code' profile tools and provides a structured
    view of available file/git tools so the caller (or LangGraph agent loop)
    can drive the implementation.

    Args:
        task: Description of the coding task to perform.
        mcp_registry: Live ``MCPToolRegistry``.  When ``None`` creates one
            targeting ``http://localhost:8000``.

    Returns:
        A summary string describing what was done, including any file paths
        modified and commit hashes if applicable.
    """
    if mcp_registry is None:
        from orchestrator.mcp import MCPToolRegistry
        mcp_registry = MCPToolRegistry("http://localhost:8000")

    tools = {t.name: t for t in mcp_registry.load_tools(profile="code")}
    _log_available(tools)

    results: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Inventory relevant files
    # ------------------------------------------------------------------
    if "filesystem_list_directory" in tools:
        listing = tools["filesystem_list_directory"].run({"path": "."})
        results.append(f"Directory listing:\n{listing}")
        logger.debug("Directory listed: %d chars", len(listing))

    # ------------------------------------------------------------------
    # Steps 2-4 are driven by the LangGraph ReAct loop when this worker
    # is used inside the orchestrator.  When called standalone, we return
    # the context so the caller can drive further tool calls.
    # ------------------------------------------------------------------
    summary_parts = [
        f"Code worker ready for task: {task}",
        f"Available tools: {', '.join(tools.keys()) or 'none (vllm-mlx not running)'}",
    ]
    if results:
        summary_parts.extend(results)

    return "\n\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_available(tools: dict) -> None:
    logger.debug("Code worker: %d MCP tools: %s", len(tools), list(tools.keys()))
    if not tools:
        logger.warning(
            "Code worker: no MCP tools — "
            "is vllm-mlx running with --mcp-config mcp-configs/mcp-code.json?"
        )
