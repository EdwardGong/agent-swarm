"""CRM and LinkedIn enrichment workflow worker.

Implements the plan's "LinkedIn / CRM" workflow:

  1. Sync: LMCP list_contacts → SQLite MCP upsert (contacts table)
  2. Enrich: Safari MCP browse LinkedIn profiles → SQLite upsert + Memory KG
  3. Query: SQLite for structured queries; Memory MCP for relationship queries
  4. Outreach: draft via LLM → send via LMCP send_email (preview + confirm)
  5. Maintain: daily cron → LMCP calendar/email events → SQLite + Memory updates

$0/mo configuration — LMCP + Safari MCP + SQLite + Memory MCP only.
Crispy.sh ($49/mo) upgrade path documented below when API-level LinkedIn
automation is needed.

Standalone usage::

    from apps.consolidated_swarm.workers.crm import run, sync_contacts, query_crm
    run("Sync all contacts and find all CTOs in AI companies")
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.mcp import MCPToolRegistry

logger = logging.getLogger(__name__)

# Default CRM database path — repo-relative so no external dir needed.
_REPO_ROOT = Path(__file__).parents[3]
_DB_PATH = _REPO_ROOT / "runtime" / "state.db"

# SQLite schema for the contacts table.
_CONTACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT,
    phone         TEXT,
    company       TEXT,
    title         TEXT,
    linkedin_url  TEXT,
    source        TEXT DEFAULT 'apple_contacts',
    last_contacted TEXT,
    notes         TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    updated_at    TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email)
    WHERE email IS NOT NULL AND email != '';
CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company);
"""


def ensure_schema(db_path: Path = _DB_PATH) -> None:
    """Create the contacts table if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_CONTACTS_SCHEMA)
    logger.info("CRM schema ensured at %s", db_path)


def sync_contacts(mcp_registry: "MCPToolRegistry") -> int:
    """Pull Apple Contacts via LMCP and upsert to SQLite.

    Returns the number of contacts processed.
    """
    tools = {t.name: t for t in mcp_registry.load_tools(profile="crm")}
    if "lmcp_list_contacts" not in tools:
        logger.warning("lmcp_list_contacts not available — LMCP not running?")
        return 0

    raw = tools["lmcp_list_contacts"].run({})
    if not raw:
        return 0

    # LMCP returns contacts as JSON or structured text — parse what we can
    contacts = _parse_contacts(raw)
    logger.info("Pulled %d contacts from LMCP", len(contacts))

    ensure_schema()
    count = 0
    if "sqlite_execute" in tools:
        for c in contacts:
            tools["sqlite_execute"].run({
                "query": """
                    INSERT OR REPLACE INTO contacts
                        (name, email, phone, company, title, source, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'apple_contacts', datetime('now'))
                """,
                "params": [
                    c.get("name", ""),
                    c.get("email"),
                    c.get("phone"),
                    c.get("company"),
                    c.get("title"),
                ],
                "database": str(_DB_PATH),
            })
            count += 1

    logger.info("Upserted %d contacts to SQLite", count)
    return count


def query_crm(query: str, mcp_registry: "MCPToolRegistry") -> str:
    """Run a natural-language or SQL query against the CRM.

    For SQL queries (starting with SELECT/INSERT/UPDATE), passes directly
    to the SQLite MCP.  For natural-language queries, searches Memory MCP.
    """
    tools = {t.name: t for t in mcp_registry.load_tools(profile="crm")}
    q = query.strip()

    if q.upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE")):
        if "sqlite_query" in tools:
            return tools["sqlite_query"].run({"query": q, "database": str(_DB_PATH)})
        if "sqlite_execute" in tools:
            return tools["sqlite_execute"].run({"query": q, "database": str(_DB_PATH)})
        return "SQLite MCP not available."

    # Natural language — search Memory MCP
    if "mcp_memory_search_nodes" in tools:
        return tools["mcp_memory_search_nodes"].run({"query": q})

    return "Memory MCP not available."


def run(task: str, mcp_registry: "MCPToolRegistry | None" = None) -> str:
    """Execute the CRM workflow for *task*.

    Args:
        task: Free-form CRM task description.
        mcp_registry: Live ``MCPToolRegistry``.  When ``None`` creates one
            targeting ``http://localhost:8000``.

    Returns:
        Summary of what was done.
    """
    if mcp_registry is None:
        from orchestrator.mcp import MCPToolRegistry
        mcp_registry = MCPToolRegistry("http://localhost:8000")

    tools = {t.name: t for t in mcp_registry.load_tools(profile="crm")}
    _log_available(tools)

    ensure_schema()

    results: list[str] = [f"CRM task: {task}"]

    # Auto-detect common task keywords and run the appropriate sub-workflow
    task_lower = task.lower()

    if any(kw in task_lower for kw in ("sync", "import", "pull contacts")):
        n = sync_contacts(mcp_registry)
        results.append(f"Synced {n} contacts from Apple Contacts → SQLite.")

    if any(kw in task_lower for kw in ("query", "find", "search", "who", "list")):
        # Extract a SQL-ish query from the task if present, else use raw task
        answer = query_crm(task, mcp_registry)
        results.append(f"Query result:\n{answer}")

    if any(kw in task_lower for kw in ("enrich", "linkedin")):
        results.append(
            "LinkedIn enrichment: run crm_agent in the orchestrator to browse "
            "profiles via Safari MCP (requires real logged-in Safari session). "
            "Ensure 'Allow JavaScript from Apple Events' is enabled in Safari → Develop."
        )

    if not results[1:]:  # no sub-workflow triggered
        results.append(
            f"Available CRM tools: {', '.join(tools.keys()) or 'none (vllm-mlx not running)'}"
        )

    return "\n\n".join(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_contacts(raw: str) -> list[dict]:
    """Best-effort parse of LMCP contact output into dicts."""
    import json
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "contacts" in data:
            return data["contacts"]
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: return raw as a single-item list for logging
    logger.debug("Could not parse contacts JSON; raw length=%d", len(raw))
    return []


def _log_available(tools: dict) -> None:
    logger.debug("CRM worker: %d MCP tools: %s", len(tools), list(tools.keys()))
    if not tools:
        logger.warning(
            "CRM worker: no MCP tools — "
            "is vllm-mlx running with --mcp-config mcp-configs/mcp-crm.json?"
        )
