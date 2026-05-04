"""MCP tool loading for vllm-mlx native MCP support.

When vllm-mlx is started with ``--mcp-config mcp.json`` it exposes:
- ``GET /v1/mcp/status``   — liveness / loaded-server summary
- ``GET /v1/mcp/tools``    — flat list of every tool across all MCP servers
- ``POST /v1/mcp/execute`` — execute a single tool by name + arguments

``MCPToolRegistry`` fetches that manifest and wraps each tool as a
LangChain ``StructuredTool`` so workers can pass the list directly to
``create_react_agent``.  Results from ``/v1/mcp/execute`` are translated
back to plain strings that fit into the ReAct message stream.

Usage::

    registry = MCPToolRegistry("http://localhost:8000")

    # All tools from all loaded MCP servers
    all_tools = registry.load_tools()

    # Tools filtered by profile name (if vllm-mlx supports profile param)
    research_tools = registry.load_tools(profile="research")

    # Invalidate cache (e.g. after server restart)
    registry.clear_cache()
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# Default timeout for MCP HTTP calls (seconds).
_CONNECT_TIMEOUT = 5
_EXECUTE_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """Build a minimal pydantic model from a JSON Schema ``properties`` dict.

    Supports the common subset used by MCP tools:
    - ``type``: string / number / integer / boolean / array / object
    - ``description``: forwarded to ``Field``
    - ``required``: marks fields as non-optional

    Anything more exotic falls back to ``Any``.
    """
    props: dict[str, dict] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    _TYPE_MAP = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }

    fields: dict[str, Any] = {}
    for field_name, field_def in props.items():
        raw_type = field_def.get("type", "string")
        py_type = _TYPE_MAP.get(raw_type, Any)
        desc = field_def.get("description", "")

        if field_name in required:
            fields[field_name] = (py_type, Field(..., description=desc))
        else:
            fields[field_name] = (py_type, Field(default=None, description=desc))

    if not fields:
        # No schema properties — accept a single freeform string input
        fields["input"] = (str, Field(default="", description="Tool input"))

    return create_model(f"{tool_name.replace('.', '_').replace('-', '_')}_Args", **fields)


def _extract_text(result: Any) -> str:
    """Coerce an /v1/mcp/execute response body to a plain string."""
    if isinstance(result, dict):
        # MCP standard: {"content": [{"type": "text", "text": "..."}]}
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)
        # Fallback: serialise the whole dict
        return json.dumps(result, indent=2)
    return str(result)


# ---------------------------------------------------------------------------
# Main registry
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    """Fetches MCP tool definitions from a running vllm-mlx server and
    returns them as LangChain ``StructuredTool`` objects.

    Results are cached per profile key after the first fetch so that
    subsequent calls to ``load_tools()`` within the same session are free.
    Call ``clear_cache()`` to force a re-fetch (e.g. after a server restart).
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        # Normalise: strip trailing slash, drop /v1 if present
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._cache: dict[str, list[StructuredTool]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_tools(self, profile: str | None = None) -> list[StructuredTool]:
        """Return LangChain tools for *profile* (or all tools if None).

        Args:
            profile: Optional profile name passed as ``?profile=<name>``
                to ``/v1/mcp/tools``.  Use this to scope which MCP servers'
                tools are returned (e.g. ``"research"``, ``"crm"``).
        """
        cache_key = profile or "__all__"
        if cache_key in self._cache:
            return self._cache[cache_key]

        tools_data = self._fetch_tools(profile)
        lc_tools = [self._wrap(td) for td in tools_data]
        self._cache[cache_key] = lc_tools
        logger.info(
            "MCPToolRegistry: loaded %d tools (profile=%s)", len(lc_tools), profile
        )
        return lc_tools

    def status(self) -> dict:
        """Return raw JSON from ``/v1/mcp/status``."""
        resp = requests.get(
            f"{self._base_url}/v1/mcp/status",
            timeout=_CONNECT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def clear_cache(self) -> None:
        """Invalidate all cached tool lists."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_tools(self, profile: str | None) -> list[dict]:
        params: dict[str, str] = {}
        if profile:
            params["profile"] = profile
        try:
            resp = requests.get(
                f"{self._base_url}/v1/mcp/tools",
                params=params,
                timeout=_CONNECT_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("tools", [])
        except requests.exceptions.ConnectionError:
            logger.warning(
                "MCPToolRegistry: vllm-mlx not reachable at %s — returning empty tool list",
                self._base_url,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCPToolRegistry: failed to fetch tools: %s", exc)
            return []

    def _wrap(self, tool_def: dict) -> StructuredTool:
        """Convert a single MCP tool manifest entry to a LangChain StructuredTool."""
        name: str = tool_def["name"]
        description: str = tool_def.get("description", "")
        schema: dict = tool_def.get("inputSchema", {})
        args_schema = _json_schema_to_pydantic(name, schema)

        base_url = self._base_url  # capture for closure

        def _run(**kwargs: Any) -> str:
            # Strip None-valued optional args to avoid confusing MCP servers
            clean_args = {k: v for k, v in kwargs.items() if v is not None}
            # Remove the dummy "input" key added for schema-less tools
            clean_args.pop("input", None)

            try:
                resp = requests.post(
                    f"{base_url}/v1/mcp/execute",
                    json={"tool": name, "arguments": clean_args},
                    timeout=_EXECUTE_TIMEOUT,
                )
                resp.raise_for_status()
                return _extract_text(resp.json())
            except requests.exceptions.Timeout:
                return f"[MCP tool '{name}' timed out after {_EXECUTE_TIMEOUT}s]"
            except Exception as exc:  # noqa: BLE001
                return f"[MCP tool '{name}' error: {exc}]"

        _run.__name__ = name  # LangChain uses __name__ as the tool name fallback

        return StructuredTool.from_function(
            func=_run,
            name=name,
            description=description,
            args_schema=args_schema,
        )
