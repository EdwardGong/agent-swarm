"""Tool registry for the orchestration layer.

Tools are grouped into named ``ToolSet`` objects.  The ``ToolRegistry``
lets agents declare which tool-sets they need by name or capability tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ToolSet:
    """A named, tagged collection of LangChain tools."""

    name: str
    tools: list[Any]  # list[BaseTool], kept as Any to avoid import coupling
    tags: set[str] = field(default_factory=set)
    description: str = ""


class ToolRegistry:
    """Central store for tool-sets.  Thread-safe for read-heavy usage."""

    def __init__(self) -> None:
        self._sets: dict[str, ToolSet] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, toolset: ToolSet) -> None:
        """Register a tool-set by name."""
        self._sets[toolset.name] = toolset

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSet:
        """Retrieve a tool-set by exact name."""
        if name not in self._sets:
            raise KeyError(f"ToolSet '{name}' not registered. Available: {list(self._sets)}")
        return self._sets[name]

    def get_tools(self, name: str) -> list[Any]:
        """Shorthand: return just the tool list for a named set."""
        return self.get(name).tools

    def find_by_tag(self, tag: str) -> list[ToolSet]:
        """Return all tool-sets that carry the given tag."""
        return [ts for ts in self._sets.values() if tag in ts.tags]

    def all_tools_for_tags(self, tags: Sequence[str]) -> list[Any]:
        """Collect all tools from tool-sets matching *any* of the tags."""
        seen_names: set[str] = set()
        tools: list[Any] = []
        for tag in tags:
            for ts in self.find_by_tag(tag):
                if ts.name not in seen_names:
                    seen_names.add(ts.name)
                    tools.extend(ts.tools)
        return tools

    @property
    def names(self) -> list[str]:
        return list(self._sets)
