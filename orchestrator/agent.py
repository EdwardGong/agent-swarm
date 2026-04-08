"""Declarative agent specifications and a runtime registry.

An ``AgentSpec`` describes *what* an agent is — name, capabilities,
tools, model tier, and system prompt.  The framework uses this metadata
to dynamically build graph nodes and generate the orchestrator prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class AgentSpec:
    """Declarative specification of an agent in the swarm.

    Attributes:
        name: Unique identifier used as the graph node name.
        description: One-liner the orchestrator sees when choosing agents.
        capabilities: Free-form tags for searching / grouping.
        toolset_names: Names of ``ToolSet`` objects this agent needs.
        model_tier: Key into the model-tier map (e.g. 'worker_fast').
        system_prompt: Optional custom system prompt.  If omitted the
            framework provides a sensible default.
    """

    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    toolset_names: list[str] = field(default_factory=list)
    model_tier: str = "worker"
    system_prompt: str | None = None


class AgentRegistry:
    """Runtime registry of agent specifications."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: AgentSpec) -> None:
        """Register an agent spec."""
        self._agents[spec.name] = spec

    def register_many(self, specs: Sequence[AgentSpec]) -> None:
        for spec in specs:
            self.register(spec)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> AgentSpec:
        if name not in self._agents:
            raise KeyError(
                f"Agent '{name}' not registered. Available: {list(self._agents)}"
            )
        return self._agents[name]

    @property
    def names(self) -> list[str]:
        return list(self._agents)

    @property
    def all(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def find_by_capability(self, cap: str) -> list[AgentSpec]:
        """Return agents that declare the given capability tag."""
        return [a for a in self._agents.values() if cap in a.capabilities]

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def describe_for_prompt(self) -> str:
        """Generate a description block for the orchestrator system prompt.

        Lists every registered agent with its name, description, and
        capabilities so the LLM can make informed routing decisions.
        """
        lines: list[str] = []
        for spec in self._agents.values():
            caps = ", ".join(spec.capabilities) if spec.capabilities else "general"
            lines.append(f'- "{spec.name}": {spec.description} (capabilities: {caps})')
        return "\n".join(lines)
