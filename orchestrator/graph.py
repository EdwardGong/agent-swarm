"""Dynamic LangGraph workflow builder.

Constructs a full orchestration graph from the agent and tool registries
at runtime — no hardcoded agent names anywhere.

Flow:
  1. User query enters the orchestrator node.
  2. The router picks the next agent (or finishes).
  3. The chosen worker runs with its declared tools.
  4. Control returns to the orchestrator for the next decision.
  5. Repeat until done or max iterations reached.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from orchestrator.agent import AgentRegistry
from orchestrator.config import OrchestratorConfig
from orchestrator.models import ModelProvider
from orchestrator.router import Router
from orchestrator.state import OrchestratorState
from orchestrator.tools import ToolRegistry


_DEFAULT_WORKER_SYSTEM = (
    "You are a specialist worker agent. Complete the assigned task "
    "using your tools. Be concise and thorough. Return your findings "
    "as a clear summary."
)


def build_graph(
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    model_provider: ModelProvider,
    router: Router,
    config: OrchestratorConfig,
) -> Any:
    """Build and compile a LangGraph orchestration graph.

    Every registered agent becomes a graph node.  Routing is fully
    dynamic — driven by the ``router`` instance.
    """

    max_iter = config.max_iterations
    _counter: dict[str, int] = {"n": 0}  # mutable counter for closure

    # ------------------------------------------------------------------
    # Orchestrator node
    # ------------------------------------------------------------------

    def orchestrator_node(state: OrchestratorState) -> dict:
        return router.route(state)

    # ------------------------------------------------------------------
    # Generic worker node factory
    # ------------------------------------------------------------------

    def _make_worker_node(agent_name: str):
        """Return a graph-node function for the given agent spec."""

        def worker_node(state: OrchestratorState) -> dict:
            spec = agent_registry.get(agent_name)
            llm = model_provider.get_llm(spec.model_tier)

            # Collect tools from declared tool-set names
            tools: list[Any] = []
            for ts_name in spec.toolset_names:
                tools.extend(tool_registry.get_tools(ts_name))

            agent = create_react_agent(llm, tools=tools)
            result = agent.invoke({
                "messages": [
                    SystemMessage(content=spec.system_prompt or _DEFAULT_WORKER_SYSTEM),
                    HumanMessage(content=state.sub_task),
                ]
            })

            last_msg = result["messages"][-1]
            summary = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            return {"results": [f"[{agent_name}] {summary}"]}

        worker_node.__name__ = agent_name  # nicer debugging
        return worker_node

    # ------------------------------------------------------------------
    # Conditional routing
    # ------------------------------------------------------------------

    agent_names = agent_registry.names

    def route_after_orchestrator(state: OrchestratorState) -> str:
        _counter["n"] += 1
        if state.done or _counter["n"] > max_iter:
            _counter["n"] = 0
            return "end"
        return state.next_agent if state.next_agent in agent_names else "end"

    # ------------------------------------------------------------------
    # Assemble the graph
    # ------------------------------------------------------------------

    graph = StateGraph(OrchestratorState)
    graph.add_node("orchestrator", orchestrator_node)

    routing_map: dict[str, str] = {"end": END}

    for name in agent_names:
        graph.add_node(name, _make_worker_node(name))
        graph.add_edge(name, "orchestrator")
        routing_map[name] = name

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", route_after_orchestrator, routing_map)

    return graph.compile()
