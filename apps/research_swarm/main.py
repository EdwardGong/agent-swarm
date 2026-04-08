#!/usr/bin/env python3
"""Research Swarm — example application built on the orchestration layer.

Usage:
    python -m apps.research_swarm.main "Quick question here"
    python -m apps.research_swarm.main --interactive
    python -m apps.research_swarm.main --research "Deep research topic here"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from orchestrator import (
    AgentRegistry,
    OrchestratorConfig,
    OllamaProvider,
    LLMRouter,
    ToolRegistry,
    build_graph,
)
from apps.research_swarm.agents import register_all as register_agents
from apps.research_swarm.tools import register_all as register_tools


CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _build_app():
    """Wire up the orchestration framework with research-swarm agents."""
    config = OrchestratorConfig.from_yaml(CONFIG_PATH)
    provider = OllamaProvider(config)

    agent_registry = AgentRegistry()
    register_agents(agent_registry)

    tool_registry = ToolRegistry()
    register_tools(tool_registry)

    router = LLMRouter(provider, agent_registry)

    return build_graph(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        model_provider=provider,
        router=router,
        config=config,
    )


def run_query(query: str, research_mode: bool = False) -> str:
    """Run a single query through the research swarm."""
    app = _build_app()

    if research_mode:
        query = (
            f"RESEARCH TASK: {query}\n\n"
            f"Instructions: Use the research_agent to deeply investigate this topic. "
            f"The research_agent should search broadly, read key articles, and save "
            f"important findings to memory. After research is complete, use the "
            f"writer_agent to produce a comprehensive report saved to "
            f"~/workspace/ai/agent-swarm/reports/"
        )
        print("\n🔬 Deep Research Mode\n")
    else:
        print("\n🧠 Orchestrator thinking...\n")

    agent_names = ("search_agent", "file_agent", "code_agent",
                   "research_agent", "writer_agent")

    final_answer = ""
    for event in app.stream(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 50},
    ):
        for node_name, node_output in event.items():
            if node_name == "orchestrator":
                next_ag = node_output.get("next_agent", "")
                sub = node_output.get("sub_task", "")
                if next_ag and next_ag != "done":
                    icon = {"research_agent": "🔬", "writer_agent": "✍️ "}.get(next_ag, "📋")
                    print(f"  {icon} Delegating to [{next_ag}]: {sub}")
                if node_output.get("done"):
                    msgs = node_output.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content
            elif node_name in agent_names:
                results = node_output.get("results", [])
                for r in results:
                    preview = r[:300] + "..." if len(r) > 300 else r
                    print(f"  ✅ {node_name} result: {preview}")

    return final_answer


def interactive_mode():
    """Run the swarm in an interactive loop."""
    print("=" * 60)
    print("  🐝 Research Swarm — Interactive Mode")
    print("  Powered by the Agent Orchestration Layer")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\n🟢 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        answer = run_query(query)
        print(f"\n🤖 Swarm: {answer}")


def main():
    parser = argparse.ArgumentParser(
        description="Research Swarm — built on the Agent Orchestration Layer"
    )
    parser.add_argument("query", nargs="?", help="Single query to run")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--research", "-r", type=str, metavar="TOPIC",
                        help="Deep research mode — investigate a topic and produce a report")
    args = parser.parse_args()

    if args.research:
        answer = run_query(args.research, research_mode=True)
        print(f"\n🤖 Research complete: {answer}")
    elif args.interactive:
        interactive_mode()
    elif args.query:
        answer = run_query(args.query)
        print(f"\n🤖 Answer: {answer}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
