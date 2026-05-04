#!/usr/bin/env python3
"""Consolidated Agent Swarm — M5 Max 128GB entry point.

Usage:
    # Single query (auto-routed to the right agent):
    python -m apps.consolidated_swarm.main "Your task here"

    # Interactive loop:
    python -m apps.consolidated_swarm.main --interactive

    # Deep research mode (forces research_agent, then content_agent for report):
    python -m apps.consolidated_swarm.main --research "Topic to investigate"

    # CRM sync:
    python -m apps.consolidated_swarm.main --crm "Sync all contacts and enrich from LinkedIn"

    # Deck / presentation:
    python -m apps.consolidated_swarm.main --content "Build a pitch deck for project X"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from orchestrator import (
    AgentRegistry,
    OrchestratorConfig,
    VLLMMLXProvider,
    LLMRouter,
    ToolRegistry,
    ToolSet,
    build_graph,
    MCPToolRegistry,
)
from apps.consolidated_swarm.agents import register_all as register_agents, MCP_PROFILE

CONFIG_PATH = Path(__file__).parent / "config.yaml"

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent ↔ icon map (cosmetic only)
# ---------------------------------------------------------------------------

_ICONS = {
    "research_agent": "🔬",
    "code_agent":     "💻",
    "crm_agent":      "🗂️",
    "content_agent":  "📊",
}


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------

def _build_app():
    """Wire the orchestration framework with consolidated-swarm config."""
    config = OrchestratorConfig.from_yaml(CONFIG_PATH)
    provider = VLLMMLXProvider(config)

    mcp_base = config.extra.get("mcp_base_url", "http://localhost:8000")
    mcp_registry = MCPToolRegistry(mcp_base)

    agent_registry = AgentRegistry()
    register_agents(agent_registry)

    # Build ToolSets from MCP profiles.  Each agent's toolset_names entry
    # maps to a ToolSet whose tools are fetched live from vllm-mlx.
    # Falls back to an empty list when vllm-mlx is not running (e.g. tests).
    tool_registry = ToolRegistry()
    for agent_name, profile in MCP_PROFILE.items():
        toolset_name = f"mcp_{profile}"
        mcp_tools = mcp_registry.load_tools(profile=profile)
        tool_registry.register(ToolSet(
            name=toolset_name,
            tools=mcp_tools,
            tags={profile, "mcp"},
            description=f"MCP tools for the {profile} profile.",
        ))

    router = LLMRouter(provider, agent_registry)

    return build_graph(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        model_provider=provider,
        router=router,
        config=config,
    )


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def run_query(query: str) -> str:
    """Route *query* through the swarm and return the final answer."""
    app = _build_app()
    print("\n🧠 Orchestrator thinking...\n")

    agent_names = set(MCP_PROFILE.keys())
    final_answer = ""

    for event in app.stream(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 80},
    ):
        for node_name, node_output in event.items():
            if node_name == "orchestrator":
                next_ag = node_output.get("next_agent", "")
                sub = node_output.get("sub_task", "")
                if next_ag and next_ag != "done":
                    icon = _ICONS.get(next_ag, "📋")
                    print(f"  {icon} [{next_ag}]: {sub}")
                if node_output.get("done"):
                    msgs = node_output.get("messages", [])
                    if msgs:
                        final_answer = msgs[-1].content
            elif node_name in agent_names:
                for r in node_output.get("results", []):
                    preview = r[:300] + "..." if len(r) > 300 else r
                    print(f"  ✅ {node_name}: {preview}")

    return final_answer


# ---------------------------------------------------------------------------
# Preset modes (wrap query with directive framing)
# ---------------------------------------------------------------------------

def run_research(topic: str) -> str:
    query = (
        f"RESEARCH TASK: {topic}\n\n"
        "Use research_agent to investigate this topic deeply — search broadly, "
        "extract key articles, store findings to FAISS and Memory MCP. "
        "Then use content_agent to synthesise a comprehensive Markdown report "
        "saved to <repo>/runtime/workspace/reports/."
    )
    print(f"\n🔬 Deep Research Mode: {topic}\n")
    return run_query(query)


def run_crm(task: str) -> str:
    query = (
        f"CRM TASK: {task}\n\n"
        "Use crm_agent to complete this task. "
        "Always show a draft or preview before sending any message or email."
    )
    print(f"\n🗂️  CRM Mode\n")
    return run_query(query)


def run_content(task: str) -> str:
    query = (
        f"CONTENT TASK: {task}\n\n"
        "Use content_agent to build the requested deliverable. "
        "Pull research findings from FAISS and CRM data from SQLite as needed. "
        "Save all output files to <repo>/runtime/workspace/."
    )
    print(f"\n📊 Content Mode\n")
    return run_query(query)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode():
    print("=" * 62)
    print("  🐝 Consolidated Agent Swarm — M5 Max 128GB")
    print("  research · code · crm · content")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 62)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Consolidated Agent Swarm — M5 Max 128GB"
    )
    parser.add_argument("query", nargs="?", help="Single query to route")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive loop")
    parser.add_argument("--research", "-r", metavar="TOPIC",
                        help="Deep research mode — investigate a topic and produce a report")
    parser.add_argument("--crm", metavar="TASK",
                        help="CRM mode — contact sync, enrichment, outreach")
    parser.add_argument("--content", metavar="TASK",
                        help="Content mode — decks, reports, presentations")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.research:
        answer = run_research(args.research)
        print(f"\n🤖 Research complete:\n{answer}")
    elif args.crm:
        answer = run_crm(args.crm)
        print(f"\n🤖 CRM result:\n{answer}")
    elif args.content:
        answer = run_content(args.content)
        print(f"\n🤖 Content ready:\n{answer}")
    elif args.interactive:
        interactive_mode()
    elif args.query:
        answer = run_query(args.query)
        print(f"\n🤖 Answer:\n{answer}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
