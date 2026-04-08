"""Agent specifications for the research swarm application.

Each agent is a declarative ``AgentSpec`` — no node functions here.
The framework builds graph nodes automatically from these specs.
"""

from __future__ import annotations

from orchestrator.agent import AgentSpec, AgentRegistry


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM = """\
You are a deep research agent. Your job is to thoroughly investigate a topic.

Workflow:
1. First, check if there's relevant past research using recall_research.
2. Run multiple searches using multi_search to explore the topic broadly.
3. For the most promising results, use extract_article to read the full content.
4. Save key findings to memory using save_to_memory (with source URL and topic).
5. Return a comprehensive summary of what you found, with source URLs.

Be thorough — follow leads, cross-reference information, and dig deep.
Always cite your sources with URLs.
Always save important findings to memory so they can be recalled later."""

WRITER_SYSTEM = """\
You are a research report writer. Your job is to synthesize research
findings into a clear, well-structured report.

Workflow:
1. FIRST, use recall_research to retrieve ALL relevant findings from memory.
   Try multiple recall queries to get comprehensive coverage.
2. Read through all findings carefully — each has a source URL and content.
3. Organize the information into a coherent structure.
4. Write the report in Markdown format with:
   - Executive summary (2-3 paragraphs)
   - Detailed findings organized by theme with SPECIFIC facts, names, numbers
   - Direct quotes where impactful
   - Sources/references section listing EVERY real URL from the findings
5. Use write_report to save the report.

CRITICAL RULES:
- NEVER use placeholder URLs like example.com. Only use real URLs from the findings.
- NEVER fabricate information. Only include facts from the retrieved findings.
- Include specific details: project names, version numbers, GitHub stars, dates.
- If findings are sparse, say so honestly rather than padding with generic content."""


# ---------------------------------------------------------------------------
# Agent specs
# ---------------------------------------------------------------------------

AGENTS = [
    AgentSpec(
        name="search_agent",
        description="Quick web search and URL fetching for simple lookups.",
        capabilities=["search", "web", "lookup"],
        toolset_names=["search"],
        model_tier="worker_fast",
    ),
    AgentSpec(
        name="file_agent",
        description="Reading, writing, and listing files on the local machine.",
        capabilities=["files", "read", "write", "filesystem"],
        toolset_names=["files"],
        model_tier="worker_light",
    ),
    AgentSpec(
        name="code_agent",
        description="Writing, reviewing, and executing code.",
        capabilities=["code", "programming", "development", "execute"],
        toolset_names=["code"],
        model_tier="coder",
    ),
    AgentSpec(
        name="research_agent",
        description=(
            "Deep research — multi-step web search, article extraction, "
            "and saving findings to memory. Use for ANY information "
            "gathering task."
        ),
        capabilities=["research", "investigate", "deep-search"],
        toolset_names=["research"],
        model_tier="worker_fast",
        system_prompt=RESEARCH_SYSTEM,
    ),
    AgentSpec(
        name="writer_agent",
        description=(
            "Synthesizes research into structured reports with citations. "
            "Use AFTER the research_agent has gathered enough information."
        ),
        capabilities=["writing", "reports", "synthesis"],
        toolset_names=["writer"],
        model_tier="worker",
        system_prompt=WRITER_SYSTEM,
    ),
]


def register_all(registry: AgentRegistry) -> None:
    """Register all research-swarm agents into the given registry."""
    registry.register_many(AGENTS)
