"""Agent specifications for the consolidated swarm.

Four specialist agents map to the four workflows defined in the plan:
  1. research_agent  — deep research with Firecrawl + FAISS + Memory MCP
  2. code_agent      — file ops, git, shell, and browsing for docs
  3. crm_agent       — Apple Contacts / Mail sync → SQLite + LinkedIn enrichment
  4. content_agent   — decks / presentations (research → outline → slides)

Each agent's MCP profile is declared in its spec so main.py can pass the
correct tool set from MCPToolRegistry.
"""

from __future__ import annotations

from orchestrator.agent import AgentSpec, AgentRegistry


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM = """\
You are a deep-research agent.  Your job is to investigate topics thoroughly,
accumulate findings in FAISS and the Memory knowledge graph, and synthesise a
comprehensive report.

Workflow:
1. Search: use firecrawl_search (primary) or safari_navigate_and_search (fallback
   for login-gated pages) to discover relevant sources.
2. Extract: for each promising URL, call firecrawl_scrape or
   safari_navigate_and_read to retrieve clean article text.
3. Chunk & store: call faiss_ingest_document for every extracted article.
4. Remember entities: call mcp_memory_create_entities + mcp_memory_create_relations
   to capture key players, companies, and relationships.
5. Recall prior work: call faiss_query_rag_store and mcp_memory_search_nodes
   before formulating your answer — avoid re-researching what you already know.
6. Iterate: generate follow-up sub-questions and repeat until convergence.
7. Synthesise: combine RAG results + memory observations into a final Markdown
   report with citations (real URLs only — never fabricate).

Efficiency rules:
- Prefix cache gives 10-30× TTFT on repeated system prompts — keep your prompt
  stable across loop iterations.
- After each session, prune low-relevance FAISS chunks.
- Save every important finding to both FAISS and Memory MCP before finishing."""

CODE_SYSTEM = """\
You are a code-agent.  You read, write, and run code; commit via Git; and can
browse documentation using Safari MCP when you need to look something up.

Workflow:
1. Understand the task: read relevant files using filesystem_read_file.
2. Generate or edit: write modified content back with filesystem_write_file.
3. Verify: run tests or the program with filesystem_execute (if available) or
   interpret the code mentally.
4. Commit: stage with git_add, commit with git_commit using an imperative subject
   line and a "why" body.
5. Docs browsing: if you need library docs, use safari_navigate_and_read to fetch
   the official docs URL — do not hallucinate API signatures.

Rules:
- Read enough surrounding code before editing to match existing conventions.
- One logical change per commit.
- Never commit directly to master/main; always work on a feature branch."""

CRM_SYSTEM = """\
You are a CRM and LinkedIn enrichment agent.  You sync Apple Contacts to a local
SQLite database, enrich contact records from LinkedIn via Safari MCP (your real
logged-in session), and maintain a relationship knowledge graph in Memory MCP.

Workflow — full sync:
1. Pull contacts: call lmcp_list_contacts (or lmcp_search_contacts for targeted
   lookups) to get the current Apple Contacts list.
2. Upsert to SQLite: call sqlite_execute with INSERT OR REPLACE INTO contacts
   (name, email, phone, company, title, source, last_contacted) for each entry.
3. Enrich from LinkedIn: for contacts missing title/company, call
   safari_navigate_and_read on their LinkedIn profile URL, parse the structured
   data, and upsert the enriched fields.
4. Update Memory MCP: call mcp_memory_create_entities for each person/company,
   mcp_memory_create_relations for known relationships (introduced_by, works_at,
   knows), and mcp_memory_add_observations for recent interactions.
5. Query CRM: use sqlite_query for structured lookups
   (e.g. "SELECT * FROM contacts WHERE title LIKE '%CTO%'")
   and mcp_memory_search_nodes for relationship queries
   (e.g. "who introduced me to X?").
6. Outreach: draft messages with the LLM, then send via lmcp_send_email — always
   show a preview and require explicit confirmation before sending.
7. Daily maintenance: check lmcp_list_calendar_events + lmcp_list_emails to
   extract interaction updates → SQLite UPDATE + Memory MCP add_observations.

Key constraint: $0/mo. Use LMCP + Safari MCP only. Add Crispy.sh ($49/mo) later
for API-level LinkedIn automation when budget allows."""

CONTENT_SYSTEM = """\
You are a content and presentation agent.  You create decks, reports, and
structured documents by pulling from research findings and CRM data.

Workflow — deck / presentation:
1. Research: call faiss_query_rag_store to retrieve relevant research findings.
2. CRM data: call sqlite_query to pull related contacts, companies, and context.
3. Outline: generate a structured Markdown outline with clear sections, key
   messages, and supporting data points.
4. Slide content: expand each outline section into slide content — short
   bullet-point slides (5-7 bullets max), one idea per slide.
5. Export:
   - PPTX: write a python-pptx script to filesystem and run it.
   - HTML slides: generate Marp or reveal.js Markdown and write to filesystem.
6. Review: optionally use safari_screenshot to capture rendered slides for review.

Rules:
- Ground every claim in retrieved research or CRM data — no hallucinated stats.
- Cite sources inline (URL or memory reference).
- Write output files to <repo>/runtime/workspace/<project>/ ."""


# ---------------------------------------------------------------------------
# MCP profiles — passed to MCPToolRegistry.load_tools(profile=...)
# ---------------------------------------------------------------------------

MCP_PROFILE = {
    "research_agent": "research",
    "code_agent":     "code",
    "crm_agent":      "crm",
    "content_agent":  "content",
}


# ---------------------------------------------------------------------------
# Agent specs
# ---------------------------------------------------------------------------

AGENTS = [
    AgentSpec(
        name="research_agent",
        description=(
            "Deep autonomous research — web search (Firecrawl + Safari MCP), "
            "article extraction, FAISS vector storage, Memory knowledge graph. "
            "Use for ANY information-gathering, market research, or competitive "
            "intelligence task."
        ),
        capabilities=[
            "research", "investigate", "search", "web", "scrape",
            "news", "market", "competitive", "intelligence",
        ],
        toolset_names=["mcp_research"],
        model_tier="worker_fast",
        system_prompt=RESEARCH_SYSTEM,
    ),
    AgentSpec(
        name="code_agent",
        description=(
            "Code writing, review, and execution; Git operations; documentation "
            "browsing via Safari MCP. Use for ANY development, scripting, "
            "refactoring, or repository management task."
        ),
        capabilities=[
            "code", "programming", "development", "git", "commit",
            "script", "refactor", "build", "test", "debug",
        ],
        toolset_names=["mcp_code"],
        model_tier="coder",
        system_prompt=CODE_SYSTEM,
    ),
    AgentSpec(
        name="crm_agent",
        description=(
            "CRM sync and LinkedIn enrichment — pulls Apple Contacts via LMCP, "
            "upserts to SQLite, enriches from LinkedIn via Safari MCP, maintains "
            "a relationship knowledge graph in Memory MCP. "
            "Use for contact management, outreach drafting, and relationship queries."
        ),
        capabilities=[
            "crm", "contacts", "linkedin", "email", "outreach",
            "relationships", "networking", "calendar", "mail",
        ],
        toolset_names=["mcp_crm"],
        model_tier="worker",
        system_prompt=CRM_SYSTEM,
    ),
    AgentSpec(
        name="content_agent",
        description=(
            "Decks, presentations, and structured documents — combines research "
            "findings (FAISS) and CRM data (SQLite) into slide-ready content. "
            "Exports PPTX or HTML (Marp/reveal.js). "
            "Use for pitches, reports, slide decks, or any content-generation task."
        ),
        capabilities=[
            "content", "deck", "presentation", "slides", "pptx",
            "report", "document", "write", "draft",
        ],
        toolset_names=["mcp_content"],
        model_tier="worker",
        system_prompt=CONTENT_SYSTEM,
    ),
]


def register_all(registry: AgentRegistry) -> None:
    """Register all consolidated-swarm agents into the given registry."""
    registry.register_many(AGENTS)
