"""Content and presentation workflow worker.

Implements the plan's "Decks / Presentations" workflow:

  1. Research worker gathers data (FAISS RAG)
  2. CRM worker pulls relevant contact/company data (SQLite)
  3. Content worker generates structured Markdown outline → slide content
  4. Export: python-pptx for PPTX, or Marp/reveal.js for HTML slides
  5. Safari MCP screenshot each slide for review (optional)

Output is written to <repo>/runtime/workspace/<project>/.

Standalone usage::

    from apps.consolidated_swarm.workers.content import run
    run("Build a pitch deck for the agent-swarm project")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.mcp import MCPToolRegistry

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[3]
_WORKSPACE = _REPO_ROOT / "runtime" / "workspace"

# Marp front-matter template for HTML slides
_MARP_HEADER = """\
---
marp: true
theme: default
paginate: true
---

"""


def run(task: str, mcp_registry: "MCPToolRegistry | None" = None) -> str:
    """Execute the content generation workflow for *task*.

    Args:
        task: Description of the content to produce (e.g. "pitch deck for X").
        mcp_registry: Live ``MCPToolRegistry``.  When ``None`` creates one
            targeting ``http://localhost:8000``.

    Returns:
        Path to the generated content file(s).
    """
    if mcp_registry is None:
        from orchestrator.mcp import MCPToolRegistry
        mcp_registry = MCPToolRegistry("http://localhost:8000")

    tools = {t.name: t for t in mcp_registry.load_tools(profile="content")}
    _log_available(tools)

    # ------------------------------------------------------------------
    # Step 1: Pull research findings from FAISS
    # ------------------------------------------------------------------
    research_context = ""
    if "faiss_query_rag_store" in tools:
        research_context = tools["faiss_query_rag_store"].run({
            "query": task,
            "k": 8,
        })

    # ------------------------------------------------------------------
    # Step 2: Pull CRM context from SQLite
    # ------------------------------------------------------------------
    crm_context = ""
    if "sqlite_query" in tools:
        crm_context = tools["sqlite_query"].run({
            "query": "SELECT name, company, title FROM contacts LIMIT 20",
            "database": str(Path(__file__).parents[3] / "runtime" / "state.db"),
        })

    # ------------------------------------------------------------------
    # Step 3-4: Generate Marp slides and write to disk
    # ------------------------------------------------------------------
    slides_md = _generate_marp_outline(task, research_context, crm_context)

    slug = task[:50].lower().replace(" ", "-").replace("/", "-")
    project_dir = _WORKSPACE / slug
    project_dir.mkdir(parents=True, exist_ok=True)

    md_path = project_dir / f"{slug}.md"
    md_path.write_text(slides_md, encoding="utf-8")

    # Write via Filesystem MCP as well (so the agent loop can refine it)
    if "filesystem_write_file" in tools:
        tools["filesystem_write_file"].run({
            "path": str(md_path),
            "content": slides_md,
        })

    results = [
        f"Content generated: {md_path}",
        f"Render as HTML: marp {md_path} --html",
        f"Render as PDF: marp {md_path} --pdf",
        f"Convert to PPTX: marp {md_path} --pptx",
    ]

    # ------------------------------------------------------------------
    # Step 5: Optional Safari screenshot (non-blocking)
    # ------------------------------------------------------------------
    if "safari_screenshot" in tools:
        try:
            # Open local Marp preview if running, else skip
            tools["safari_screenshot"].run({"url": f"file://{md_path}"})
            results.append("Safari screenshot taken.")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Safari screenshot skipped: %s", exc)

    return "\n".join(results)


def generate_pptx_script(title: str, slides: list[dict]) -> str:
    """Return a python-pptx script string for the given slide data.

    Each slide dict: {"title": str, "bullets": list[str]}.
    The script can be written to disk and executed via the code_agent.
    """
    lines = [
        "from pptx import Presentation",
        "from pptx.util import Inches, Pt",
        "",
        "prs = Presentation()",
        "slide_layout = prs.slide_layouts[1]  # title + content",
        "",
    ]
    for i, slide in enumerate(slides):
        stitle = slide.get("title", f"Slide {i+1}").replace('"', '\\"')
        lines.append(f"# Slide {i+1}: {stitle}")
        lines.append("slide = prs.slides.add_slide(slide_layout)")
        lines.append(f'slide.shapes.title.text = "{stitle}"')
        lines.append("tf = slide.placeholders[1].text_frame")
        lines.append("tf.text = ''")
        for bullet in slide.get("bullets", []):
            b = bullet.replace('"', '\\"')
            lines.append(f'p = tf.add_paragraph()')
            lines.append(f'p.text = "{b}"')
            lines.append("p.level = 0")
        lines.append("")

    slug = title[:30].lower().replace(" ", "_")
    lines += [
        f'prs.save(str(Path(__file__).parents[3] / "runtime" / "workspace" / "{slug}.pptx"))',
        'print("PPTX saved.")',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_marp_outline(task: str, research: str, crm: str) -> str:
    """Build a skeleton Marp Markdown presentation.

    In the full agent loop the LLM expands this skeleton — this function
    provides the structural scaffold the agent populates with real content.
    """
    import time
    date = time.strftime("%Y-%m-%d")

    slides = [
        f"# {task}\n*{date}*",
        "---\n## Agenda\n- Background\n- Key Findings\n- Implications\n- Next Steps",
        "---\n## Background\n*(content_agent: expand from research context)*",
        "---\n## Key Findings\n*(content_agent: expand from FAISS findings)*",
        "---\n## Implications\n*(content_agent: derive from findings + CRM context)*",
        "---\n## Next Steps\n*(content_agent: actionable recommendations)*",
        "---\n## Thank You\n*Questions?*",
    ]

    context_appendix = []
    if research:
        context_appendix.append(f"<!-- RESEARCH CONTEXT (for agent expansion):\n{research[:2000]}\n-->")
    if crm:
        context_appendix.append(f"<!-- CRM CONTEXT:\n{crm[:500]}\n-->")

    return _MARP_HEADER + "\n\n".join(slides) + "\n\n" + "\n\n".join(context_appendix)


def _log_available(tools: dict) -> None:
    logger.debug("Content worker: %d MCP tools: %s", len(tools), list(tools.keys()))
    if not tools:
        logger.warning(
            "Content worker: no MCP tools — "
            "is vllm-mlx running with --mcp-config mcp-configs/mcp-content.json?"
        )
