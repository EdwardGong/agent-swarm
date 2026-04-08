"""Concrete tool functions for the research swarm application.

Defines the actual tool implementations and registers them as named
ToolSets that agents can reference.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from orchestrator.tools import ToolSet, ToolRegistry


# ===================================================================
# Tool implementations
# ===================================================================


# --- Web Search -------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results.

    Args:
        query: The search query string.
    """
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a web page.

    Args:
        url: The URL to fetch.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000]
    except Exception as e:
        return f"Fetch error: {e}"


@tool
def multi_search(queries: list[str]) -> str:
    """Run multiple web searches and return combined results.

    Args:
        queries: List of search query strings (max 5).
    """
    try:
        from ddgs import DDGS
        all_results = []
        for q in queries[:5]:
            results = list(DDGS().text(q, max_results=3))
            for r in results:
                all_results.append(
                    f"[Query: {q}]\n**{r['title']}**\n{r['href']}\n{r['body']}\n"
                )
        return "\n".join(all_results) if all_results else "No results found."
    except Exception as e:
        return f"Multi-search error: {e}"


@tool
def extract_article(url: str) -> str:
    """Fetch a web page and extract the main article content.

    Args:
        url: The URL of the article to extract.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "iframe", "form", "button"]):
            tag.decompose()
        article = (soup.find("article") or soup.find("main") or
                   soup.find("div", class_="content") or soup.find("body"))
        text = article.get_text(separator="\n", strip=True) if article else ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean = "\n".join(lines)
        if len(clean) > 12000:
            clean = clean[:12000] + f"\n\n... [truncated, {len(clean)} chars total]"
        return f"Source: {url}\n\n{clean}" if clean else f"Could not extract content from {url}"
    except Exception as e:
        return f"Extract error: {e}"


# --- File Operations --------------------------------------------------

@tool
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file.
    """
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"File not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 10000:
            return content[:10000] + f"\n\n... [truncated, {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Read error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        path: Absolute or relative path to the file.
        content: The text content to write.
    """
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"Write error: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path.

    Args:
        path: Directory path to list. Defaults to current directory.
    """
    try:
        p = Path(path).expanduser()
        if not p.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(p.iterdir())
        lines = []
        for e in entries[:100]:
            prefix = "📁 " if e.is_dir() else "📄 "
            lines.append(f"{prefix}{e.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"List error: {e}"


# --- Shell ------------------------------------------------------------

@tool
def run_shell(command: str) -> str:
    """Run a shell command and return its output.

    Args:
        command: The shell command to execute.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.path.expanduser("~"),
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code {result.returncode})"
        return output[:5000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s."
    except Exception as e:
        return f"Shell error: {e}"


# --- Memory / Research ------------------------------------------------

@tool
def save_to_memory(content: str, source: str, topic: str) -> str:
    """Save a research finding to persistent memory for later recall.

    Args:
        content: The key finding or information to remember.
        source: Where this came from (URL, document name, etc.).
        topic: The research topic this relates to.
    """
    from apps.research_swarm.memory import save_finding
    return save_finding(content, source, topic)


@tool
def recall_research(query: str, topic: str = "") -> str:
    """Search past research findings stored in memory.

    Args:
        query: What to search for in past research.
        topic: Optional topic filter to narrow results.
    """
    from apps.research_swarm.memory import recall_formatted
    return recall_formatted(query, k=5, topic=topic or None)


@tool
def write_report(path: str, title: str, content: str) -> str:
    """Write a research report as a Markdown file.

    Args:
        path: File path for the report.
        title: Report title.
        content: The full report content in Markdown format.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    report = f"# {title}\n\n*Generated: {__import__('time').strftime('%Y-%m-%d %H:%M')}*\n\n{content}"
    p.write_text(report, encoding="utf-8")
    return f"Report written to {p} ({len(report)} chars)"


# ===================================================================
# ToolSet registration
# ===================================================================

def register_all(registry: ToolRegistry) -> None:
    """Register all research-swarm tool-sets into the given registry."""
    registry.register(ToolSet(
        name="search",
        tools=[web_search, fetch_url],
        tags={"web", "search"},
        description="Basic web search and URL fetching.",
    ))
    registry.register(ToolSet(
        name="files",
        tools=[read_file, write_file, list_directory],
        tags={"files", "io"},
        description="Local file read/write/list operations.",
    ))
    registry.register(ToolSet(
        name="shell",
        tools=[run_shell],
        tags={"shell", "code"},
        description="Shell command execution.",
    ))
    registry.register(ToolSet(
        name="code",
        tools=[read_file, write_file, run_shell],
        tags={"code", "dev"},
        description="Code reading, writing, and execution.",
    ))
    registry.register(ToolSet(
        name="research",
        tools=[multi_search, extract_article, web_search, fetch_url,
               save_to_memory, recall_research],
        tags={"research", "web"},
        description="Deep research: multi-search, article extraction, memory.",
    ))
    registry.register(ToolSet(
        name="writer",
        tools=[recall_research, read_file, write_report],
        tags={"writing", "reports"},
        description="Report writing with memory recall.",
    ))
