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


# --- Market / Financial -----------------------------------------------

@tool
def get_commodity_prices(symbols: list[str]) -> str:
    """Get current commodity prices from public APIs.

    Args:
        symbols: Commodity symbols, e.g. ['gold', 'silver', 'copper', 'oil', 'natural_gas', 'wheat', 'corn'].
    """
    try:
        import requests as req
        results = []
        # Use metals.dev free API for precious metals
        metals_map = {"gold": "XAU", "silver": "XAG", "platinum": "XPT", "palladium": "XPD"}
        other_symbols = []

        metals_to_fetch = {s: metals_map[s] for s in symbols if s in metals_map}
        if metals_to_fetch:
            try:
                resp = req.get("https://api.metals.dev/v1/latest?api_key=demo&currency=USD&unit=toz", timeout=10)
                if resp.ok:
                    data = resp.json().get("metals", {})
                    for name, code in metals_to_fetch.items():
                        price = data.get(code)
                        if price:
                            results.append(f"{name.upper()}: ${price:,.2f}/oz")
                        else:
                            results.append(f"{name.upper()}: price unavailable")
                else:
                    results.append(f"Metals API error: HTTP {resp.status_code}")
            except Exception as e:
                results.append(f"Metals API error: {e}")

        for s in symbols:
            if s not in metals_map:
                other_symbols.append(s)

        # Fallback: use DuckDuckGo for non-metal commodities
        if other_symbols:
            try:
                from ddgs import DDGS
                for sym in other_symbols[:5]:
                    hits = list(DDGS().text(f"{sym} commodity price today USD", max_results=2))
                    if hits:
                        results.append(f"{sym.upper()}: {hits[0]['body'][:200]}")
                    else:
                        results.append(f"{sym.upper()}: no price data found")
            except Exception as e:
                results.append(f"Search fallback error: {e}")

        return "\n".join(results) if results else "No price data available."
    except Exception as e:
        return f"Commodity price error: {e}"


@tool
def get_crypto_prices(symbols: list[str]) -> str:
    """Get current cryptocurrency prices from CoinGecko.

    Args:
        symbols: Crypto IDs, e.g. ['bitcoin', 'ethereum', 'solana'].
    """
    try:
        import requests as req
        ids = ",".join(s.lower() for s in symbols[:10])
        resp = req.get(
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={ids}&vs_currencies=usd&include_24hr_change=true"
            f"&include_market_cap=true",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        lines = []
        for sym in symbols:
            key = sym.lower()
            if key in data:
                d = data[key]
                price = d.get("usd", "N/A")
                change = d.get("usd_24h_change")
                mcap = d.get("usd_market_cap")
                change_str = f" ({change:+.1f}% 24h)" if change is not None else ""
                mcap_str = f" | MCap: ${mcap:,.0f}" if mcap else ""
                lines.append(f"{sym.upper()}: ${price:,.2f}{change_str}{mcap_str}")
            else:
                lines.append(f"{sym.upper()}: not found on CoinGecko")
        return "\n".join(lines)
    except Exception as e:
        return f"Crypto price error: {e}"


@tool
def get_fear_greed_index() -> str:
    """Get the current Crypto Fear & Greed Index."""
    try:
        import requests as req
        resp = req.get("https://api.alternative.me/fng/?limit=7", timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return "Fear & Greed data unavailable."

        lines = ["Crypto Fear & Greed Index (last 7 days):"]
        for entry in data:
            val = entry.get("value", "?")
            label = entry.get("value_classification", "?")
            ts = entry.get("timestamp", "")
            date = ""
            if ts:
                import datetime
                date = datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
            lines.append(f"  {date}: {val} ({label})")
        return "\n".join(lines)
    except Exception as e:
        return f"Fear & Greed error: {e}"


@tool
def get_market_news(query: str) -> str:
    """Search for recent financial/market news on a topic.

    Args:
        query: The market topic to search for (e.g. 'copper futures', 'bitcoin ETF').
    """
    try:
        from ddgs import DDGS
        results = list(DDGS().news(query, max_results=8))
        if not results:
            return "No market news found."
        lines = []
        for r in results:
            date = r.get("date", "")[:10]
            lines.append(f"[{date}] **{r['title']}**\n{r.get('url', '')}\n{r['body'][:200]}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Market news error: {e}"


# --- Memory / Research ------------------------------------------------

@tool
def save_to_memory(content: str, source: str, topic: str, namespace: str = "general") -> str:
    """Save a research finding to persistent memory for later recall.

    Args:
        content: The key finding or information to remember.
        source: Where this came from (URL, document name, etc.).
        topic: The research topic this relates to.
        namespace: Memory namespace to isolate domains (e.g. 'crypto', 'commodities').
    """
    from apps.research_swarm.memory import save_finding
    return save_finding(content, source, topic, namespace=namespace)


@tool
def recall_research(query: str, topic: str = "", namespace: str = "general") -> str:
    """Search past research findings stored in memory.

    Args:
        query: What to search for in past research.
        topic: Optional topic filter to narrow results.
        namespace: Memory namespace to search in (e.g. 'crypto', 'commodities').
    """
    from apps.research_swarm.memory import recall_formatted
    return recall_formatted(query, k=5, topic=topic or None, namespace=namespace)


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
    registry.register(ToolSet(
        name="market",
        tools=[get_commodity_prices, get_crypto_prices, get_fear_greed_index,
               get_market_news, web_search, save_to_memory, recall_research],
        tags={"market", "finance", "crypto", "commodities"},
        description="Financial data: commodity/crypto prices, sentiment, market news.",
    ))
