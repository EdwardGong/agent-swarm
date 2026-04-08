# Agent Orchestration Layer

A generic framework for building multi-agent systems on LangGraph.
Agents, tools, models, and routing strategies are all pluggable —
define your agents as data and the framework wires up the graph.

Ships with a **research swarm** application featuring 6 specialist agents,
namespaced persistent memory, financial market tools, and an automated
sweep scheduler for recurring intelligence gathering.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Application Layer  (e.g. apps/research_swarm/)     │
│  ┌────────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ AgentSpecs │  │ ToolSets │  │ config.yaml   │   │
│  └─────┬──────┘  └────┬─────┘  └──────┬────────┘   │
└────────┼──────────────┼───────────────┼─────────────┘
         │              │               │
┌────────▼──────────────▼───────────────▼─────────────┐
│  Orchestration Layer  (orchestrator/)               │
│                                                     │
│  AgentRegistry ─── Router ─── ModelProvider         │
│       │              │              │               │
│  ToolRegistry    LLMRouter      OllamaProvider      │
│                  RuleRouter     (your provider)      │
│       │              │              │               │
│       └──────── GraphBuilder ───────┘               │
│                      │                              │
│               LangGraph (dynamic)                   │
└─────────────────────────────────────────────────────┘
```

## Core Concepts

- **AgentSpec** — declarative description of an agent (name, capabilities, tools, model tier, prompt)
- **AgentRegistry** — runtime store; auto-generates the orchestrator prompt from registered agents
- **ToolSet / ToolRegistry** — named, tagged collections of LangChain tools
- **ModelProvider** — protocol for any LLM backend (ships with `OllamaProvider`)
- **Router** — pluggable routing strategy (`LLMRouter` or `RuleRouter`)
- **Namespaced Memory** — ChromaDB-backed persistent memory with domain isolation
- **build_graph()** — assembles a LangGraph workflow dynamically from the registries

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve &

# 3. Run the research swarm
python -m apps.research_swarm.main "Your task here"

# 4. Interactive mode
python -m apps.research_swarm.main --interactive

# 5. Deep research mode (multi-step research → report)
python -m apps.research_swarm.main --research "Topic to investigate"
```

## Agents

The research swarm ships with 6 specialist agents. The orchestrator
automatically routes queries to the right one:

- **market_agent** — live commodity/crypto prices, Fear & Greed index, market news. Saves findings to namespaced memory for accumulation over time.
- **research_agent** — deep multi-step web research with article extraction and persistent memory.
- **writer_agent** — synthesizes research findings into structured Markdown reports with citations.
- **search_agent** — quick web search and URL fetching for simple lookups.
- **code_agent** — code writing, review, and shell execution.
- **file_agent** — local file read/write/list operations.

## Namespaced Memory

Research findings are stored in ChromaDB with **namespace isolation**.
Each domain (crypto, commodities, AI, etc.) gets its own collection,
preventing cross-contamination and enabling accumulated intelligence.

```python
# Agents specify the namespace when saving/recalling
save_to_memory(content, source, topic, namespace="crypto")
recall_research(query, namespace="crypto")
```

The 10th research sweep on a topic is dramatically better than the 1st
because the agent recalls all prior findings before searching.

## Automated Sweep Scheduler

Define recurring research jobs in `sweeps.yaml` and run them on a schedule.
Each sweep targets a specific memory namespace and report directory.

```yaml
sweeps:
  - name: crypto-market-pulse
    query: "Check BTC/ETH/SOL prices, Fear & Greed, and top crypto news..."
    schedule: "daily"
    namespace: crypto
    report_dir: ~/workspace/ai/agent-swarm/reports/crypto
    enabled: true
```

```bash
# Run as a daemon (fires sweeps on schedule):
python -m apps.research_swarm.scheduler

# Run all enabled sweeps once now:
python -m apps.research_swarm.scheduler --run-now

# Run a specific sweep:
python -m apps.research_swarm.scheduler --run-now --sweep crypto-market-pulse

# List configured sweeps:
python -m apps.research_swarm.scheduler --list
```

Schedule options: `hourly`, `daily`, `weekly`, `every 30m`, `every 4h`.

## Building Your Own App

```python
from orchestrator import (
    AgentSpec, AgentRegistry, ToolSet, ToolRegistry,
    OrchestratorConfig, OllamaProvider, LLMRouter, build_graph,
)

# 1. Define agents
registry = AgentRegistry()
registry.register(AgentSpec(
    name="my_agent",
    description="Does something useful.",
    capabilities=["search"],
    toolset_names=["my_tools"],
    model_tier="worker_fast",
))

# 2. Register tools
tools = ToolRegistry()
tools.register(ToolSet(name="my_tools", tools=[...]))

# 3. Wire up and run
config = OrchestratorConfig()
provider = OllamaProvider(config)
router = LLMRouter(provider, registry)
app = build_graph(registry, tools, provider, router, config)
```

## Testing

55 tests, no Ollama or network access required:

```bash
PYTHONPATH=. pytest tests/ -v
```

## Project Structure

```
agent-swarm/
├── orchestrator/              # Core framework (generic, reusable)
│   ├── agent.py               # AgentSpec, AgentRegistry
│   ├── graph.py               # Dynamic graph builder
│   ├── router.py              # LLMRouter, RuleRouter
│   ├── state.py               # OrchestratorState
│   ├── models.py              # ModelProvider protocol + OllamaProvider
│   ├── tools.py               # ToolSet, ToolRegistry
│   └── config.py              # OrchestratorConfig
├── apps/
│   └── research_swarm/        # Example application
│       ├── agents.py           # 6 agent specs (market, research, writer, ...)
│       ├── tools.py            # Tool implementations (web, file, market, memory)
│       ├── memory.py           # Namespaced ChromaDB-backed research memory
│       ├── scheduler.py        # Automated sweep scheduler (daemon or one-shot)
│       ├── sweeps.yaml         # Recurring research job definitions
│       ├── main.py             # CLI entry point
│       └── config.yaml         # Model tiers and options
├── tests/
│   └── test_orchestrator.py   # 55 tests covering framework + app
├── ARCHITECTURE.md            # Detailed design documentation
├── README.md
└── requirements.txt
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.
