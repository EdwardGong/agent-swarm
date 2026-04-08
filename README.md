# Agent Orchestration Layer

A generic framework for building multi-agent systems on LangGraph.
Agents, tools, models, and routing strategies are all pluggable —
define your agents as data and the framework wires up the graph.

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
- **build_graph()** — assembles a LangGraph workflow dynamically from the registries

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve &

# 3. Run the example research swarm
python -m apps.research_swarm.main "Your task here"

# 4. Or run interactively
python -m apps.research_swarm.main --interactive
```

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

## Project Structure

```
agent-swarm/
├── orchestrator/              # Core framework
│   ├── agent.py               # AgentSpec, AgentRegistry
│   ├── graph.py               # Dynamic graph builder
│   ├── router.py              # LLMRouter, RuleRouter
│   ├── state.py               # OrchestratorState
│   ├── models.py              # ModelProvider protocol + OllamaProvider
│   ├── tools.py               # ToolSet, ToolRegistry
│   └── config.py              # OrchestratorConfig
├── apps/
│   └── research_swarm/        # Example application
│       ├── agents.py           # Agent specs for research swarm
│       ├── tools.py            # Tool implementations + registration
│       ├── memory.py           # ChromaDB-backed research memory
│       ├── main.py             # CLI entry point
│       └── config.yaml         # Model tiers, options
├── README.md
└── requirements.txt
```
