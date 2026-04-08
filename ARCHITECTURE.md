# Architecture — Agent Orchestration Layer

## Overview

This project provides a **two-layer architecture** for building multi-agent
systems on top of LangGraph:

1. **Orchestration Layer** (`orchestrator/`) — a generic, reusable framework
   with no hardcoded agent names, tools, or prompts.
2. **Application Layer** (`apps/`) — concrete applications that register their
   agents and tools into the framework and run.

The key insight is that **agents are data, not code**.  An agent is described by
an `AgentSpec` dataclass — name, description, capabilities, tool requirements,
model tier, and system prompt.  The framework reads these specs at startup and
dynamically builds the LangGraph workflow, the routing logic, and the
orchestrator's system prompt.

## Component Map

```
                 ┌────────────────────────────────┐
                 │     Application (apps/)         │
                 │                                 │
                 │  AgentSpecs   ToolSets   YAML   │
                 └───────┬──────────┬────────┬─────┘
                         │          │        │
  ┌──────────────────────▼──────────▼────────▼──────────────┐
  │              Orchestration Layer (orchestrator/)         │
  │                                                         │
  │  ┌───────────────┐  ┌────────────┐  ┌───────────────┐  │
  │  │ AgentRegistry │  │ToolRegistry│  │OrchestratorCfg│  │
  │  └───────┬───────┘  └─────┬──────┘  └───────┬───────┘  │
  │          │                │                  │          │
  │  ┌───────▼────────────────▼──────────────────▼───────┐  │
  │  │              Router (LLM or Rule)                 │  │
  │  │  • Auto-generates system prompt from registry     │  │
  │  │  • Picks next agent or declares "done"            │  │
  │  └──────────────────────┬────────────────────────────┘  │
  │                         │                               │
  │  ┌──────────────────────▼────────────────────────────┐  │
  │  │           Dynamic Graph Builder                   │  │
  │  │  • Orchestrator node → conditional edge → workers │  │
  │  │  • Worker nodes auto-created from AgentSpecs      │  │
  │  │  • All workers loop back to orchestrator          │  │
  │  └──────────────────────┬────────────────────────────┘  │
  │                         │                               │
  │  ┌──────────────────────▼────────────────────────────┐  │
  │  │           ModelProvider (protocol)                 │  │
  │  │  • OllamaProvider (ships by default)              │  │
  │  │  • Swap in OpenAI, Anthropic, vLLM, etc.          │  │
  │  └───────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
```

## Core Components

### `AgentSpec` / `AgentRegistry` (`orchestrator/agent.py`)

An `AgentSpec` is a pure-data declaration of an agent:

```python
AgentSpec(
    name="research_agent",
    description="Deep research with multi-step web search.",
    capabilities=["research", "investigate"],
    toolset_names=["research"],
    model_tier="worker_fast",
    system_prompt="You are a deep research agent...",
)
```

The `AgentRegistry` stores specs at runtime and provides:
- `find_by_capability(tag)` — discover agents by what they can do.
- `describe_for_prompt()` — auto-generates the "Available specialists" block
  that the LLM orchestrator sees, so routing adapts automatically when agents
  are added or removed.

### `ToolSet` / `ToolRegistry` (`orchestrator/tools.py`)

Tools are grouped into named, tagged sets.  Agents reference tool-sets by name
in their `toolset_names` field.  The framework collects the actual tool objects
at graph-build time.

```python
ToolSet(name="search", tools=[web_search, fetch_url], tags={"web"})
```

### `Router` (`orchestrator/router.py`)

A `Router` is a protocol with one method: `route(state) -> dict`.  Two
implementations ship by default:

- **`LLMRouter`** — sends the conversation state plus auto-generated agent
  descriptions to an LLM, which returns a JSON decision
  (`next_agent` / `sub_task` / `final_answer`).
- **`RuleRouter`** — matches keywords in the user message against agent
  capability tags.  Useful for testing or deterministic pipelines.

### `ModelProvider` (`orchestrator/models.py`)

A protocol that decouples the framework from any specific LLM backend:

```python
class ModelProvider(Protocol):
    def get_llm(self, tier: str, **kwargs) -> BaseChatModel: ...
```

`OllamaProvider` is the default, reading model names from
`OrchestratorConfig.model_tiers`.  To use a different backend, implement the
protocol and pass it to `build_graph()`.

### `OrchestratorConfig` (`orchestrator/config.py`)

Centralized configuration — can be constructed in code or loaded from YAML:

```yaml
routing_strategy: llm
max_iterations: 10
model_tiers:
  orchestrator: "qwen2.5:32b"
  worker_fast: "qwen2.5:7b-instruct-q8_0"
```

### `build_graph()` (`orchestrator/graph.py`)

The graph builder takes the four registries/providers and assembles a
LangGraph `StateGraph`:

1. Adds an **orchestrator node** that calls the router.
2. For each registered agent, creates a **worker node** (via
   `create_react_agent`) with the agent's declared tools and model tier.
3. Wires conditional edges: orchestrator → chosen worker → back to
   orchestrator.
4. Enforces `max_iterations` to prevent infinite loops.

No agent names are hardcoded — everything is driven by the registry contents.

### `OrchestratorState` (`orchestrator/state.py`)

Shared state flowing through the graph:

| Field        | Purpose                                      |
|-------------|----------------------------------------------|
| `messages`  | Chat history (LangGraph `add_messages` reducer) |
| `next_agent`| Which worker the orchestrator chose           |
| `sub_task`  | Task description for the chosen worker        |
| `results`   | Accumulated worker outputs (append reducer)   |
| `done`      | Whether the orchestrator considers task complete |
| `extra`     | App-defined dict for custom state             |

## Data Flow

```
User query
    │
    ▼
[Orchestrator Node]  ◄──────────────────────┐
    │                                        │
    │  Router.route(state) → decision        │
    │                                        │
    ├── next_agent == "done" ──► END         │
    │                                        │
    ├── next_agent == "agent_x"              │
    │       │                                │
    │       ▼                                │
    │  [Worker Node: agent_x]                │
    │    • Looks up AgentSpec                 │
    │    • Gets LLM via ModelProvider         │
    │    • Collects tools from ToolRegistry   │
    │    • Runs ReAct agent on sub_task       │
    │    • Appends result to state.results    │
    │       │                                │
    │       └────────────────────────────────┘
    │
    └── max_iterations exceeded ──► END
```

## Building a New Application

1. **Define agents** — create `AgentSpec` instances describing your agents.
2. **Define tools** — implement tool functions, group them into `ToolSet`s,
   register them in a `ToolRegistry`.
3. **Write a config** — either a YAML file or an `OrchestratorConfig` in code.
4. **Wire up** — create registries, choose a router, call `build_graph()`.

See `apps/research_swarm/` for a complete working example.

## Testing

Tests live in `tests/` and run without Ollama or network access:

```bash
PYTHONPATH=. pytest tests/ -v
```

LLM calls are mocked via `FakeProvider` and `unittest.mock.patch`.  The test
suite validates:

- All registry operations (register, lookup, capability search, prompt gen)
- Both routers (LLM delegation/done/fallback, Rule keyword matching)
- Config loading from YAML and programmatic construction
- Dynamic graph compilation and execution
- Research swarm integration (agent→toolset→config cross-references)
