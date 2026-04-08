"""Tests for the Agent Orchestration Layer and the Research Swarm app.

All tests run without Ollama or network access — LLM calls are mocked.
Run with:  PYTHONPATH=. pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.agent import AgentSpec, AgentRegistry
from orchestrator.config import OrchestratorConfig
from orchestrator.graph import build_graph
from orchestrator.models import ModelProvider, OllamaProvider
from orchestrator.router import LLMRouter, RuleRouter
from orchestrator.state import OrchestratorState
from orchestrator.tools import ToolSet, ToolRegistry


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sample_spec():
    return AgentSpec(
        name="test_agent",
        description="A test agent.",
        capabilities=["testing", "demo"],
        toolset_names=["test_tools"],
        model_tier="worker_fast",
        system_prompt="You are a test agent.",
    )


@pytest.fixture
def agent_registry(sample_spec):
    reg = AgentRegistry()
    reg.register(sample_spec)
    return reg


@pytest.fixture
def tool_registry():
    reg = ToolRegistry()
    reg.register(ToolSet(
        name="test_tools",
        tools=["fake_tool_a", "fake_tool_b"],
        tags={"testing", "demo"},
        description="Tools for testing.",
    ))
    reg.register(ToolSet(
        name="other_tools",
        tools=["fake_tool_c"],
        tags={"other"},
        description="Other tools.",
    ))
    return reg


@pytest.fixture
def config():
    return OrchestratorConfig(
        max_iterations=5,
        model_tiers={"orchestrator": "test-model", "worker_fast": "test-model"},
    )


class FakeProvider:
    """A ModelProvider that returns a mock LLM with a scripted response."""

    def __init__(self, response_content: str = ""):
        self.response_content = response_content
        self.last_tier = None

    def get_llm(self, tier: str, **kwargs: Any):
        self.last_tier = tier
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.response_content
        mock_llm.invoke.return_value = mock_response
        return mock_llm


# ===================================================================
# AgentSpec + AgentRegistry
# ===================================================================


class TestAgentSpec:
    def test_defaults(self):
        spec = AgentSpec(name="a", description="d")
        assert spec.model_tier == "worker"
        assert spec.capabilities == []
        assert spec.toolset_names == []
        assert spec.system_prompt is None

    def test_custom_fields(self, sample_spec):
        assert sample_spec.name == "test_agent"
        assert sample_spec.model_tier == "worker_fast"
        assert "testing" in sample_spec.capabilities
        assert sample_spec.system_prompt == "You are a test agent."


class TestAgentRegistry:
    def test_register_and_get(self, agent_registry, sample_spec):
        assert agent_registry.get("test_agent") is sample_spec

    def test_get_missing_raises(self, agent_registry):
        with pytest.raises(KeyError, match="not registered"):
            agent_registry.get("nonexistent")

    def test_names(self, agent_registry):
        assert agent_registry.names == ["test_agent"]

    def test_all(self, agent_registry, sample_spec):
        assert agent_registry.all == [sample_spec]

    def test_register_many(self):
        reg = AgentRegistry()
        specs = [
            AgentSpec(name="a", description="A"),
            AgentSpec(name="b", description="B"),
        ]
        reg.register_many(specs)
        assert reg.names == ["a", "b"]

    def test_find_by_capability(self, agent_registry):
        found = agent_registry.find_by_capability("testing")
        assert len(found) == 1
        assert found[0].name == "test_agent"

        assert agent_registry.find_by_capability("nonexistent") == []

    def test_describe_for_prompt(self, agent_registry):
        desc = agent_registry.describe_for_prompt()
        assert '"test_agent"' in desc
        assert "A test agent." in desc
        assert "testing" in desc

    def test_overwrite_on_re_register(self):
        reg = AgentRegistry()
        reg.register(AgentSpec(name="a", description="v1"))
        reg.register(AgentSpec(name="a", description="v2"))
        assert reg.get("a").description == "v2"
        assert len(reg.all) == 1


# ===================================================================
# ToolSet + ToolRegistry
# ===================================================================


class TestToolRegistry:
    def test_register_and_get(self, tool_registry):
        ts = tool_registry.get("test_tools")
        assert ts.name == "test_tools"
        assert len(ts.tools) == 2

    def test_get_missing_raises(self, tool_registry):
        with pytest.raises(KeyError, match="not registered"):
            tool_registry.get("nope")

    def test_get_tools(self, tool_registry):
        tools = tool_registry.get_tools("test_tools")
        assert tools == ["fake_tool_a", "fake_tool_b"]

    def test_find_by_tag(self, tool_registry):
        found = tool_registry.find_by_tag("testing")
        assert len(found) == 1
        assert found[0].name == "test_tools"

        assert tool_registry.find_by_tag("nonexistent") == []

    def test_all_tools_for_tags(self, tool_registry):
        tools = tool_registry.all_tools_for_tags(["testing", "other"])
        assert "fake_tool_a" in tools
        assert "fake_tool_b" in tools
        assert "fake_tool_c" in tools

    def test_all_tools_for_tags_deduplicates(self, tool_registry):
        # "testing" and "demo" both match "test_tools" — should not duplicate
        tools = tool_registry.all_tools_for_tags(["testing", "demo"])
        assert tools == ["fake_tool_a", "fake_tool_b"]

    def test_names(self, tool_registry):
        assert set(tool_registry.names) == {"test_tools", "other_tools"}


# ===================================================================
# OrchestratorConfig
# ===================================================================


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.routing_strategy == "llm"
        assert cfg.max_iterations == 10
        assert "orchestrator" in cfg.model_tiers

    def test_custom_values(self, config):
        assert config.max_iterations == 5
        assert config.model_tiers["worker_fast"] == "test-model"

    def test_from_yaml(self):
        yaml_content = (
            "routing_strategy: rule\n"
            "max_iterations: 3\n"
            "model_tiers:\n"
            "  orchestrator: tiny-model\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = OrchestratorConfig.from_yaml(f.name)

        assert cfg.routing_strategy == "rule"
        assert cfg.max_iterations == 3
        assert cfg.model_tiers == {"orchestrator": "tiny-model"}
        Path(f.name).unlink()

    def test_from_yaml_ignores_unknown_keys(self):
        yaml_content = "max_iterations: 7\nunknown_field: should_be_ignored\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = OrchestratorConfig.from_yaml(f.name)

        assert cfg.max_iterations == 7
        Path(f.name).unlink()

    def test_research_swarm_config_loads(self):
        cfg_path = Path(__file__).parent.parent / "apps" / "research_swarm" / "config.yaml"
        if cfg_path.exists():
            cfg = OrchestratorConfig.from_yaml(cfg_path)
            assert cfg.max_iterations == 10
            assert cfg.model_tiers["orchestrator"] == "qwen2.5:32b"


# ===================================================================
# RuleRouter
# ===================================================================


class TestRuleRouter:
    def _make_registry(self):
        reg = AgentRegistry()
        reg.register(AgentSpec(
            name="search_agent", description="Search.", capabilities=["search", "web"],
        ))
        reg.register(AgentSpec(
            name="code_agent", description="Code.", capabilities=["code", "dev"],
        ))
        return reg

    def test_routes_to_matching_agent(self):
        reg = self._make_registry()
        router = RuleRouter(reg)
        state = OrchestratorState(messages=[HumanMessage(content="search the web for AI news")])
        result = router.route(state)
        assert result["next_agent"] == "search_agent"
        assert result["done"] is False

    def test_routes_code_keyword(self):
        reg = self._make_registry()
        router = RuleRouter(reg)
        state = OrchestratorState(messages=[HumanMessage(content="write some code")])
        result = router.route(state)
        assert result["next_agent"] == "code_agent"

    def test_no_match_returns_done(self):
        reg = self._make_registry()
        router = RuleRouter(reg)
        state = OrchestratorState(messages=[HumanMessage(content="hello there")])
        result = router.route(state)
        assert result["next_agent"] == "done"
        assert result["done"] is True

    def test_case_insensitive(self):
        reg = self._make_registry()
        router = RuleRouter(reg)
        state = OrchestratorState(messages=[HumanMessage(content="SEARCH for something")])
        result = router.route(state)
        assert result["next_agent"] == "search_agent"


# ===================================================================
# LLMRouter
# ===================================================================


class TestLLMRouter:
    def _make_router(self, llm_response: str):
        reg = AgentRegistry()
        reg.register(AgentSpec(name="agent_a", description="A."))
        provider = FakeProvider(llm_response)
        return LLMRouter(provider, reg), provider

    def test_delegates_to_agent(self):
        response = json.dumps({
            "thinking": "need agent_a",
            "next_agent": "agent_a",
            "sub_task": "do something",
        })
        router, provider = self._make_router(response)
        state = OrchestratorState(messages=[HumanMessage(content="test")])
        result = router.route(state)

        assert result["next_agent"] == "agent_a"
        assert result["sub_task"] == "do something"
        assert result["done"] is False
        assert provider.last_tier == "orchestrator"

    def test_done_with_final_answer(self):
        response = json.dumps({
            "thinking": "all done",
            "next_agent": "done",
            "final_answer": "Here is the answer.",
        })
        router, _ = self._make_router(response)
        state = OrchestratorState(messages=[HumanMessage(content="test")])
        result = router.route(state)

        assert result["done"] is True
        assert result["messages"][0].content == "Here is the answer."

    def test_handles_code_fenced_json(self):
        response = "```json\n" + json.dumps({
            "thinking": "ok",
            "next_agent": "done",
            "final_answer": "fenced",
        }) + "\n```"
        router, _ = self._make_router(response)
        state = OrchestratorState(messages=[HumanMessage(content="test")])
        result = router.route(state)

        assert result["done"] is True
        assert result["messages"][0].content == "fenced"

    def test_invalid_json_falls_back_to_done(self):
        router, _ = self._make_router("this is not json at all")
        state = OrchestratorState(messages=[HumanMessage(content="test")])
        result = router.route(state)

        assert result["done"] is True
        assert result["messages"][0].content == "this is not json at all"

    def test_includes_previous_results(self):
        response = json.dumps({"next_agent": "done", "final_answer": "ok"})
        router, _ = self._make_router(response)
        state = OrchestratorState(
            messages=[HumanMessage(content="test")],
            results=["[agent_a] found something"],
        )
        result = router.route(state)
        assert result["done"] is True

    def test_system_prompt_contains_agent_descriptions(self):
        reg = AgentRegistry()
        reg.register(AgentSpec(name="special", description="Very special.", capabilities=["magic"]))
        provider = FakeProvider(json.dumps({"next_agent": "done", "final_answer": "x"}))
        router = LLMRouter(provider, reg)

        prompt = router._build_system_prompt()
        assert '"special"' in prompt
        assert "Very special." in prompt
        assert "magic" in prompt


# ===================================================================
# OrchestratorState
# ===================================================================


class TestOrchestratorState:
    def test_defaults(self):
        state = OrchestratorState()
        assert state.messages == []
        assert state.next_agent == ""
        assert state.sub_task == ""
        assert state.results == []
        assert state.done is False
        assert state.extra == {}

    def test_extra_state(self):
        state = OrchestratorState(extra={"app_key": "app_value"})
        assert state.extra["app_key"] == "app_value"


# ===================================================================
# ModelProvider protocol
# ===================================================================


class TestModelProvider:
    def test_fake_provider_satisfies_protocol(self):
        provider = FakeProvider("test")
        assert isinstance(provider, ModelProvider)

    def test_ollama_provider_unknown_tier(self):
        cfg = OrchestratorConfig(model_tiers={"known": "model"})
        provider = OllamaProvider(cfg)
        with pytest.raises(ValueError, match="Unknown model tier"):
            provider.get_llm("unknown_tier")


# ===================================================================
# build_graph (dynamic graph builder)
# ===================================================================


class TestBuildGraph:
    def test_graph_compiles_with_registered_agents(self, config):
        """Graph should compile and contain orchestrator + agent nodes."""
        agent_reg = AgentRegistry()
        agent_reg.register(AgentSpec(
            name="echo_agent", description="Echo.", toolset_names=["echo_tools"],
            model_tier="worker_fast",
        ))

        tool_reg = ToolRegistry()
        tool_reg.register(ToolSet(name="echo_tools", tools=[]))

        provider = FakeProvider()

        # Use a simple router that always returns done
        class DoneRouter:
            def route(self, state):
                return {
                    "next_agent": "done",
                    "sub_task": "",
                    "done": True,
                    "messages": [AIMessage(content="immediate done")],
                }

        graph = build_graph(agent_reg, tool_reg, provider, DoneRouter(), config)
        assert graph is not None

        # Run it — should go orchestrator -> done immediately
        result = graph.invoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"recursion_limit": 10},
        )
        assert result["done"] is True

    def test_max_iterations_stops_loop(self):
        """Graph should stop after max_iterations even if router never says done."""
        agent_reg = AgentRegistry()
        agent_reg.register(AgentSpec(
            name="loop_agent", description="Loop.", toolset_names=["noop_tools"],
            model_tier="worker_fast",
        ))

        tool_reg = ToolRegistry()
        tool_reg.register(ToolSet(name="noop_tools", tools=[]))

        cfg = OrchestratorConfig(
            max_iterations=2,
            model_tiers={"worker_fast": "test"},
        )

        # Router always delegates to loop_agent — never finishes
        class LoopRouter:
            def route(self, state):
                return {
                    "next_agent": "loop_agent",
                    "sub_task": "keep going",
                    "done": False,
                }

        # Mock create_react_agent so worker nodes don't need a real LLM
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="looped")]
        }

        with patch("orchestrator.graph.create_react_agent", return_value=mock_agent):
            provider = FakeProvider()
            graph = build_graph(agent_reg, tool_reg, provider, LoopRouter(), cfg)

            result = graph.invoke(
                {"messages": [HumanMessage(content="loop test")]},
                config={"recursion_limit": 50},
            )

        # Should have accumulated results from worker runs before stopping
        assert len(result["results"]) >= 1
        assert any("loop_agent" in r for r in result["results"])


# ===================================================================
# Research Swarm app registration
# ===================================================================


class TestResearchSwarmRegistration:
    def test_agents_register(self):
        from apps.research_swarm.agents import register_all as register_agents
        reg = AgentRegistry()
        register_agents(reg)

        expected = {"search_agent", "file_agent", "code_agent",
                    "research_agent", "writer_agent"}
        assert set(reg.names) == expected

    def test_agent_specs_have_required_fields(self):
        from apps.research_swarm.agents import register_all as register_agents
        reg = AgentRegistry()
        register_agents(reg)

        for name in reg.names:
            spec = reg.get(name)
            assert spec.description, f"{name} missing description"
            assert spec.toolset_names, f"{name} missing toolset_names"
            assert spec.model_tier, f"{name} missing model_tier"

    def test_tools_register(self):
        from apps.research_swarm.tools import register_all as register_tools
        reg = ToolRegistry()
        register_tools(reg)

        expected = {"search", "files", "shell", "code", "research", "writer"}
        assert set(reg.names) == expected

    def test_toolsets_have_actual_tools(self):
        from apps.research_swarm.tools import register_all as register_tools
        reg = ToolRegistry()
        register_tools(reg)

        for name in reg.names:
            ts = reg.get(name)
            assert len(ts.tools) > 0, f"ToolSet '{name}' has no tools"

    def test_agent_toolsets_exist_in_tool_registry(self):
        """Every toolset_name referenced by an agent must be registered."""
        from apps.research_swarm.agents import register_all as register_agents
        from apps.research_swarm.tools import register_all as register_tools

        agent_reg = AgentRegistry()
        register_agents(agent_reg)
        tool_reg = ToolRegistry()
        register_tools(tool_reg)

        for spec in agent_reg.all:
            for ts_name in spec.toolset_names:
                assert ts_name in tool_reg.names, (
                    f"Agent '{spec.name}' references toolset '{ts_name}' "
                    f"which is not registered"
                )

    def test_agent_model_tiers_exist_in_config(self):
        """Every model_tier referenced by an agent must be in the config."""
        from apps.research_swarm.agents import register_all as register_agents

        agent_reg = AgentRegistry()
        register_agents(agent_reg)

        cfg_path = Path(__file__).parent.parent / "apps" / "research_swarm" / "config.yaml"
        cfg = OrchestratorConfig.from_yaml(cfg_path)

        for spec in agent_reg.all:
            assert spec.model_tier in cfg.model_tiers, (
                f"Agent '{spec.name}' uses tier '{spec.model_tier}' "
                f"not found in config"
            )

    def test_full_graph_compiles(self):
        """The research swarm graph should compile without errors."""
        from apps.research_swarm.agents import register_all as register_agents
        from apps.research_swarm.tools import register_all as register_tools

        cfg_path = Path(__file__).parent.parent / "apps" / "research_swarm" / "config.yaml"
        cfg = OrchestratorConfig.from_yaml(cfg_path)

        agent_reg = AgentRegistry()
        register_agents(agent_reg)
        tool_reg = ToolRegistry()
        register_tools(tool_reg)

        provider = FakeProvider()
        router = RuleRouter(agent_reg)

        graph = build_graph(agent_reg, tool_reg, provider, router, cfg)
        assert graph is not None
