"""Integration tests for the Consolidated Agent Swarm (M5 Max 128GB).

Covers:
  - VLLMMLXProvider — instantiation, get_llm returns ChatOpenAI, bad-tier error
  - MCPToolRegistry — graceful fallback when vllm-mlx is unreachable,
                      schema-to-pydantic helper, _extract_text coercion,
                      tool wrapping shape
  - Consolidated swarm — config loads from YAML, register_all produces 4
                         agents, graph compiles with FakeProvider + empty tools
  - Workers — all four modules are importable and expose a ``run`` callable
  - CRM worker — ensure_schema creates the contacts table in a temp DB

All tests run without a live vllm-mlx server or network access.
Run with:  PYTHONPATH=. pytest tests/ -v
"""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agent import AgentRegistry
from orchestrator.config import OrchestratorConfig
from orchestrator.graph import build_graph
from orchestrator.mcp import MCPToolRegistry, _json_schema_to_pydantic, _extract_text
from orchestrator.models import VLLMMLXProvider
from orchestrator.router import LLMRouter
from orchestrator.tools import ToolSet, ToolRegistry


CONFIG_PATH = Path(__file__).parent.parent / "apps" / "consolidated_swarm" / "config.yaml"

# Mark for tests that require langchain_openai (may not be installed in lightweight envs)
_needs_lc_openai = pytest.mark.skipif(
    importlib.util.find_spec("langchain_openai") is None,
    reason="langchain_openai not installed",
)


# ---------------------------------------------------------------------------
# Shared FakeProvider (mirrors the one in test_orchestrator.py)
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, response_content: str = ""):
        self.response_content = response_content

    def get_llm(self, tier: str, **kwargs: Any):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.response_content
        mock_llm.invoke.return_value = mock_response
        return mock_llm


# ===================================================================
# VLLMMLXProvider
# ===================================================================


class TestVLLMMLXProvider:
    def _make_config(self, extra: dict | None = None) -> OrchestratorConfig:
        return OrchestratorConfig(
            model_tiers={
                "orchestrator": "qwen3.6-35b-opus-abl-mxfp4",
                "worker": "qwen3.6-35b-opus-abl-mxfp4",
            },
            extra=extra or {"vllm_mlx_base_url": "http://localhost:8000/v1"},
        )

    def test_instantiation(self):
        config = self._make_config()
        provider = VLLMMLXProvider(config)
        assert provider._base_url == "http://localhost:8000/v1"

    def test_default_base_url(self):
        config = OrchestratorConfig(
            model_tiers={"orchestrator": "qwen3.6-35b-opus-abl-mxfp4"}
        )
        provider = VLLMMLXProvider(config)
        assert provider._base_url == "http://localhost:8000/v1"

    def test_custom_base_url(self):
        config = self._make_config(
            extra={"vllm_mlx_base_url": "http://myhost:9000/v1"}
        )
        provider = VLLMMLXProvider(config)
        assert provider._base_url == "http://myhost:9000/v1"

    @_needs_lc_openai
    def test_get_llm_returns_chat_openai(self):
        config = self._make_config()
        provider = VLLMMLXProvider(config)
        from langchain_openai import ChatOpenAI
        with patch("langchain_openai.ChatOpenAI.__init__", return_value=None):
            llm = provider.get_llm("orchestrator")
            assert isinstance(llm, ChatOpenAI)

    def test_get_llm_unknown_tier_raises(self):
        """Tier validation happens before the langchain_openai import — no dependency."""
        config = self._make_config()
        provider = VLLMMLXProvider(config)
        with pytest.raises(ValueError, match="Unknown model tier"):
            provider.get_llm("nonexistent_tier")

    @_needs_lc_openai
    def test_get_llm_passes_model_name(self):
        config = self._make_config()
        provider = VLLMMLXProvider(config)
        captured: dict[str, Any] = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        with patch("langchain_openai.ChatOpenAI.__init__", fake_init):
            provider.get_llm("orchestrator")

        assert captured.get("model") == "qwen3.6-35b-opus-abl-mxfp4"
        assert captured.get("base_url") == "http://localhost:8000/v1"
        assert captured.get("temperature") == 0.0

    @_needs_lc_openai
    def test_max_tokens_forwarded_when_set(self):
        config = OrchestratorConfig(
            model_tiers={"worker": "qwen3.6-35b-opus-abl-mxfp4"},
            model_options={"temperature": 0.0, "max_tokens": 4096},
        )
        provider = VLLMMLXProvider(config)
        captured: dict[str, Any] = {}

        def fake_init(self, **kwargs):
            captured.update(kwargs)

        with patch("langchain_openai.ChatOpenAI.__init__", fake_init):
            provider.get_llm("worker")

        assert captured.get("max_tokens") == 4096


# ===================================================================
# MCPToolRegistry helpers
# ===================================================================


class TestMCPHelpers:
    def test_json_schema_to_pydantic_basic(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results"},
            },
            "required": ["query"],
        }
        model = _json_schema_to_pydantic("test_tool", schema)
        # required field has no default
        instance = model(query="hello", k=5)
        assert instance.query == "hello"  # type: ignore[attr-defined]
        assert instance.k == 5  # type: ignore[attr-defined]

    def test_json_schema_optional_field_defaults_none(self):
        schema = {
            "properties": {"limit": {"type": "integer"}},
            "required": [],
        }
        model = _json_schema_to_pydantic("t", schema)
        instance = model()
        assert instance.limit is None  # type: ignore[attr-defined]

    def test_json_schema_empty_adds_input_field(self):
        model = _json_schema_to_pydantic("empty_tool", {})
        instance = model()
        assert hasattr(instance, "input")

    def test_json_schema_name_sanitisation(self):
        """Tool names with dots or dashes must produce valid model class names."""
        model = _json_schema_to_pydantic("lmcp.send-email", {})
        assert "Args" in model.__name__

    def test_extract_text_mcp_content_list(self):
        result = {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}
        assert _extract_text(result) == "hello\nworld"

    def test_extract_text_no_text_items_falls_back_to_json(self):
        result = {"content": [{"type": "image", "data": "..."}]}
        text = _extract_text(result)
        assert "image" in text

    def test_extract_text_plain_string(self):
        assert _extract_text("plain") == "plain"

    def test_extract_text_empty_content_list(self):
        result = {"content": []}
        text = _extract_text(result)
        assert isinstance(text, str)


# ===================================================================
# MCPToolRegistry — connection behaviour
# ===================================================================


class TestMCPToolRegistry:
    def test_graceful_fallback_when_unreachable(self):
        """Registry must return [] (not raise) when vllm-mlx is not running."""
        registry = MCPToolRegistry("http://localhost:19999")  # no server here
        tools = registry.load_tools()
        assert tools == []

    def test_graceful_fallback_with_profile(self):
        registry = MCPToolRegistry("http://localhost:19999")
        tools = registry.load_tools(profile="research")
        assert tools == []

    def test_cache_is_populated_after_first_call(self):
        registry = MCPToolRegistry("http://localhost:19999")
        registry.load_tools()
        assert "__all__" in registry._cache

    def test_clear_cache(self):
        registry = MCPToolRegistry("http://localhost:19999")
        registry.load_tools()
        registry.clear_cache()
        assert registry._cache == {}

    def test_wrap_produces_structured_tool(self):
        """_wrap should produce a StructuredTool regardless of schema complexity."""
        registry = MCPToolRegistry("http://localhost:19999")
        tool_def = {
            "name": "filesystem_read_file",
            "description": "Read a file.",
            "inputSchema": {
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        }
        st = registry._wrap(tool_def)
        assert st.name == "filesystem_read_file"
        assert "Read a file" in st.description

    def test_wrap_tool_error_returns_string(self):
        """Tool execution errors must be returned as strings, not raised."""
        registry = MCPToolRegistry("http://localhost:19999")
        tool_def = {
            "name": "broken_tool",
            "description": "Breaks.",
            "inputSchema": {},
        }
        st = registry._wrap(tool_def)
        result = st.run({})
        assert isinstance(result, str)
        assert "error" in result.lower() or "timed out" in result.lower()

    def test_load_tools_with_mocked_server(self):
        """Full happy path: mock /v1/mcp/tools and verify tool list is built."""
        import requests

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tools": [
                {
                    "name": "faiss_query_rag_store",
                    "description": "Query the FAISS index.",
                    "inputSchema": {
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
                {
                    "name": "safari_navigate_and_read",
                    "description": "Browse a URL.",
                    "inputSchema": {
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            registry = MCPToolRegistry("http://localhost:8000")
            tools = registry.load_tools(profile="research")

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "faiss_query_rag_store" in names
        assert "safari_navigate_and_read" in names


# ===================================================================
# Consolidated swarm — config, agents, graph
# ===================================================================


class TestConsolidatedSwarmConfig:
    def test_config_loads_from_yaml(self):
        assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"
        config = OrchestratorConfig.from_yaml(CONFIG_PATH)
        assert config.model_provider == "vllm_mlx"
        assert config.routing_strategy == "llm"
        assert config.max_iterations == 20

    def test_config_has_correct_model_tiers(self):
        config = OrchestratorConfig.from_yaml(CONFIG_PATH)
        for tier in ("orchestrator", "worker", "worker_fast", "coder", "reasoner"):
            assert tier in config.model_tiers, f"Missing tier: {tier}"
            assert config.model_tiers[tier] == "qwen3.6-35b-opus-abl-mxfp4"

    def test_config_extra_has_mcp_base_url(self):
        config = OrchestratorConfig.from_yaml(CONFIG_PATH)
        assert "mcp_base_url" in config.extra
        assert config.extra["mcp_base_url"] == "http://localhost:8000"


class TestConsolidatedSwarmAgents:
    def test_register_all_produces_four_agents(self):
        from apps.consolidated_swarm.agents import register_all, MCP_PROFILE

        registry = AgentRegistry()
        register_all(registry)
        assert len(registry.all) == 4

    def test_all_expected_agents_registered(self):
        from apps.consolidated_swarm.agents import register_all

        registry = AgentRegistry()
        register_all(registry)
        names = set(registry.names)
        assert names == {"research_agent", "code_agent", "crm_agent", "content_agent"}

    def test_mcp_profile_covers_all_agents(self):
        from apps.consolidated_swarm.agents import register_all, MCP_PROFILE

        registry = AgentRegistry()
        register_all(registry)
        for name in registry.names:
            assert name in MCP_PROFILE, f"{name} missing from MCP_PROFILE"

    def test_agent_capabilities_non_empty(self):
        from apps.consolidated_swarm.agents import AGENTS

        for spec in AGENTS:
            assert spec.capabilities, f"{spec.name} has no capabilities"

    def test_agent_system_prompts_non_empty(self):
        from apps.consolidated_swarm.agents import AGENTS

        for spec in AGENTS:
            assert spec.system_prompt, f"{spec.name} has no system_prompt"

    def test_agent_toolset_names_reference_mcp_profile(self):
        from apps.consolidated_swarm.agents import AGENTS, MCP_PROFILE

        for spec in AGENTS:
            profile = MCP_PROFILE.get(spec.name, "")
            expected_toolset = f"mcp_{profile}"
            assert expected_toolset in spec.toolset_names, (
                f"{spec.name}: expected toolset '{expected_toolset}' in {spec.toolset_names}"
            )


class TestConsolidatedSwarmGraph:
    def _build(self) -> Any:
        from apps.consolidated_swarm.agents import register_all, MCP_PROFILE

        config = OrchestratorConfig.from_yaml(CONFIG_PATH)
        provider = FakeProvider(
            response_content='{"thinking":"","next_agent":"done","sub_task":"","final_answer":"ok"}'
        )

        agent_registry = AgentRegistry()
        register_all(agent_registry)

        # Empty tool-sets: MCPToolRegistry is offline in tests
        tool_registry = ToolRegistry()
        for _, profile in MCP_PROFILE.items():
            tool_registry.register(ToolSet(
                name=f"mcp_{profile}", tools=[], tags={profile}, description=""
            ))

        router = LLMRouter(provider, agent_registry)
        return build_graph(agent_registry, tool_registry, provider, router, config)

    def test_graph_compiles_without_error(self):
        app = self._build()
        assert app is not None

    def test_graph_runs_single_turn(self):
        from langchain_core.messages import HumanMessage
        app = self._build()
        events = list(app.stream(
            {"messages": [HumanMessage(content="ping")]},
            config={"recursion_limit": 10},
        ))
        assert len(events) >= 1


# ===================================================================
# Workers — importability + standalone smoke tests
# ===================================================================


class TestWorkerImports:
    def test_research_worker_importable(self):
        from apps.consolidated_swarm.workers import research
        assert callable(research.run)

    def test_coder_worker_importable(self):
        from apps.consolidated_swarm.workers import coder
        assert callable(coder.run)

    def test_crm_worker_importable(self):
        from apps.consolidated_swarm.workers import crm
        assert callable(crm.run)

    def test_content_worker_importable(self):
        from apps.consolidated_swarm.workers import content
        assert callable(content.run)

    def test_content_generate_pptx_script(self):
        from apps.consolidated_swarm.workers.content import generate_pptx_script
        slides = [{"title": "Intro", "bullets": ["Point 1", "Point 2"]}]
        script = generate_pptx_script("Test Deck", slides)
        assert "from pptx import Presentation" in script
        assert "Intro" in script
        assert "Point 1" in script


class TestCRMWorkerSchema:
    def test_ensure_schema_creates_table(self):
        """ensure_schema must be idempotent and create the contacts table."""
        from apps.consolidated_swarm.workers.crm import ensure_schema, _CONTACTS_SCHEMA

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            ensure_schema(db_path)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='contacts'"
                ).fetchall()
            assert len(rows) == 1

    def test_ensure_schema_idempotent(self):
        """Calling ensure_schema twice must not raise."""
        from apps.consolidated_swarm.workers.crm import ensure_schema

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            ensure_schema(db_path)
            ensure_schema(db_path)  # second call — must not raise

    def test_contacts_table_has_expected_columns(self):
        from apps.consolidated_swarm.workers.crm import ensure_schema

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            ensure_schema(db_path)

            with sqlite3.connect(db_path) as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(contacts)")}

        expected = {"name", "email", "phone", "company", "title",
                    "linkedin_url", "source", "last_contacted", "notes"}
        assert expected.issubset(cols)

    def test_query_crm_sql_fallback_without_mcp(self):
        """query_crm with no MCP tools must return a 'not available' message."""
        from apps.consolidated_swarm.workers.crm import query_crm
        from unittest.mock import MagicMock

        registry = MagicMock()
        registry.load_tools.return_value = []  # no tools

        result = query_crm("SELECT 1", registry)
        assert "not available" in result.lower()
