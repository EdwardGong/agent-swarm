"""Agent Orchestration Layer — a generic framework for multi-agent systems."""

from orchestrator.agent import AgentSpec, AgentRegistry
from orchestrator.config import OrchestratorConfig
from orchestrator.graph import build_graph
from orchestrator.mcp import MCPToolRegistry
from orchestrator.models import ModelProvider, OllamaProvider, LlamaCppProvider, VLLMMLXProvider
from orchestrator.router import Router, LLMRouter, RuleRouter
from orchestrator.state import OrchestratorState
from orchestrator.tools import ToolSet, ToolRegistry

__all__ = [
    "AgentSpec",
    "AgentRegistry",
    "OrchestratorConfig",
    "build_graph",
    "MCPToolRegistry",
    "ModelProvider",
    "OllamaProvider",
    "LlamaCppProvider",
    "VLLMMLXProvider",
    "Router",
    "LLMRouter",
    "RuleRouter",
    "OrchestratorState",
    "ToolSet",
    "ToolRegistry",
]
