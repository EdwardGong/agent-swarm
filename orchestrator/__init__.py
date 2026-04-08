"""Agent Orchestration Layer — a generic framework for multi-agent systems."""

from orchestrator.agent import AgentSpec, AgentRegistry
from orchestrator.config import OrchestratorConfig
from orchestrator.graph import build_graph
from orchestrator.models import ModelProvider, OllamaProvider
from orchestrator.router import Router, LLMRouter, RuleRouter
from orchestrator.state import OrchestratorState
from orchestrator.tools import ToolSet, ToolRegistry

__all__ = [
    "AgentSpec",
    "AgentRegistry",
    "OrchestratorConfig",
    "build_graph",
    "ModelProvider",
    "OllamaProvider",
    "Router",
    "LLMRouter",
    "RuleRouter",
    "OrchestratorState",
    "ToolSet",
    "ToolRegistry",
]
