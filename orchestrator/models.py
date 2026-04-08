"""Provider-agnostic model layer.

Defines a ``ModelProvider`` protocol so the framework is not coupled to
any single inference backend.  Ships with ``OllamaProvider`` as the
default implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from orchestrator.config import OrchestratorConfig


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ModelProvider(Protocol):
    """Interface that any model backend must implement."""

    def get_llm(self, tier: str, **kwargs: Any) -> BaseChatModel:
        """Return a chat model for the given tier (e.g. 'orchestrator', 'worker')."""
        ...


# ---------------------------------------------------------------------------
# Ollama implementation
# ---------------------------------------------------------------------------

class OllamaProvider:
    """Model provider backed by a local Ollama server."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._tiers = config.model_tiers
        self._defaults = config.model_options

    def get_llm(self, tier: str, **kwargs: Any) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        model = self._tiers.get(tier)
        if model is None:
            raise ValueError(
                f"Unknown model tier '{tier}'. Available: {list(self._tiers)}"
            )

        params: dict[str, Any] = {
            "model": model,
            "temperature": self._defaults.get("temperature", 0.0),
            "keep_alive": self._defaults.get("keep_alive", "5m"),
        }
        params.update(kwargs)
        return ChatOllama(**params)
