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


# ---------------------------------------------------------------------------
# vllm-mlx implementation (native MCP support via --mcp-config)
# ---------------------------------------------------------------------------

class VLLMMLXProvider:
    """Model provider backed by a vllm-mlx server.

    vllm-mlx exposes an OpenAI-compatible API at /v1 and optionally runs
    MCP servers natively when started with ``--mcp-config mcp.json``.
    Tool discovery and execution are then available via
    ``/v1/mcp/tools`` and ``/v1/mcp/execute``.

    Config example::

        model_provider: vllm_mlx
        vllm_mlx_base_url: http://localhost:8000/v1
        model_tiers:
          orchestrator: "qwen3.6-35b-opus-distilled-abl-mxfp4"
          worker: "qwen3.6-35b-opus-distilled-abl-mxfp4"
          worker_fast: "qwen3.6-35b-opus-distilled-abl-mxfp4"
          coder: "qwen3.6-35b-opus-distilled-abl-mxfp4"
          reasoner: "qwen3.6-35b-opus-distilled-abl-mxfp4"
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._tiers = config.model_tiers
        self._defaults = config.model_options
        self._base_url = config.extra.get(
            "vllm_mlx_base_url", "http://localhost:8000/v1"
        )

    def get_llm(self, tier: str, **kwargs: Any) -> BaseChatModel:
        model = self._tiers.get(tier)
        if model is None:
            raise ValueError(
                f"Unknown model tier '{tier}'. Available: {list(self._tiers)}"
            )

        from langchain_openai import ChatOpenAI

        max_tok = self._defaults.get("max_tokens", 4096)

        params: dict[str, Any] = {
            "model": model,
            "base_url": self._base_url,
            "api_key": "not-needed",  # vllm-mlx ignores auth by default
            "temperature": self._defaults.get("temperature", 0.0),
            "request_timeout": self._defaults.get("request_timeout", 180),
            # langchain-openai >=1.2 sends max_tokens as max_completion_tokens,
            # which vllm-mlx does not recognise.  Pass via extra_body so the
            # raw "max_tokens" field reaches the server.
            "extra_body": {"max_tokens": max_tok},
        }
        params.update(kwargs)
        return ChatOpenAI(**params)


# ---------------------------------------------------------------------------
# llama-server implementation (direct llama.cpp, no Ollama wrapper)
# ---------------------------------------------------------------------------

class LlamaCppProvider:
    """Model provider backed by a llama-server instance.

    Uses the OpenAI-compatible API that llama-server exposes at /v1.
    This bypasses Ollama entirely, giving direct access to llama.cpp
    with full architecture support (e.g. Gated DeltaNet / qwen35).

    Config example::

        model_provider: llamacpp
        llamacpp_base_url: http://127.0.0.1:8080/v1
        model_tiers:
          orchestrator: "local"     # name is arbitrary, llama-server
          worker: "local"           # serves whatever model it was
          worker_fast: "local"      # started with
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._tiers = config.model_tiers
        self._defaults = config.model_options
        self._base_url = config.extra.get(
            "llamacpp_base_url", "http://127.0.0.1:8080/v1"
        )

    def get_llm(self, tier: str, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        model = self._tiers.get(tier)
        if model is None:
            raise ValueError(
                f"Unknown model tier '{tier}'. Available: {list(self._tiers)}"
            )

        params: dict[str, Any] = {
            "model": model,
            "base_url": self._base_url,
            "api_key": "not-needed",  # llama-server ignores this
            "temperature": self._defaults.get("temperature", 0.0),
        }
        params.update(kwargs)
        return ChatOpenAI(**params)
