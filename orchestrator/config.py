"""Configuration for the orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OrchestratorConfig:
    """Central configuration for an orchestrated agent system.

    Can be constructed programmatically or loaded from a YAML file.
    """

    # Routing
    routing_strategy: str = "llm"  # "llm" | "rule"
    max_iterations: int = 10

    # Model provider
    model_provider: str = "ollama"  # extensible — key used by provider registry
    model_tiers: dict[str, str] = field(default_factory=lambda: {
        "orchestrator": "qwen2.5:32b",
        "worker": "qwen2.5:14b",
        "worker_fast": "qwen2.5:7b-instruct-q8_0",
        "worker_light": "qwen2.5:3b",
        "worker_micro": "qwen2.5:1.5b",
        "coder": "qwen2.5-coder:32b",
        "reasoner": "qwen2.5:72b-instruct-q4_K_M",
    })
    model_options: dict[str, Any] = field(default_factory=lambda: {
        "keep_alive": "5m",
        "temperature": 0.0,
    })

    # Optional extras apps can attach
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> OrchestratorConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
