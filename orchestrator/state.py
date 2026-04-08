"""Shared state schema for the orchestration graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class OrchestratorState(BaseModel):
    """State that flows through the orchestration graph.

    Core fields are used by the framework.  Applications can store
    additional data in ``extra``.
    """

    # Chat messages — uses LangGraph's reducer to append correctly
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Which specialist the orchestrator wants to delegate to next
    next_agent: str = ""

    # The sub-task description assigned by the orchestrator
    sub_task: str = ""

    # Accumulates results from worker agents
    results: Annotated[list[str], operator.add] = Field(default_factory=list)

    # Whether the orchestrator considers the task complete
    done: bool = False

    # Application-defined extra state (not touched by the framework)
    extra: dict[str, Any] = Field(default_factory=dict)
