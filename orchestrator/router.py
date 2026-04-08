"""Pluggable routing strategies for the orchestrator.

A ``Router`` decides which agent to invoke next (or whether the task is
done).  Two built-in implementations are provided:

* ``LLMRouter`` — asks an LLM to pick the next agent (current approach,
  generalized).  The system prompt is *auto-generated* from the agent
  registry so no hardcoded agent names leak into the core.
* ``RuleRouter`` — deterministic keyword/capability matching for
  lightweight, predictable routing.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestrator.agent import AgentRegistry
from orchestrator.models import ModelProvider
from orchestrator.state import OrchestratorState


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Router(Protocol):
    """Interface every routing strategy must satisfy."""

    def route(self, state: OrchestratorState) -> dict[str, Any]:
        """Return a dict with at least ``next_agent``, ``sub_task``,
        ``done``, and optionally ``messages``.
        """
        ...


# ---------------------------------------------------------------------------
# LLM-based router
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are the Orchestrator of a multi-agent system. Your job is to:
1. Understand the user's request.
2. Break it into sub-tasks if needed.
3. Delegate each sub-task to the right specialist agent.
4. Synthesize results into a final answer.

Available specialists:
{agent_descriptions}

Respond with a JSON object (and ONLY the JSON, no markdown fences):
{{
  "thinking": "your reasoning about what to do next",
  "next_agent": "<agent_name>" | "done",
  "sub_task": "description of what that agent should do",
  "final_answer": "your final answer to the user (only when next_agent is 'done')"
}}

Rules:
- Delegate ONE sub-task at a time.
- After receiving a result, decide if more delegation is needed or if you're done.
- When done, set next_agent to "done" and provide final_answer.
- Be concise in sub_task descriptions — the specialist is capable.
"""


class LLMRouter:
    """Routes via an LLM whose system prompt is auto-generated from the
    agent registry.
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: AgentRegistry,
        tier: str = "orchestrator",
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._tier = tier

    def _build_system_prompt(self) -> str:
        return _SYSTEM_TEMPLATE.format(
            agent_descriptions=self._registry.describe_for_prompt(),
        )

    def route(self, state: OrchestratorState) -> dict[str, Any]:
        llm = self._provider.get_llm(self._tier)

        messages: list[Any] = [SystemMessage(content=self._build_system_prompt())]
        messages.extend(state.messages)

        if state.results:
            results_text = "\n\n".join(
                f"[Result {i+1}]: {r}" for i, r in enumerate(state.results)
            )
            messages.append(
                HumanMessage(
                    content=f"Previous agent results:\n{results_text}\n\nDecide what to do next."
                )
            )

        response = llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON (handle models that wrap in code fences)
        try:
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            decision = json.loads(content)
        except json.JSONDecodeError:
            return {
                "next_agent": "done",
                "sub_task": "",
                "done": True,
                "messages": [AIMessage(content=content)],
            }

        next_agent = decision.get("next_agent", "done")
        sub_task = decision.get("sub_task", "")
        final_answer = decision.get("final_answer", "")

        if next_agent == "done":
            return {
                "next_agent": "done",
                "sub_task": "",
                "done": True,
                "messages": [AIMessage(content=final_answer or content)],
            }

        return {
            "next_agent": next_agent,
            "sub_task": sub_task,
            "done": False,
        }


# ---------------------------------------------------------------------------
# Rule-based router
# ---------------------------------------------------------------------------

class RuleRouter:
    """Deterministic router that matches keywords in the sub-task / user
    message against agent capability tags.

    Useful for testing or when LLM routing overhead is undesirable.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def route(self, state: OrchestratorState) -> dict[str, Any]:
        # Simple heuristic: scan the last user message for capability keywords
        text = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                text = msg.content.lower()
                break

        for spec in self._registry.all:
            for cap in spec.capabilities:
                if cap.lower() in text:
                    return {
                        "next_agent": spec.name,
                        "sub_task": text,
                        "done": False,
                    }

        # No match — finish immediately
        return {
            "next_agent": "done",
            "sub_task": "",
            "done": True,
            "messages": [AIMessage(content="No suitable agent found for this request.")],
        }
