"""Standalone workflow runners for each consolidated swarm agent.

Each module exposes a ``run(task, mcp_registry=None) -> str`` function that
implements the full multi-step workflow described in the plan.  These can be
invoked directly (for standalone use) or via the LangGraph orchestrator.
"""
