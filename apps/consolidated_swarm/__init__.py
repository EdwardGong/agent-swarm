"""Consolidated Agent Swarm — M5 Max 128GB configuration.

Runs four specialist agents (research, code, CRM, content) through a LangGraph
state machine, backed by a single vllm-mlx inference server serving
qwen3.6-35b-opus-distilled-abl-mxfp4.  MCP tools are managed natively by
vllm-mlx; agents receive the full MCP tool list for their profile.
"""
