# WARP.md

Project-scoped rules for `agent-swarm`. These apply to any agent working in this repo and override personal/global rules where they conflict.

## Use verbose model naming conventions
- Use the verbose canonical form: when referencing AI/ML models, include abliteration state (`abl`), quantization format and bit-width (e.g., `Q4_K_M`, `GPTQ-4bit`), fine-tune lineage, and parameter count. Do not abbreviate or normalize these away, even on second mention.
