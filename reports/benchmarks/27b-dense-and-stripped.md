# Benchmark comparison (20260428-005845)

- prompts (5 cycled): `Design a distributed message queue system with partitioning,...`; `Explain the differences between cooperative and preemptive m...`; `Write a step-by-step plan to migrate a monolithic Postgres d...`; `Describe how Raft achieves consensus and what failure modes ...`; `Compare HNSW, IVF, and ScaNN as vector indexing strategies f...`
- max_tokens: 1024
- runs per model: 3
- thinking: False

| rank | model | gen mean | median | min | max | CV% | TTFT ms | wall s | peak GB | disk GB | vs best | load s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `qwen3.6-35b-opus-abl-mxfp4-mlx` | 87.6 | 87.8 | 86.9 | 88.2 | 0.77 | 79.2 | 11.26 | 21.29 | 17.16 | 100.0% | 4.8 |
| 2 | `qwen3.6-35b-optiq-4bit-mlx` | 70.6 | 70.7 | 69.8 | 71.3 | 1.09 | 85.2 | 14.50 | 21.29 | 19.67 | 80.6% | 6.7 |
| 3 | `qwen3.6-27b-4bit-mlx` | 18.6 | 18.9 | 17.1 | 19.8 | 7.37 | 361.4 | 55.20 | 16.37 | 14.95 | 21.2% | 11.3 |
| 4 | `qwen3.6-27b-abl-4.5bit-msq-mlx` | 18.2 | 18.3 | 18.0 | 18.5 | 1.47 | 364.7 | 56.16 | 17.26 | 15.78 | 20.8% | 5.1 |
