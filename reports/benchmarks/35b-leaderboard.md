# Benchmark comparison (20260427-083255)

- prompts (5 cycled): `Design a distributed message queue system with partitioning,...`; `Explain the differences between cooperative and preemptive m...`; `Write a step-by-step plan to migrate a monolithic Postgres d...`; `Describe how Raft achieves consensus and what failure modes ...`; `Compare HNSW, IVF, and ScaNN as vector indexing strategies f...`
- max_tokens: 1024
- runs per model: 3
- thinking: False

| rank | model | gen tok/s (mean) | min | max | stdev | prompt tok/s | peak GB | vs best | load s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `qwen3.6-35b-opus-abl-mxfp4-vlm-mlx` | 77.1 | 76.9 | 77.4 | 0.28 | 432.0 | 20.58 | 100.0% | 5.1 |
| 2 | `qwen3.6-35b-mxfp4-mlx` | 76.7 | 76.5 | 76.9 | 0.23 | 435.6 | 20.58 | 99.4% | 4.9 |
| 3 | `qwen3.6-35b-opus-abl-4bit-vlm-mlx` | 75.0 | 74.7 | 75.2 | 0.26 | 424.1 | 20.58 | 97.3% | 8.6 |
| 4 | `qwen3.6-35b-abl-4.4bit-msq-mlx` | 53.6 | 52.7 | 55.3 | 1.48 | 222.3 | 21.37 | 69.5% | 5.6 |
