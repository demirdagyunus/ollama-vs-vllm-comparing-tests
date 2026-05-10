# Experiment knobs (compact)

| Aspect | Ollama | vLLM |
| --- | --- | --- |
| Model | `qwen3:4b` | `Qwen/Qwen3-4B` |
| Temperature | 0.7 (scenario 2 lowers to ~0.1) | Mirrors Ollama per scenario script |
| Top-p | 0.9 | 0.9 |
| Context clamp | Scenario scripts set 2048–4096 | `--max-model-len 4096` in compose hints |
| Batching | llama.cpp internal queue | Continuous batch + chunked prefill |

Tune docker-compose environment variables (`VLLM_TP_SIZE`) when tensor-parallel widths differ.
