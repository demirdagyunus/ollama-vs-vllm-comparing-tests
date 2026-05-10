# Scenario synopsis

## Scenario 1 — Baseline Q&A
Scripts: `cases/test-scenario-1-ollama.py`, `...-vllm.py`
Focus: percentile latency ladders, naive throughput ceilings, coarse resource traces.

## Scenario 2 — Complex reasoning corpus
Synthetic math/code/QA curricula with verifier heuristics capturing partial credit.

## Scenario 3 — Streaming / multi-turn dialogs
TTFT spectra, coherence placeholders, conversational completion ratios.

## Scenario 4 — Stress + endurance telemetry
Historical archives emphasise vLLM; Ollama driver exists but may halt under identical peak loads unless operators scale `OLLAMA_NUM_PARALLEL`.
