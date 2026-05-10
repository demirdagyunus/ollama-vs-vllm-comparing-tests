# Known artefacts & caveats

1. **`total_requests` spikes on connection storms** — Ollama JSON may log thousands of `Failed to connect` entries; scholarly tables should annotate *effective attempts* when comparing against vLLM.
2. **GPU counters occasionally zero-out** — some exports fall back silently to CPU-bound execution; corroborate manually with `nvidia-smi`.
4. **Trial count vs. manuscript narrative** — manuscript Section 4 may mention averaging over five trials; bundled JSON snapshots store a single run per scenario. Figures expose **intra-run** distribution (percentiles, min/max, peaks) where the driver records it, or cross-concurrency spread where applicable, rather than inter-trial standard deviation.
