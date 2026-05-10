# Methodological anchor

Controlled experiments juxtapose synthetic prompt mixes executed through mirrored Python harnesses against `ollama/qwen3:4b` and `vLLM/Qwen/Qwen3-4B`.

Each workload records client-visible latency tails, throughput, heuristic accuracy gauges (reasoning tracks), conversational streaming artefacts, coarse CPU/RAM footprints, GPU proxies when tooling surfaces them, and stress envelopes (gradual ramp + hourly endurance regimes).

Warm-up executions are mandated but truncated from aggregates; pacing between escalations aligns with argparse defaults baked into scenario drivers.
