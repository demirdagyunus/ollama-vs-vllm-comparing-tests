# Contributing

Issues and merge requests documenting hardware deltas are welcome. GPU SKU, driver, and quantization choices materially reshape latency envelopes.

Suggested workflow:

1. Fork → feature branch `feat/<topic>`.
2. Run `./scripts/setup_environment.sh` when touching scenario drivers.
3. Regenerate affected figures selectively: `./scripts/generate_figures.sh --only 12,21`.
4. Attach before/after JSON excerpts in the PR narrative when metrics shift.

Coding standards: PEP 8; avoid bare `except` unless mirroring upstream vendor samples.
