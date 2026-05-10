# Figure regeneration toolkit

`generate_all.py` ingests the latest JSON export per scenario (lexicographic newest `*_data_*.json` glob) and emits:

- `output/composite/` — multi-panel canvases mirroring manuscript layouts
- `output/clean/` — decomposed single-metric panels when provided

## Usage

```bash
# from repository root
python3 reporters/paper_figures/generate_all.py --help
python3 reporters/paper_figures/generate_all.py --only 7,13,21 --no-pdf
```

## Scenario coverage

| Figures | Scenario | Notes |
| --- | --- | --- |
| 3–5 | 1 | Latency triptychs, throughput / token slopes, efficiency |
| 6–9 | 2 | Reasoning fidelity + reliability overlays |
| 10–14 | 3 | Streaming QoS + TTFT analyses |
| 15–18 | 4 (`vLLM`) | Breaking points, timelines, dashboards |
| 19–21 | Cross | Macro radar + KPI overview |

Consult `docs/figure_mapping.md` for JSON column references.

## Variability semantics

Error bars and shaded regions are derived from native distribution statistics persisted by each scenario driver — typically min, max, and percentile envelopes (p50/p75/p90/p95/p99). They do *not* represent 5-trial standard deviation, since archived JSON exports contain a single canonical run per scenario. Figures whose source metric is a single point estimate (success rate without a cross-level series, accuracy splits, derived ratios) intentionally omit error bars. Where sample standard deviation across **concurrency levels** is shown (Figure 20 scenario bars, reasoning accuracy spread in Figure 19), it summarizes variation *across displayed load points* in the bundled snapshot, not inter-trial repetition.
