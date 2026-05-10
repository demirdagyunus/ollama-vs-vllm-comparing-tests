# Figure lineage

Automated composites ship under `reporters/paper_figures/output/` after `./scripts/generate_figures.sh`.

| Fig. span | Scenario JSON folders | Modules |
| --- | --- | --- |
| 3–5 | `cases/results/scenario-1/` | `figure_03` … `figure_05` |
| 6–9 | `scenario-2` | `_06` … `_09` |
| 10–14 | `scenario-3` | `_10` … `_14` |
| 15–18 | `scenario-4/` (`vllm_stress_*`) | `_15` … `_18` |
| 19–21 | Cross-scenario aggregation | `_19` … `_21` |

Regenerate selectively: `python reporters/paper_figures/generate_all.py --only 13 --no-pdf`.
