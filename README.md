# Replication package — Ollama vs. vLLM production-style serving

This repository bundles the scenario drivers, archived JSON measurement logs, HTML summarisers, and matplotlib regenerators supporting the MDPI manuscript:

> **A Systematic Benchmark of Ollama and vLLM for Scalable LLM Serving under Diverse Workloads**  
> Authors: Betül Ay and Yunus Emre Demirdağ (Fırat University, Elazığ, Türkiye)

## Repository map


| Path                                         | Role                                                           |
| -------------------------------------------- | -------------------------------------------------------------- |
| `cases/test-scenario-{1-4}-{ollama,vllm}.py` | Load generators + metric aggregation                           |
| `cases/results/scenario-*/*.json`            | Immutable experiment exports                                   |
| `reporters/paper_figures/`                   | Figure 3–21 generators (`generate_all.py`)                     |
| `docker/`                                    | Optional tri-service compose (Ollama + vLLM + benchmark image) |
| `docs/`                                      | Expanded methodology + troubleshooting                         |
| `scripts/`                                   | Setup, execution, and reproduction helpers                     |


## Quick start (host Python)

```bash
git clone https://github.com/demirdagyunus/ollama-vs-vllm-comparing-tests.git
cd ollama-vs-vllm-comparing-tests
cp .env.example .env  # adjust endpoints
./scripts/setup_environment.sh
./scripts/generate_figures.sh        # uses bundled JSON; no GPU required
```

## Re-running workloads (GPU mandatory)

```bash
./scripts/run_scenario.sh 1 ollama
./scripts/run_scenario.sh 1 vllm
# ... repeat for scenarios 2–4 (hours of wall-clock time)
```

`scripts/reproduce_paper.sh` chains setup → full matrix → figure export with an interactive acknowledgement gate.

## Docker option

```bash
docker compose -f docker/docker-compose.yml up --build ollama vllm
```

See `docker/README.md` for model pull instructions and tensor-parallel knobs.

## Figures

```bash
python reporters/paper_figures/generate_all.py --only 3,21 --no-pdf
```

Outputs default to `reporters/paper_figures/output/{composite,clean}/`.

## Citation metadata

`CITATION.cff` exposes GitHub citation buttons; update ORCID + DOI placeholders after acceptance.

## Maintainer contacts

- Betül Ay — `betulay@firat.edu.tr`  
- Yunus Emre Demirdağ — `demirdag.emre.y@gmail.com`

## License

MIT — see `LICENSE`.