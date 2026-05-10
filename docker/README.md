# Docker-assisted reproduction notes

Three images are declared:

| Image | Dockerfile | Responsibility |
| --- | --- | --- |
| `ollama` | `Dockerfile.ollama` | Hosts GGUF checkpoints via upstream `ollama/ollama` |
| `vllm` | `Dockerfile.vllm` | Serves HF weights with chunked pre-fill + TP |
| `benchmark` | `Dockerfile.benchmark` | Installs pinned Python deps to execute drivers / figures |

Quick start (from repo root):

```bash
docker compose -f docker/docker-compose.yml up --build ollama vllm
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen3:4b
```

The benchmark matrix can run for multiple hours once both serving endpoints are warmed up. Figures can be regenerated from archived JSON artefacts without rerunning GPUs — see `scripts/generate_figures.sh`.
