#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

for s in 1 2 3 4; do
  ./scripts/run_scenario.sh "${s}" ollama || echo "[warn] scenario ${s} ollama failed — continuing."
  ./scripts/run_scenario.sh "${s}" vllm || echo "[warn] scenario ${s} vllm failed — continuing."
done
