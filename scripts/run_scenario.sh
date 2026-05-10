#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.env"
fi

SCEN="${1:-}"
BACKEND="${2:-}"
if [[ -z "${SCEN}" || -z "${BACKEND}" ]]; then
  echo "usage: ./scripts/run_scenario.sh <1-4> <ollama|vllm>"
  exit 1
fi

export PYTHONPATH="${ROOT}"
ARGS=()
b="$(printf '%s' "${BACKEND}" | tr '[:upper:]' '[:lower:]')"
TARGET="${ROOT}/cases/test-scenario-${SCEN}-${b}.py"
if [[ ! -f "${TARGET}" ]]; then
  echo "[err] Scenario script missing: ${TARGET}"
  exit 3
fi

case "${b}" in
  ollama)
    ARGS+=( "--url" "${OLLAMA_BASE_URL:-http://127.0.0.1:11434}" "--model" "${OLLAMA_MODEL:-qwen3:4b}" )
    ;;
  vllm)
    ARGS+=( "--url" "${VLLM_BASE_URL:-http://127.0.0.1:8000}" "--model" "${VLLM_MODEL:-Qwen/Qwen3-4B}" )
    ;;
  *)
    echo "[err] Unsupported backend '${BACKEND}'"
    exit 2
    ;;
 esac

OUT_DIR="${OUT_ROOT:-cases/results/scenario-${SCEN}}"
ARGS+=( "--output-dir" "${OUT_DIR}" )

python3 "${TARGET}" "${ARGS[@]}"
