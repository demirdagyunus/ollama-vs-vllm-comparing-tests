#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.env"
fi

echo "[setup] Repository root resolved to ${ROOT}"
python3 -m pip install -r requirements.txt
echo "[setup] Pip dependencies installed."
