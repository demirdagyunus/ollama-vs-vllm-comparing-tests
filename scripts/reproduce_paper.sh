#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[reproduce] Installing dependencies ..."
./scripts/setup_environment.sh

echo "[reproduce] Executing workloads (hours of GPU runtime possible) ..."
read -rp "Proceed with exhaustive scenario execution? [y/N] " ack
case "${ack}" in
  y|Y)
    ;;
  *)
    echo "[reproduce] Aborted workload phase — rerun with acknowledge."
    exit 1
    ;;
 esac

./scripts/run_all_tests.sh
echo "[reproduce] Generating Figures 03–21 ..."
./scripts/generate_figures.sh
