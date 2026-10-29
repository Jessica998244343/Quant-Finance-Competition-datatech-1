#!/usr/bin/env bash
# Activate the phase-1 research environment.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"
echo "python: $(python -V)  OMP=${OMP_NUM_THREADS} threads  PYTHONPATH=${PYTHONPATH}"
