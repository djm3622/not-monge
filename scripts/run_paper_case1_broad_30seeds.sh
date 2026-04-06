#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache}"

SEEDS_DEFAULT="686499,928801,48156,431753,526655,480953,178898,700645,62001,192385,911170,307368,546159,381730,917274,910268,301123,218738,982909,870264,948176,164608,892045,767301,576270,525000,7029,644131,480217,389379"

SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
SOLVERS="${SOLVERS:-gaussian,mm,mmv2,tw2,mm_b,qc}"
FIGURE_SOLVERS="${FIGURE_SOLVERS:-mm,mmv2}"
CACHE_VERSION="${CACHE_VERSION:-paper_ref_d64_b256_s25k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_case1_full_compare_broad_30seeds_v3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
EVAL_ITEMS="${EVAL_ITEMS:-4096}"
VISUALIZATION_ITEMS="${VISUALIZATION_ITEMS:-512}"

ARGS=(
  --seeds "$SEEDS"
  --solvers "$SOLVERS"
  --figure-solvers "$FIGURE_SOLVERS"
  --cache-version "$CACHE_VERSION"
  --output-root "$OUTPUT_ROOT"
  --eval-items "$EVAL_ITEMS"
  --visualization-items "$VISUALIZATION_ITEMS"
  --device "$TRAIN_DEVICE"
)

if [[ "${RERUN:-0}" == "1" ]]; then
  ARGS+=(--rerun)
fi

cat <<EOF
Running case study 1 broad comparison
  python: $PYTHON
  seeds: $SEEDS
  solvers: $SOLVERS
  figure_solvers: $FIGURE_SOLVERS
  cache_version: $CACHE_VERSION
  output_root: $OUTPUT_ROOT
  device: $TRAIN_DEVICE
EOF

exec "$PYTHON" scripts/paper_case1_full_compare.py "${ARGS[@]}"
