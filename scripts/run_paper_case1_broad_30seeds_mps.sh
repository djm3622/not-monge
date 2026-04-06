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

if ! "$PYTHON" - <<'PY'
import sys
import torch

available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
sys.exit(0 if available else 1)
PY
then
  echo "MPS is not available for this Python environment." >&2
  echo "Set PYTHON_BIN to the environment you want to use, or run the CPU runner instead." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper_case1_full_compare_broad_30seeds_mps_v1}"

exec "$ROOT_DIR/scripts/run_paper_case1_broad_30seeds.sh" "$@"
