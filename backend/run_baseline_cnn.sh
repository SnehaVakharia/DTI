#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate

INSTALL_DEPS="${INSTALL_DEPS:-0}"
if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install -r backend/requirements-train.txt
fi

DATA_PATH="${DATA_PATH:-backend/data/davis_binary_dti.csv}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SEED="${SEED:-42}"
MAX_SMILES_LEN="${MAX_SMILES_LEN:-120}"
MAX_PROTEIN_LEN="${MAX_PROTEIN_LEN:-800}"
DAVIS_THRESHOLD_NM="${DAVIS_THRESHOLD_NM:-300}"
RUN_ID="${RUN_ID:-}"

CMD=(
  python backend/train_baseline_cnn.py
  --prepare-davis-data
  --data-path "$DATA_PATH"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --seed "$SEED"
  --max-smiles-len "$MAX_SMILES_LEN"
  --max-protein-len "$MAX_PROTEIN_LEN"
  --davis-threshold-nm "$DAVIS_THRESHOLD_NM"
)

if [[ -n "$RUN_ID" ]]; then
  CMD+=(--run-id "$RUN_ID")
fi

"${CMD[@]}"
