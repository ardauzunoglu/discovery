#!/usr/bin/env bash
# Stage 2 (materials science): reasoning-trace SFT. Each merged stage-1 model
# is trained on the stage-2 data with the same monofact ratio.
# Arguments are forwarded to train/train.py and override the defaults below.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_KEY="${MODEL_KEY:-gpt-oss-20b}"
export INIT_MODELS_ROOT="${INIT_MODELS_ROOT:-${REPO_DIR}/outputs/materials_science/stage1/merged/${MODEL_KEY}}"
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/materials_science/data/stage2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/materials_science/stage2/adapters/${MODEL_KEY}}"

exec bash "${REPO_DIR}/train/train_variants.sh" \
  --num_train_epochs 1 \
  --max_length 32768 \
  --attn_implementation flash_attention_2 \
  "$@"
