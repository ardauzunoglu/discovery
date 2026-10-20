#!/usr/bin/env bash
# Stage 1 (materials science): formula-only SFT of the base model on every
# monofact-ratio/variant split.
# Arguments are forwarded to train/train.py and override the defaults below.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"
MODEL_KEY="${MODEL_KEY:-$(basename -- "${BASE_MODEL}")}"
export DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/materials_science/data/stage1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/materials_science/stage1/adapters/${MODEL_KEY}}"

exec bash "${REPO_DIR}/train/train_variants.sh" \
  --num_train_epochs 1 \
  --max_length 8192 \
  --attn_implementation eager \
  "$@"
