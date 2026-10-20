#!/usr/bin/env bash
# Merge the stage2 LoRA adapters into standalone models.
# Arguments are forwarded to train/merge_lora_checkpoint.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_KEY="${MODEL_KEY:-gpt-oss-20b}"
export ADAPTERS_ROOT="${ADAPTERS_ROOT:-${REPO_DIR}/outputs/materials_science/stage2/adapters/${MODEL_KEY}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/materials_science/stage2/merged/${MODEL_KEY}}"

exec bash "${REPO_DIR}/train/merge_lora_variants.sh" "$@"
