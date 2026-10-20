#!/usr/bin/env bash
# Extract predicted formulas from an evaluation run and label each one as
# correct, a known formula from the released data, or a formula outside it.
# Usage: extract_prediction_formulas.sh EVAL_DIR [extra extract_prediction_formulas.py args]

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
eval_dir="${1:?usage: extract_prediction_formulas.sh EVAL_DIR [ARGS...]}"
shift

python "${script_dir}/extract_prediction_formulas.py" \
  "${eval_dir}/predictions.jsonl" \
  --splits-dir "${SPLITS_DIR:-${repo_dir}/data/materials_science/data}" \
  --output "${eval_dir}/formulas.jsonl" \
  "$@"
