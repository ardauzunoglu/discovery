#!/usr/bin/env bash
# Generate stage-2 materials reasoning traces with a teacher model. The teacher
# sees each example's constraints, a composition-blind view of its reference
# CIF, and the known answer, and writes a reasoning trace for that answer.
#
# Required: INPUT (source examples as .jsonl or .parquet in the stage-1 schema)
# and the provider's API key (ANTHROPIC_API_KEY or OPENAI_API_KEY). Reference
# CIFs are resolved through MATSCI_REFERENCE_ROOT. Output is resumable.
# Extra arguments are forwarded to generate_distillation_data_cif.py.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
input="${INPUT:?Set INPUT to the source examples (.jsonl or .parquet)}"
input_name="$(basename -- "${input%.*}")"

cif_size_args=()
if [[ -n "${MAX_CIF_CHARS:-}" ]]; then
  cif_size_args=(--max-cif-chars "${MAX_CIF_CHARS}")
fi

python "${script_dir}/generate_distillation_data_cif.py" \
  --input "${input}" \
  --provider "${PROVIDER:-anthropic}" \
  --model "${MODEL:-claude-opus-5}" \
  --output "${OUTPUT:-${repo_dir}/outputs/materials_science/distillation/${input_name}_with_cif.jsonl}" \
  --max-concurrency "${MAX_CONCURRENCY:-20}" \
  --max-tokens "${MAX_TOKENS:-32768}" \
  --cif-reasoning-mode "${CIF_REASONING_MODE:-direct}" \
  --reasoning-effort "${REASONING_EFFORT:-medium}" \
  --resume \
  --allow-max-tokens-increase-on-resume \
  --skip-failed \
  "${cif_size_args[@]}" \
  "$@"
