#!/usr/bin/env bash
# Generate stage-2 MPRA reasoning traces with a teacher model. The teacher
# sees each design request plus the known edit and writes a forward reasoning
# trace that arrives at it.
#
# Required: INPUT (source examples as .parquet in the stage-1 schema) and the
# provider's API key (ANTHROPIC_API_KEY or OPENAI_API_KEY). Output is
# resumable. Extra arguments are forwarded to generate_distillation_data.py.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
input="${INPUT:?Set INPUT to the source examples (.parquet)}"
input_name="$(basename -- "${input%.*}")"

python "${script_dir}/generate_distillation_data.py" \
  --input "${input}" \
  --provider "${PROVIDER:-anthropic}" \
  --model "${MODEL:-claude-opus-5}" \
  --output "${OUTPUT:-${repo_dir}/outputs/regulatory_genomics/distillation/${input_name}_reasoning.jsonl}" \
  --max-concurrency "${MAX_CONCURRENCY:-10}" \
  --max-tokens "${MAX_TOKENS:-32768}" \
  --reasoning-effort "${REASONING_EFFORT:-medium}" \
  --resume \
  "$@"
