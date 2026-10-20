#!/usr/bin/env bash
# Merge every trained LoRA adapter in a ratio/variant tree into a standalone
# Hugging Face model that vLLM (evaluation) and stage-2 training can load:
#
#   ADAPTERS_ROOT/ratio_*/variant_*/  ->  OUTPUT_ROOT/ratio_*/variant_*/
#
# Extra arguments are forwarded to merge_lora_checkpoint.py.

set -euo pipefail

export PYTHONUNBUFFERED=1

TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ADAPTERS_ROOT="${ADAPTERS_ROOT:?ADAPTERS_ROOT must be set}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RATIO_FILTER="${RATIO_FILTER:-all}"
VARIANT_FILTER="${VARIANT_FILTER:-all}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

matches_filter() {
  local value="$1" filter="$2" candidate
  [[ "${filter}" == "all" ]] && return 0
  for candidate in ${filter//,/ }; do
    [[ "${value}" == "${candidate}" ]] && return 0
  done
  return 1
}

[[ -d "${ADAPTERS_ROOT}" ]] || { echo "[error] Missing adapters root: ${ADAPTERS_ROOT}" >&2; exit 1; }

shopt -s nullglob
matched=0
merged=0
for ratio_dir in "${ADAPTERS_ROOT}"/ratio_*; do
  [[ -d "${ratio_dir}" ]] || continue
  ratio_name="$(basename -- "${ratio_dir}")"
  matches_filter "${ratio_name}" "${RATIO_FILTER}" || continue
  for variant_dir in "${ratio_dir}"/variant_*; do
    [[ -d "${variant_dir}" ]] || continue
    variant_name="$(basename -- "${variant_dir}")"
    matches_filter "${variant_name}" "${VARIANT_FILTER}" || continue
    if [[ ! -f "${variant_dir}/adapter_config.json" ]]; then
      echo "[skip] ${ratio_name}/${variant_name}: no adapter_config.json" >&2
      continue
    fi
    matched=$((matched + 1))

    output_dir="${OUTPUT_ROOT}/${ratio_name}/${variant_name}"
    # Large models are saved as model-0000N-of-0000M.safetensors shards.
    if [[ -f "${output_dir}/config.json" ]] && compgen -G "${output_dir}/model*.safetensors" >/dev/null; then
      echo "[skip] ${ratio_name}/${variant_name}: merged model already exists" >&2
      continue
    fi

    command=(
      "${PYTHON_BIN}" "${TRAIN_DIR}/merge_lora_checkpoint.py"
      --adapter_model_path "${variant_dir}"
      --output_dir "${output_dir}"
      --trust_remote_code
      --attn_implementation eager
      --dtype bfloat16
      --device cpu
      "$@"
    )
    merged=$((merged + 1))
    echo "[start] ${ratio_name}/${variant_name} -> ${output_dir}" >&2
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run]' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2
    else
      mkdir -p -- "${output_dir}"
      "${command[@]}"
    fi
  done
done

if ((matched == 0)); then
  echo "[error] No trained adapters matched the filters under ${ADAPTERS_ROOT}" >&2
  exit 1
fi
echo "[done] matched=${matched} merged=${merged}" >&2
