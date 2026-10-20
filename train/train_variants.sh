#!/usr/bin/env bash
# Run LoRA SFT once per ratio/variant split of a dataset tree:
#
#   DATA_ROOT/ratio_*/variant_*/train.parquet  ->  OUTPUT_ROOT/ratio_*/variant_*/
#
# Stage 1 trains every data variant from one base model (BASE_MODEL).
# Stage 2 sets INIT_MODELS_ROOT to a merged stage-1 tree; each merged model
# INIT_MODELS_ROOT/ratio_X/variant_Y is trained on DATA_ROOT/ratio_X/DATA_VARIANT.
#
# The per-stage train.sh scripts set these variables and the stage-specific
# hyperparameters; arguments after -- are forwarded to train.py last, so they
# override any default.

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT must point to a ratio_*/variant_* dataset tree}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
BASE_MODEL="${BASE_MODEL:-}"
INIT_MODELS_ROOT="${INIT_MODELS_ROOT:-}"
DATA_VARIANT="${DATA_VARIANT:-variant_000}"
EVAL_FILE="${EVAL_FILE:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
FSDP_CONFIG="${FSDP_CONFIG:-${TRAIN_DIR}/fsdp_transformer_auto_wrap.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-gpu}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LORA_TARGET_PARAMETERS="${LORA_TARGET_PARAMETERS:-auto}"
RATIO_FILTER="${RATIO_FILTER:-all}"
VARIANT_FILTER="${VARIANT_FILTER:-all}"
RATIO_ORDER="${RATIO_ORDER:-ascending}"
RATIO_MIDPOINT="${RATIO_MIDPOINT:-ratio_0p500}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [[ "${1:-}" == "--" ]]; then
  shift
fi
train_args=("$@")

if [[ -n "${INIT_MODELS_ROOT}" ]]; then
  [[ -d "${INIT_MODELS_ROOT}" ]] || { echo "[error] Missing INIT_MODELS_ROOT: ${INIT_MODELS_ROOT}" >&2; exit 1; }
elif [[ -z "${BASE_MODEL}" ]]; then
  echo "[error] Set BASE_MODEL (stage 1) or INIT_MODELS_ROOT (stage 2)." >&2
  exit 1
fi
[[ -d "${DATA_ROOT}" ]] || { echo "[error] Missing DATA_ROOT: ${DATA_ROOT} (run download_data.sh first)" >&2; exit 1; }
[[ -f "${FSDP_CONFIG}" ]] || { echo "[error] Missing FSDP config: ${FSDP_CONFIG}" >&2; exit 1; }
[[ -z "${EVAL_FILE}" || -f "${EVAL_FILE}" ]] || { echo "[error] Missing EVAL_FILE: ${EVAL_FILE}" >&2; exit 1; }

# Filters are "all" or a comma/space-separated list of directory names.
matches_filter() {
  local value="$1" filter="$2" candidate
  [[ "${filter}" == "all" ]] && return 0
  for candidate in ${filter//,/ }; do
    [[ "${value}" == "${candidate}" ]] && return 0
  done
  return 1
}

is_completed_adapter() {
  local output_dir="$1"
  [[ -f "${output_dir}/adapter_config.json" ]] &&
    [[ -f "${output_dir}/adapter_model.safetensors" || -f "${output_dir}/adapter_model.bin" ]]
}

shopt -s nullglob
if [[ -n "${INIT_MODELS_ROOT}" ]]; then
  ratio_dirs=("${INIT_MODELS_ROOT}"/ratio_*)
else
  ratio_dirs=("${DATA_ROOT}"/ratio_*)
fi

# ascending, descending, or midpoint (RATIO_MIDPOINT first, then fanning
# outward). Different orders let several nodes share one sweep.
ordered_ratio_dirs=()
case "${RATIO_ORDER}" in
  ascending)
    ordered_ratio_dirs=("${ratio_dirs[@]}")
    ;;
  descending)
    for ((index = ${#ratio_dirs[@]} - 1; index >= 0; index--)); do
      ordered_ratio_dirs+=("${ratio_dirs[index]}")
    done
    ;;
  midpoint)
    midpoint_index=-1
    for index in "${!ratio_dirs[@]}"; do
      if [[ "$(basename -- "${ratio_dirs[index]}")" == "${RATIO_MIDPOINT}" ]]; then
        midpoint_index="${index}"
        break
      fi
    done
    ((midpoint_index >= 0)) || { echo "[error] RATIO_MIDPOINT not found: ${RATIO_MIDPOINT}" >&2; exit 1; }
    ordered_ratio_dirs+=("${ratio_dirs[midpoint_index]}")
    for ((offset = 1; ${#ordered_ratio_dirs[@]} < ${#ratio_dirs[@]}; offset++)); do
      ((midpoint_index - offset >= 0)) && ordered_ratio_dirs+=("${ratio_dirs[midpoint_index - offset]}")
      ((midpoint_index + offset < ${#ratio_dirs[@]})) && ordered_ratio_dirs+=("${ratio_dirs[midpoint_index + offset]}")
    done
    ;;
  *)
    echo "[error] RATIO_ORDER must be ascending, descending, or midpoint; got: ${RATIO_ORDER}" >&2
    exit 2
    ;;
esac

echo "[config] data_root=${DATA_ROOT}" >&2
echo "[config] initialization=${INIT_MODELS_ROOT:-${BASE_MODEL}}" >&2
echo "[config] output_root=${OUTPUT_ROOT}" >&2
echo "[config] ratio_filter=${RATIO_FILTER} variant_filter=${VARIANT_FILTER} ratio_order=${RATIO_ORDER}" >&2

matched=0
launched=0
skipped=0
failures=0
for ratio_dir in "${ordered_ratio_dirs[@]}"; do
  [[ -d "${ratio_dir}" ]] || continue
  ratio_name="$(basename -- "${ratio_dir}")"
  matches_filter "${ratio_name}" "${RATIO_FILTER}" || continue

  for variant_dir in "${ratio_dir}"/variant_*; do
    [[ -d "${variant_dir}" ]] || continue
    variant_name="$(basename -- "${variant_dir}")"
    matches_filter "${variant_name}" "${VARIANT_FILTER}" || continue
    matched=$((matched + 1))

    if [[ -n "${INIT_MODELS_ROOT}" ]]; then
      model="${variant_dir}"
      train_file="${DATA_ROOT}/${ratio_name}/${DATA_VARIANT}/train.parquet"
      if [[ ! -f "${model}/config.json" ]]; then
        echo "[skip] ${ratio_name}/${variant_name}: no merged model in ${model}" >&2
        skipped=$((skipped + 1))
        continue
      fi
    else
      model="${BASE_MODEL}"
      train_file="${variant_dir}/train.parquet"
    fi
    [[ -f "${train_file}" ]] || { echo "[error] Missing training data: ${train_file}" >&2; exit 1; }

    output_dir="${OUTPUT_ROOT}/${ratio_name}/${variant_name}"
    if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed_adapter "${output_dir}"; then
      echo "[skip] ${ratio_name}/${variant_name}: adapter already exists" >&2
      skipped=$((skipped + 1))
      continue
    fi

    command=(
      "${PYTHON_BIN}" -m torch.distributed.run
      --standalone
      --nproc-per-node "${NPROC_PER_NODE}"
      "${TRAIN_DIR}/train.py"
      --model_name_or_path "${model}"
      --tokenizer_name_or_path "${model}"
      --train_file "${train_file}"
      --output_dir "${output_dir}"
      --run_name "${ratio_name}_${variant_name}"
      --trust_remote_code
      --bf16
      --fsdp
      --fsdp_config "${FSDP_CONFIG}"
      --learning_rate 1e-4
      --lr_scheduler_type cosine_with_min_lr
      --lr_scheduler_kwargs '{"min_lr": 1e-5}'
      --per_device_train_batch_size 1
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
      --use_simple_chat_template_if_missing
      --use_lora
      --lora_r 32
      --lora_alpha 64
      --lora_dropout 0.0
      --lora_target_modules all-linear
      --lora_target_parameters "${LORA_TARGET_PARAMETERS}"
      --logging_steps 5
      --save_steps 20
      --save_total_limit 3
    )
    if [[ -n "${EVAL_FILE}" ]]; then
      command+=(
        --eval_file "${EVAL_FILE}"
        --max_eval_samples "${MAX_EVAL_SAMPLES}"
        --per_device_eval_batch_size 1
        --eval_steps 20
      )
    fi
    command+=("${train_args[@]}")

    launched=$((launched + 1))
    echo "[start] ${ratio_name}/${variant_name}" >&2
    echo "        model=${model}" >&2
    echo "        train=${train_file}" >&2
    echo "        output=${output_dir}" >&2
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run]' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2
      continue
    fi

    mkdir -p -- "${output_dir}"
    if "${command[@]}"; then
      echo "[done] ${ratio_name}/${variant_name}" >&2
    else
      status=$?
      failures=$((failures + 1))
      echo "[failed] ${ratio_name}/${variant_name} (exit ${status})" >&2
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit "${status}"
    fi
  done
done

if ((matched == 0)); then
  echo "[error] No ratio/variant directories matched RATIO_FILTER=${RATIO_FILTER} VARIANT_FILTER=${VARIANT_FILTER}" >&2
  exit 1
fi
echo "[done] matched=${matched} launched=${launched} skipped=${skipped} failures=${failures}" >&2
((failures == 0))
