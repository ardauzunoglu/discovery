#!/usr/bin/env bash
set -euo pipefail

# Evaluate every merged checkpoint of a training sweep (or an explicit model
# list) at a caller-selected pass@k. CoT mode evaluates stage-2 models;
# candidate-only mode evaluates stage-1 models with the formula-only prompt.
#
# Unrecognized arguments (or everything after --) are forwarded to
# evaluate_matsci_vllm.sh and ultimately evaluate_matsci_vllm.py.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
evaluator_launcher="${EVALUATOR_LAUNCHER:-${script_dir}/evaluate_matsci_vllm.sh}"
model_key_default="${MODEL_KEY:-gpt-oss-20b}"

usage() {
  cat <<'EOF'
Usage: evaluate_matsci_vllm_models.sh [OPTIONS] [-- EVALUATOR_ARGS...]

Options:
  --mode MODE       cot or candidate-only (default: cot)
  --k K             Positive pass@k candidate count (default: 16)
  --all             Run cot and candidate-only at pass@1 and pass@16
  --base-models     Evaluate untrained base models instead of SFT checkpoints
  --reverse         Evaluate the selected model/checkpoint list in reverse order
  --incomplete-only Evaluate only runs that are incomplete for MAX_EXAMPLES
                    (enabled by default; set INCOMPLETE_ONLY=0 to disable)
  --from-incomplete-midpoint
                    Find all incomplete runs and evaluate from the midpoint of
                    that filtered list through its end
  --dry-run         Print the resolved runs without launching the evaluator
  -h, --help        Show this help

Examples:
  # CoT pass@1 (greedy, one candidate)
  ./evaluate_matsci_vllm_models.sh --mode cot --k 1

  # Candidate-only pass@16
  ./evaluate_matsci_vllm_models.sh --mode candidate-only --k 16

  # Select explicit checkpoints and forward an evaluator option
  MODELS_TO_EVALUATE="/path/model_a /path/model_b" \
    ./evaluate_matsci_vllm_models.sh --mode cot --k 1 -- --overwrite

Environment:
  EVALUATION_MODE       Alternative to --mode
  PASS_K                Alternative to --k
  MODELS_TO_EVALUATE    Space- or comma-separated model/checkpoint list
  FINETUNE_ROOTS        Space- or comma-separated merged-checkpoint sweep roots
                        (default: outputs/materials_science/stage{1,2}/merged/MODEL_KEY)
  RATIOS_TO_EVALUATE    Optional ratio names, e.g. "ratio_0p000 ratio_0p100"
  VARIANTS_TO_EVALUATE  Optional variants (default: all three)
  EVALUATION_SUITE_ROOT Output parent (default: outputs/materials_science/evaluation)
  INPUT_PARQUET         Evaluation set (default: the downloaded evaluation split)
  MAX_EXAMPLES          Number of evaluation rows (default: all of them)
  MAX_RUN_ATTEMPTS      Attempts per model before continuing (default: 3)
  RETRY_DELAY_SECONDS   Delay between attempts (default: 30)
  INCOMPLETE_ONLY       Filter to incomplete runs (default: 1; set 0 to disable)
  FROM_INCOMPLETE_MIDPOINT
                        Set to 1 to enable --from-incomplete-midpoint
  EVALUATION_OUTPUT_ROOT, OUTPUT_SUFFIX, and the sampling/engine variables of
  evaluate_matsci_vllm.sh are also honored.
EOF
}

mode="${EVALUATION_MODE:-cot}"
pass_k="${PASS_K:-16}"
all_panels=false
reverse=false
incomplete_only=false
from_incomplete_midpoint=false
dry_run=false
base_models=false
forwarded_args=()
max_run_attempts="${MAX_RUN_ATTEMPTS:-3}"
retry_delay_seconds="${RETRY_DELAY_SECONDS:-30}"

case "${INCOMPLETE_ONLY:-1}" in
  0|false|no) ;;
  1|true|yes) incomplete_only=true ;;
  *)
    echo "INCOMPLETE_ONLY must be 0/1, false/true, or no/yes; got: ${INCOMPLETE_ONLY}" >&2
    exit 2
    ;;
esac

case "${FROM_INCOMPLETE_MIDPOINT:-0}" in
  0|false|no) ;;
  1|true|yes) from_incomplete_midpoint=true ;;
  *)
    echo "FROM_INCOMPLETE_MIDPOINT must be 0/1, false/true, or no/yes; got: ${FROM_INCOMPLETE_MIDPOINT}" >&2
    exit 2
    ;;
esac

if [[ ! "${max_run_attempts}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RUN_ATTEMPTS must be a positive integer; got: ${max_run_attempts}" >&2
  exit 2
fi
if [[ ! "${retry_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a nonnegative integer; got: ${retry_delay_seconds}" >&2
  exit 2
fi

while (($#)); do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires a value" >&2; exit 2; }
      mode="$2"
      shift 2
      ;;
    --mode=*)
      mode="${1#*=}"
      shift
      ;;
    --k)
      [[ $# -ge 2 ]] || { echo "--k requires a value" >&2; exit 2; }
      pass_k="$2"
      shift 2
      ;;
    --k=*)
      pass_k="${1#*=}"
      shift
      ;;
    --all)
      all_panels=true
      shift
      ;;
    --reverse)
      reverse=true
      shift
      ;;
    --incomplete-only)
      incomplete_only=true
      shift
      ;;
    --from-incomplete-midpoint)
      from_incomplete_midpoint=true
      shift
      ;;
    --base-models)
      base_models=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      forwarded_args+=("$@")
      break
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if ${all_panels}; then
  if ${base_models}; then
    echo "--all cannot be combined with --base-models" >&2
    exit 2
  fi
  shared_args=()
  ${reverse} && shared_args+=(--reverse)
  ${incomplete_only} && shared_args+=(--incomplete-only)
  ${from_incomplete_midpoint} && shared_args+=(--from-incomplete-midpoint)
  ${dry_run} && shared_args+=(--dry-run)
  panel_modes=(cot candidate-only cot candidate-only)
  panel_ks=(1 1 16 16)
  failed_panels=0
  for ((panel_index=0; panel_index<${#panel_modes[@]}; panel_index++)); do
    if bash "${script_dir}/evaluate_matsci_vllm_models.sh" \
        --mode "${panel_modes[panel_index]}" \
        --k "${panel_ks[panel_index]}" \
        "${shared_args[@]}" \
        -- "${forwarded_args[@]}"; then
      echo "==> Panel completed: mode=${panel_modes[panel_index]} pass@${panel_ks[panel_index]}" >&2
    else
      panel_exit_code=$?
      ((failed_panels += 1))
      echo "[error] Panel finished with failures: mode=${panel_modes[panel_index]} pass@${panel_ks[panel_index]} exit=${panel_exit_code}; continuing" >&2
    fi
  done
  if ((failed_panels > 0)); then
    echo "[error] Matrix completed with ${failed_panels} panel(s) containing failed runs" >&2
    exit 1
  fi
  echo "==> All four panels completed successfully" >&2
  exit 0
fi

case "${mode}" in
  cot)
    mode_label="cot"
    ;;
  candidate-only|candidate_only|nocot|no-cot)
    mode="candidate-only"
    mode_label="nocot"
    ;;
  *)
    echo "--mode must be cot or candidate-only; got: ${mode}" >&2
    exit 2
    ;;
esac

if ${base_models} && [[ "${mode}" != cot ]]; then
  echo "--base-models is only supported with --mode cot" >&2
  exit 2
fi

if [[ ! "${pass_k}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--k must be a positive integer; got: ${pass_k}" >&2
  exit 2
fi

export INPUT_PARQUET="${INPUT_PARQUET:-${repo_dir}/data/materials_science/data/evaluation/test.parquet}"
if [[ ! -f "${INPUT_PARQUET}" ]]; then
  echo "Evaluation parquet not found: ${INPUT_PARQUET} (run download_data.sh first)" >&2
  exit 1
fi
if [[ -z "${MAX_EXAMPLES:-}" ]]; then
  MAX_EXAMPLES="$(python -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "${INPUT_PARQUET}")"
fi
export MAX_EXAMPLES

# An explicit model list always wins. Otherwise candidate-only mode evaluates
# the merged stage-1 sweep and CoT mode the merged stage-2 sweep;
# --base-models selects the untrained models.
models_text="${MODELS_TO_EVALUATE:-}"
finetune_roots_text="${FINETUNE_ROOTS:-}"
if [[ -z "${models_text}" && -z "${finetune_roots_text}" ]] && ! ${base_models}; then
  if [[ "${mode}" == candidate-only ]]; then
    finetune_roots_text="${repo_dir}/outputs/materials_science/stage1/merged/${model_key_default}"
  else
    finetune_roots_text="${repo_dir}/outputs/materials_science/stage2/merged/${model_key_default}"
  fi
fi

models=()
if [[ -n "${models_text}" ]]; then
  models_text="${models_text//,/ }"
  read -r -a models <<< "${models_text}"
elif [[ -n "${finetune_roots_text}" ]]; then
  finetune_roots_text="${finetune_roots_text//,/ }"
  read -r -a finetune_roots <<< "${finetune_roots_text}"
  variants_text="${VARIANTS_TO_EVALUATE:-variant_000 variant_001 variant_002}"
  variants_text="${variants_text//,/ }"
  read -r -a variants <<< "${variants_text}"
  if ((${#variants[@]} == 0)); then
    echo "VARIANTS_TO_EVALUATE must name at least one variant" >&2
    exit 2
  fi

  for finetune_root in "${finetune_roots[@]}"; do
    if [[ ! -d "${finetune_root}" ]]; then
      echo "Merged checkpoint root does not exist: ${finetune_root}" >&2
      exit 1
    fi
    finetune_root="$(cd -- "${finetune_root}" && pwd)"
    ratios_text="${RATIOS_TO_EVALUATE:-}"
    ratio_dirs=()
    if [[ -n "${ratios_text}" ]]; then
      ratios_text="${ratios_text//,/ }"
      read -r -a ratio_names <<< "${ratios_text}"
      for ratio_name in "${ratio_names[@]}"; do
        ratio_dirs+=("${finetune_root}/${ratio_name}")
      done
    else
      shopt -s nullglob
      ratio_dirs=("${finetune_root}"/ratio_*)
      shopt -u nullglob
    fi

    for ratio_dir in "${ratio_dirs[@]}"; do
      for variant in "${variants[@]}"; do
        checkpoint="${ratio_dir}/${variant}"
        [[ -d "${checkpoint}" ]] && models+=("${checkpoint}")
      done
    done
  done
else
  models=(
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
  )
fi

if ${reverse}; then
  reversed_models=()
  for ((index=${#models[@]} - 1; index >= 0; index--)); do
    reversed_models+=("${models[index]}")
  done
  models=("${reversed_models[@]}")
fi

if ((${#models[@]} == 0)); then
  echo "No models/checkpoints selected" >&2
  [[ -n "${finetune_roots_text}" ]] && echo "Searched under: ${finetune_roots_text}" >&2
  exit 1
fi

prompt_args=()
if [[ "${mode}" == candidate-only ]]; then
  prompt_args=(
    --include-cif
    --cif-prompt-layout "${CIF_PROMPT_LAYOUT:-legacy}"
    --candidate-only-prompt
  )
fi

sampling_args=()
if [[ "${pass_k}" == 1 ]]; then
  export NUM_CANDIDATES=1
  export TEMPERATURE=0.0
  export TOP_P=1.0
  export TOP_K=-1
  export MIN_P=0.0
  export REQUEST_CHUNK_SIZE="${REQUEST_CHUNK_SIZE:-1000}"
  export SAVE_EVERY="${SAVE_EVERY:-1000}"
else
  export NUM_CANDIDATES="${pass_k}"
  export TEMPERATURE="${TEMPERATURE:-1.0}"
  export TOP_P="${TOP_P:-1.0}"
  export TOP_K="${TOP_K:--1}"
  export MIN_P="${MIN_P:-0.0}"
fi

sampling_args=(
  --n "${NUM_CANDIDATES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --min-p "${MIN_P}"
  --repetition-penalty "${REPETITION_PENALTY:-1.0}"
  --presence-penalty "${PRESENCE_PENALTY:-0.0}"
  --frequency-penalty "${FREQUENCY_PENALTY:-0.0}"
  --max-tokens "${MAX_NEW_TOKENS:-8192}"
  --seed "${SEED:-42}"
)

evaluation_suite_root="${EVALUATION_SUITE_ROOT:-${repo_dir}/outputs/materials_science/evaluation}"
if ${base_models}; then
  default_output_root="${evaluation_suite_root}/p${pass_k}-no-train"
else
  default_output_root="${evaluation_suite_root}/p${pass_k}-${mode_label}-train"
fi
if [[ "${pass_k}" == 1 && "${mode}" == candidate-only ]]; then
  default_output_suffix="-candidate-only-deterministic-n1"
elif [[ "${pass_k}" == 1 ]]; then
  default_output_suffix="-cif-deterministic-n1"
elif [[ "${mode}" == candidate-only ]]; then
  default_output_suffix="-candidate-only"
else
  default_output_suffix="-cif"
fi
evaluation_output_root="${EVALUATION_OUTPUT_ROOT:-${default_output_root}}"
output_suffix="${OUTPUT_SUFFIX:-${default_output_suffix}}"
if ${base_models}; then
  output_suffix="${OUTPUT_SUFFIX:-}"
fi

model_key() {
  local model_path="$1"
  case "${model_path}" in
    *gpt-oss-120b*|*gpt_oss_120b*) printf '%s\n' gpt_oss_120b ;;
    *gpt-oss-20b*|*gpt_oss_20b*) printf '%s\n' gpt_oss_20b ;;
    *gemma-3-27b*|*gemma3_27b*) printf '%s\n' gemma3_27b ;;
    *gemma-3-12b*|*gemma3_12b*) printf '%s\n' gemma3_12b ;;
    *)
      local fallback="${model_path##*/}"
      printf '%s\n' "${fallback//[^[:alnum:]_.-]/_}"
      ;;
  esac
}

model_output_dir() {
  local model_path="$1"
  local key ratio_name variant_name model_name
  key="$(model_key "${model_path}")"
  variant_name="${model_path##*/}"
  ratio_name="$(basename -- "$(dirname -- "${model_path}")")"
  model_name="${model_path##*/}"

  # Released checkpoints encode the sweep coordinates in their folder name,
  # e.g. gpt-oss-20b-stage2-ratio0.3-v0.
  if [[ "${model_name}" =~ -ratio([0-9.]+)-v([0-9]+)$ ]]; then
    ratio_name="ratio_$(printf '%.3f' "${BASH_REMATCH[1]}" | tr . p)"
    variant_name="variant_$(printf '%03d' "${BASH_REMATCH[2]}")"
  fi
  if [[ "${ratio_name}" == ratio_* && "${variant_name}" == variant_* ]]; then
    printf '%s/%s/%s/%s%s\n' \
      "${evaluation_output_root}" "${key}" "${ratio_name}" \
      "${variant_name}" "${output_suffix}"
  else
    printf '%s/%s%s\n' "${evaluation_output_root}" "${key}" "${output_suffix}"
  fi
}

run_is_complete() {
  local output_dir="$1"
  local metrics_path="${output_dir}/metrics.json"
  [[ -f "${metrics_path}" ]] || return 1

  python - "${metrics_path}" "${MAX_EXAMPLES}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        metrics = json.load(handle)
    requested = int(sys.argv[2])
    completed = int(metrics.get("completed_examples", -1))
    selected = int(metrics.get("selected_examples", -1))
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)

complete = (
    metrics.get("status") == "complete"
    and completed == selected
    and selected >= requested
)
raise SystemExit(0 if complete else 1)
PY
}

if ${incomplete_only} || ${from_incomplete_midpoint}; then
  incomplete_models=()
  for model in "${models[@]}"; do
    output_dir="$(model_output_dir "${model}")"
    if ! run_is_complete "${output_dir}"; then
      incomplete_models+=("${model}")
    fi
  done

  incomplete_count="${#incomplete_models[@]}"
  if ((incomplete_count == 0)); then
    echo "==> All ${#models[@]} selected runs are complete for ${MAX_EXAMPLES} examples; nothing to do" >&2
    exit 0
  fi

  if ${from_incomplete_midpoint}; then
    # Use the lower middle for an even-sized list: 16 incomplete runs start at
    # the 8th run (array index 7), matching the intended overlapping midpoint.
    midpoint_index=$(((incomplete_count - 1) / 2))
    midpoint_ordinal=$((midpoint_index + 1))
    models=("${incomplete_models[@]:midpoint_index}")
    echo "==> Found ${incomplete_count} incomplete run(s); starting at incomplete run ${midpoint_ordinal}: ${models[0]}" >&2
    echo "==> Evaluating ${#models[@]} incomplete run(s) from the midpoint through the end" >&2
  else
    models=("${incomplete_models[@]}")
    echo "==> Found ${incomplete_count} incomplete run(s); skipping completed runs" >&2
  fi
fi

echo "==> Evaluation preset: mode=${mode_label} pass@${pass_k}" >&2
echo "==> Output root: ${evaluation_output_root}" >&2
echo "==> Models: ${#models[@]}" >&2
echo "==> Dataset prefix: rows 1-${MAX_EXAMPLES}" >&2
echo "==> Retry policy: attempts=${max_run_attempts} delay=${retry_delay_seconds}s; exhausted runs are recorded and skipped" >&2
echo "==> Decoding: n=${NUM_CANDIDATES} temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} min_p=${MIN_P}" >&2

failed_runs=()
for model in "${models[@]}"; do
  output_dir="$(model_output_dir "${model}")"
  echo "==> Evaluating ${model} -> ${output_dir}" >&2
  command=(
    "${evaluator_launcher}"
    "${prompt_args[@]}"
    "${sampling_args[@]}"
    "${forwarded_args[@]}"
  )
  if ${dry_run}; then
    printf 'MAX_EXAMPLES=%q MODEL=%q OUTPUT_DIR=%q' \
      "${MAX_EXAMPLES}" "${model}" "${output_dir}"
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    run_succeeded=false
    run_exit_code=0
    for ((attempt=1; attempt<=max_run_attempts; attempt++)); do
      echo "==> Attempt ${attempt}/${max_run_attempts}: ${model}" >&2
      if MODEL="${model}" \
          OUTPUT_DIR="${output_dir}" \
          "${command[@]}"; then
        run_succeeded=true
        break
      else
        run_exit_code=$?
      fi
      if ((attempt < max_run_attempts)); then
        echo "[warning] Run failed with exit ${run_exit_code}; retrying in ${retry_delay_seconds}s: ${model}" >&2
        sleep "${retry_delay_seconds}"
      fi
    done
    if [[ "${run_succeeded}" != true ]]; then
      failed_runs+=("${model}|${output_dir}|${run_exit_code}")
      echo "[error] Exhausted ${max_run_attempts} attempts; continuing to next model: ${model}" >&2
    fi
  fi
done

if ((${#failed_runs[@]} > 0)); then
  echo "[error] ${#failed_runs[@]} model run(s) failed in mode=${mode_label} pass@${pass_k}:" >&2
  for failure in "${failed_runs[@]}"; do
    IFS='|' read -r failed_model failed_output failed_code <<< "${failure}"
    echo "  exit=${failed_code} model=${failed_model} output=${failed_output}" >&2
  done
  exit 1
fi

echo "==> Panel completed without failed runs: mode=${mode_label} pass@${pass_k}" >&2
