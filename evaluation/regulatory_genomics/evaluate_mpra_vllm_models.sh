#!/usr/bin/env bash
set -euo pipefail

# Evaluate every merged checkpoint of a training sweep (or an explicit model
# list). CoT mode evaluates stage-2 models; candidate-only mode evaluates
# stage-1 models. Only the structured final answer is scored.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
evaluator_launcher="${EVALUATOR_LAUNCHER:-${script_dir}/evaluate_mpra_vllm.sh}"
model_key_default="${MODEL_KEY:-gpt-oss-20b}"

usage() {
  cat <<'EOF'
Usage: evaluate_mpra_vllm_models.sh [OPTIONS] [-- EVALUATION_ARGS...]

  --mode MODE       cot or candidate-only (default: cot)
  --k K             1 or 16 (default: 16)
  --pass-at-1       Backward-compatible alias for --k 1
  --pass-at-16      Backward-compatible alias for --k 16
  --start-index N   Zero-based first parquet row (default: 0)
  --max-examples N  Number of consecutive rows (default: all rows)
  --resume           Continue partial runs and skip complete runs (default)
  --no-resume        Do not continue or skip existing runs
  --reverse         Evaluate the selected models in reverse order
  --dry-run         Print resolved runs without launching vLLM

Environment overrides: EVALUATION_MODE, PASS_K, MODELS_TO_EVALUATE,
FINETUNE_ROOTS (default: outputs/regulatory_genomics/stage{1,2}/merged/MODEL_KEY),
RATIOS_TO_EVALUATE, VARIANTS_TO_EVALUATE, and EVALUATION_OUTPUT_ROOT.
INPUT_PARQUET overrides the downloaded evaluation split.

Examples:
  ./evaluate_mpra_vllm_models.sh --mode cot --k 1
  ./evaluate_mpra_vllm_models.sh --mode candidate-only --k 16
EOF
}

mode="${EVALUATION_MODE:-cot}"
pass_k="${PASS_K:-16}"
start_index=0; max_examples=""; resume=true; reverse=false; dry_run=false; evaluation_args=()
while (($#)); do
  case "$1" in
    --mode) [[ $# -ge 2 ]] || { echo "--mode requires a value" >&2; exit 2; }; mode="$2"; shift 2 ;;
    --mode=*) mode="${1#*=}"; shift ;;
    --k) [[ $# -ge 2 ]] || { echo "--k requires a value" >&2; exit 2; }; pass_k="$2"; shift 2 ;;
    --k=*) pass_k="${1#*=}"; shift ;;
    --pass-at-1) pass_k=1; shift ;;
    --pass-at-16) pass_k=16; shift ;;
    --start-index) [[ $# -ge 2 ]] || { echo "--start-index requires a value" >&2; exit 2; }; start_index="$2"; shift 2 ;;
    --start-index=*) start_index="${1#*=}"; shift ;;
    --max-examples) [[ $# -ge 2 ]] || { echo "--max-examples requires a value" >&2; exit 2; }; max_examples="$2"; shift 2 ;;
    --max-examples=*) max_examples="${1#*=}"; shift ;;
    --resume) resume=true; shift ;;
    --no-resume) resume=false; shift ;;
    --reverse) reverse=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; evaluation_args+=("$@"); break ;;
    *) evaluation_args+=("$1"); shift ;;
  esac
done

case "${mode}" in
  cot) mode_label=cot ;;
  candidate-only|candidate_only|nocot|no-cot) mode=candidate-only; mode_label=nocot ;;
  *) echo "--mode must be cot or candidate-only; got: ${mode}" >&2; exit 2 ;;
esac
case "${pass_k}" in 1|16) ;; *) echo "--k must be 1 or 16; got: ${pass_k}" >&2; exit 2 ;; esac
[[ "${start_index}" =~ ^[0-9]+$ ]] || { echo "--start-index must be a nonnegative integer" >&2; exit 2; }
input_parquet="${INPUT_PARQUET:-${repo_dir}/data/regulatory_genomics/data/evaluation/test.parquet}"
[[ -f "${input_parquet}" ]] || { echo "Evaluation parquet not found: ${input_parquet} (run download_data.sh first)" >&2; exit 1; }
export INPUT_PARQUET="${input_parquet}"
if [[ -z "${max_examples}" ]]; then
  total_rows="$(python -c 'import sys, pyarrow.parquet as pq; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' "${input_parquet}")"
  max_examples=$((total_rows - start_index))
fi
[[ "${max_examples}" =~ ^[1-9][0-9]*$ ]] || { echo "--max-examples must be a positive integer" >&2; exit 2; }
for argument in "${evaluation_args[@]}"; do
  case "${argument%%=*}" in
    --n|--temperature|--max-tokens|--save-every) echo "${argument%%=*} is controlled by --mode/--k" >&2; exit 2 ;;
  esac
done

models_text="${MODELS_TO_EVALUATE:-}"; roots_text="${FINETUNE_ROOTS:-}"
if [[ -z "${models_text}" && -z "${roots_text}" ]]; then
  if [[ "${mode}" == cot ]]; then
    roots_text="${repo_dir}/outputs/regulatory_genomics/stage2/merged/${model_key_default}"
  else
    roots_text="${repo_dir}/outputs/regulatory_genomics/stage1/merged/${model_key_default}"
  fi
fi

models=()
if [[ -n "${models_text}" ]]; then
  models_text="${models_text//,/ }"; read -r -a models <<< "${models_text}"
else
  roots_text="${roots_text//,/ }"; read -r -a roots <<< "${roots_text}"
  variants_text="${VARIANTS_TO_EVALUATE:-variant_000 variant_001 variant_002}"
  variants_text="${variants_text//,/ }"; read -r -a variants <<< "${variants_text}"
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || { echo "Fine-tune root does not exist: ${root}" >&2; exit 1; }
    if [[ -n "${RATIOS_TO_EVALUATE:-}" ]]; then
      ratios_text="${RATIOS_TO_EVALUATE//,/ }"; read -r -a ratio_names <<< "${ratios_text}"
      ratio_dirs=(); for ratio in "${ratio_names[@]}"; do ratio_dirs+=("${root}/${ratio}"); done
    else
      shopt -s nullglob; ratio_dirs=("${root}"/ratio_*); shopt -u nullglob
    fi
    for ratio_dir in "${ratio_dirs[@]}"; do
      for variant in "${variants[@]}"; do
        candidate="${ratio_dir}/${variant}"
        [[ -d "${candidate}" ]] || continue
        shopt -s nullglob
        model_weights=("${candidate}"/model*.safetensors "${candidate}"/pytorch_model*.bin)
        shopt -u nullglob
        if [[ ! -f "${candidate}/config.json" || ${#model_weights[@]} -eq 0 ]]; then
          echo "[skip] Incomplete merged model: ${candidate}" >&2
          continue
        fi
        models+=("${candidate}")
      done
    done
  done
fi
((${#models[@]})) || { echo "No ${mode_label} checkpoints found under: ${roots_text:-MODELS_TO_EVALUATE}" >&2; exit 1; }
if ${reverse}; then
  reversed_models=()
  for ((index=${#models[@]} - 1; index >= 0; index--)); do
    reversed_models+=("${models[index]}")
  done
  models=("${reversed_models[@]}")
fi

prompt_args=()
if [[ "${mode}" == cot ]]; then
  max_tokens="${COT_MAX_TOKENS:-8192}"
  prompt_args=(--cot-prompt)
else
  max_tokens="${NOCOT_MAX_TOKENS:-256}"
fi
if [[ "${pass_k}" == 1 ]]; then
  mode_args=(--n 1 --temperature 0.0 --max-tokens "${max_tokens}" --save-every 1000)
else
  mode_args=(--n 16 --max-tokens "${max_tokens}" --save-every 100)
fi
first_row=$((start_index + 1)); last_row=$((start_index + max_examples))
output_root="${EVALUATION_OUTPUT_ROOT:-${repo_dir}/outputs/regulatory_genomics/evaluation/p${pass_k}-${mode_label}-rows-${first_row}-${last_row}}"

echo "==> Evaluation preset: mode=${mode_label} pass@${pass_k}; models=${#models[@]}" >&2
for model in "${models[@]}"; do
  variant="${model##*/}"; ratio="$(basename -- "$(dirname -- "${model}")")"
  family="$(basename -- "$(dirname -- "$(dirname -- "${model}")")")"
  if [[ "${ratio}" == ratio_* && "${variant}" == variant_* ]]; then
    output_dir="${output_root}/${family}/${ratio}/${variant}"
  else
    model_key="${model##*/}"; model_key="${model_key//[^[:alnum:]_.-]/_}"
    output_dir="${output_root}/${model_key}"
  fi

  resume_args=(--no-resume)
  resume_tensor_parallel_size=""
  resume_model=""
  if ${resume}; then
    resume_args=(--resume)
    progress=unknown
    if [[ -f "${output_dir}/predictions.jsonl" && -f "${output_dir}/run_config.json" ]]; then
      resume_details="$({ python - "${output_dir}" "${start_index}" "${max_examples}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
start_index = int(sys.argv[2])
max_examples = int(sys.argv[3])

try:
    config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    selected = config.get("input", {})
    tensor_parallel_size = config.get("engine", {}).get("tensor_parallel_size", "")
    metrics_path = output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    completed = metrics.get("examples_completed", "?")
    total = metrics.get("examples_total", selected.get("selected_rows", "?"))
    compatible_selection = (
        selected.get("selected_start_index") == start_index
        and selected.get("selected_rows") == max_examples
    )
    is_complete = (
        compatible_selection
        and metrics.get("status") == "complete"
        and completed == total == max_examples
    )
    state = "complete" if is_complete else "partial"
    print(f"{state}\t{completed}\t{total}\t{tensor_parallel_size}\t{config.get('model', '')}")
except Exception:
    print("unknown\t?\t?\t\t")
PY
      } 2>/dev/null)"
      IFS=$'\t' read -r progress completed_examples total_examples resume_tensor_parallel_size resume_model <<< "${resume_details}"
    fi

    if [[ "${progress}" == complete ]]; then
      echo "[skip complete] ${model} (${completed_examples}/${total_examples}) -> ${output_dir}" >&2
      continue
    elif [[ "${progress}" == partial ]]; then
      echo "[resume] ${model} (${completed_examples}/${total_examples} complete) -> ${output_dir}" >&2
    elif [[ -f "${output_dir}/predictions.jsonl" ]]; then
      echo "[resume] ${model} (existing predictions; evaluator will validate them) -> ${output_dir}" >&2
    fi
  fi

  command=("${evaluator_launcher}" "${evaluation_args[@]}" --start-index "${start_index}" --max-examples "${max_examples}" "${prompt_args[@]}" "${mode_args[@]}" "${resume_args[@]}")
  # Engine settings are part of the evaluator's resume compatibility check.
  # Preserve the setting used by an existing partial run even if launcher
  # defaults have changed since that run started.
  if [[ -n "${resume_tensor_parallel_size}" ]]; then
    command+=(--tensor-parallel-size "${resume_tensor_parallel_size}")
  fi
  launch_model="${resume_model:-${model}}"
  if ${dry_run}; then
    printf 'MODEL=%q OUTPUT_DIR=%q' "${launch_model}" "${output_dir}"; printf ' %q' "${command[@]}"; printf '\n'
  else
    echo "==> Evaluating ${model} -> ${output_dir}" >&2
    MODEL="${launch_model}" OUTPUT_DIR="${output_dir}" "${command[@]}"
  fi
done
