#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
# All tensor-parallel workers run on one node. Avoid stale cluster hostname
# addresses when PyTorch creates its auxiliary Gloo process group.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
evaluator_script="${EVALUATOR_SCRIPT:-${script_dir}/evaluate_matsci_vllm.py}"
model="${MODEL:-openai/gpt-oss-20b}"
input_parquet="${INPUT_PARQUET:-${repo_dir}/data/materials_science/data/evaluation/test.parquet}"
model_name="${model%/}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/materials_science/evaluation/${model_name##*/}}"

# Local checkpoint names need not contain the model family. Prefer the saved architecture metadata for preset selection, while keeping
# path-based matching for remote model IDs and older checkpoint layouts.
preset_model="${model}"
if [[ -f "${model}/config.json" ]]; then
  model_type="$(python - "${model}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("model_type", ""))
PY
)"
  case "${model_type}" in
    gpt_oss|llama|gemma*) preset_model="${model_type}" ;;
  esac
fi

case "${preset_model}" in
  *gpt*|*gpt-oss-20b*|*gpt-oss-120b*)
    temperature=1.0; top_p=1.0; top_k=-1; reasoning_effort=medium; max_new_tokens=8192
    ;;
  *llama*|*Llama-3.1-8B-Instruct*|*llama-3.1-8b-instruct*|*Llama-3.1-70B-Instruct*|*llama-3.1-70b-instruct*)
    temperature=0.6; top_p=0.9; top_k=-1; reasoning_effort=""; max_new_tokens=8192
    ;;
  *gemma*|*gemma-3-27b-it*|*gemma-3-12b-it*)
    temperature=1.0; top_p=1.0; top_k=-1; reasoning_effort=""; max_new_tokens=8192
    ;;
  *)
    echo "No evaluation preset for model: ${model}" >&2
    exit 2
    ;;
esac

# Every preset can be overridden from the environment for cluster sweeps.
temperature="${TEMPERATURE:-${temperature}}"
top_p="${TOP_P:-${top_p}}"
top_k="${TOP_K:-${top_k}}"
max_new_tokens="${MAX_NEW_TOKENS:-${max_new_tokens}}"
num_candidates="${NUM_CANDIDATES:-16}"
seed="${SEED:-42}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-4}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.90}"
request_chunk_size="${REQUEST_CHUNK_SIZE:-1024}"
save_every="${SAVE_EVERY:-100}"
cif_prompt_layout="${CIF_PROMPT_LAYOUT:-sft_task_first}"

reasoning_args=()
if [[ -n "${reasoning_effort}" ]]; then
  reasoning_args=(--reasoning-effort "${reasoning_effort}")
fi
# By default every row of the evaluation set is used.
selection_args=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  selection_args=(--max-examples "${MAX_EXAMPLES}")
fi

python "${evaluator_script}" \
  --input-parquet "${input_parquet}" \
  --model "${model}" \
  --output-dir "${output_dir}" \
  --chat-template-source model \
  "${reasoning_args[@]}" \
  "${selection_args[@]}" \
  --dtype bfloat16 \
  --max-tokens "${max_new_tokens}" \
  --temperature "${temperature}" \
  --top-p "${top_p}" \
  --top-k "${top_k}" \
  --repetition-penalty 1.0 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --n "${num_candidates}" \
  --seed "${seed}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --request-chunk-size "${request_chunk_size}" \
  --save-every "${save_every}" \
  --disable-custom-all-reduce \
  --enforce-eager \
  --include-cif \
  --cif-prompt-layout "${cif_prompt_layout}" \
  --resume \
  "$@"
