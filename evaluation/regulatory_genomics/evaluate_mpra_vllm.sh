#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
model="${MODEL:-openai/gpt-oss-20b}"
model_name="${model%/}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/regulatory_genomics/evaluation/${model_name##*/}}"
input_parquet="${INPUT_PARQUET:-${repo_dir}/data/regulatory_genomics/data/evaluation/test.parquet}"
# The answer format is read from the Parquet metadata unless overridden.
answer_format="${ANSWER_FORMAT:-auto}"

case "${model}" in
  *gpt-oss-20b*|*gpt_oss_20b*)
    temperature=1.0
    top_p=1.0
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=medium
    max_new_tokens=256
    ;;
  *gpt-oss-120b*)
    temperature=1.0
    top_p=1.0
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=medium
    max_new_tokens=256
    ;;
  *Llama-3.1-8B-Instruct*|*llama-3.1-8b-instruct*)
    temperature=0.6
    top_p=0.9
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=""
    max_new_tokens=256
    ;;
  *Llama-3.1-70B-Instruct*|*llama-3.1-70b-instruct*)
    temperature=0.6
    top_p=0.9
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=""
    max_new_tokens=256
    ;;
  *gemma-3-27b-it*)
    temperature=1.0
    top_p=1.0
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=""
    max_new_tokens=256
    ;;
  *gemma-3-12b-it*|*gemma_3_12b_it*)
    temperature=1.0
    top_p=1.0
    top_k=-1
    repetition_penalty=1.0
    presence_penalty=0.0
    frequency_penalty=0.0
    reasoning_effort=""
    max_new_tokens=256
    ;;
  *)
    echo "No evaluation preset for model: ${model}" >&2
    exit 2
    ;;
esac

reasoning_args=()
if [[ -n "${reasoning_effort}" ]]; then
  reasoning_args=(--reasoning-effort "${reasoning_effort}")
fi
# By default every row of the evaluation set is used.
selection_args=()
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  selection_args=(--max-examples "${MAX_EXAMPLES}")
fi

num_candidates=16
seed=42

# vLLM throughput settings
tensor_parallel_size=1
gpu_memory_utilization=0.90
request_chunk_size=1024
save_every=100

python "${script_dir}/evaluate_mpra_vllm.py" \
  --input-parquet "${input_parquet}" \
  --answer-format "${answer_format}" \
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
  --repetition-penalty "${repetition_penalty}" \
  --presence-penalty "${presence_penalty}" \
  --frequency-penalty "${frequency_penalty}" \
  --n "${num_candidates}" \
  --seed "${seed}" \
  --tensor-parallel-size "${tensor_parallel_size}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --request-chunk-size "${request_chunk_size}" \
  --save-every "${save_every}" \
  --disable-custom-all-reduce \
  --enforce-eager \
  "$@"
