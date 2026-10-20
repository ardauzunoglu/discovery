#!/usr/bin/env bash
# Re-parse saved generations and recompute metrics without calling a model.
# Usage: rescore_matsci_evaluations.sh [EVAL_DIR_OR_ROOT ...] [--overwrite]
# With no paths, every run under outputs/materials_science/evaluation is checked.
# Without --overwrite this is a dry run that only reports metric changes.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"

paths=()
options=()
for argument in "$@"; do
  case "${argument}" in
    -*) options+=("${argument}") ;;
    *) paths+=("${argument}") ;;
  esac
done
if ((${#paths[@]} == 0)); then
  paths=("${repo_dir}/outputs/materials_science/evaluation")
fi

python "${script_dir}/rescore_matsci_evaluations.py" "${paths[@]}" "${options[@]}"
