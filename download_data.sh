#!/usr/bin/env bash
# Download the training and evaluation data into ./data.
# Usage: ./download_data.sh [materials_science] [regulatory_genomics]  (default: both)

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
domains=("$@")
if ((${#domains[@]} == 0)); then
  domains=(materials_science regulatory_genomics)
fi

for domain in "${domains[@]}"; do
  case "${domain}" in
    materials_science) repo_id=ardauzunoglu/materials-discovery ;;
    regulatory_genomics) repo_id=ardauzunoglu/regulatory-genomics ;;
    *) echo "Unknown domain: ${domain}" >&2; exit 2 ;;
  esac
  echo "==> ${repo_id} -> data/${domain}" >&2
  hf download "${repo_id}" --repo-type dataset --local-dir "${repo_dir}/data/${domain}"
done
