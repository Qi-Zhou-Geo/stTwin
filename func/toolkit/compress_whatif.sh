# Last Update: Last modified: 2026-08-07T10:49:42
# Author: Qi Zhou


#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/storage/vast-gfz-hpc-01/home/qizhou/3paper/stTwin/pipeline/run_whatif_Rs2d=0.05/v0dot4}"
OUT_FILE="${2:-/storage/vast-gfz-hpc-01/home/qizhou/3paper/stTwin/pipeline/run_whatif_Rs2d=0.05/whatif_selected_nc.tar.gz}"

cd "$BASE_DIR"

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

find . -type f \( \
  -name "climate_forcing.nc" -o \
  -name "hydro_output.nc" -o \
  -name "sed_output.nc" \
\) -print0 > "$tmp_list"

n_files="$(tr -cd '\0' < "$tmp_list" | wc -c | tr -d ' ')"

if [ "$n_files" -eq 0 ]; then
  echo "No selected .nc files found."
  exit 1
fi

echo "Found $n_files files."
echo "Creating: $OUT_FILE"

tar --null -czf "$OUT_FILE" -T "$tmp_list"

echo "Done."
du -h "$OUT_FILE"