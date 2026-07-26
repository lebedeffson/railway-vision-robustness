#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-data/new_scenes_sources}"
OUT="${2:-data/new_scenes_extracted}"
mkdir -p "$OUT"

if [ -f "$ROOT/railgoerl24/Annotated_RGB_data.7z" ]; then
  command -v 7z >/dev/null 2>&1 || { echo "7z is required" >&2; exit 2; }
  7z t "$ROOT/railgoerl24/Annotated_RGB_data.7z"
  mkdir -p "$OUT/railgoerl24"
  7z x -aos -o"$OUT/railgoerl24" "$ROOT/railgoerl24/Annotated_RGB_data.7z"
fi

for name in annotations images; do
  z="$ROOT/railbench_object/${name}.zip"
  if [ -f "$z" ]; then
    command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 2; }
    unzip -t "$z"
    mkdir -p "$OUT/railbench_object/$name"
    unzip -n "$z" -d "$OUT/railbench_object/$name"
  fi
done

echo "Extraction finished. Run the project CPU audit before accepting any scene."
