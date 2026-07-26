#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/new_scenes_sources}"
mkdir -p "$ROOT/railgoerl24" "$ROOT/railbench_object" "$ROOT/raileye3d"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 2; }; }
need curl
need sha256sum

fetch() {
  local url="$1"
  local out="$2"
  echo "Downloading: $url"
  curl -L -C - --fail --retry 20 --retry-delay 5 --connect-timeout 30 -o "$out" "$url"
  sha256sum "$out" | tee "$out.sha256"
}

# CC0 source. Raw frames remain local by project policy.
fetch \
  "https://download.data.fid-move.de/dzsf/railgoerl24/Annotated_RGB_data.7z" \
  "$ROOT/railgoerl24/Annotated_RGB_data.7z"

# Public annotation repository; image data still requires a separate email request.
if command -v git >/dev/null 2>&1; then
  if [ ! -d "$ROOT/raileye3d/annotations/.git" ]; then
    git clone --depth 1 https://github.com/raileye3d/raileye3d_dataset.git "$ROOT/raileye3d/annotations"
  else
    git -C "$ROOT/raileye3d/annotations" pull --ff-only
  fi
fi

# RAIL-BENCH is optional/static and requires explicit terms acceptance.
if [ "${I_ACCEPT_RAILBENCH_NONCOMMERCIAL_TERMS:-NO}" = "YES" ]; then
  fetch "https://www.mrt.kit.edu/railbench_data/object_and_rail/data_overview.csv" \
        "$ROOT/railbench_object/data_overview.csv"
  fetch "https://www.mrt.kit.edu/railbench_data/object_and_rail/annotations.zip" \
        "$ROOT/railbench_object/annotations.zip"
  fetch "https://www.mrt.kit.edu/railbench_data/object_and_rail/images.zip" \
        "$ROOT/railbench_object/images.zip"
else
  echo "RAIL-BENCH skipped: set I_ACCEPT_RAILBENCH_NONCOMMERCIAL_TERMS=YES after reviewing official terms." >&2
fi

echo "Downloads completed. Do not commit raw archives or extracted images."
