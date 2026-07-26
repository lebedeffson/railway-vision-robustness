#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${1:-$PROJECT_ROOT/data/new_scenes_sources}"
RAILGOERL_HOST="download.data.fid-move.de"
RAILGOERL_URL="https://$RAILGOERL_HOST/dzsf/railgoerl24/Annotated_RGB_data.7z"
RAILGOERL_BYTES=4055218180
MINIMUM_FREE_BYTES=$((12 * 1024 * 1024 * 1024))

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 2
  }
}

need curl
need git
need sha256sum
need python

available="$(df --output=avail -B1 "$PROJECT_ROOT" | tail -1 | tr -d ' ')"
if (( available < MINIMUM_FREE_BYTES )); then
  echo "Insufficient free space: $available bytes" >&2
  exit 3
fi

mkdir -p "$SOURCE_ROOT/railgoerl24" "$SOURCE_ROOT/raileye3d"

resolve_args=()
if ! getent ahosts "$RAILGOERL_HOST" >/dev/null 2>&1; then
  resolved_ip="$(
    curl -fsS --connect-timeout 15 --max-time 30 \
      -H "accept: application/dns-json" \
      "https://cloudflare-dns.com/dns-query?name=$RAILGOERL_HOST&type=A" |
      python -c '
import json, sys
payload = json.load(sys.stdin)
addresses = [
    row["data"] for row in payload.get("Answer", [])
    if row.get("type") == 1
]
if not addresses:
    raise SystemExit("No verified A record returned")
print(addresses[0])
'
  )"
  resolve_args=(--resolve "$RAILGOERL_HOST:443:$resolved_ip")
fi

annotation_root="$SOURCE_ROOT/raileye3d/annotations"
if [[ ! -d "$annotation_root/.git" ]]; then
  git clone --depth 1 \
    https://github.com/raileye3d/raileye3d_dataset.git \
    "$annotation_root"
else
  git -C "$annotation_root" pull --ff-only
fi
git -C "$annotation_root" rev-parse HEAD |
  tee "$SOURCE_ROOT/raileye3d/ANNOTATIONS_COMMIT.txt"

archive="$SOURCE_ROOT/railgoerl24/Annotated_RGB_data.7z"
download="$archive.download"
if [[ -e "$download" ]]; then
  echo "Incomplete non-resumable download already exists: $download" >&2
  echo "Preserve or move it before starting a fresh attempt." >&2
  exit 4
fi

if ! curl "${resolve_args[@]}" --http1.1 -L --fail \
  --connect-timeout 30 --keepalive-time 15 --tcp-nodelay \
  -o "$download" "$RAILGOERL_URL"; then
  echo "RailGoerl24 download failed; partial file preserved at $download." >&2
  echo "The official host does not support HTTP Range, so it cannot resume." >&2
  exit 4
fi

actual_bytes="$(stat -c %s "$download")"
if (( actual_bytes != RAILGOERL_BYTES )); then
  echo "Unexpected RailGoerl24 archive size: $actual_bytes" >&2
  exit 4
fi
mv "$download" "$archive"
sha256sum "$archive" | tee "$archive.sha256"

echo "Open-source acquisition finished; restricted image sources remain untouched."
