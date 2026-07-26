#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_path="${REVIEW_ASSISTANT_VENV:-${project_root}/.venv-review-assistant}"
checkpoint=""
verifier=""
encoder=""
database="${project_root}/data/review_assistant.sqlite"
port="8501"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --verifier) verifier="$2"; shift 2 ;;
    --encoder) encoder="$2"; shift 2 ;;
    --database) database="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${venv_path}/bin/python" ]]; then
  echo "Environment missing. Run scripts/review_assistant/install.sh first." >&2
  exit 1
fi
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "Frozen detector checkpoint is required. Expected SHA-256:" >&2
  echo "5a8483b1d40938f49b05225cfe4b3c92414e874fe72008746396ea32bcfb0a79" >&2
  exit 1
fi
if [[ -n "${verifier}" && ! -f "${verifier}" ]]; then
  echo "Verifier file does not exist. Expected SHA-256:" >&2
  echo "8ee0e1ad4960963b87b9eef01c4f039c649b4e33bd2155eeb4d2a7c4a43b999d" >&2
  exit 1
fi
if [[ -n "${encoder}" && ! -f "${encoder}" ]]; then
  echo "Verifier encoder does not exist. Expected SHA-256:" >&2
  echo "c16f796a297b0418cf315212a7d12a00b25825d857782da84f24653909723de6" >&2
  exit 1
fi

export REVIEW_ASSISTANT_CHECKPOINT="${checkpoint}"
export REVIEW_ASSISTANT_VERIFIER="${verifier}"
export REVIEW_ASSISTANT_ENCODER="${encoder}"
export REVIEW_ASSISTANT_DATABASE="${database}"

cd "${project_root}"
exec "${venv_path}/bin/python" -m streamlit run \
  src/review_assistant/web_app.py \
  --server.address 0.0.0.0 \
  --server.port "${port}" \
  --server.headless true
