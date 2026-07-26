#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_path="${REVIEW_ASSISTANT_VENV:-${project_root}/.venv-review-assistant}"

python3 -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --upgrade pip
"${venv_path}/bin/python" -m pip install -r "${project_root}/requirements.txt"

echo "Installed Railway Person Review Assistant into ${venv_path}"
echo "Models were not downloaded. Supply local frozen model paths to run.sh."
