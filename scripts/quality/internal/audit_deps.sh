#!/usr/bin/env bash

set -euo pipefail

internal_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/quality/internal/lib.sh
source "${internal_dir}/lib.sh"

lib_require_venv
lib_activate_venv

python -m pip install --upgrade 'pip>=26.1.2'

if ! command -v pip-audit &>/dev/null; then
	pip install pip-audit
fi

python -m pip install --upgrade 'msgpack>=1.2.1'

pip-audit -l --skip-editable
