#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "usage: run_python.sh <module>" >&2
	exit 1
fi

module="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
	echo "Missing .venv. Create it first: python -m venv .venv && pip install -e '.[dev]'" >&2
	exit 1
fi

cd "${repo_root}"
exec "${python_bin}" -m "precipitation_predictor.${module}" "$@"
