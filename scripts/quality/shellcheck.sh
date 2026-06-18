#!/usr/bin/env bash

set -euo pipefail

quality_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/quality/internal/lib.sh
source "${quality_dir}/internal/lib.sh"

lib_require_shell_tools

CHECK_ONLY=true
for arg in "$@"; do
	case "${arg}" in
	--fix) CHECK_ONLY=false ;;
	esac
done

lib_shell_targets

if [[ "${CHECK_ONLY}" == false ]]; then
	shfmt -i 0 -bn -w "${LIB_SHELL_TARGETS[@]}"
fi

shfmt -i 0 -bn -d "${LIB_SHELL_TARGETS[@]}"
shellcheck "${LIB_SHELL_TARGETS[@]}"
