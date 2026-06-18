#!/usr/bin/env bash
# Install shellcheck and shfmt on CachyOS / Arch. Run from repo root; requires sudo once.
set -euo pipefail

sc_ok=false
shfmt_ok=false
command -v shellcheck >/dev/null 2>&1 && sc_ok=true
command -v shfmt >/dev/null 2>&1 && shfmt_ok=true

if [[ "${sc_ok}" == true && "${shfmt_ok}" == true ]]; then
	echo "shellcheck $(shellcheck --version | grep 'version:') already installed"
	echo "shfmt $(shfmt --version) already installed"
	exit 0
fi

echo "Installing shellcheck and shfmt (sudo required)..."
sudo pacman -S --needed --noconfirm shellcheck shfmt

echo ""
echo "shellcheck $(shellcheck --version | grep 'version:')"
echo "shfmt $(shfmt --version)"
