#!/usr/bin/env bash

set -u

# Quality gate — audit, ruff, shell, basedpyright.
# --fix: ruff autofix+format and shfmt write; ignored when CI=true.

quality_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
internal_dir="${quality_dir}/internal"
scripts_dir="$(cd "${quality_dir}/.." && pwd)"
repo_root="$(cd "${scripts_dir}/.." && pwd)"

# shellcheck source=scripts/quality/internal/gate.sh
source "${internal_dir}/gate.sh"

FIX=false
for arg in "$@"; do
	case "${arg}" in
	--fix)
		FIX=true
		;;
	esac
done

if [[ "${CI:-}" == "true" && "${FIX}" == true ]]; then
	echo "note: --fix ignored in CI (check-only mode)"
	FIX=false
fi

GATE_PLANNED_STEPS=4
gate_init

# shellcheck source=scripts/quality/internal/lib.sh
source "${internal_dir}/lib.sh"
lib_require_venv
PYTHON="${LIB_REPO_ROOT}/.venv/bin/python"
cd "${repo_root}" || exit

set +e

# --- 1. audit ---
gate_step_start "audit"
audit_output="$("${internal_dir}/audit_deps.sh" 2>&1)"
audit_exit=$?
printf '%s\n' "${audit_output}"
if [[ "${audit_exit}" -eq 0 ]]; then
	emit_out="$(printf '%s' "${audit_output}" | "${PYTHON}" "${internal_dir}/gate_emit.py" audit 2>&1)"
	summary=""
	while IFS= read -r line; do
		if [[ "${line}" == GATE_SUMMARY* ]]; then
			summary="${line}"
		elif [[ "${line}" == ::* ]]; then
			echo "${line}"
		fi
	done <<<"${emit_out}"
	gate_apply_emit_summary "${summary:-GATE_SUMMARY errors=0 warnings=0}"
else
	gate_gha_error "" "" "" "pip-audit" "dependency audit failed (exit ${audit_exit})"
	gate_record_fail 1 0
	gate_add_detail "[audit] exit ${audit_exit}"
fi

# --- 2. ruff ---
gate_step_start "ruff"
if [[ "${FIX}" == true ]]; then
	ruff_output="$("${quality_dir}/ruff.sh" 2>&1)"
	ruff_exit=$?
	printf '%s\n' "${ruff_output}"
	if [[ "${ruff_exit}" -eq 0 ]]; then
		gate_record_pass
	else
		gate_gha_error "" "" "" "ruff" "ruff fix/format failed (exit ${ruff_exit})"
		gate_record_fail 1 0
		gate_add_detail "[ruff] exit ${ruff_exit}"
	fi
else
	lib_activate_venv
	lib_ruff_targets
	ruff_check_out="$(ruff check "${LIB_RUFF_TARGETS[@]}" --output-format=github 2>&1)"
	printf '%s\n' "${ruff_check_out}"
	emit_out="$(printf '%s\n' "${ruff_check_out}" | "${PYTHON}" "${internal_dir}/gate_emit.py" ruff-github 2>&1)"
	summary=""
	while IFS= read -r line; do
		if [[ "${line}" == GATE_SUMMARY* ]]; then
			summary="${line}"
		fi
	done <<<"${emit_out}"
	ruff_format_out="$(ruff format --check "${LIB_RUFF_TARGETS[@]}" 2>&1)"
	ruff_format_exit=$?
	if [[ -n "${ruff_format_out}" ]]; then
		printf '%s\n' "${ruff_format_out}"
	fi
	ruff_errors=0
	ruff_warnings=0
	if [[ -n "${summary}" ]]; then
		ruff_errors="$(echo "${summary}" | sed -n 's/.*errors=\([0-9]*\).*/\1/p')"
		ruff_warnings="$(echo "${summary}" | sed -n 's/.*warnings=\([0-9]*\).*/\1/p')"
	fi
	if [[ "${ruff_format_exit}" -ne 0 ]]; then
		ruff_errors=$((ruff_errors + 1))
		gate_gha_error "" "" "" "ruff" "format check failed"
		gate_add_detail "[ruff] format check failed"
	fi
	if [[ "${ruff_errors}" -gt 0 || "${ruff_format_exit}" -ne 0 ]]; then
		gate_record_fail "${ruff_errors:-1}" "${ruff_warnings:-0}"
	elif [[ "${ruff_warnings:-0}" -gt 0 ]]; then
		gate_record_step "warn" 0 "${ruff_warnings}"
	else
		gate_record_pass
	fi
fi

# --- 3. shell ---
gate_step_start "shell"
if [[ "${FIX}" == true ]]; then
	shell_output="$("${quality_dir}/shellcheck.sh" --fix 2>&1)"
else
	shell_output="$("${quality_dir}/shellcheck.sh" 2>&1)"
fi
shell_exit=$?
printf '%s\n' "${shell_output}"
if [[ "${shell_exit}" -eq 0 ]]; then
	gate_record_pass
else
	gate_gha_error "" "" "" "shell" "shell lint/format failed (exit ${shell_exit})"
	gate_record_fail 1 0
	gate_add_detail "[shell] exit ${shell_exit}"
fi

# --- 4. pyright ---
gate_step_start "basedpyright"
pyright_stderr="$(mktemp)"
pyright_json="$("${quality_dir}/pyright.sh" --outputjson 2>"${pyright_stderr}")"
pyright_exit=$?
emit_out="$(printf '%s' "${pyright_json}" | "${PYTHON}" "${internal_dir}/gate_emit.py" pyright 2>&1)"
summary=""
while IFS= read -r line; do
	if [[ "${line}" == GATE_SUMMARY* ]]; then
		summary="${line}"
	elif [[ "${line}" == ::* ]]; then
		echo "${line}"
		if [[ "${line}" == *"basedpyright returned invalid JSON"* ]] && [[ -s "${pyright_stderr}" ]]; then
			cat "${pyright_stderr}" >&2
		fi
	fi
done <<<"${emit_out}"
rm -f "${pyright_stderr}"
if [[ -n "${summary}" ]]; then
	gate_apply_emit_summary "${summary}"
else
	if [[ "${pyright_exit}" -eq 0 ]]; then
		gate_record_pass
	else
		gate_gha_error "" "" "" "basedpyright" "type check failed (exit ${pyright_exit})"
		gate_record_fail 1 0
	fi
fi

gate_exit
