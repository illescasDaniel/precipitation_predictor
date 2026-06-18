# Agent Working Notes

## Python environment

- **Requires Python 3.13+** (see `.python-version`).
- Never install packages globally for this project.
- Use the virtual environment at `.venv` before `pip`, `python`, tests, or linters.
- Bootstrap: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.

## Package layout

- Application code: `src/precipitation_predictor/`
- Entry modules at package root (`predict_bilbao.py`, `benchmark.py`, etc.); library code in `internal/`
- Import as `precipitation_predictor.*` (not `src.*`).
- AEMET JSON data: `data/` (paths are relative to repo root)
- Generated outputs: `results/` (charts and metrics from predict/seasonality scripts)
- AEMET API key: repo-root `.env` (`AEMET_API_KEY`; see `.env.example`)
- Run entry scripts from the repository root:

| Script | Purpose |
|--------|---------|
| `./scripts/extract_aemet_data.sh` | Fetch raw AEMET JSON into `data/` |
| `./scripts/predict_bilbao.sh` | Train/predict Bilbao for sample dates |
| `./scripts/export_bilbao_model.sh` | Train and export Bilbao XGBoost bundle (`.ubj` + manifest); optional `--prediction-date`, default output dir uses `max_date` |
| `./scripts/predict_bilbao_from_model.sh` | Forecast from an exported model bundle (no retraining) |
| `./scripts/visualize_seasonality.sh` | Multi-city seasonality chart |
| `./scripts/benchmark.sh` | Expanding-window cross-validation → `results/benchmark/` |

## Python formatting

- **Tabs** for indentation (enforced by Ruff).
- Run Ruff from the repository root.

## Quality gate (after substantive changes)

After substantive code changes, run the combined check script and fix all reported issues:

1. `./scripts/quality/checks.sh --fix` — Ruff autofix + format, shell format, full gate
2. `./scripts/quality/checks.sh` — confirm clean (check-only, same as CI)

**Cursor agent sandbox:** run checks with **full permissions** (`required_permissions: ["all"]`). Default sandbox may block `.venv/`.

Individual steps:

| Step | Script |
|------|--------|
| Combined gate | `./scripts/quality/checks.sh` |
| Ruff only | `./scripts/quality/ruff.sh` |
| Shell lint | `./scripts/quality/shellcheck.sh` |
| Typecheck | `./scripts/quality/pyright.sh` |

Install missing shell tools: `./scripts/install/install_shellcheck.sh`

## Type checking

- [basedpyright](https://docs.basedpyright.com/) in **strict** mode on `src/` (`pyproject.toml` → `[tool.basedpyright]`).
- Pandas/matplotlib-heavy code uses targeted relaxations for third-party inference (`reportUnknownMemberType`, etc.).
- Fix type errors when changing typed APIs; use `TYPE_CHECKING` imports for annotation-only dependencies.

## Documentation

| Topic | Doc |
|-------|-----|
| Overview & setup | [README.md](README.md) |
| Methodology | [docs/methodology.md](docs/methodology.md) |
