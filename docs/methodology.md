# Methodology

This document describes the forecasting approach used in this project. It is meant as a companion to the [README](../README.md).

## Context and goals

Recurring natural disasters — especially floods — make regional precipitation awareness essential. This project analyzes rainfall patterns for a chosen locality, identifies wet and dry cycles, and forecasts imminent precipitation events.

**General objective**: build a predictive model for short-term precipitation using multiple climatic variables.

**Specific objectives**:

1. Study monthly precipitation patterns in the study area.
2. Analyze rain and drought periods via rolling-sum visualizations.
3. Develop a model that anticipates precipitation in the coming days.
4. Evaluate accuracy on rain-occurrence prediction.

## Seasonality study

Precipitation varies between regions but shares a similar seasonal shape across Spain: peaks around November and sustained wet months, with marked summer dryness.

Comparing five cities (Bilbao, San Sebastián, Valencia, Madrid, Málaga) via 7-day rolling precipitation sums led to selecting **Bilbao**:

- Continuous daily records from December 1949.
- High annual precipitation — informative for modeling wet climates.

## Data preparation

**Source**: AEMET Open Data, Bilbao station (`historical_climate_data_BILBAO_*.json` under `data/`).

**Storage**: JSON shards in `data/` are the canonical, version-controlled export. Runtime scripts query **`data/climate.sqlite`**, a SQLite database built from those same JSON files (also checked into the repo). Rebuild with `./scripts/import_climate_db.sh` after JSON changes, or upsert via `./scripts/extract_aemet_data.sh` when fetching new data from AEMET.

**Extraction** (raw JSON into `data/`): `./scripts/extract_aemet_data.sh` (`extract_aemet_data.py`) reads `AEMET_API_KEY` from repo-root `.env`.

Register for an API key at [AEMET Open Data](https://opendata.aemet.es/centrodedescargas/inicio), copy `.env.example` to `.env`, and set `AEMET_API_KEY`. The API limits requests to six-month windows; the extractor splits longer ranges automatically.

**Cleaning pipeline** (`internal/process_data.py`):

| Step | Detail |
|------|--------|
| Dates & numbers | Parse ISO dates; convert Spanish decimal notation (`Babel`) |
| Precipitation codes | `Ip` → 0.05 mm, `Acum` → 0 mm |
| Missing values | Impute with historical column means |
| Column selection | Drop redundant metadata and low-value fields (e.g. wind direction for final model) |
| Renaming | International column names (`Column` enum) |

## Temporal features

Future observations are unavailable at forecast time, so **lag** and **rolling** features are derived from history (and prior predictions during multi-step forecasting).

**Final configuration** (after experimentation):

| Feature type | Column | Window / lag (days) |
|--------------|--------|---------------------|
| Lag | Precipitation | 14, 16, 18, 20 |
| Lag | Average humidity | 14 |
| Rolling std | Precipitation | 14, 25 |
| Calendar | Day of year | — |

**Forecast horizon**: 14 days.

Implementation: `create_temporal_features()` for bulk history; `Model._create_features()` for sequential day-by-day prediction (rolling features depend on prior predictions).

## Model selection

All algorithms implement a common `Model` protocol (`fit_predict`) for fair comparison.

| Algorithm | Role |
|-----------|------|
| **XGBoost** | Selected — Tweedie objective, fast inference, best MAE |
| LightGBM | Strong alternative; slower per fold |
| LSTM | Sequence model; long training; flatter outputs |
| Prophet | Trend/seasonality baseline; weaker on short horizon |

**XGBoost training**: train/test split by date; lag + rolling features as *X*, daily precipitation as *y*; hyperparameters tuned for Tweedie regression.

**Sequential prediction**: for each future day, features are rebuilt from historical data plus predictions already made — required for precipitation rolling statistics.

## Evaluation

**Cross-validation**: expanding window — e.g. predict 1951 using data through 1950, then predict 1952 using 1950–1951, and so on through 2024.

**Metrics**:

- **MAE** (mm) — primary comparison metric in benchmarks.
- **Custom rain error** — penalizes false negatives (missed rain) and false positives more heavily than small amount errors.

**Exogenous experiments** (not retained):

| Variable | Source | Outcome |
|----------|--------|---------|
| ENSO index | NOAA / IRI | Slight degradation; removed |
| San Sebastián precipitation | Nearby station lags & rolling std | No improvement; removed |

## Visualization

- **Continuous**: predicted line vs actual scatter; uncertainty band from residual standard deviation (widens with horizon; negative band clipped at 0 mm).
- **Categorical**: discretize mm into levels for interpretability:

| Level | Range (mm) |
|-------|------------|
| 0 | &lt; 2 |
| 1 | 2 – 5 |
| 2 | 5 – 10 |
| 3 | 10 – 50 |
| 4 | &gt; 50 |

## Feature importance

XGBoost feature weights highlight:

1. **Rolling precipitation standard deviation** — recent variability dominates.
2. **Day of year** — seasonality.

Lower-importance features were kept: removing them reduced validation accuracy despite simpler models.

## Limitations and outlook

- Computational budget constrained exhaustive neural tuning (LSTM epochs).
- The model complements — does not replace — operational numerical weather prediction.
- XGBoost delivers practical 14-day forecasts in seconds on modest hardware.

## References

- AEMET Open Data: https://opendata.aemet.es/centrodedescargas/inicio
