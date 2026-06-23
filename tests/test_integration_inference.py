from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from precipitation_predictor.config import (
	BILBAO_IDEMA,
	CLIMATE_DB_PATH,
	DEFAULT_DATE_FEATURES,
	DEFAULT_FEATURES,
	FORECAST_HORIZON,
	INFERENCE_LOOKBACK_DAYS,
	MAX_FEATURE_PERIOD,
	SEED,
	inference_data_window,
)
from precipitation_predictor.internal.climate_db import load_station_records
from precipitation_predictor.internal.process_data import create_temporal_features, process_data
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.model_bundle import load_model_bundle, save_model_bundle
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper


REPO_ROOT = Path(__file__).resolve().parents[1]
PAST_MODEL_DIR = REPO_ROOT / "results/bilbao/models/2023-09-11"
FUTURE_MODEL_DIR = REPO_ROOT / "results/bilbao/models/2025-02-25"
CLIMATE_DB = REPO_ROOT / CLIMATE_DB_PATH


pytestmark = pytest.mark.integration


def _require_integration_assets() -> None:
	if not CLIMATE_DB.is_file():
		pytest.skip(f"Missing {CLIMATE_DB}; run ./scripts/import_climate_db.sh")
	if not PAST_MODEL_DIR.is_dir() or not FUTURE_MODEL_DIR.is_dir():
		pytest.skip("Missing exported model bundles under results/bilbao/models/")


def _forecast_from_bundle(
	model_dir: Path,
	*,
	use_trailing_window: bool,
) -> tuple[list[float], float | None]:
	_require_integration_assets()
	model, manifest = load_model_bundle(model_dir)
	max_date = pd.Timestamp(manifest.max_date)
	min_date = pd.Timestamp(manifest.min_date)

	if use_trailing_window:
		start_date, end_date = inference_data_window(manifest.max_date, manifest.forecast_horizon)
		raw = load_station_records(BILBAO_IDEMA, db_path=CLIMATE_DB, start_date=start_date, end_date=end_date)
	else:
		raw = load_station_records(BILBAO_IDEMA, db_path=CLIMATE_DB)

	df = create_temporal_features(
		process_data(raw),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)
	train, test = model.train_test_split(df, min_date, max_date, manifest.forecast_horizon)
	forecast = model.forecast(
		train,
		test,
		max_date,
		manifest.forecast_horizon,
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
	)

	mae: float | None = None
	if not test.empty:
		from sklearn.metrics import mean_absolute_error

		actual = test[Column.PRECIPITATION].values
		predicted = forecast[: len(test)]
		mae = float(mean_absolute_error(actual, predicted))

	return forecast, mae


@pytest.mark.integration
def test_past_model_trailing_window_matches_full_history(capsys: pytest.CaptureFixture[str]) -> None:
	_require_integration_assets()
	trailing, _ = _forecast_from_bundle(PAST_MODEL_DIR, use_trailing_window=True)
	capsys.readouterr()
	full, _ = _forecast_from_bundle(PAST_MODEL_DIR, use_trailing_window=False)
	capsys.readouterr()
	assert trailing == full


@pytest.mark.integration
def test_past_model_matches_reference_mae(capsys: pytest.CaptureFixture[str]) -> None:
	_require_integration_assets()
	_, mae = _forecast_from_bundle(PAST_MODEL_DIR, use_trailing_window=True)
	capsys.readouterr()
	assert mae is not None
	assert mae == pytest.approx(5.04, abs=0.01)


@pytest.mark.integration
def test_future_model_produces_fourteen_day_forecast(capsys: pytest.CaptureFixture[str]) -> None:
	_require_integration_assets()
	forecast, mae = _forecast_from_bundle(FUTURE_MODEL_DIR, use_trailing_window=True)
	capsys.readouterr()
	assert len(forecast) == FORECAST_HORIZON
	assert all(value >= 0 for value in forecast)
	assert mae is None


@pytest.mark.integration
def test_model_bundle_save_load_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
	_require_integration_assets()
	_, manifest = load_model_bundle(PAST_MODEL_DIR)
	raw = load_station_records(
		BILBAO_IDEMA,
		db_path=CLIMATE_DB,
		start_date=manifest.max_date - timedelta(days=INFERENCE_LOOKBACK_DAYS),
		end_date=manifest.max_date,
	)
	df = create_temporal_features(
		process_data(raw),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)
	max_date = pd.Timestamp(manifest.max_date)
	min_date = cast(pd.Timestamp, df[Column.DATE].min())
	model = XGBoostWrapper(SEED)
	train, test = model.train_test_split(df, min_date, max_date, FORECAST_HORIZON)
	model.fit(train, test, DEFAULT_FEATURES, DEFAULT_DATE_FEATURES)
	forecast_before = model.forecast(
		train,
		test,
		max_date,
		FORECAST_HORIZON,
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
	)

	bundle_dir = save_model_bundle(
		model,
		tmp_path / "bundle",
		seed=SEED,
		min_date=min_date.date(),
		max_date=manifest.max_date,
		forecast_horizon=FORECAST_HORIZON,
		additional_date_features=DEFAULT_DATE_FEATURES,
	)
	loaded, loaded_manifest = load_model_bundle(bundle_dir)
	capsys.readouterr()

	forecast_after = loaded.forecast(
		train,
		test,
		max_date,
		FORECAST_HORIZON,
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
	)

	assert loaded_manifest.max_date == manifest.max_date
	assert loaded.feature_columns == model.feature_columns
	assert forecast_before == forecast_after
