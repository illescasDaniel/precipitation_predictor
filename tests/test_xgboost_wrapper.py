from __future__ import annotations

from datetime import date, timedelta
from typing import cast

import pandas as pd

from precipitation_predictor.config import DEFAULT_DATE_FEATURES, DEFAULT_FEATURES, FORECAST_HORIZON, MAX_FEATURE_PERIOD, SEED
from precipitation_predictor.internal.process_data import create_temporal_features, process_data
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper
from fixtures_data import aemet_record


def _synthetic_dataframe(days: int = 80) -> pd.DataFrame:
	start = date(2024, 1, 1)
	raw = [aemet_record((start + timedelta(days=offset)).isoformat(), prec=f"{offset % 5},0") for offset in range(days)]
	return create_temporal_features(
		process_data(raw),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)


def test_fit_predict_matches_fit_then_forecast() -> None:
	df = _synthetic_dataframe()
	min_date = cast(pd.Timestamp, df[Column.DATE].min())
	max_date = cast(pd.Timestamp, df[Column.DATE].max() - pd.Timedelta(days=FORECAST_HORIZON))

	combined = XGBoostWrapper(SEED)
	combined_forecast = combined.fit_predict(
		df,
		min_date,
		max_date,
		FORECAST_HORIZON,
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
	)

	split = XGBoostWrapper(SEED)
	train, test = split.train_test_split(df, min_date, max_date, FORECAST_HORIZON)
	split.fit(train, test, DEFAULT_FEATURES, DEFAULT_DATE_FEATURES)
	split_forecast = split.forecast(
		train,
		test,
		max_date,
		FORECAST_HORIZON,
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
	)

	assert combined_forecast == split_forecast
	assert len(combined_forecast) == FORECAST_HORIZON
	assert all(value >= 0 for value in combined_forecast)
