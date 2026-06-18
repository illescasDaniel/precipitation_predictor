from __future__ import annotations

from datetime import date, timedelta

import pytest

from precipitation_predictor.config import (
	FORECAST_HORIZON,
	INFERENCE_LOOKBACK_DAYS,
	MAX_FEATURE_PERIOD,
	inference_data_window,
)


def test_inference_data_window_span() -> None:
	max_date = date(2025, 2, 25)
	start, end = inference_data_window(max_date, FORECAST_HORIZON)
	assert start == max_date - timedelta(days=INFERENCE_LOOKBACK_DAYS)
	assert end == max_date + timedelta(days=FORECAST_HORIZON)


def test_inference_lookback_covers_feature_windows() -> None:
	assert INFERENCE_LOOKBACK_DAYS >= 2 * MAX_FEATURE_PERIOD
