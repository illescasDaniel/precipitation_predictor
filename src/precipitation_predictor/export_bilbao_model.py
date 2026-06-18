import argparse
import random
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from precipitation_predictor.config import (
	BILBAO_IDEMA,
	BILBAO_MODELS_DIR,
	DEFAULT_DATE_FEATURES,
	DEFAULT_FEATURES,
	FORECAST_HORIZON,
	MAX_FEATURE_PERIOD,
	SEED,
)
from precipitation_predictor.internal.climate_db import load_station_records
from precipitation_predictor.internal.process_data import create_temporal_features, process_data
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.model_bundle import save_model_bundle
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper
from precipitation_predictor.utils.pandas_utils import configure_pandas


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and export a Bilbao XGBoost model bundle.")
	parser.add_argument(
		"--prediction-date",
		default=None,
		help="First forecast day (ISO date). Training data ends the day before. "
		"If omitted, trains on all available data (max date in the dataset).",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help=f"Bundle output directory (default: {BILBAO_MODELS_DIR}/{{max_date}}).",
	)
	return parser.parse_args()


def resolve_max_date(df: pd.DataFrame, prediction_date: str | None) -> pd.Timestamp:
	if prediction_date is not None:
		first_forecast_day = pd.to_datetime(prediction_date).date()
		return cast(pd.Timestamp, pd.Timestamp(first_forecast_day) - pd.Timedelta(days=1))
	return cast(pd.Timestamp, df[Column.DATE].max())


def main() -> None:
	args = parse_args()
	random.seed(SEED)
	np.random.seed(SEED)
	configure_pandas()

	data = load_station_records(BILBAO_IDEMA)
	df = create_temporal_features(
		process_data(data),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)
	min_date = cast(pd.Timestamp, df[Column.DATE].min())
	max_date = resolve_max_date(df, args.prediction_date)
	output_dir = Path(args.output_dir or f"{BILBAO_MODELS_DIR}/{max_date.date()}")

	model = XGBoostWrapper(SEED)
	train, test = model.train_test_split(df, min_date, max_date, FORECAST_HORIZON)
	model.fit(train, test, DEFAULT_FEATURES, DEFAULT_DATE_FEATURES)

	bundle_dir = save_model_bundle(
		model,
		output_dir,
		seed=SEED,
		min_date=min_date.date(),
		max_date=max_date.date(),
		forecast_horizon=FORECAST_HORIZON,
		additional_date_features=DEFAULT_DATE_FEATURES,
	)

	print(f"Saved model bundle to {bundle_dir}")
	print(f"- Features: {len(model.feature_columns)}")
	print(f"- Training period: {min_date.date()} to {max_date.date()}")
	if args.prediction_date is None:
		print(f"- Full-history export (last available date: {max_date.date()})")


if __name__ == "__main__":
	main()
