import argparse
import random
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from precipitation_predictor.config import (
	BILBAO_RESULTS_DIR,
	DEFAULT_DATE_FEATURES,
	DEFAULT_FEATURES,
	FORECAST_HORIZON,
	MAX_FEATURE_PERIOD,
	PREDICTION_DATES,
	SEED,
	STD_RESIDUALS_LIST,
	bilbao_data_files,
)
from precipitation_predictor.internal.prediction import run_prediction
from precipitation_predictor.internal.process_data import create_temporal_features, load_data, process_data
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper
from precipitation_predictor.utils.pandas_utils import configure_pandas


def build_xgb() -> XGBoostWrapper:
	return XGBoostWrapper(SEED)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and predict Bilbao precipitation for sample dates.")
	parser.add_argument(
		"--output-dir",
		default=BILBAO_RESULTS_DIR,
		help=f"Directory for saved charts and metrics (default: {BILBAO_RESULTS_DIR}).",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Open matplotlib windows for each chart (default: save only).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	random.seed(SEED)
	np.random.seed(SEED)
	configure_pandas()

	data = load_data(bilbao_data_files())
	df = create_temporal_features(
		process_data(data),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)

	metrics_blocks: list[str] = []
	for date_str in PREDICTION_DATES:
		end_date = pd.to_datetime(date_str) - pd.Timedelta(days=1)
		metrics = run_prediction(
			df=df,
			min_date=cast(pd.Timestamp, df[Column.DATE].min()),
			max_date=cast(pd.Timestamp, end_date),
			model=build_xgb(),
			forecast_horizon=FORECAST_HORIZON,
			features=DEFAULT_FEATURES,
			additional_date_features=DEFAULT_DATE_FEATURES,
			std_residuals_list=STD_RESIDUALS_LIST,
			output_dir=output_dir,
			prediction_label=date_str,
			show=args.show,
		)
		if metrics is not None:
			metrics_blocks.append(metrics.format_lines())

	if metrics_blocks:
		output_dir.mkdir(parents=True, exist_ok=True)
		metrics_path = output_dir / "metrics.txt"
		metrics_path.write_text("\n".join(metrics_blocks) + "\n", encoding="utf-8")
		print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()
