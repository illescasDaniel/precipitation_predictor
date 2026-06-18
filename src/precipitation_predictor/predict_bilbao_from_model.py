import argparse
import random
from datetime import timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from precipitation_predictor.config import (
	BILBAO_RESULTS_DIR,
	DEFAULT_DATE_FEATURES,
	DEFAULT_FEATURES,
	MAX_FEATURE_PERIOD,
	SEED,
	STD_RESIDUALS_LIST,
	bilbao_data_files,
)
from precipitation_predictor.internal.prediction import run_forecast
from precipitation_predictor.internal.process_data import create_temporal_features, load_data, process_data
from precipitation_predictor.models.model_bundle import load_model_bundle
from precipitation_predictor.utils.pandas_utils import configure_pandas


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run Bilbao precipitation forecasts from an exported model bundle.")
	parser.add_argument(
		"--model-dir",
		required=True,
		help="Directory containing model.ubj and manifest.json.",
	)
	parser.add_argument(
		"--output-dir",
		default=BILBAO_RESULTS_DIR,
		help=f"Directory for saved charts (default: {BILBAO_RESULTS_DIR}).",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Open matplotlib windows for each chart (default: save only).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	random.seed(SEED)
	np.random.seed(SEED)
	configure_pandas()

	model, manifest = load_model_bundle(args.model_dir)
	output_dir = Path(args.output_dir)
	prediction_label = (manifest.max_date + timedelta(days=1)).isoformat()

	data = load_data(bilbao_data_files())
	df = create_temporal_features(
		process_data(data),
		DEFAULT_FEATURES,
		DEFAULT_DATE_FEATURES,
		MAX_FEATURE_PERIOD,
	)

	min_date = pd.Timestamp(manifest.min_date)
	max_date = pd.Timestamp(manifest.max_date)

	metrics = run_forecast(
		df=df,
		min_date=cast(pd.Timestamp, min_date),
		max_date=cast(pd.Timestamp, max_date),
		model=model,
		forecast_horizon=manifest.forecast_horizon,
		features=DEFAULT_FEATURES,
		additional_date_features=DEFAULT_DATE_FEATURES,
		std_residuals_list=STD_RESIDUALS_LIST,
		output_dir=output_dir,
		prediction_label=prediction_label,
		show=args.show,
	)

	if metrics is not None:
		output_dir.mkdir(parents=True, exist_ok=True)
		metrics_path = output_dir / f"{prediction_label}-metrics-from-model.txt"
		metrics_path.write_text(metrics.format_lines() + "\n", encoding="utf-8")
		print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
	main()
