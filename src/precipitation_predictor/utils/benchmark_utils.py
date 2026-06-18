import os
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, cast

import pandas as pd
from matplotlib import pyplot as plt

from precipitation_predictor.internal.prediction import run_prediction_benchmark
from precipitation_predictor.internal.process_data import create_temporal_features
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.model_protocol import Model
from precipitation_predictor.utils.plot_utils import plot_feature_importance_df


class BenchmarkData(NamedTuple):
	features: list[Feature]
	additional_date_features: list[str]
	forecast_horizon: int


BenchmarkDataList = list[BenchmarkData]


def run_cv_benchmark(
	df: pd.DataFrame,
	forecast_horizon: Optional[int],
	initial_data: BenchmarkDataList,
	output_file: str,
	feature_importance_figure_folder: str,
	model_builder: Callable[[], Model[Any]],
	years: list[int] | None = None,
	verbose: bool = True,
):
	if years is None:
		years = [2024]
	start_time = time.time()

	predictions: list[
		tuple[float, float, float, list[float], list[Feature], list[str], int, Optional[pd.DataFrame]]
	] = []

	df_copy = df.copy()

	for features, additional_date_features, forecast_horizon in initial_data:
		df = df_copy
		max_period = max(features, key=lambda x: x.window).window if features else 0
		df = create_temporal_features(df, features, additional_date_features, max_period)

		overall_mean, overall_std, custom_metric_mean, std_residuals_list, feature_importance_df = (
			run_prediction_benchmark(
				df=df,
				min_date=cast(pd.Timestamp, df[Column.DATE].min()),
				years=years,
				forecast_horizon=forecast_horizon,
				features=features,
				additional_date_features=additional_date_features,
				model_builder=model_builder,
				year_min_day=None,
				year_max_day=None,
			)
		)

		predictions.append(
			(
				overall_mean,
				overall_std,
				custom_metric_mean,
				std_residuals_list,
				features,
				additional_date_features,
				forecast_horizon,
				feature_importance_df,
			)
		)

		if verbose:
			print(
				f"Error: {custom_metric_mean}\nMAE ± std: {overall_mean:.4f} ± {overall_std:.4f}\nStd residuals list: {std_residuals_list}\nForecast Horizon: {forecast_horizon}\nFeatures: {features}\nDate features: {additional_date_features}"
			)
			if feature_importance_df is not None:
				print(feature_importance_df.to_string(index=False))
				print("-------")

	results = sorted(predictions, key=lambda x: x[0])

	for file in os.listdir(feature_importance_figure_folder):
		if file.startswith("feature_importances_") and file.endswith(".png"):
			os.remove(os.path.join(feature_importance_figure_folder, file))

	end_time = time.time()
	total_seconds = end_time - start_time
	minutes = int(total_seconds // 60)
	seconds = int(total_seconds % 60)

	output_path = Path(output_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	Path(feature_importance_figure_folder).mkdir(parents=True, exist_ok=True)

	with output_path.open("w") as f:
		i = 0
		f.write(f"Total running time: {minutes}min, {seconds}s\n")
		for (
			overall_mean,
			overall_std,
			custom_metric_mean,
			std_residuals_list,
			features,
			additional_date_features,
			forecast_horizon,
			feature_importance_df,
		) in results:
			f.write(
				f"Error: {custom_metric_mean}\nMAE ± std: {overall_mean:.4f} ± {overall_std:.4f}\nStd residuals list: {std_residuals_list}\nForecast Horizon: {forecast_horizon}\nFeatures: {features}\nDate features: {additional_date_features}\n\n"
			)
			if feature_importance_df is not None:
				fig = plot_feature_importance_df(feature_importance_df)
				fig.savefig(os.path.join(feature_importance_figure_folder, f"feature_importances_{i}"))
				plt.close(fig)
			i += 1

	if verbose:
		print(f"Total running time: {minutes}min, {seconds}s")
