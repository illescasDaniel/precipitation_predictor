import inspect
import os
import random
import time
from math import ceil
from typing import Any, Callable, Optional, cast

import pandas as pd
from matplotlib import pyplot as plt

from precipitation_predictor.internal.prediction import run_prediction_benchmark
from precipitation_predictor.internal.process_data import create_temporal_features
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.feature_type import FeatureType
from precipitation_predictor.models.model_protocol import Model
from precipitation_predictor.utils.benchmark_utils import BenchmarkData, BenchmarkDataList
from precipitation_predictor.utils.plot_utils import plot_feature_importance_df


def get_static_members(cls: type) -> dict[str, object]:
	all_members = vars(cls)
	static_members = {
		name: value
		for name, value in all_members.items()
		if not (name.startswith("__") and name.endswith("__"))
		and not inspect.isfunction(value)
		and not isinstance(value, staticmethod)
	}
	return static_members


def generate_random_configs(
	columns: list[str], feature_types: list[FeatureType], iterations: int, forecast_horizon: Optional[int]
) -> BenchmarkDataList:
	output: BenchmarkDataList = []
	for _ in range(iterations):
		features, new_forecast_horizon = generate_random_features_v2(columns, feature_types, forecast_horizon)
		additional_date_features = generate_random_additional_date_features_v2()
		output.append(BenchmarkData(features, additional_date_features, new_forecast_horizon))
	return output


def generate_random_features_v2(
	columns: list[str], feature_types: list[FeatureType], forecast_horizon: Optional[int]
) -> tuple[list[Feature], int]:
	features: list[Feature] = []
	forecast_horizon = 14
	for col in random.sample(columns, k=random.randint(2, 5)):
		for feature_type in random.sample(feature_types, k=random.randint(1, 2)):
			values = [14, 16, 18, 20] if random.choice([True, False]) else [14]
			for p in values:

				def get_feature() -> Feature:  # type: ignore
					match feature_type:
						case FeatureType.LAG:
							return Feature.lag(col, p)
						case FeatureType.ROLLING_MEAN:
							return Feature.rolling_mean(col, p)
						case FeatureType.ROLLING_MAX:
							return Feature.rolling_max(col, p)
						case FeatureType.ROLLING_MIN:
							return Feature.rolling_min(col, p)
						case FeatureType.ROLLING_STD:
							return Feature.rolling_std(col, p)
						case FeatureType.ROLLING_SUM:
							return Feature.rolling_sum(col, p)

				new_f = get_feature()
				features.append(new_f)
	return features, forecast_horizon


def generate_random_additional_date_features_v2():
	return [Column.DAY_OF_YEAR]


def generate_exploration_configs(
	iterations: int,
	forecast_horizon: Optional[int],
	initial_data: BenchmarkDataList,
) -> BenchmarkDataList:
	if iterations > 0:
		columns: list[str] = [
			str(c)
			for c in get_static_members(Column).values()
			if c
			not in [Column.DATE, Column.DAY_OF_YEAR, Column.WEEK_OF_YEAR, Column.YEAR, Column.QUARTER, Column.MONTH]
		]
		feature_types = list(get_static_members(FeatureType)["_member_map_"].values())
		return [*initial_data, *generate_random_configs(columns, feature_types, iterations, forecast_horizon)]
	return [*initial_data]


def run_config_exploration(
	df: pd.DataFrame,
	iterations: int,
	forecast_horizon: Optional[int],
	initial_data: BenchmarkDataList,
	output_file: str,
	feature_importance_figure_folder: str,
	model_builder: Callable[[], Model[Any]],
	training_min_date: Optional[pd.Timestamp] = None,
	years: list[int] | None = None,
	year_min_day: Optional[int] = None,
	year_max_day: Optional[int] = None,
	filter_best_results_percent: float = 0.2,
	verbose: bool = True,
):
	if years is None:
		years = [2024]
	start_time = time.time()

	features_list_and_additional_date_features = generate_exploration_configs(
		iterations=iterations, forecast_horizon=forecast_horizon, initial_data=initial_data
	)

	predictions: list[
		tuple[float, float, float, list[float], list[Feature], list[str], int, Optional[pd.DataFrame]]
	] = []

	df_copy = df.copy()

	for features, additional_date_features, forecast_horizon in features_list_and_additional_date_features:
		df = df_copy
		max_period = max(features, key=lambda x: x.window).window if features else 0
		df = create_temporal_features(df, features, additional_date_features, max_period)

		overall_mean, overall_std, custom_metric_mean, std_residuals_list, feature_importance_df = (
			run_prediction_benchmark(
				df=df,
				min_date=cast(pd.Timestamp, training_min_date or df[Column.DATE].min()),
				years=years,
				forecast_horizon=forecast_horizon,
				features=features,
				additional_date_features=additional_date_features,
				model_builder=model_builder,
				year_min_day=year_min_day,
				year_max_day=year_max_day,
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

	results_percent = 1 if iterations <= 0 else filter_best_results_percent
	with open(output_file, "w") as f:
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
		) in results[: ceil(len(predictions) * results_percent)]:
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
