from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from precipitation_predictor.internal.custom_metrics import calculate_custom_rain_error
from precipitation_predictor.models.booster_model import BoosterModel
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.model_protocol import Model
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper
from precipitation_predictor.utils.plot_utils import (
	plot_feature_importance,
	visualize_results,
	visualize_results_as_levels,
)


@dataclass(frozen=True)
class PredictionMetrics:
	custom_rain_error: float
	mae_error: float
	period_start: date
	period_end: date

	def format_lines(self) -> str:
		return (
			"\nModel Performance:\n"
			f"- Custom Rain Error: {self.custom_rain_error:.2f}\n"
			f"- MAE Error: {self.mae_error:.2f}\n"
			f"- Prediction Period: {self.period_start} to {self.period_end}"
		)


def _render_prediction_results(
	model: Model[Any],
	forecast: list[float],
	max_date: pd.Timestamp,
	std_residuals_list: Optional[list[float]],
	output_dir: str | Path | None,
	prediction_label: str,
	show: bool,
) -> Optional[PredictionMetrics]:
	metrics: Optional[PredictionMetrics] = None

	if not model.test_.empty:
		y_true = model.test_[Column.PRECIPITATION].values
		y_pred = np.array(forecast[: len(y_true)])  # type: ignore # Convert to NumPy array

		custom_error = calculate_custom_rain_error(y_true, y_pred)  # type: ignore
		mae_error = mean_absolute_error(y_true, y_pred)  # type: ignore

		metrics = PredictionMetrics(
			custom_rain_error=custom_error,
			mae_error=mae_error,
			period_start=cast(pd.Timestamp, model.future_dates_[0]).date(),
			period_end=cast(pd.Timestamp, model.future_dates_[-1]).date(),
		)
		print(metrics.format_lines())

	output_path = Path(output_dir) if output_dir is not None else None

	visualize_results(
		model.model_name,
		model.train_,
		model.future_dates_,
		forecast,
		model.test_,
		max_date,
		std_residuals_list,
		metrics.mae_error if metrics is not None else None,
		output_path=output_path / f"{prediction_label}-precipitation.png" if output_path else None,
		show=show,
	)
	visualize_results_as_levels(
		model.model_name,
		model.train_,
		model.future_dates_,
		forecast,
		model.test_,
		max_date,
		std_residuals_list,
		metrics.mae_error if metrics is not None else None,
		output_path=output_path / f"{prediction_label}-levels.png" if output_path else None,
		show=show,
	)

	if isinstance(model, BoosterModel):
		plot_feature_importance(
			model,
			output_path=output_path / f"{prediction_label}-feature-importance.png" if output_path else None,
			show=show,
		)

	return metrics


def run_prediction(
	df: pd.DataFrame,
	min_date: pd.Timestamp,
	max_date: pd.Timestamp,
	model: Model[Any],
	forecast_horizon: int,
	features: list[Feature],
	additional_date_features: list[str],
	std_residuals_list: Optional[list[float]] = None,
	output_dir: str | Path | None = None,
	prediction_label: str = "prediction",
	show: bool = False,
) -> Optional[PredictionMetrics]:
	forecast = model.fit_predict(df, min_date, max_date, forecast_horizon, features, additional_date_features)
	return _render_prediction_results(model, forecast, max_date, std_residuals_list, output_dir, prediction_label, show)


def run_forecast(
	df: pd.DataFrame,
	min_date: pd.Timestamp,
	max_date: pd.Timestamp,
	model: XGBoostWrapper,
	forecast_horizon: int,
	features: list[Feature],
	additional_date_features: list[str],
	std_residuals_list: Optional[list[float]] = None,
	output_dir: str | Path | None = None,
	prediction_label: str = "prediction",
	show: bool = False,
) -> Optional[PredictionMetrics]:
	train, test = model.train_test_split(df, min_date, max_date, forecast_horizon)
	forecast = model.forecast(train, test, max_date, forecast_horizon, features, additional_date_features)
	return _render_prediction_results(model, forecast, max_date, std_residuals_list, output_dir, prediction_label, show)


def run_prediction_benchmark(
	df: pd.DataFrame,
	min_date: pd.Timestamp,
	years: list[int],
	forecast_horizon: int,
	features: list[Feature],
	additional_date_features: list[str],
	model_builder: Callable[[], Model[Any]],
	year_min_day: Optional[int] = None,
	year_max_day: Optional[int] = None,
) -> tuple[float, float, float, list[float], Optional[pd.DataFrame]]:
	"""
	Runs a prediction benchmark using an expanding window approach for time series forecasting (Expanding Window Cross-Validation for Time Series Forecasting).
	Each iteration extends the training period up to a dynamically increasing `max_date`
	while maintaining a fixed `min_date`. The model is then tested on the subsequent forecast horizon.

	Returns the overall mean MAE, its standard deviation across years, the global mean custom metric,
	and an optional feature importance dataframe.

	Parameters:
		df (pd.DataFrame): Data containing time series values.
		min_date (pd.Timestamp): The fixed minimum date for the training period.
		years (list[int]): Years over which the evaluation is performed.
		forecast_horizon (int): Forecast prediction horizon in days.
		features (list[Feature]): Features used for the model.
		additional_date_features (list[str]): Extra date-based features.
		model_builder (Callable[[], Model]): Function to construct the model instance.
		year_min_day (Optional[int]): Minimum day of the year to start predictions.
		year_max_day (Optional[int]): Maximum day of the year for predictions.
	Returns:
		tuple:
			- Overall mean MAE (float)
			- Standard deviation of yearly MAEs (float)
			- Global mean custom metric (float)
			- List of standard deviations for each day in the forecast horizon (list[float])
			- Optional feature importance dataframe (pd.DataFrame)
	"""
	yearly_mae_results: dict[int, list[float]] = {year: [] for year in years}
	custom_metric_results: list[float] = []
	feature_importance_d: dict[str, list[float]] = {}
	residuals_list_by_step: list[list[float]] = [[] for _ in range(forecast_horizon)]  # Residuals per step

	for year in years:
		min_day = year_min_day or 0
		max_day = year_max_day or (days_in_year(year) + 1)
		for start_day in range(min_day, max_day, forecast_horizon):
			max_date = cast(pd.Timestamp, pd.Timestamp(f"{year - 1}-12-31") + pd.Timedelta(days=start_day))
			model = model_builder()

			# Forecasting
			forecast = model.fit_predict(df, min_date, max_date, forecast_horizon, features, additional_date_features)

			if isinstance(model, BoosterModel):
				for k, v in zip(model.feature_names_, model.feature_importances_, strict=True):
					feature_importance_d.setdefault(k, []).append(v)

			# Metrics and residuals
			if not model.test_.empty:
				actual = model.test_[Column.PRECIPITATION].values
				predicted = np.array(forecast[: len(model.test_)])  # type: ignore

				# MAE
				mae: float = mean_absolute_error(actual, predicted)  # type: ignore
				yearly_mae_results[year].append(mae)

				# Custom metric
				custom_metric = calculate_custom_rain_error(actual, predicted)  # type: ignore
				custom_metric_results.append(custom_metric)

				# Collect residuals by forecast step
				residuals = actual - predicted
				for i in range(len(residuals)):
					if i < forecast_horizon:  # Guard against overflows
						residuals_list_by_step[i].append(residuals[i])

	# Compute MAE statistics
	yearly_mae_means = [np.mean(yearly_mae_results[year]) for year in years if yearly_mae_results[year]]
	overall_mean = float(np.mean(yearly_mae_means)) if yearly_mae_means else 0.0
	overall_std = float(np.std(yearly_mae_means)) if len(yearly_mae_means) > 1 else 0.0

	# Custom metric mean
	custom_metric_mean = float(np.mean(custom_metric_results)) if custom_metric_results else 0.0

	# Compute std for each forecast step
	std_residuals_list = [
		float(np.std(step_residuals)) if step_residuals else 0.0 for step_residuals in residuals_list_by_step
	]

	# Feature importance
	if feature_importance_d:
		feature_importance = pd.DataFrame(
			{"feature": feature_importance_d.keys(), "importance": [np.mean(v) for v in feature_importance_d.values()]}
		).sort_values("importance", ascending=False)
		return overall_mean, overall_std, custom_metric_mean, std_residuals_list, feature_importance

	return overall_mean, overall_std, custom_metric_mean, std_residuals_list, None  # Now returns a list


def days_in_year(year: int) -> int:
	return pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31").size
