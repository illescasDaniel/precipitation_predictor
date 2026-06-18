from math import ceil
from pathlib import Path
from typing import Iterable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from precipitation_predictor.config import SEASONALITY_OUTPUT_PATH
from precipitation_predictor.models.booster_model import BoosterModel
from precipitation_predictor.models.column import Column


def visualize_average_precipitation_across_all_years(
	data_list: list[tuple[str, pd.DataFrame]],
	output_path: str | Path = SEASONALITY_OUTPUT_PATH,
	show: bool = False,
) -> Path:
	fig = plt.figure(figsize=(12, 6))

	colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta"]  # Add more colors if needed

	# Plot data for each place
	for idx, (place_name, df) in enumerate(data_list):
		df = df.copy()
		df[Column.DAY_OF_YEAR] = df[Column.DATE].dt.dayofyear
		y_label = "_precipitation_rolling_7d_sum"

		df[y_label] = df[Column.PRECIPITATION].rolling(7, min_periods=1).sum()

		# Group by day of year and calculate mean across all years
		avg_across_all_years_7_day_rolling = df.groupby(Column.DAY_OF_YEAR)[y_label].mean().reset_index()

		# Plot with a unique color and label for the legend
		sns.lineplot(
			data=avg_across_all_years_7_day_rolling,
			x=Column.DAY_OF_YEAR,
			y=y_label,
			color=colors[idx % len(colors)],
			linewidth=1,
			label=place_name,
		)

	# Set x-ticks to correspond to the start of each month
	month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
	month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "(end)"]
	plt.xticks(ticks=month_starts, labels=month_labels)

	plt.title("7-Day Rolling Sum of Precipitation Across All Years")
	plt.xlabel("Month")
	plt.ylabel("Precipitation (mm)")
	plt.legend(title="Location", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=5, borderaxespad=0.0)

	plt.tight_layout()

	out = Path(output_path)
	out.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out, dpi=150, bbox_inches="tight")
	print(f"Saved seasonality chart to {out}")

	if show:
		plt.show()
	else:
		plt.close(fig)

	return out


def visualize_results(
	model_name: str,
	train: pd.DataFrame,
	future_dates: pd.DatetimeIndex,
	forecast: Iterable[float],
	test: pd.DataFrame,
	max_date: Optional[pd.Timestamp],
	std_residuals_list: Optional[list[float]] = None,
	mae_error: Optional[float] = None,
	output_path: str | Path | None = None,
	show: bool = False,
) -> None:
	"""
	Visualizes historical precipitation, predictions, actual values, and an optional prediction band.

	Parameters:
		model_name (str): Name of the model for the title.
		train (pd.DataFrame): Historical training data with DATE and PRECIPITATION columns.
		future_dates (Iterable): Dates for the forecast.
		forecast (Iterable): Predicted precipitation values.
		test (pd.DataFrame): Test data with actual values (DATE and PRECIPITATION).
		max_date (Optional[pd.Timestamp]): Maximum date for context.
		std_residuals_list (Optional[list[float]]): List of standard deviations for each day in the forecast horizon.
		mae_error (Optional[float]): Mean Absolute Error to display in the plot.
	"""
	fig = plt.figure(figsize=(14, 8))

	# Plot historical data
	sns.lineplot(data=train, x=Column.DATE, y=Column.PRECIPITATION, label="Historical")

	# Create prediction DataFrame
	pred_df = pd.DataFrame({Column.DATE: future_dates, "prediction": forecast})

	# Add progressive prediction band
	if std_residuals_list is not None:
		deviations = std_residuals_list[: len(future_dates)]
		pred_df["lower_band"] = np.maximum(pred_df["prediction"] - deviations, 0)
		pred_df["upper_band"] = pred_df["prediction"] + deviations
		plt.fill_between(
			pred_df[Column.DATE],
			pred_df["lower_band"],
			pred_df["upper_band"],
			color="red",
			alpha=0.2,
			label="Prediction Band (±1σ)",
		)

	# Plot predictions
	sns.lineplot(
		data=pred_df,
		x=Column.DATE,
		y="prediction",
		label="Predictions",
		color="red",
	)

	# Plot actual values if available
	if not test.empty:
		sns.scatterplot(
			data=test,
			x=Column.DATE,
			y=Column.PRECIPITATION,
			label="Actual Values",
			color="green",
			s=100,
		)

	# Set temporal boundaries
	focus_start = max_date - pd.Timedelta(days=30) if max_date else future_dates[0] - pd.Timedelta(days=30)
	plt.xlim(focus_start, cast(pd.Timestamp, future_dates[-1]))  # pyright: ignore[reportArgumentType]

	# Determine y-axis limits and add buffer
	displayed_df = train[(train[Column.DATE] >= focus_start) & (train[Column.DATE] < future_dates[-1])]
	try:
		max_y = max(
			max(displayed_df[Column.PRECIPITATION]),
			max(pred_df["prediction"]),
			max(test[Column.PRECIPITATION]) if not test.empty else -float("inf"),
		)
		min_y = min(
			min(displayed_df[Column.PRECIPITATION]),
			min(pred_df["lower_band"] if "lower_band" in pred_df else pred_df["prediction"]),
			min(test[Column.PRECIPITATION]) if not test.empty else float("inf"),
		)
	except Exception:
		max_y = max(max(displayed_df[Column.PRECIPITATION]), max(pred_df["prediction"]))
		min_y = min(
			min(displayed_df[Column.PRECIPITATION]),
			min(pred_df["lower_band"] if "lower_band" in pred_df else pred_df["prediction"]),
		)

	max_y = max(ceil(max_y) + 0.5, 10)
	min_y = -0.5
	plt.ylim(bottom=min_y, top=max_y)

	# Add horizontal grid lines at integer values
	plt.grid(axis="y", linestyle="--", alpha=0.7)
	plt.yticks(range(int(min_y), int(max_y) + 1, 1))

	# Set x-axis ticks
	all_xticks = sorted(set(displayed_df[Column.DATE]).union(set(pred_df[Column.DATE])))
	plt.xticks(ticks=all_xticks, rotation=90)

	# Titles and labels
	plt.title(f"{model_name} - 30-Day Context + Predictions")
	plt.ylabel("Precipitation (mm)")

	# Adjust subplot to leave space at the bottom
	plt.subplots_adjust(bottom=0.25)

	# Adjust legend position to figure coordinates
	plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.05), bbox_transform=plt.gcf().transFigure, ncol=4)

	# Add MAE box below the plot if provided
	if mae_error is not None:
		plt.figtext(
			0.84,
			0.07,
			f"MAE: {mae_error:.2f}",
			ha="center",
			va="center",
			fontsize=16,
			bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
		)

	_save_or_show_figure(fig, output_path, "precipitation chart", show)


def visualize_results_as_levels(
	model_name: str,
	train: pd.DataFrame,
	future_dates: pd.DatetimeIndex,
	forecast: Iterable[float],
	test: pd.DataFrame,
	max_date: Optional[pd.Timestamp],
	std_residuals_list: Optional[list[float]] = None,
	mae_error: Optional[float] = None,  # Added parameter
	output_path: str | Path | None = None,
	show: bool = False,
) -> None:
	"""
	Visualizes historical precipitation levels, predictions, actual values, and an optional prediction band.

	Parameters:
		model_name (str): Name of the model for the title.
		train (pd.DataFrame): Historical training data with DATE and PRECIPITATION columns (in mm).
		future_dates (Iterable): Dates for the forecast.
		forecast (Iterable): Predicted precipitation values (in mm).
		test (pd.DataFrame): Test data with actual values (DATE and PRECIPITATION in mm).
		max_date (Optional[pd.Timestamp]): Maximum date for context.
		std_residuals_list (Optional[list[float]]): List of standard deviations for each day in the forecast horizon.
		mae_error (Optional[float]): Mean Absolute Error to display below the plot.
	"""

	def mm_to_level(mm: float) -> int:
		if mm < 2:
			return 0
		elif 2 <= mm <= 5:
			return 1
		elif 5 < mm <= 10:
			return 2
		elif 10 < mm < 50:
			return 3
		else:
			return 4

	# Prepare data copies
	train = train.copy()
	test = test.copy()
	train.loc[:, Column.PRECIPITATION] = train[Column.PRECIPITATION].apply(mm_to_level)
	forecast_level = list(map(mm_to_level, forecast))

	fig = plt.figure(figsize=(13, 4))

	# Plot historical data
	sns.lineplot(data=train, x=Column.DATE, y=Column.PRECIPITATION, label="Historical")

	# Create prediction DataFrame
	pred_df = pd.DataFrame({Column.DATE: future_dates, "prediction": forecast_level})

	# Add progressive prediction band if std_residuals_list is provided
	if std_residuals_list is not None:
		deviations = std_residuals_list[: len(future_dates)]
		forecast_mm = list(forecast)  # Original forecast in mm
		lower_band_mm = [max(0, f - deviations[i]) for i, f in enumerate(forecast_mm)]
		upper_band_mm = [f + deviations[i] for i, f in enumerate(forecast_mm)]
		lower_band_level = [mm_to_level(mm) for mm in lower_band_mm]
		upper_band_level = [mm_to_level(mm) for mm in upper_band_mm]

		pred_df["lower_band"] = lower_band_level
		pred_df["upper_band"] = upper_band_level

		plt.fill_between(
			pred_df[Column.DATE],
			pred_df["lower_band"],
			pred_df["upper_band"],
			step="mid",
			color="red",
			alpha=0.2,
			label="Prediction Band (±1σ)",
		)

	# Plot predictions
	sns.lineplot(
		data=pred_df,
		x=Column.DATE,
		y="prediction",
		label="Predictions",
		color="red",
	)

	# Plot actual values if available
	if not test.empty:
		test.loc[:, Column.PRECIPITATION] = test[Column.PRECIPITATION].apply(mm_to_level)
		sns.scatterplot(
			data=test,
			x=Column.DATE,
			y=Column.PRECIPITATION,
			label="Actual Values",
			color="green",
			s=100,
		)

	# Set temporal boundaries
	focus_start = max_date - pd.Timedelta(days=30) if max_date else future_dates[0] - pd.Timedelta(days=30)
	plt.xlim(focus_start, cast(pd.Timestamp, future_dates[-1]))  # pyright: ignore[reportArgumentType]

	# Set y-axis for levels
	plt.ylim(-0.5, 4.5)
	plt.yticks([0, 1, 2, 3, 4])

	# Add horizontal grid lines
	plt.grid(axis="y", linestyle="--", alpha=0.7)

	# Set x-axis ticks
	displayed_df = train[(train[Column.DATE] >= focus_start) & (train[Column.DATE] < future_dates[-1])]
	all_xticks = sorted(set(displayed_df[Column.DATE]).union(set(pred_df[Column.DATE])))
	plt.xticks(ticks=all_xticks, rotation=90)

	# Titles and labels
	plt.title(f"{model_name} - 30-Day Context + Predictions")
	plt.ylabel("Precipitation Level")

	# Adjust layout to leave space at the bottom and left
	plt.subplots_adjust(left=0.15, bottom=0.25)  # Adjusted bottom from 0.45 to 0.25

	# Adjust legend position to figure coordinates
	plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.05), bbox_transform=plt.gcf().transFigure, ncol=4)

	# Add level description on the left
	level_description = "Levels:\n0: <2mm\n1: 2-5mm\n2: 5-10mm\n3: 10-50mm\n4: >50mm"
	plt.figtext(
		0.015,
		0.5,
		level_description,
		ha="left",
		va="center",
		fontsize=9,
		bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
	)

	# Add MAE box below the plot if provided
	if mae_error is not None:
		plt.figtext(
			0.84,
			0.07,
			f"MAE: {mae_error:.2f}",
			ha="center",
			va="center",
			fontsize=16,
			bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
		)

	# Adjust layout
	plt.subplots_adjust(left=0.15, bottom=0.45)
	_save_or_show_figure(fig, output_path, "precipitation levels chart", show)


def _save_or_show_figure(fig: Figure, output_path: str | Path | None, label: str, show: bool) -> None:
	if output_path is not None:
		out = Path(output_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(out, dpi=150, bbox_inches="tight")
		print(f"Saved {label} to {out}")

	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_feature_importance(
	model: BoosterModel,
	output_path: str | Path | None = None,
	show: bool = False,
) -> Figure:
	importance = pd.DataFrame({"feature": model.feature_names_, "importance": model.feature_importances_}).sort_values(
		"importance", ascending=False
	)
	fig = plot_feature_importance_df(importance)
	_save_or_show_figure(fig, output_path, "feature importance chart", show)
	return fig


def plot_feature_importance_df(importance: pd.DataFrame) -> Figure:
	fig = plt.figure(figsize=(10, 10))
	sns.barplot(data=importance, x="importance", y="feature")
	plt.title("Feature Importance")
	plt.tight_layout()
	return fig
