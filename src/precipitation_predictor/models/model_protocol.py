from typing import Any, Generic, Protocol, TypeVar

import pandas as pd

from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.feature_type import FeatureType


T = TypeVar("T", bound=Any)


class Model(Protocol, Generic[T]):
	model_name: str
	model: T
	future_dates_: pd.DatetimeIndex
	test_: pd.DataFrame
	train_: pd.DataFrame

	def fit_predict(
		self,
		df: pd.DataFrame,
		min_date: pd.Timestamp,
		max_date: pd.Timestamp,
		periods: int,
		features: list[Feature],
		additional_date_features: list[str],
	) -> list[float]: ...

	def _train_test_split(
		self, df: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp, periods: int
	) -> tuple[pd.DataFrame, pd.DataFrame]:
		train = df.loc[(df[Column.DATE] >= min_date) & (df[Column.DATE] <= max_date)].copy()
		test = df.loc[(df[Column.DATE] > max_date) & (df[Column.DATE] <= max_date + pd.Timedelta(days=periods))].copy()
		return train, test

	def _create_features(
		self,
		future: pd.DataFrame,
		historical_data: pd.DataFrame,  # includes train + predictions
		features: list[Feature],
		additional_date_features: list[str],
	):
		new_columns = {}

		# Add date-related features
		if Column.DAY_OF_YEAR in additional_date_features:
			new_columns[Column.DAY_OF_YEAR] = future[Column.DATE].dt.dayofyear
		if Column.WEEK_OF_YEAR in additional_date_features:
			new_columns[Column.WEEK_OF_YEAR] = future[Column.DATE].dt.isocalendar().week.astype(int)
		if Column.MONTH in additional_date_features:
			new_columns[Column.MONTH] = future[Column.DATE].dt.month
		if Column.YEAR in additional_date_features:
			new_columns[Column.YEAR] = future[Column.DATE].dt.year
		if Column.QUARTER in additional_date_features:
			new_columns[Column.QUARTER] = future[Column.DATE].dt.quarter

		# Define feature addition functions
		def add_lag_feature(lag_col_name: str, feature_name: str, time_window: int):
			new_columns[lag_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[historical_data[Column.DATE] == x - pd.Timedelta(days=time_window)][
						feature_name
					].iloc[0]
					if not historical_data[historical_data[Column.DATE] == x - pd.Timedelta(days=time_window)].empty
					else 0
				)
			)

		def add_rolling_mean_feature(rolling_col_name: str, feature_name: str, time_window: int):
			new_columns[rolling_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].mean()
					if len(
						historical_data[
							(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
							& (historical_data[Column.DATE] < x)
						]
					)
					> 1
					else historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].iloc[-1]
					if not historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					].empty
					else 0
				)
			)

		def add_rolling_std_feature(rolling_col_name: str, feature_name: str, time_window: int):
			new_columns[rolling_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].std()
					if len(
						historical_data[
							(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
							& (historical_data[Column.DATE] < x)
						]
					)
					> 1
					else 0
				)
			)

		def add_rolling_max_feature(rolling_col_name: str, feature_name: str, time_window: int):
			new_columns[rolling_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].max()
					if len(
						historical_data[
							(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
							& (historical_data[Column.DATE] < x)
						]
					)
					> 1
					else historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].iloc[-1]
					if not historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					].empty
					else 0
				)
			)

		def add_rolling_min_feature(rolling_col_name: str, feature_name: str, time_window: int):
			new_columns[rolling_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].min()
					if len(
						historical_data[
							(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
							& (historical_data[Column.DATE] < x)
						]
					)
					> 1
					else historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].iloc[-1]
					if not historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					].empty
					else 0
				)
			)

		def add_rolling_sum_feature(rolling_col_name: str, feature_name: str, time_window: int):
			new_columns[rolling_col_name] = future[Column.DATE].apply(
				lambda x: (
					historical_data[
						(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
						& (historical_data[Column.DATE] < x)
					][feature_name].sum()
					if len(
						historical_data[
							(historical_data[Column.DATE] >= x - pd.Timedelta(days=time_window))
							& (historical_data[Column.DATE] < x)
						]
					)
					> 0
					else 0
				)
			)

		# Add feature-related columns
		for f in features:
			match f.feature_type:
				case FeatureType.LAG:
					add_lag_feature(f.new_col, f.col, time_window=f.window)
				case FeatureType.ROLLING_MEAN:
					add_rolling_mean_feature(f.new_col, f.col, time_window=f.window)
				case FeatureType.ROLLING_STD:
					add_rolling_std_feature(f.new_col, f.col, time_window=f.window)
				case FeatureType.ROLLING_MAX:
					add_rolling_max_feature(f.new_col, f.col, time_window=f.window)
				case FeatureType.ROLLING_MIN:
					add_rolling_min_feature(f.new_col, f.col, time_window=f.window)
				case FeatureType.ROLLING_SUM:
					add_rolling_sum_feature(f.new_col, f.col, time_window=f.window)

		# Concatenate all new columns to the future DataFrame
		new_df = pd.DataFrame(new_columns)
		future = pd.concat([future, new_df], axis=1)
		return future
