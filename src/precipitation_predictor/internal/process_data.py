import json
from typing import Any, cast

import numpy as np
import pandas as pd
from babel.numbers import parse_decimal

from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.feature_type import FeatureType


def load_data(file_paths: list[str]) -> list[dict[str, Any]]:
	combined_data = []
	for file_path in file_paths:
		try:
			with open(file_path, "r", encoding="utf-8") as f:
				data = json.load(f)
				combined_data.extend(data)
			print(f"Successfully loaded data from {file_path}")
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
	return combined_data


def process_data(data: list[dict[str, Any]]) -> pd.DataFrame:

	df = pd.DataFrame(data)
	df.drop(["indicativo", "nombre", "provincia", "altitud"], axis=1, inplace=True)

	numbers_locale = "es_ES"

	def parse_prec(value: str):
		match value:
			case "Ip":
				return 0.05
			case "Acum":
				return 0
			case _:
				return parse_number(value)

	def parse_number(number: object) -> float | object:
		if number is None:
			return 0
		if isinstance(number, (float, int)) and pd.isna(number):
			return number
		if number in ("", "0", 0, None):
			return 0
		return float(parse_decimal(str(number), locale=numbers_locale))

	def process_numeric_serie(series: pd.Series) -> pd.Series:
		numeric_serie_with_na = series.apply(parse_number)
		if bool(numeric_serie_with_na.isna().any()):
			mean_value = numeric_serie_with_na.mean(skipna=True)
			if mean_value is None or (isinstance(mean_value, float) and pd.isna(mean_value)):
				mean_value = 0
			return cast(pd.Series, numeric_serie_with_na.fillna(mean_value))
		return cast(pd.Series, numeric_serie_with_na)

	def process_prec_serie(series: pd.Series) -> pd.Series:
		numeric_serie_with_na = series.apply(parse_prec)
		if bool(numeric_serie_with_na.isna().any()):
			mean_value = numeric_serie_with_na.mean(skipna=True)
			if mean_value is None or (isinstance(mean_value, float) and pd.isna(mean_value)):
				mean_value = 0
			return cast(pd.Series, numeric_serie_with_na.fillna(mean_value))
		return cast(pd.Series, numeric_serie_with_na)

	# Essential features

	df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")
	df["prec"] = process_prec_serie(pd.Series(df["prec"]))
	# Features to test

	numeric_cols = ["sol", "tmed", "tmin", "tmax", "velmedia", "racha", "presMax", "presMin", "hrMedia"]
	for col in numeric_cols:
		df[col] = process_numeric_serie(pd.Series(df[col]))

	# direction_labels = [
	# 	'North', 'NorthEast', 'East', 'SouthEast',
	# 	'South', 'SouthWest', 'West', 'NorthWest'
	# ]
	# Define the degree boundaries for each direction
	degree_boundaries = [0, 45, 90, 135, 180, 225, 270, 315, 360]

	def map_direction(value: float):
		if pd.isna(value):
			return pd.NA
		degrees = value * 10  # Convert tens of degrees to degrees
		for i in range(len(degree_boundaries) - 1):
			if degree_boundaries[i] <= degrees < degree_boundaries[i + 1]:
				return i
		return 0  # Edge case for exactly 360 degrees

	dir_numeric = pd.to_numeric(df["dir"], errors="coerce")
	dir_cleaned = dir_numeric.mask(dir_numeric.isin(np.array([88, 99])))
	mapped_dir = dir_cleaned.apply(map_direction)
	df["dir"] = mapped_dir.fillna(mapped_dir.mean(skipna=True)).astype(int)

	# Remove some other features

	df.drop(
		["horatmin", "horatmax", "horaPresMin", "horaHrMin", "horaPresMax", "horaHrMax", "horaracha", "hrMax", "hrMin"],
		axis=1,
		errors="ignore",
		inplace=True,
	)

	# Rename features

	df.rename(
		columns={
			"fecha": Column.DATE,
			"tmed": Column.AVERAGE_TEMPERATURE,
			"prec": Column.PRECIPITATION,
			"tmin": Column.MIN_TEMPERATURE,
			"tmax": Column.MAX_TEMPERATURE,
			"dir": Column.WIND_DIRECTION,
			"velmedia": Column.AVERAGE_WIND_SPEED,
			"racha": Column.MAX_WIND_GUST,
			"presMax": Column.MAX_PRESSURE,
			"presMin": Column.MIN_PRESSURE,
			"hrMedia": Column.AVERAGE_HUMIDITY,
			"sol": Column.SUNSHINE,
		},
		inplace=True,
	)

	return df


def create_lag_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].shift(time_window).fillna(0)


def create_rolling_mean_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].rolling(time_window, min_periods=1).mean().fillna(0)


def create_rolling_std_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].rolling(time_window, min_periods=1).std().fillna(0)


def create_rolling_max_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].rolling(time_window, min_periods=1).max().fillna(0)


def create_rolling_min_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].rolling(time_window, min_periods=1).min().fillna(0)


def create_rolling_sum_feature(df: pd.DataFrame, column_name: str, time_window: int):
	return df[column_name].rolling(time_window, min_periods=1).sum().fillna(0)


def create_temporal_features(
	df: pd.DataFrame, features: list[Feature], additional_date_features: list[str], max_period: int
) -> pd.DataFrame:
	new_columns = {}

	if Column.DAY_OF_YEAR in additional_date_features:
		new_columns[Column.DAY_OF_YEAR] = df[Column.DATE].dt.dayofyear
	if Column.WEEK_OF_YEAR in additional_date_features:
		new_columns[Column.WEEK_OF_YEAR] = df[Column.DATE].dt.isocalendar().week.astype(int)
	if Column.MONTH in additional_date_features:
		new_columns[Column.MONTH] = df[Column.DATE].dt.month
	if Column.YEAR in additional_date_features:
		new_columns[Column.YEAR] = df[Column.DATE].dt.year
	if Column.QUARTER in additional_date_features:
		new_columns[Column.QUARTER] = df[Column.DATE].dt.quarter

	for f in features:
		match f.feature_type:
			case FeatureType.LAG:
				new_columns[f.new_col] = create_lag_feature(df, f.col, f.window)
			case FeatureType.ROLLING_MEAN:
				new_columns[f.new_col] = create_rolling_mean_feature(df, f.col, f.window)
			case FeatureType.ROLLING_STD:
				new_columns[f.new_col] = create_rolling_std_feature(df, f.col, f.window)
			case FeatureType.ROLLING_MAX:
				new_columns[f.new_col] = create_rolling_max_feature(df, f.col, f.window)
			case FeatureType.ROLLING_MIN:
				new_columns[f.new_col] = create_rolling_min_feature(df, f.col, f.window)
			case FeatureType.ROLLING_SUM:
				new_columns[f.new_col] = create_rolling_sum_feature(df, f.col, f.window)

	new_df = pd.DataFrame(new_columns)
	df = pd.concat([df, new_df], axis=1)

	df.drop(index=range(max_period), inplace=True)
	df.dropna(inplace=True)
	df.reset_index(drop=True, inplace=True)
	return df
