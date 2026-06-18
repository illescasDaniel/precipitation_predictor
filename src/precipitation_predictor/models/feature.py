from dataclasses import dataclass

from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature_type import FeatureType


@dataclass
class Feature:
	new_col: str
	col: str
	window: int
	feature_type: FeatureType

	@staticmethod
	def lag(col: str, window: int) -> "Feature":
		return Feature(Column.lag_col_name(col, window), col, window, FeatureType.LAG)

	@staticmethod
	def rolling_mean(col: str, window: int) -> "Feature":
		return Feature(Column.rolling_mean_col_name(col, window), col, window, FeatureType.ROLLING_MEAN)

	@staticmethod
	def rolling_std(col: str, window: int) -> "Feature":
		return Feature(Column.rolling_std_col_name(col, window), col, window, FeatureType.ROLLING_STD)

	@staticmethod
	def rolling_max(col: str, window: int) -> "Feature":
		return Feature(Column.rolling_max_col_name(col, window), col, window, FeatureType.ROLLING_MAX)

	@staticmethod
	def rolling_min(col: str, window: int) -> "Feature":
		return Feature(Column.rolling_min_col_name(col, window), col, window, FeatureType.ROLLING_MIN)

	@staticmethod
	def rolling_sum(col: str, window: int) -> "Feature":
		return Feature(Column.rolling_sum_col_name(col, window), col, window, FeatureType.ROLLING_SUM)
