class Column:
	DATE = "date"
	AVERAGE_TEMPERATURE = "average_temperature"
	PRECIPITATION = "precipitation"
	MAX_WIND_GUST = "max_wind_gust"
	MIN_TEMPERATURE = "min_temperature"
	MAX_TEMPERATURE = "max_temperature"
	WIND_DIRECTION = "wind_direction"
	AVERAGE_WIND_SPEED = "average_wind_speed"
	MIN_PRESSURE = "min_pressure"
	MAX_PRESSURE = "max_pressure"
	AVERAGE_HUMIDITY = "average_humidity"
	SUNSHINE = "sunshine"
	# Temporal features
	DAY_OF_YEAR = "day_of_year"
	WEEK_OF_YEAR = "week_of_year"
	MONTH = "month"
	YEAR = "year"
	QUARTER = "quarter"

	@staticmethod
	def lag_col_name(col: str, period: int) -> str:
		return f"{col}_lag_{period}d"

	@staticmethod
	def rolling_mean_col_name(col: str, period: int) -> str:
		return f"{col}_rolling_{period}d_mean"

	@staticmethod
	def rolling_std_col_name(col: str, period: int) -> str:
		return f"{col}_rolling_{period}d_std"

	@staticmethod
	def rolling_max_col_name(col: str, period: int) -> str:
		return f"{col}_rolling_{period}d_max"

	@staticmethod
	def rolling_min_col_name(col: str, period: int) -> str:
		return f"{col}_rolling_{period}d_min"

	@staticmethod
	def rolling_sum_col_name(col: str, period: int) -> str:
		return f"{col}_rolling_{period}d_sum"
