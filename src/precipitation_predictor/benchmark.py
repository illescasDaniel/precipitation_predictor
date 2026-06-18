from precipitation_predictor.config import BENCHMARK_OUTPUT_PATH, BENCHMARK_RESULTS_DIR
from precipitation_predictor.internal.process_data import load_data, process_data
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper
from precipitation_predictor.utils.benchmark_utils import (
	BenchmarkData,
	BenchmarkDataList,
	run_cv_benchmark,
)
from precipitation_predictor.utils.pandas_utils import configure_pandas


if __name__ == "__main__":
	configure_pandas()

	initial_data: BenchmarkDataList = [
		BenchmarkData(
			features=[
				*[Feature.lag(Column.PRECIPITATION, p) for p in [14, 16, 18, 20]],
				*[Feature.lag(Column.AVERAGE_HUMIDITY, p) for p in [14]],
				*[Feature.rolling_std(Column.PRECIPITATION, p) for p in [14, 25]],
			],
			additional_date_features=[
				Column.DAY_OF_YEAR,
			],
			forecast_horizon=14,
		),
	]

	# Load, process and run benchmark
	data = load_data([f"./data/Bilbao/historical_climate_data_BILBAO_{i}.json" for i in range(0, 11)])
	df = process_data(data)

	seed = 28

	def build_xgb() -> XGBoostWrapper:
		return XGBoostWrapper(seed)

	# cross validation with training data
	run_cv_benchmark(
		df=df,
		forecast_horizon=None,
		initial_data=initial_data,
		output_file=BENCHMARK_OUTPUT_PATH,
		feature_importance_figure_folder=BENCHMARK_RESULTS_DIR,
		model_builder=build_xgb,
		# years=[2024],
		years=[y for y in range(1951, 2025)],
	)
	# why 1951-2024?
	# - Start date: Since the data starts in December 1949, you need to allow for the lag days (from the forecast horizon) to account for the time series features.
	# - First year to predict: After accounting for the lag days, you can start predictions from 1951, as you'd need the entire 1950 year for the model to be trained with the lag features.
	# - Last year: Since 2025 isn't fully available, 2024 is indeed the latest year you can predict for.
