from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature


SEED = 28
FORECAST_HORIZON = 14

DEFAULT_FEATURES = [
	*[Feature.lag(Column.PRECIPITATION, p) for p in [14, 16, 18, 20]],
	*[Feature.lag(Column.AVERAGE_HUMIDITY, p) for p in [14]],
	*[Feature.rolling_std(Column.PRECIPITATION, p) for p in [14, 25]],
]
DEFAULT_DATE_FEATURES = [Column.DAY_OF_YEAR]
MAX_FEATURE_PERIOD = max((f.window for f in DEFAULT_FEATURES), default=0)

RESULTS_DIR = "./results"
BILBAO_RESULTS_DIR = f"{RESULTS_DIR}/bilbao"
SEASONALITY_OUTPUT_PATH = f"{RESULTS_DIR}/seasonality/7d-rolling-sum-prec.png"
BENCHMARK_RESULTS_DIR = f"{RESULTS_DIR}/benchmark"
BENCHMARK_OUTPUT_PATH = f"{BENCHMARK_RESULTS_DIR}/output.txt"

PREDICTION_DATES = ["2022-07-27", "2024-03-14", "2023-08-27", "2023-09-11", "2025-02-26"]

# Per-forecast-day residual spread (mm) for Bilbao chart uncertainty bands (±1σ).
# One entry per day in FORECAST_HORIZON: std(actual − predicted) pooled across all
# expanding-window CV folds in run_prediction_benchmark (see internal/prediction.py).
# Snapshot from benchmark.py with DEFAULT_FEATURES, DEFAULT_DATE_FEATURES, Bilbao data,
# years 1951–2024, and XGBoostWrapper(SEED). Regenerate via ./scripts/benchmark.sh and
# copy the "Std residuals list" line from results/benchmark/output.txt.
STD_RESIDUALS_LIST = [
	7.753461613310481,
	8.64236262051793,
	8.607975573892729,
	6.949425690207271,
	7.312719993199972,
	7.736787647758962,
	7.495041549314444,
	7.389810418776884,
	8.492416456844554,
	7.575075174160229,
	6.751153569483533,
	7.16921973460898,
	7.901264913747243,
	9.960110861955693,
]


def bilbao_data_files() -> list[str]:
	return [f"./data/Bilbao/historical_climate_data_BILBAO_{i}.json" for i in range(0, 11)]


def city_data_files(city_dir: str, file_prefix: str, start: int, end: int) -> list[str]:
	return [f"./data/{city_dir}/{file_prefix}_{i}.json" for i in range(start, end)]


def seasonality_city_paths() -> list[tuple[str, list[str]]]:
	return [
		("Bilbao", bilbao_data_files()),
		(
			"San Sebastian",
			city_data_files("SanSebastian", "historical_climate_data_SAN_SEBASTIAN", 1, 7),
		),
		("Valencia", city_data_files("Valencia", "historical_climate_data_Valencia", 1, 8)),
		("Madrid", city_data_files("Madrid", "historical_climate_data_Madrid", 1, 8)),
		("Malaga", city_data_files("Malaga", "historical_climate_data_Malaga", 1, 9)),
	]
