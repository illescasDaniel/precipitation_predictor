from precipitation_predictor.config import SEASONALITY_OUTPUT_PATH, seasonality_stations
from precipitation_predictor.internal.climate_db import load_station_records
from precipitation_predictor.internal.process_data import process_data
from precipitation_predictor.utils.pandas_utils import configure_pandas_visualization
from precipitation_predictor.utils.plot_utils import visualize_average_precipitation_across_all_years


def main() -> None:
	configure_pandas_visualization()

	city_data = []
	for place_name, idema in seasonality_stations():
		data = load_station_records(idema)
		city_data.append((place_name, process_data(data)))

	visualize_average_precipitation_across_all_years(city_data, output_path=SEASONALITY_OUTPUT_PATH)


if __name__ == "__main__":
	main()
