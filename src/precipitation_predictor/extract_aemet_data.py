import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from precipitation_predictor.config import CLIMATE_DB_PATH
from precipitation_predictor.internal.climate_db import get_connection, upsert_records


AEMET_REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class AemetStation:
	city_dir: str
	file_prefix: str
	idema: str

	def output_path(self, shard: int) -> str:
		return f"./data/{self.city_dir}/{self.file_prefix}_{shard}.json"


AEMET_STATIONS: dict[str, AemetStation] = {
	"bilbao": AemetStation("Bilbao", "historical_climate_data_BILBAO", "1082"),
	"san-sebastian": AemetStation("SanSebastian", "historical_climate_data_SAN_SEBASTIAN", "1024E"),
	"valencia": AemetStation("Valencia", "historical_climate_data_Valencia", "8416"),
	"madrid": AemetStation("Madrid", "historical_climate_data_Madrid", "3195"),
	"malaga": AemetStation("Malaga", "historical_climate_data_Malaga", "6155A"),
}


def get_aemet_api_key(*, env_file: Path | None = None) -> str:
	load_dotenv(env_file)
	api_key = os.environ.get("AEMET_API_KEY", "").strip()
	if not api_key:
		raise ValueError("AEMET_API_KEY is not set. Copy .env.example to .env and add your AEMET Open Data API key.")
	return api_key


def split_date_ranges(
	start_date_str: str,
	end_date_str: str,
	*,
	max_interval_months: int = 6,
) -> list[tuple[str, str]]:
	start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
	end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
	intervals: list[tuple[str, str]] = []

	current_start = start_date
	while current_start < end_date:
		current_end = min(
			current_start + relativedelta(months=max_interval_months),
			end_date,
		)
		intervals.append(
			(
				current_start.strftime("%Y-%m-%d"),
				current_end.strftime("%Y-%m-%d"),
			)
		)
		current_start = current_end + timedelta(days=1)

	return intervals


def fetch_climate_data(
	api_key: str,
	start_date_str: str,
	end_date_str: str,
	idema: str,
) -> list[dict[str, Any]] | None:
	headers = {
		"accept": "application/json",
		"api_key": api_key,
	}

	try:
		url = (
			"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/"
			f"fechaini/{start_date_str}T00:00:00UTC/"
			f"fechafin/{end_date_str}T00:00:00UTC/estacion/{idema}"
		)

		response = requests.get(url, headers=headers, timeout=AEMET_REQUEST_TIMEOUT_SECONDS)
		response.raise_for_status()

		hateoas_response = response.json()
		if hateoas_response.get("estado") != 200:
			print(f"Error in {start_date_str}-{end_date_str}: {hateoas_response.get('descripcion')}")
			return None

		data_response = requests.get(
			hateoas_response["datos"],
			headers=headers,
			timeout=AEMET_REQUEST_TIMEOUT_SECONDS,
		)
		data_response.raise_for_status()
		return cast(list[dict[str, Any]], data_response.json())

	except requests.exceptions.RequestException as error:
		print(f"Error fetching {start_date_str}-{end_date_str}: {error}")
		return None


def fetch_historical_data(
	api_key: str,
	overall_start: str,
	overall_end: str,
	idema: str,
	*,
	output_file: str | None = None,
) -> list[dict[str, Any]]:
	all_data: list[dict[str, Any]] = []

	for start, end in split_date_ranges(overall_start, overall_end):
		print(f"Fetching data from {start} to {end}")
		data = fetch_climate_data(api_key, start, end, idema)
		if data is None:
			print(f"Failed to retrieve data for {start}-{end}")
			break

		all_data.extend(data)
		print(f"Successfully retrieved {len(data)} records")

	if output_file is not None:
		try:
			output_path = Path(output_file)
			output_path.parent.mkdir(parents=True, exist_ok=True)
			with output_path.open("w", encoding="utf-8") as file:
				json.dump(all_data, file, ensure_ascii=False, indent=2)
			print(f"\nData successfully saved to {output_file}")
		except OSError as error:
			print(f"Failed to write output file: {error}")

	if all_data:
		conn = get_connection(CLIMATE_DB_PATH)
		try:
			upserted = upsert_records(conn, all_data)
			print(f"Upserted {upserted} records into {CLIMATE_DB_PATH}")
		finally:
			conn.close()

	return all_data


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch daily climatological records from AEMET Open Data and save raw JSON."
	)
	parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
	parser.add_argument(
		"--end",
		default=datetime.today().strftime("%Y-%m-%d"),
		help="End date (YYYY-MM-DD). Defaults to today.",
	)

	station_group = parser.add_mutually_exclusive_group(required=True)
	station_group.add_argument(
		"--city",
		choices=sorted(AEMET_STATIONS),
		help="Known city station preset.",
	)
	station_group.add_argument("--idema", help="AEMET station idema code for custom downloads.")

	parser.add_argument(
		"--shard",
		type=int,
		help="Shard index for --city output path (e.g. 8 -> ..._Malaga_8.json).",
	)
	parser.add_argument(
		"--output",
		help="Output JSON path. Required with --idema; optional override with --city.",
	)

	return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> tuple[str, str]:
	if args.city is not None:
		station = AEMET_STATIONS[args.city]
		if args.shard is None:
			raise SystemExit("--shard is required when using --city.")
		output_file = args.output or station.output_path(args.shard)
		return station.idema, output_file

	if args.output is None:
		raise SystemExit("--output is required when using --idema.")

	return args.idema, args.output


def main() -> None:
	args = parse_args()
	idema, output_file = resolve_output_path(args)

	try:
		api_key = get_aemet_api_key()
	except ValueError as error:
		raise SystemExit(str(error)) from error

	records = fetch_historical_data(
		api_key,
		args.start,
		args.end,
		idema,
		output_file=output_file,
	)
	print(f"\nTotal records retrieved: {len(records)}")


if __name__ == "__main__":
	main()
