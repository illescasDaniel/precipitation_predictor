import argparse

from precipitation_predictor.config import CLIMATE_DB_PATH
from precipitation_predictor.internal.climate_db import (
	discover_climate_json_files,
	get_connection,
	import_json_files,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Import AEMET climate JSON shards into the local SQLite database.")
	parser.add_argument(
		"--data-dir",
		default="./data",
		help="Root directory to scan for *.json climate files (default: ./data).",
	)
	parser.add_argument(
		"--db-path",
		default=CLIMATE_DB_PATH,
		help=f"SQLite database path (default: {CLIMATE_DB_PATH}).",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	json_files = discover_climate_json_files(args.data_dir)
	if not json_files:
		raise SystemExit(f"No JSON files found under {args.data_dir}")

	conn = get_connection(args.db_path)
	try:
		imported = import_json_files(conn, json_files)
	finally:
		conn.close()

	total = sum(imported.values())
	print(f"\nImported {total} records from {len(imported)} files into {args.db_path}")


if __name__ == "__main__":
	main()
