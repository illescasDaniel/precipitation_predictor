import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence, cast


DEFAULT_CLIMATE_DB_PATH = Path("./data/climate.sqlite")


def get_connection(db_path: str | Path = DEFAULT_CLIMATE_DB_PATH) -> sqlite3.Connection:
	path = Path(db_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(path)
	conn.row_factory = sqlite3.Row
	init_db(conn)
	return conn


def init_db(conn: sqlite3.Connection) -> None:
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS daily_climate (
			idema TEXT NOT NULL,
			fecha TEXT NOT NULL,
			record_json TEXT NOT NULL,
			PRIMARY KEY (idema, fecha)
		)
		"""
	)
	conn.execute(
		"""
		CREATE INDEX IF NOT EXISTS idx_daily_climate_idema_fecha
		ON daily_climate (idema, fecha)
		"""
	)
	conn.commit()


def upsert_records(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
	if not records:
		return 0

	rows: list[tuple[str, str, str]] = []
	for record in records:
		idema = str(record.get("indicativo", "")).strip()
		fecha = str(record.get("fecha", "")).strip()
		if not idema or not fecha:
			continue
		rows.append((idema, fecha, json.dumps(record, ensure_ascii=False)))

	conn.executemany(
		"""
		INSERT INTO daily_climate (idema, fecha, record_json)
		VALUES (?, ?, ?)
		ON CONFLICT(idema, fecha) DO UPDATE SET record_json = excluded.record_json
		""",
		rows,
	)
	conn.commit()
	return len(rows)


def import_json_file(conn: sqlite3.Connection, file_path: str | Path) -> int:
	path = Path(file_path)
	with path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, list):
		raise ValueError(f"Expected a JSON array in {path}")
	records = [cast(dict[str, Any], record) for record in data if isinstance(record, dict)]
	return upsert_records(conn, records)


def import_json_files(conn: sqlite3.Connection, file_paths: Sequence[str | Path]) -> dict[str, int]:
	imported_by_file: dict[str, int] = {}
	for file_path in file_paths:
		count = import_json_file(conn, file_path)
		imported_by_file[str(file_path)] = count
		print(f"Imported {count} records from {file_path}")
	return imported_by_file


def discover_climate_json_files(data_dir: str | Path = "./data") -> list[Path]:
	root = Path(data_dir)
	return sorted(path for path in root.glob("**/*.json") if path.is_file())


def fetch_records(
	conn: sqlite3.Connection,
	idema: str,
	*,
	start_date: date | None = None,
	end_date: date | None = None,
) -> list[dict[str, Any]]:
	query = "SELECT record_json FROM daily_climate WHERE idema = ?"
	params: list[str] = [idema]

	if start_date is not None:
		query += " AND fecha >= ?"
		params.append(start_date.isoformat())
	if end_date is not None:
		query += " AND fecha <= ?"
		params.append(end_date.isoformat())

	query += " ORDER BY fecha"
	rows = conn.execute(query, params).fetchall()
	return [cast(dict[str, Any], json.loads(row["record_json"])) for row in rows]


def load_station_records(
	idema: str,
	*,
	db_path: str | Path = DEFAULT_CLIMATE_DB_PATH,
	start_date: date | None = None,
	end_date: date | None = None,
	lookback_days: int | None = None,
) -> list[dict[str, Any]]:
	path = Path(db_path)
	if not path.is_file():
		raise FileNotFoundError(
			f"Climate database not found at {path}. Run ./scripts/import_climate_db.sh to rebuild it from data/**/*.json."
		)

	conn = get_connection(path)
	try:
		resolved_start = start_date
		resolved_end = end_date
		if lookback_days is not None:
			if resolved_end is None:
				row = conn.execute(
					"SELECT MAX(fecha) AS max_fecha FROM daily_climate WHERE idema = ?",
					(idema,),
				).fetchone()
				if row is None or row["max_fecha"] is None:
					return []
				resolved_end = date.fromisoformat(cast(str, row["max_fecha"]))
			resolved_start = resolved_end - timedelta(days=lookback_days)

		records = fetch_records(conn, idema, start_date=resolved_start, end_date=resolved_end)
		range_suffix = ""
		if resolved_start is not None or resolved_end is not None:
			range_suffix = f" ({resolved_start} to {resolved_end})"
		print(f"Loaded {len(records)} records for station {idema}{range_suffix}")
		return records
	finally:
		conn.close()
