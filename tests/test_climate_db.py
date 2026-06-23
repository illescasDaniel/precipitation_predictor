from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fixtures_data import aemet_record, make_daily_records
from precipitation_predictor.internal.climate_db import (
	fetch_records,
	get_connection,
	import_json_file,
	load_station_records,
	upsert_records,
)


@pytest.fixture
def climate_db(tmp_path: Path) -> Path:
	db_path = tmp_path / "climate.sqlite"
	conn = get_connection(db_path)
	try:
		upsert_records(conn, make_daily_records(date(2024, 1, 1), 90))
	finally:
		conn.close()
	return db_path


def test_upsert_deduplicates_by_station_and_date(climate_db: Path) -> None:
	conn = get_connection(climate_db)
	try:
		upsert_records(conn, [aemet_record("2024-01-01", prec="9,9")])
		records = fetch_records(conn, "1082", start_date=date(2024, 1, 1), end_date=date(2024, 1, 1))
	finally:
		conn.close()

	assert len(records) == 1
	assert records[0]["prec"] == "9,9"


def test_fetch_records_respects_date_range(climate_db: Path) -> None:
	conn = get_connection(climate_db)
	try:
		records = fetch_records(
			conn,
			"1082",
			start_date=date(2024, 1, 10),
			end_date=date(2024, 1, 20),
		)
	finally:
		conn.close()

	assert len(records) == 11
	assert records[0]["fecha"] == "2024-01-10"
	assert records[-1]["fecha"] == "2024-01-20"


def test_import_json_file_roundtrip(tmp_path: Path) -> None:
	json_path = tmp_path / "sample.json"
	json_path.write_text(
		'[{"fecha":"2024-06-01","indicativo":"1082","prec":"0,0","hrMedia":"50"}]',
		encoding="utf-8",
	)
	db_path = tmp_path / "climate.sqlite"
	conn = get_connection(db_path)
	try:
		imported = import_json_file(conn, json_path)
		records = fetch_records(conn, "1082")
	finally:
		conn.close()

	assert imported == 1
	assert records[0]["fecha"] == "2024-06-01"


def test_load_station_records_lookback(climate_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
	records = load_station_records(
		"1082",
		db_path=climate_db,
		end_date=date(2024, 3, 1),
		lookback_days=10,
	)
	assert len(records) == 11
	assert records[0]["fecha"] == "2024-02-20"
	assert records[-1]["fecha"] == "2024-03-01"
	assert "Loaded 11 records" in capsys.readouterr().out


def test_load_station_records_missing_db(tmp_path: Path) -> None:
	with pytest.raises(FileNotFoundError, match="import_climate_db"):
		load_station_records("1082", db_path=tmp_path / "missing.sqlite")
