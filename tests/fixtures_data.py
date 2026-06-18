from __future__ import annotations

from datetime import date, timedelta


def aemet_record(fecha: str, *, prec: str = "1,0", hr_media: str = "70") -> dict[str, str]:
	return {
		"fecha": fecha,
		"indicativo": "1082",
		"nombre": "BILBAO AEROPUERTO",
		"provincia": "BIZKAIA",
		"altitud": "42",
		"tmed": "10,0",
		"prec": prec,
		"tmin": "5,0",
		"horatmin": "05:00",
		"tmax": "15,0",
		"horatmax": "14:00",
		"dir": "35",
		"velmedia": "1,4",
		"racha": "6,1",
		"sol": "6,0",
		"presMax": "1030,0",
		"presMin": "1020,0",
		"hrMedia": hr_media,
	}


def make_daily_records(start: date, days: int) -> list[dict[str, str]]:
	return [aemet_record((start + timedelta(days=offset)).isoformat()) for offset in range(days)]
