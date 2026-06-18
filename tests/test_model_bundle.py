from __future__ import annotations

from datetime import date

import pytest

from precipitation_predictor.models.model_bundle import MANIFEST_VERSION, ModelManifest


def test_manifest_roundtrip_json() -> None:
	manifest = ModelManifest(
		version=MANIFEST_VERSION,
		model_name="XGBoost",
		seed=28,
		min_date=date(1949, 12, 26),
		max_date=date(2023, 9, 10),
		forecast_horizon=14,
		feature_columns=["precipitation_lag_14d", "day_of_year"],
		additional_date_features=["day_of_year"],
	)
	restored = ModelManifest.from_json(manifest.to_json())
	assert restored == manifest


@pytest.mark.parametrize(
	("payload", "match"),
	[
		({"version": 99}, "Unsupported manifest version"),
		({"version": 1, "model_name": 1}, "manifest.model_name"),
		({"version": 1, "model_name": "X", "seed": "x"}, "manifest.seed"),
	],
)
def test_manifest_rejects_invalid_payload(payload: dict[str, object], match: str) -> None:
	with pytest.raises(ValueError, match=match):
		ModelManifest.from_dict(payload)
