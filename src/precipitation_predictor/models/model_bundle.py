import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Self

from precipitation_predictor.models.xgboost_wrapper import XGBoostWrapper


MODEL_FILENAME = "model.ubj"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ModelManifest:
	version: int
	model_name: str
	seed: int
	min_date: date
	max_date: date
	forecast_horizon: int
	feature_columns: list[str]
	additional_date_features: list[str]

	def to_json(self) -> str:
		payload = {
			"version": self.version,
			"model_name": self.model_name,
			"seed": self.seed,
			"min_date": self.min_date.isoformat(),
			"max_date": self.max_date.isoformat(),
			"forecast_horizon": self.forecast_horizon,
			"feature_columns": self.feature_columns,
			"additional_date_features": self.additional_date_features,
		}
		return json.dumps(payload, indent=2)

	@classmethod
	def from_dict(cls, data: dict[str, object]) -> Self:
		version = data.get("version")
		model_name = data.get("model_name")
		seed = data.get("seed")
		min_date = data.get("min_date")
		max_date = data.get("max_date")
		forecast_horizon = data.get("forecast_horizon")
		feature_columns = data.get("feature_columns")
		additional_date_features = data.get("additional_date_features")

		if not isinstance(version, int):
			raise ValueError("manifest.version must be an int")
		if version != MANIFEST_VERSION:
			raise ValueError(f"Unsupported manifest version: {version}")
		if not isinstance(model_name, str):
			raise ValueError("manifest.model_name must be a str")
		if not isinstance(seed, int):
			raise ValueError("manifest.seed must be an int")
		if not isinstance(min_date, str) or not isinstance(max_date, str):
			raise ValueError("manifest min_date and max_date must be ISO date strings")
		if not isinstance(forecast_horizon, int):
			raise ValueError("manifest.forecast_horizon must be an int")
		if not isinstance(feature_columns, list) or not all(isinstance(c, str) for c in feature_columns):
			raise ValueError("manifest.feature_columns must be a list of str")
		if not isinstance(additional_date_features, list) or not all(
			isinstance(c, str) for c in additional_date_features
		):
			raise ValueError("manifest.additional_date_features must be a list of str")

		return cls(
			version=version,
			model_name=model_name,
			seed=seed,
			min_date=date.fromisoformat(min_date),
			max_date=date.fromisoformat(max_date),
			forecast_horizon=forecast_horizon,
			feature_columns=feature_columns,
			additional_date_features=additional_date_features,
		)

	@classmethod
	def from_json(cls, text: str) -> Self:
		data = json.loads(text)
		if not isinstance(data, dict):
			raise ValueError("manifest root must be a JSON object")
		return cls.from_dict(data)


def save_model_bundle(
	wrapper: XGBoostWrapper,
	output_dir: str | Path,
	*,
	seed: int,
	min_date: date,
	max_date: date,
	forecast_horizon: int,
	additional_date_features: list[str],
) -> Path:
	bundle_dir = Path(output_dir)
	bundle_dir.mkdir(parents=True, exist_ok=True)

	model_path = bundle_dir / MODEL_FILENAME
	manifest_path = bundle_dir / MANIFEST_FILENAME

	wrapper.model.save_model(str(model_path))

	manifest = ModelManifest(
		version=MANIFEST_VERSION,
		model_name=wrapper.model_name,
		seed=seed,
		min_date=min_date,
		max_date=max_date,
		forecast_horizon=forecast_horizon,
		feature_columns=wrapper.feature_columns,
		additional_date_features=additional_date_features,
	)
	manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
	return bundle_dir


def load_model_bundle(bundle_dir: str | Path) -> tuple[XGBoostWrapper, ModelManifest]:
	path = Path(bundle_dir)
	model_path = path / MODEL_FILENAME
	manifest_path = path / MANIFEST_FILENAME

	if not model_path.is_file():
		raise FileNotFoundError(f"Missing model file: {model_path}")
	if not manifest_path.is_file():
		raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

	manifest = ModelManifest.from_json(manifest_path.read_text(encoding="utf-8"))
	wrapper = XGBoostWrapper(manifest.seed)
	wrapper.model.load_model(str(model_path))
	wrapper.feature_columns = manifest.feature_columns
	return wrapper, manifest
