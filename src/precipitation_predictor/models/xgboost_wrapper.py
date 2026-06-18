import numpy as np
import pandas as pd
import xgboost as xgb

from precipitation_predictor.models.booster_model import BoosterModel
from precipitation_predictor.models.column import Column
from precipitation_predictor.models.feature import Feature
from precipitation_predictor.models.model_protocol import Model


class XGBoostWrapper(Model[xgb.XGBRegressor], BoosterModel):
	def __init__(self, seed: int, use_eval_sets: bool = False) -> None:
		self.model_name = "XGBoost"
		self.__use_eval_sets = use_eval_sets
		self._feature_columns: list[str] = []
		self.model = xgb.XGBRegressor(
			objective="reg:tweedie",
			n_estimators=300,
			learning_rate=0.05,
			max_depth=4,
			subsample=0.9,
			colsample_bytree=0.7,
			min_child_weight=2,
			gamma=0.1,
			random_state=seed,
			n_jobs=-1,
			tweedie_variance_power=1.9,
		)

	def fit(
		self,
		train: pd.DataFrame,
		test: pd.DataFrame,
		features: list[Feature],
		additional_date_features: list[str],
	) -> None:
		included_features = [f.new_col for f in features]
		included_features += additional_date_features
		X_train, y_train = train.loc[:, [*included_features]].copy(), train[Column.PRECIPITATION]
		self._feature_columns = list(X_train.columns)

		if self.__use_eval_sets:
			X_test, y_test = test.loc[:, [*included_features]].copy(), test[Column.PRECIPITATION]
			self.model.fit(
				X_train,
				y_train,
				eval_set=[(X_train, y_train), (X_test, y_test)],
				early_stopping_rounds=10,
				verbose=False,
			)
		else:
			self.model.fit(X_train, y_train)

	def forecast(
		self,
		train: pd.DataFrame,
		test: pd.DataFrame,
		max_date: pd.Timestamp,
		periods: int,
		features: list[Feature],
		additional_date_features: list[str],
	) -> list[float]:
		if not self._feature_columns:
			raise RuntimeError("Model must be fitted or loaded before forecasting.")

		self.future_dates_ = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=periods)
		forecast: list[float] = []
		historical_data = train.copy()

		for future_date in self.future_dates_:
			future = pd.DataFrame({Column.DATE: [future_date]})
			future = self._create_features(future, historical_data, features, additional_date_features)

			pred = self.model.predict(future[self._feature_columns])[0]
			pred = max(0, pred)  # type: ignore
			forecast.append(pred)

			new_row = future.iloc[0].copy()
			new_row[Column.PRECIPITATION] = pred
			historical_data = pd.concat([historical_data, new_row.to_frame().T], ignore_index=True)

		self.test_ = test
		self.train_ = train
		return forecast

	def fit_predict(
		self,
		df: pd.DataFrame,
		min_date: pd.Timestamp,
		max_date: pd.Timestamp,
		periods: int,
		features: list[Feature],
		additional_date_features: list[str],
	) -> list[float]:
		train, test = super().train_test_split(df, min_date, max_date, periods)
		self.fit(train, test, features, additional_date_features)
		return self.forecast(train, test, max_date, periods, features, additional_date_features)

	@property
	def feature_columns(self) -> list[str]:
		return list(self._feature_columns)

	@feature_columns.setter
	def feature_columns(self, columns: list[str]) -> None:
		self._feature_columns = list(columns)

	@property
	def feature_names_(self) -> list[str]:
		return list(self.model.get_booster().feature_names or [])

	@property
	def feature_importances_(self) -> np.ndarray:
		return self.model.feature_importances_
