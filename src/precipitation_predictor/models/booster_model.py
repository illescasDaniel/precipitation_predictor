from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BoosterModel(Protocol):
	@property
	def feature_names_(self) -> list[str]: ...
	@property
	def feature_importances_(self) -> np.ndarray: ...
