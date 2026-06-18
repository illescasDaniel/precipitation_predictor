from enum import Enum, auto


class FeatureType(Enum):
	LAG = auto()
	ROLLING_MEAN = auto()
	ROLLING_STD = auto()
	ROLLING_MAX = auto()
	ROLLING_MIN = auto()
	ROLLING_SUM = auto()

	def __str__(self) -> str:
		return f"{self.__class__.__name__}.{self.name}"

	def __repr__(self) -> str:
		return self.__str__()
