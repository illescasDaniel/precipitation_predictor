import pandas as pd


def configure_pandas_visualization():
	pd.set_option("display.max_columns", None)


def configure_pandas():
	pd.set_option("future.no_silent_downcasting", True)
