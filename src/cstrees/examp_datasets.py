from importlib import resources
import pandas as pd

from . import datasets


def sachs_observational():
    data_path = resources.files(datasets) / "sachs_observational.csv"
    return pd.read_csv(data_path)
