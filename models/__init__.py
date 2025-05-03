"""
Main moduule package file for importing the preprocessing module.

Written by Reece Turner, 22036698.
"""

from .preprocessing.preprocessor import DataPreprocessor
from .preprocessing.standardizer import DataStandardizer
from .preprocessing.encoding import Encoder
from .preprocessing.scaling import Scaler
from .preprocessing.imputing import Imputer

from .types.standard.linear_regression import LinearRegression
from .types.standard.knn import KNN
from .types.experimental.mars.mars import MARS
from .types.experimental.mars.mars_gpu import MARSGPU
from .types.experimental.catboost.catboost import CatBoost

from .selection.grid_search import GridSearch

from .config import (
    TARGET_VARIABLE_COL_NUM,
    RANDOM_STATE,
    TESTING_DATA_SIZE,
    datasets_processed_directory,
    datasets_raw_directory,
    requirements,
    insurance_dataset,
    gdpr_protected_cols,
    invalid_cols,
    medical_protected_cols,
    datetime_cols,
    arguments,
)

__all__ = [
    "DataPreprocessor",
    "DataStandardizer",
    "Encoder",
    "Scaler",
    "Imputer",
    "TARGET_VARIABLE_COL_NUM",
    "RANDOM_STATE",
    "TESTING_DATA_SIZE",
    "datasets_processed_directory",
    "datasets_raw_directory",
    "requirements",
    "insurance_dataset",
    "gdpr_protected_cols",
    "invalid_cols",
    "medical_protected_cols",
    "datetime_cols",
    "LinearRegression",
    "KNN",
    "arguments",
    "MARS",
    "MARSGPU",
    "CatBoost",
    "GridSearch"
]
