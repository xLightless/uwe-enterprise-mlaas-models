"""
This module contains utilities for handling the preprocessing of data.

Written by Reece Turner, 22036698.
"""

import pandas as pd
# from sklearn.preprocessing import (
#     OneHotEncoder,
#     StandardScaler,
#     OrdinalEncoder,
#     MinMaxScaler
# )
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
import numpy as np
import re


class DataPreprocessor:
    """
    The DataPreprocessor class automatically handles the preprocessing of
    linear/tabular data in the form of numerical, datetime, and
    categorical data.

    It is designed to handle the following tasks:
    - Data Cleaning
    - Data Normalisation
    - Data Standardisation
    - Data Encoding
    - Data Imputation
    - Data Scaling
    """
    def __init__(
        self,
        df: pd.DataFrame
    ):
        self._df = df
        self.df = self.process_df(df)
        self.labels = self.get_labels(df)

    def __format_column_words(
        self,
        cols: str
    ) -> str:
        """
        This internal function seperates words in a string by
        spaces and upper case each word.

        returns: str
        """
        return re.sub(
            r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', cols
        )

    def get_labels(
        self,
        df: pd.DataFrame
    ) -> dict:
        """
        Get the labels of the dataframe.

        This function processes the labels to remove any
        unnecessary characters, spaces, or other inconsistencies.

        returns: dict
        """

        # Copy the dataframe to avoid modifying the original
        # then remove any inconsistencies in the column names
        data_frame = df.copy()
        data_frame.columns = [
            self.__format_column_words(col)
                .title().replace(' ', '')
                .replace('_', '')
                for col in df.columns
        ]

        numerical_cols = data_frame.select_dtypes(
            include=[np.number]
        ).columns

        categorical_cols = data_frame.select_dtypes(
            include=[object]
        ).columns

        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }

    def get_categorical_columns(self):
        """
        Get the categorical columns of the dataframe.
        """
        return self.labels['categorical']

    def get_numerical_columns(self):
        """
        Get the numerical columns of the dataframe.
        """
        return self.labels['numerical']

    def process_df(self, df):
        """
        Processes the dataframe, returning updated rows, and column names.
        """
        labels = self.get_labels(df)
        updated_cols = [
            self.__format_column_words(col)
            .title().replace(' ', '').replace('_', '')
            for col in df.columns
        ]

        df.columns = updated_cols
        return df

    def download(
        self,
        path: str = "",
        df=None
    ):
        """
        Download the preprocessed data as a CSV file.

        - If path and df are provided, download to the specified path.
        - If only df is provided, download to the current directory.
        - If only path is provided, download the active dataframe to that path.

        returns: None
        """

        if df and len(path) > 0:
            self.df.to_csv(path, index=False)
            return None

        elif df and len(path) == 0:
            return self.df.to_csv(index=False)

        df_cols = self.get_labels(self.df)
        nums = df_cols['numerical'].tolist()
        categories = df_cols['categorical'].tolist()

        self.df.columns = nums + categories
        self.df.to_csv(path, index=False)
        return None
