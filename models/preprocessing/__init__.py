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
from sklearn.impute import SimpleImputer
import numpy as np
import re
import os


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
        self.__df = df  # original dataframe
        self.df = self.process_df(df)
        self.labels = self.get_labels(df)

    def __format_column_name_words(
        self,
        cols: str
    ) -> str:
        """
        This internal function seperates words in a string with
        spaces and upper cases each word.

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
            self.__format_column_name_words(col)
                .title().replace(' ', '')
                .replace('_', '') for col in df.columns
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

    def get_missing_value_columns(self, df):
        """
        Get the columns with missing values.
        """
        return df.columns[df.isnull().any()].tolist()

    def process_df(self, df):
        """
        Processes the dataframe, returning updated rows, and column names.
        """

        # Update the dataframe column names
        labels = self.get_labels(df)
        updated_cols = [
            self.__format_column_name_words(col)
            .title().replace(' ', '').replace('_', '')
            for col in df.columns
        ]

        if len(updated_cols) != len(df.columns):
            raise ValueError("Column names not updated.")

        df.columns = updated_cols

        missing_values_cols = self.get_missing_value_columns(df)
        if missing_values_cols:
            # Get the labels of the dataframe
            numerical_cols = labels['numerical']
            categorical_cols = labels['categorical']

            # Search for missing values
            total_missing_nums = df[numerical_cols].isnull().sum().sum()
            total_missing_cats = df[categorical_cols].isnull().sum().sum()
            print(
                "[WARNING] Missing values found. " +
                "If populating value is 0, " +
                "this does not always mean the existing columns are filled."
            )

            # Fill missing numerical values using imputation
            if numerical_cols.any():
                imputer = SimpleImputer(strategy='most_frequent')
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

            print(
                "[INFO] Populated %s numerical values." % total_missing_nums
            )

            if categorical_cols.any():
                imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = imputer.fit_transform(
                    df[categorical_cols]
                )

            print(
                "[INFO] Populated %s categorical values." % (
                    total_missing_cats
                )
            )

        return df

    def download(
        self,
        path: str = "",
        df: pd.DataFrame = None
    ):
        """
        Download the preprocessed data as a CSV file.

        - If path and df are provided, download to the specified path.
        - If only df is provided, download to the current directory.
        - If only path is provided, download the active dataframe to that path.

        returns: None
        """

        # If the previous file exists, remove it, then download the new file
        if len(path) > 0 and os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                print(
                    "[ERROR] The file you are trying to remove " +
                    "is currently open. Please close it and try again."
                )
                return False

        if df is not None and not df.empty and len(path) > 0:
            df.to_csv(path, index=False)
            return None

        elif df is not None and not df.empty and len(path) == 0:
            return df.to_csv(index=False)

        df_cols = self.get_labels(self.df)
        nums = df_cols['numerical'].tolist()
        categories = df_cols['categorical'].tolist()

        self.df.columns = nums + categories
        self.df.to_csv(path, index=False)
        return None
