"""
This module contains the functionality for scaling numerical data into a
normalised format that models can interpret.

Written by Reece Turner, 22036698.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Scaler:
    """
    The Scaler class handles various scaling techniques for numerical data
    within a dataframe.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def minmax(self, columns: list[str]) -> None:
        """
        Scale the specified columns of the dataframe using Min-Max scaling.

        Args:
            columns (list[str]): List of column names to scale.
        """
        scaler = MinMaxScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df

    def standard(self, columns: list[str]) -> None:
        """
        Scale the specified columns of the dataframe using standard scaling.

        Args:
            columns (list[str]): List of column names to scale.
        """
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df
