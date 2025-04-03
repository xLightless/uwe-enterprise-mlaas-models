"""
This module contains the hyperparameters for each model in the repository.

"""

import os
import argparse

# Dataset params
TARGET_VARIABLE_COL_NUM = 0

# Model params
RANDOM_STATE = 42
TESTING_DATA_SIZE = 0.4

# Terminal arguments and dataset paths
current_directory = os.path.dirname(
    os.path.abspath("main.py")
) + "\\"

datasets_raw_directory = current_directory + "datasets\\raw\\"
datasets_processed_directory = current_directory + "datasets\\processed\\"

# Files and Datasets
requirements = current_directory + "requirements.txt"
insurance_dataset = datasets_raw_directory + "insurance.csv"

# gdpr_protected_cols = [
#     "AccidentDate",
#     "ClaimDate",
#     "AccidentDescription",
#     "InjuryDescription",
#     "Gender",
#     "PoliceReportFiled",
#     "WitnessPresent"
# ]

# medical_protected_cols = [
#     "InjuryPrognosis",
#     "DominantInjury",
#     "Whiplash",
#     "MinorPsychologicalInjury"
# ]

gdpr_protected_cols = [
    "DriverAge"
    "Gender",
]

invalid_cols = [
    "AccidentDescription",
    "WeatherConditions",
    "AccidentDescription",
]

datetime_cols = [
    "AccidentDate",
    "ClaimDate",
]

medical_protected_cols = [
    "InjuryPrognosis"
]


def validate_protected(value):
    """ Validate the protected attributes argument. """
    if not (value.startswith("[") and value.endswith("]")):
        raise argparse.ArgumentTypeError(
            "The --protected argument must be in the format " +
            "[str, ..., n - 1] and wrapped in quotes."
        )
    # Remove brackets and split by commas
    return value[1:-1].split(",") or None


arguments = {
    "--data": {
        "type": str,
        "required": False,
        "default": insurance_dataset,
        "help": "Dataset file path (CSV format). Default: insurance dataset"
    },
    "--o": {
        "type": str,
        "default": False,
        "help": "Outputs the dataset's columns or specific column data"
    },
    "--download": {
        "action": "store_true",
        "default": False,
        "help": "Downloads the dataset to the specified path"
    },
    "--row": {
        "type": int,
        "default": False,
        "help": "Outputs a specific row from the dataset"
    },
    "--install": {
        "action": "store_true",
        "default": False,
        "help": "Installs dependencies for this service"
    },
    "--model": {
        "type": str,
        "required": False,
        "default": None,
        "help": "Specify the model to use, e.g., 'linear', 'knn'"
    },
    "--verbose": {
        "action": "store_true",
        "default": False,
        "help": "Outputs detailed model information"
    },
    "--seed": {
        "type": int,
        "default": RANDOM_STATE,
        "help": "Random state seed for reproducibility. Default: 42"
    },
    "--train": {
        "action": "store_true",
        "default": False,
        "help": "Train the specified model"
    },
    "--target": {
        "type": str,
        "default": None,
        "help": "Specify the target variable for the model"
    },
    "--protected": {
        "type": str,
        "default": validate_protected,
        "help": "Specify the protected attributes for the model"
    },
    "--test": {
        "action": "store_true",
        "default": False,
        "help": "Test the specified model"
    }
}
