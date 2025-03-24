"""
This module is the main entry point for the Engineering of AI Models.
It provides direct access to training, testing, and predictions, external
from the MLAAS. Each model should be imported here and tuned via the
configuration file, which will be utilised by the MLAAS afterwards.

To convienently train a model, simply run the following command:
    python main.py --model <model_name> --data <dataset_path> [optional args]
"""

import subprocess
import pandas as pd
import argparse
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.config import RANDOM_STATE, TARGET_VARIABLE_COL_NUM
from models.standard.linear_regression import LinearRegression
from models.standard.knn import KNN

# Directory Paths
current_directory = os.path.dirname(
    os.path.abspath("main.ipynb")
) + "\\"

datasets_directory = current_directory + "datasets\\"

# Files and Datasets
requirements = current_directory + "requirements.txt"
insurance_dataset = datasets_directory + "insurance.csv"


def install(requirements):
    """
    Install all the relevent project dependencies.
    """

    try:
        if os.path.isdir('.venv'):
            activate_script = os.path.join('.venv', 'bin', 'activate')
            subprocess.check_call(['source', activate_script], shell=True)

        with open(requirements, 'r') as f:
            requirements = f.read().splitlines()
            subprocess.check_call(['pip', 'install'] + requirements)
        print("[INFO] Installed dependencies.")

    except FileNotFoundError:
        print("[ERROR] File '%s' not found." % requirements)
    except subprocess.CalledProcessError:
        pass


def train_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Public function for training a model from the passed parameters.
    """

    model_selected = "[INFO] %s model selected."

    # Each new model created, add a case statement here so it can be
    # called in the copmmand line.
    match model_name:
        case "linear":
            model = LinearRegression()
            print(model_selected % "Linear Regression")
        case "knn":
            model = KNN()
            print(model_selected % "K-Nearest Neighbours")
        case _:
            print("[ERROR] Model %s not found." % model_name)
            exit()

    # Fit the model to the training data
    model.fit(X_train, y_train)
    return model


def evaluate_model(model_name, model, X_test, y_test, verbose):
    """
    Public function for testing a trained model.
    """

    # Make predictions on the test data
    predictions = model.predict(X_test)

    if verbose:
        print("[INFO] Predictions:", predictions)
        print("[INFO] Mean Absolute Error:", mean_absolute_error(
            y_test, predictions
        ))
        print("[INFO] Mean Squared Error:", mean_squared_error(
            y_test, predictions
        ))
        print("[INFO] R2 Score:", r2_score(
            y_test, predictions
        ))

    return predictions


def add_parser_args():
    """
    Add the parser arguments to the parser.
    """

    parser = argparse.ArgumentParser(
        description="Run various machine learning models"
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Installs dependencies for this service",
        default=False
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Specify the model to use, e.g. linear",
        default=None
    )

    # Optional arguments
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="Dataset file path (CSV format)",
        default=insurance_dataset
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Outputs model information",
        default=False
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random state seed for reproducing results (default: 42)",
        default=RANDOM_STATE
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the specified model",
        default=False
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the specified model",
        default=False
    )

    return parser


def get_data(args):
    """
    Load the dataset from the specified path.
    """

    if args.data:
        try:
            if not args.data.endswith(".csv"):
                raise FileNotFoundError
            elif not os.path.exists(args.data):
                raise FileNotFoundError
            elif os.path.exists(insurance_dataset):
                raise FileNotFoundError

            data = pd.read_csv(args.data)
        except FileNotFoundError:
            print(
                "[WARNING] Dataset in args '--data' not found. " +
                "Using the default dataset instead..."
            )
            data = pd.read_csv(insurance_dataset)
    else:
        print(
            "[ERROR] No dataset specified. " +
            "Please use --data <dataset_path> OR " +
            "leave it empty to use the default dataset."
        )

    if data.empty:
        print("[ERROR] Dataset is empty.")
        return None


def main():
    """
    Main entry point for the Engineering of AI Models.
    """

    parser = add_parser_args()
    args = parser.parse_args()
    print("*** Executing args *** \n- " + "\n- ".join(
        f"{key}: {value}" for key, value in args.__dict__.items()
    ))

    if args.install:
        print("[INFO] Installing dependencies...")
        install(requirements)
        print("[INFO] Dependencies installed. No further action can be taken.")
        return None

    if args.model is None and not args.install:
        print("[ERROR] No model specified. Please use --model <model_name>.")
        return None

    # Executes the training portion of the model on the split dataset.
    if args.train and not args.test:

        print("[INFO] Attempting to load the dataset...")
        start_time = time.time()
        data = get_data(args.data)

        if not data:
            return None

        end_time = time.time()
        print(
            "[INFO] Dataset loaded. Took %f seconds." % (
                end_time - start_time
            )
        )

        y = data.iloc[:, TARGET_VARIABLE_COL_NUM].values  # Target variable
        X = data.iloc[:, 1:].values  # Features

        # Split the data into training and testing sets
        # (60% training, 40% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.4,
            random_state=args.seed
        )

        print("[INFO] Training model...")
        start_time = time.time()
        model = train_model(
            args.model,
            X_train,
            y_train,
            X_test,
            y_test
        )
        end_time = time.time()
        print("[INFO] Training complete. Took %f seconds." % (
            end_time - start_time
        ))

    elif args.test and not args.train:
        if (os.path.exists(datasets_directory)):
            print("Evaluating %s model..." % args.model)
            evaluate_model(
                args.model,
                model,
                X_test,
                y_test,
                args.verbose
            )

    elif not args.train and not args.test:
        print(
            "[ERROR] No action specified. Please use --train or --test."
        )
    else:
        print(
            "[ERROR] You cannot perform training and testing simultaneously."
        )


if __name__ == "__main__":
    main()
