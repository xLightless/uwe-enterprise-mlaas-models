"""
This module is the main entry point for the Engineering of AI Models.
It provides direct access to training, testing, and predictions, external
from the MLAAS. Each model should be imported here and tuned via the
configuration file, which will be utilised by the MLAAS afterwards.

To convienently train a model, simply run the following command:
    python main.py --model <model_name> --data <dataset_path> [optional args]

Written by Reece Turner, 22036698.
"""

import argparse
import os
import sys
import time
import subprocess
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import (
    DataPreprocessor,
    requirements,
    datasets_processed_directory,
    datasets_raw_directory,
    insurance_dataset,
    TARGET_VARIABLE_COL_NUM,
    TESTING_DATA_SIZE,
    gdpr_protected_cols,
    invalid_cols,
    medical_protected_cols,
    datetime_cols,
    LinearRegression,
    KNN,
    arguments,
)


def install(requirements_path: str):
    """
    Install all the relevent project dependencies.
    """

    try:
        if os.path.isdir('.venv'):
            activate_script = os.path.join('.venv', 'bin', 'activate')
            subprocess.check_call(['source', activate_script], shell=True)

        with open(requirements_path, 'r', encoding='utf-8') as f:
            r = f.read().splitlines()
            subprocess.check_call(['pip', 'install'] + r)
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
            sys.exit()

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

    # Define arguments in a dictionary


    for arg, params in arguments.items():
        parser.add_argument(arg, **params)

    return parser


def load_dataset(args):
    """
    Load the dataset from the specified path.
    """
    if not args.data.endswith(".csv"):
        raise FileNotFoundError
    elif not os.path.exists(args.data):
        raise FileNotFoundError
    elif os.path.exists(insurance_dataset):
        raise FileNotFoundError

    data = pd.read_csv(args.data)
    return data


def get_data(args=None):
    """
    Load the dataset from the specified path.
    """

    if args is None:
        return None

    if args.data:
        try:
            data = load_dataset(args)
        except FileNotFoundError:
            # print(
            #     "[WARNING] Dataset in args '--data' not found. " +
            #     "Using the default dataset instead..."
            # )

            try:
                data = pd.read_csv(insurance_dataset)
                return data
            except FileNotFoundError:
                print(
                    "[ERROR] Default dataset not found. " +
                    "No further action can be taken."
                )
                sys.exit()
    else:
        print(
            "[ERROR] No dataset specified. " +
            "Please use --data <dataset_path> OR " +
            "leave it empty to use the default dataset."
        )

    if data.empty:
        print("[ERROR] Dataset is empty. No further action can be taken.")
        sys.exit()


def check_missing_values(processor: DataPreprocessor):
    """
    Check for missing values in the dataset being processed.

    - If missing values are found, print a warning message.
    """

    missing_values = processor.df.isnull().sum().sum()
    if missing_values > 0:
        print(
            "[WARNING] Missing values found in the dataset: \n" +
            "- Categorical values: %s\n" % (
                processor.df[
                    processor.get_categorical_columns()
                ].isnull().sum().sum()
            ) +

            "- Numerical values (likely already imputed): %s\n" % (
                processor.df[
                    processor.get_numerical_columns()
                ].isnull().sum().sum()
            )
        )

def run_models(args, processor: DataPreprocessor):
    """
    Run the select model in args for training or testing.
    """

    if (args.model is None and args.o) and not args.install:
        print("[ERROR] No model specified. Please use --model <model_name>.")
        return None

    # Executes the training portion of the model on the split dataset.
    if args.train and not args.test:
        y = processor.df.iloc[:, TARGET_VARIABLE_COL_NUM].values  # Target variable
        X = processor.df.iloc[:, 1:].values  # Features

        # Split the data into training and testing sets
        # (60% training, 40% testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TESTING_DATA_SIZE,
            random_state=args.seed,
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
        if (os.path.exists(datasets_raw_directory)):
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

# flake8: noqa: C901
def main():
    """
    Main entry point for the Engineering of AI Models.
    """

    parser = add_parser_args()
    args = parser.parse_args()
    # print("*** Executing args *** \n- " + "\n- ".join(
    #     f"{key}: {value}" for key, value in args.__dict__.items()
    # ))

    if (len(sys.argv) == 1):
        print(
            "[ERROR] No arguments provided. Please use --help for more information."
        )
        return None

    elif args.install:
        print("[INFO] Installing dependencies...")
        install(requirements)
        print("[INFO] Dependencies installed. No further action can be taken.")
        return None

    elif ((os.path.exists(datasets_raw_directory)) or
            (os.path.exists(args.data)) and not
            args.install):

        print("[INFO] Attempting to load and preprocess the dataset...")
        start_time = time.time()

        df = get_data(args)
        protected_cols: list = None

        if (args.protected == "."):
                print(
                    "[WARNING] No protected columns specified. " +
                      "Using predetermined instead."
                )
                protected_cols = (
                    gdpr_protected_cols +
                    invalid_cols +
                    medical_protected_cols +
                    datetime_cols
                )
        else:
            try:
                protected_cols_strip = args.protected.strip("[]").split(",")
                protected_cols = [
                    col.strip() for col in protected_cols_strip if col.strip()
                ]
            except AttributeError:
                protected_cols = None

        processor = DataPreprocessor(
            df=df,
            target_variable=args.target,
            protected_cols=protected_cols,
        )
        df_file_name = os.path.basename(args.data)

        end_time = time.time()
        print(
            "[INFO] Data Preprocessing complete. Took %f seconds." % (
                end_time - start_time
            )
        )

        # Check for missing values and tell the user, but continue
        # any further actions, for various reasons.
        check_missing_values(
            processor=processor
        )

        if args.model is not None:
            run_models(args, processor=processor)
            return None

        elif args.o == 'cols':
            if args.download:
                print("[INFO] Downloading dataset...")
                start_time = time.time()
                download_state = processor.download(
                    datasets_processed_directory +
                    df_file_name,
                    processor.df,
                )
                end_time = time.time()
                if download_state is None:
                    print("[INFO] Downloaded dataset. Took %f seconds." % (
                        end_time - start_time
                    ))
                return None

            print("[INFO] Dataset columns:")
            for idx, label in enumerate(processor.labels, start=1):
                print("\n", processor.labels[label])
            return None

        elif args.o == 'num':
            print("[INFO] Numerical columns:")
            print(processor.df[processor.get_numerical_columns()])

            total_missing_numerical = processor.df[
                processor.get_numerical_columns()
            ].isnull().sum().sum()
            print(
                "[INFO] Total missing numerical values: %s" % (
                    total_missing_numerical
                )
            )
            return None

        elif args.o == 'cat':
            print("[INFO] Categorical columns:")
            print(processor.df[processor.get_categorical_columns()])
            return None

        elif args.o == '.':
            if (args.target is not None) and (args.target in processor.df.columns):
                print(
                    f"\nDataframe target variable '{args.target}' in '{df_file_name}':\n",
                    processor.df.drop(columns=[args.target]).head()
                )
                return None
            else:
                print("\nDataset head of '%s':\n" % df_file_name, df.head())
                return None

        elif args.o and args.o != '':
            try:
                if args.row:
                    column_name = args.o
                    print("\nDataframe row of column '%s' in '%s':\n" % (
                        column_name, df_file_name
                        ),
                        df[column_name].iloc[args.row]
                    )

                    return None
                else:
                    print("\nDataframe column '%s' in '%s':\n" % (
                        args.o, df_file_name
                        ),
                        df[args.o]
                    )
                    return None
            except KeyError:
                column_name = args.o
                print(
                    "[ERROR] Column '%s' not found in '%s'." % (
                        column_name, df_file_name
                    )
                )
                return None

            except UnboundLocalError:
                column_name = args.o
                print(
                    "[ERROR] Column '%s' not found in '%s'." % (
                        args.o, df_file_name
                    )
                )
                return None

            except IndexError:
                print(
                    "[ERROR] Row '%s' not found in '%s'." % (
                        args.row, df_file_name
                    )
                )
                return None
        # return None

        # elif args.model is not None:
        #     run_models(args, processor=processor)

        else:
            print(
                "[ERROR] Something went wrong, for help type py main.py [-h]"
            )


if __name__ == "__main__":
    main()
