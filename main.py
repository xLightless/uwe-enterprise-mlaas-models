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
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.types.experimental.catboost import CatBoost

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
    CatBoost,
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
        case "cat":
            feature_names = None

            if hasattr(X_train, 'columns'):
                feature_names = X_train.columns.tolist()

            model = CatBoost(
                iterations=2000,
                learning_rate=0.02,
                depth=6,
                l2_leaf_reg=2,
                loss_function='RMSE',
                eval_metric='R2',
                early_stopping_rounds=100,
                bootstrap_type='Bayesian',
                use_best_model=True,
                subsample=0.85,
                colsample_bylevel=0.8,
                auto_feature_selection=False,
                feature_selection_method='mutual_info',
                feature_fraction=0.8,
                handle_outliers=True,
                outlier_method='clip',
                use_cv=True,
                cv_folds=5,
                optimise_hyperparams=True,
                n_iterations=30,
                use_lr_schedule=True,
                save_path=f"cat_model.pkl",
                random_state=42,
                verbose=100
            )
            print(model_selected % "CatBoost")
        case _:
            print("[ERROR] Model %s not found." % model_name)
            sys.exit()

    # Fit the model to the training data
    if model_name == "cat" and feature_names is not None:
        model.feature_names_ = feature_names
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def evaluate_model(model_name, model, X_test, y_test, verbose):
    """
    Public function for testing a trained model.
    """
    import math

    # Make predictions on the test data
    predictions = model.predict(X_test)

    if verbose:
        print("[INFO] Predictions:", predictions)
        print("[INFO] Mean Absolute Error:", mean_absolute_error(
            y_test, predictions
        ))
        mse = mean_squared_error(y_test, predictions)
        print("[INFO] Mean Squared Error:", mse)
        print("[INFO] Root Mean Squared Error (RMSE):", math.sqrt(mse))
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
    # --- Define model filename based on args.model ---
    model_filename = f"{args.model}_model.pkl" if args.model else None

    if (args.model is None and args.o) and not args.install:
        print("[ERROR] No model specified. Please use --model <model_name>.")
        return None

    # --- Prepare data split (needed for both train and test) ---
    # Ensure processor.df is not empty and has enough columns
    if processor.df is None or processor.df.empty or processor.df.shape[1] <= 1:
        print("[ERROR] Preprocessed DataFrame is invalid or empty. Cannot proceed.")
        return None
    if not (0 <= TARGET_VARIABLE_COL_NUM < processor.df.shape[1]):
         print(f"[ERROR] TARGET_VARIABLE_COL_NUM ({TARGET_VARIABLE_COL_NUM}) is out of bounds for the DataFrame shape {processor.df.shape}. Cannot proceed.")
         return None

    try:
        y = processor.df.iloc[:, TARGET_VARIABLE_COL_NUM].values  # Target variable
        X = processor.df.iloc[:, 1:].values  # Features
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TESTING_DATA_SIZE,
            random_state=args.seed,
        )
    except Exception as e:
        print(f"[ERROR] Failed during data splitting: {e}")
        return None
    # --- End data split ---

    model = None

    # Executes the training portion of the model on the split dataset.
    if args.train and not args.test:
        print("[INFO] Training model...")
        start_time = time.time()
        try:
            if not isinstance(X_train, pd.DataFrame) and hasattr(processor, 'df'):
                feature_cols = processor.df.columns.tolist()
                feature_cols.pop(TARGET_VARIABLE_COL_NUM)
                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                model = train_model(args.model, X_train_df, y_train, X_test_df, y_test)
            else:
                model = train_model(args.model, X_train, y_train, X_test, y_test)
        except Exception as e_train:
            print(f"[ERROR] Model training failed: {e_train}")
            model = None

        end_time = time.time()
        if model:
             print("[INFO] Training process complete. Took %f seconds." % (
                 end_time - start_time
             ))


    elif args.test and not args.train:
        # --- Load model and evaluate ---
        if model_filename and os.path.exists(model_filename):
            print(f"[INFO] Loading model from {model_filename} for evaluation...")
            try:
                with open(model_filename, 'rb') as f:
                    model = pickle.load(f)
                print("[INFO] Model loaded successfully.")

                # --- Now evaluate the loaded model ---
                print("Evaluating %s model..." % args.model)
                evaluate_model(
                    args.model,
                    model,
                    X_test,
                    y_test,
                    args.verbose
                )

            except ModuleNotFoundError as e_mod:
                 print(f"[ERROR] Failed to load model: Class definition not found. Ensure the model class ({e_mod.name}) is available in the environment.")
                 import traceback
                 traceback.print_exc()
            except EOFError:
                 print(f"[ERROR] Failed to load model: File {model_filename} might be corrupted or incomplete.")
            except Exception as e_load_eval:
                print(f"[ERROR] Failed to load or evaluate model: {e_load_eval}")
                import traceback
                traceback.print_exc()
        elif not model_filename:
             print("[ERROR] Cannot test: No model specified via --model.")
        else:
            print(f"[ERROR] Model file {model_filename} not found. Train the model first using --train.")
        # --- End Load and Evaluate ---


    elif not args.train and not args.test:
        print(
            "[ERROR] No action specified. Please use --train or --test."
        )
    else: # Handles args.train and args.test being true
        print(
            "[ERROR] You cannot perform training and testing simultaneously in separate steps." +
            " Training automatically saves the model. Use --test separately."
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
            protected_cols=protected_cols
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
