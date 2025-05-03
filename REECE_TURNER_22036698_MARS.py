"""
Multivariate Adaptive Regression Splines (MARS).

Written by Reece Turner, 22036698.
"""
import itertools
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import lime
import lime.lime_tabular

from models import (
    DataPreprocessor,
    RANDOM_STATE,
    TESTING_DATA_SIZE,
    insurance_dataset,
    gdpr_protected_cols,
    invalid_cols,
    medical_protected_cols,
    datetime_cols,
    MARS
)

# Update to the correct path location of this project.
PATH_TO_PROJECT = (
    "C:/Program Development/UWE Bristol/Year 3/Enterprise AI Project/"
)

# Load the dataset
PATH_TO_UWE_ENTERPRISE_MLAAS_MODELS = (
    PATH_TO_PROJECT +
    "project/backend/services/machinelearning/uwe_enterprise_mlaas_models")

sys.path.append(PATH_TO_UWE_ENTERPRISE_MLAAS_MODELS)
data = pd.read_csv(insurance_dataset)

# Data Preprocessing
protected_cols = (
    gdpr_protected_cols + medical_protected_cols + datetime_cols + invalid_cols
)

TARGET_VARIABLE = "SettlementValue"
processor = DataPreprocessor(
    df=data,
    protected_cols=protected_cols
)

if TARGET_VARIABLE not in processor.df.columns:
    print(f"Error: Target variable '{TARGET_VARIABLE}' " +
          "does not exist in the dataset.")
    sys.exit(1)

target = processor.df[TARGET_VARIABLE].copy()

if target.isnull().sum() > 0:
    print("Error: Target variable has null values.")
    sys.exit(1)

# Split the data into training and testing sets
df = processor.df.copy()
df = df.iloc[: 1500]
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=[TARGET_VARIABLE]),
    df[TARGET_VARIABLE],
    test_size=TESTING_DATA_SIZE,
    random_state=RANDOM_STATE,
)


# Check the data for missing values
def check_missing_values(X, y):
    """
    Checks for NaN or infinite values in X and y.

    Parameters:
        X (DataFrame): Features.
        y (Series): Target.

    Returns:
        bool: True if NaN or infinite values are found, False otherwise.
    """
    if X.isna().any().any() or y.isna().any():
        print("NaN values found in X or y.")
        return True
    if X.isin([float('inf'), float('-inf')]).any().any() or y.isin([
        float('inf'), float('-inf')
    ]).any():
        print("Infinite values found in X or y.")
        return True
    return False


# Check for a malformed dataset, continue if not.
MISSING_VALUES = check_missing_values(X_train, y_train) or \
    check_missing_values(X_test, y_test)

if MISSING_VALUES:
    print("Missing values detected in the dataset.")
    sys.exit(1)


# Chunk the data for parallel processing
def split_data(X, y, n_chunks):
    """
    Split the data into chunks (processors) for parallel processing.
    """
    X = np.atleast_2d(X)
    y = np.ravel(y)

    chunk_size = len(X) // n_chunks
    return [(
            X[i * chunk_size:(i + 1) * chunk_size],
            y[i * chunk_size:(i + 1) * chunk_size]
            ) for i in range(n_chunks)]


def predict(models, X):
    """
    Predict using the ensemble of models.

    Rather than averaging normally, the model can apply weights
    based on individual model performance, thus models with
    better performance should have a higher weight which improves prediction.
    """
    chunked_predictions = np.array([model.predict(X) for model in models])
    return np.mean(chunked_predictions, axis=0)


def weighted_mse(y_true, y_pred, weights):
    """
    Calculate weighted mean squared error.

    Parameters:
        y_true (ndarray): True target values.
        y_pred (ndarray): Predicted target values.
        weights (ndarray): Weights for each prediction.

    Returns:
        float: Weighted mean squared error.
    """
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.sum(weights * squared_errors) / np.sum(weights)
    return weighted_mse


def weighted_r2(y_true, y_pred, weights):
    """
    Calculate weighted R^2 score.

    Parameters:
        y_true (ndarray): True target values.
        y_pred (ndarray): Predicted target values.
        weights (ndarray): Weights for each prediction.

    Returns:
        float: Weighted R^2 score.
    """
    ss_res = ((y_true - y_pred) ** 2 * weights).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2 * weights).sum()
    r2 = 1 - ss_res / ss_tot
    return r2


def evaluate_ensemble(models, X_test, y_test, weights):
    """
    Evaluate the ensemble of models using weighted MSE and weighted R^2.
    """
    preds = np.mean([model.predict(X_test) for model in models], axis=0)
    mse = weighted_mse(y_test, preds, weights)
    r2 = weighted_r2(y_test, preds, weights)
    return mse, r2


def grid_search(
    model_class: MARS,
    fit_function,
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid,
    n_jobs=-1
):
    """
    Custom grid search with parallel processing for hyperparameter tuning.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training target variable.
        X_test (DataFrame): Testing features.
        y_test (Series): Testing target variable.
        param_grid (dict): Dictionary of hyperparameters to search.
        n_jobs (int): Number of parallel jobs. Default is -1.

    Returns:
        dict: Best parameters, their corresponding score, and the best model.
    """

    keys, values = zip(*param_grid.items())
    param_combinations = [
        dict(zip(keys, v))
        for v in itertools.product(*values)
    ]

    n_chunks = n_jobs
    data_chunks = split_data(X_train, y_train, n_chunks)
    print(f"Data chunks {len(data_chunks)}: ", [
        len(chunk) for chunk in data_chunks
    ])

    best_params = None
    avg_best_r2 = float('-inf')
    avg_best_mse = float('inf')
    best_ensemble = None
    grid_searches = []
    best_i = 0

    print(f"Combinations: {len(param_combinations)}")
    for i, params in enumerate(param_combinations):
        weights = np.ones(len(y_test))
        models = Parallel(n_jobs=n_jobs)(
            delayed(fit_function)(model_class, X_chunk, y_chunk, **params)
            for X_chunk, y_chunk in data_chunks
        )

        models = [model for model in models if model is not None]
        if not models:
            print(f"No valid models for parameters: {params}. Skipping.")
            continue

        avg_mse, avg_r2 = evaluate_ensemble(models, X_test, y_test, weights)
        print(
            f"{i}: {models[0]} - " +
            f"Weighted MSE: {avg_mse}, Weighted R^2: {avg_r2}"
        )

        grid_searches.append({
            'params': params,
            'w_mse': avg_mse,
            'w_r2': avg_r2
        })

        if avg_r2 > avg_best_r2 and avg_mse < avg_best_mse:
            avg_best_r2 = avg_r2
            avg_best_mse = avg_mse
            best_params = params
            best_ensemble = models
            best_i = i

    print(
        f"\n[{best_i}] Best Params: {best_params}" +
        f"\nWeighted MSE: {avg_best_mse}\nWeighted R^2: {avg_best_r2}"
    )
    return best_ensemble, np.array(grid_searches)


param_grid = {
    'max_terms': [100],
    'max_degree': [3],
    'min_samples_split': [4],
    'penalty': [0.1]
}


def fit_mars_model(
    model_class: MARS,
    X,
    y,
    max_terms,
    max_degree,
    min_samples_split,
    penalty
):
    """
    Fit a MARS model with the given parameters.
    """
    X = np.atleast_2d(X)
    y = np.ravel(y)
    model = model_class(
        max_terms=max_terms,
        max_degree=max_degree,
        min_samples_split=min_samples_split,
        penalty=penalty,
    )

    try:
        model.fit(X, y)
        print("Fitted model with params: " +
              f"{max_terms}, {max_degree}, {min_samples_split}, {penalty}")
    except Exception as e:
        print("Error fitting model with params: " +
              f"{max_terms}, {max_degree}, {min_samples_split}, {penalty}. " +
              f"Error: {e}")
        return None

    return model


if max(param_grid['max_terms']) <= df.shape[0]:
    start_time = time.time()
    print("Starting grid search...")
    best_ensemble, grid_searches = grid_search(
        model_class=MARS,
        fit_function=fit_mars_model,
        X_train=X_train.values,
        y_train=y_train.values,
        X_test=X_test.values,
        y_test=y_test.values,
        param_grid=param_grid,
        n_jobs=20,
    )
    end_time = time.time()
    print(f"Grid search completed in {end_time - start_time:.2f} seconds.")
else:
    print("Error: max_terms cannot be larger than the dataset rows.")
    sys.exit()


# Evaluate the best ensemble
def permutation_importance(model, X, y, metric):
    """
    Custom permutation importance function for MARS models.

    Parameters:
    model (MARS): Trained MARS model.
    X (np.ndarray): Feature matrix.
    y (np.ndarray): Target vector.
    metric (function): The metric used to evaluate the performance.

    Returns:
    list: List of tuples (feature_index, importance_score).
    """
    baseline_score = metric(y, model.predict(X))

    importances = []
    for col in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, col])
        permuted_score = metric(y, model.predict(X_permuted))
        importance = baseline_score - permuted_score
        importances.append((col, importance))

    return sorted(importances, key=lambda x: x[1], reverse=True)


feature_names = X_test.columns
importance = permutation_importance(
    best_ensemble[0],
    X_test.values,
    y_test.values,
    mean_squared_error
)

# Replace feature indices with feature names
importance = [(feature_names[idx], score) for idx, score in importance]


def graph_features():
    """
    Graph the feature importances.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        mode='regression',
        verbose=True
    )

    exp = explainer.explain_instance(
        data_row=X_test.values[0],
        predict_fn=best_ensemble[0].predict,
        num_features=30,
        top_labels=1,
        model_regressor=None,
    )

    explanation = exp.as_list()
    features, contributions = zip(*explanation)

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(features)), contributions, color='mediumseagreen')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Contribution to Prediction')
    plt.title('LIME Explanation for Feature Importance ')
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
