"""
Written implementation of the MARS model, similar to
Jerome Friedman’s MARS algorithm.

Written by Reece Turner, 22036698.
"""

import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


class MARS(BaseEstimator, RegressorMixin):
    """
    Multivariate Adaptive Regression Splines (MARS) model.

    - Knot: A point along the feature axis
    - Hinge: A vector point which is created
        from the knot to assert a multivariate bend.
    """

    def __init__(
        self,
        max_terms=20,
        max_degree=1,
        min_samples_split=3,
        penalty=3.0
    ):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.min_samples_split = min_samples_split
        self.penalty = penalty
        self.coef_ = None
        self.intercept_ = 0.0
        self.basis_ = []
        self.mse_ = float('inf')
        self.r2_ = float('-inf')
        self.params_ = {
            "max_terms": max_terms,
            "max_degree": max_degree,
            "min_samples_split": min_samples_split,
            "penalty": penalty
        }

    def _forward_pass(self, X, y):
        """
        Adds basis functions to the model
        being constructed; uses stepwise expansion.
        """

        _, n_features = X.shape
        r = y.copy()  # Residuals of the model

        # Iterate until the max terms is reached or no further improvcements
        while len(self.basis_) < self.max_terms:

            # Stores occurances of the best results
            best_split_occurance = None
            best_mse = float('inf')

            for index in range(n_features):
                knot_splits = self._generate_candidate_splits(X[:, index])

                if not any(knot_splits):
                    continue

                # Check for missing or infinite values in the feature column
                if knot_splits.size == 0:
                    continue

                # Test each knot for hinge directions
                for knot in knot_splits:
                    for direction in [1, -1]:
                        self._add_basis_function(
                            parent=None,
                            variable_index=index,
                            knot=knot,
                            direction=direction
                        )

                        # Evaluate the current model with
                        # the newly added basis function
                        basis_matrix = self._evaluate_basis(X)

                        # r = y - basis_matrix @ self.coef_ - self.intercept_
                        self.coef_ = np.linalg.lstsq(
                            basis_matrix, y - self.intercept_, rcond=None
                        )[0]

                        mse = mean_squared_error(y, r) \
                            + self.penalty * len(self.basis_)

                        if mse < best_mse:
                            best_mse = mse
                            best_split_occurance = (
                                index, knot, direction
                            )
            if best_split_occurance is None:
                break

            index, knot, direction = best_split_occurance
            self._add_basis_function(
                parent=None,
                variable_index=index,
                knot=knot,
                direction=direction
            )

            basis_matrix = self._evaluate_basis(X)
            # r = y - basis_matrix @ self.coef_ - self.intercept_
            r = self.coef_ = np.linalg.lstsq(
                basis_matrix, y - self.intercept_, rcond=None
            )[0]

    def _backward_pass(self, X, y):
        """
        Backward pass to prune the basis functions based on GCV.
        This prevents overfitting by removing basis functions
        that do not improve the model's performance.
        """
        n_basis_functions = len(self.basis_)

        # Initialize GCV with the full set of basis functions
        basis_matrix = self._evaluate_basis(X)
        residuals = y - basis_matrix @ self.coef_ - self.intercept_
        gcv_score = self._compute_gcv(residuals, n_basis_functions)

        while n_basis_functions > 0:
            best_gcv = gcv_score
            best_removed_index = None

            # Test removing each basis function and calculate the GCV score
            for idx in range(n_basis_functions):
                # Remove the column corresponding to the current basis function
                updated_basis_matrix = np.delete(basis_matrix, idx, axis=1)

                # Fit the coefficients for the updated basis matrix
                updated_coef_ = np.linalg.lstsq(
                    updated_basis_matrix, y - self.intercept_, rcond=None
                )[0]

                # Calculate residuals and GCV score for the updated model
                updated_residuals = y - updated_basis_matrix @ updated_coef_  \
                    - self.intercept_

                updated_gcv = self._compute_gcv(
                    updated_residuals, n_basis_functions - 1
                )

                # Check if removing this basis function improves the GCV score
                if updated_gcv < best_gcv:
                    best_gcv = updated_gcv
                    best_removed_index = idx

            # If no improvement is found, stop pruning
            if best_removed_index is None or best_gcv >= gcv_score:
                break

            # Remove the basis function that improves the GCV score the most
            self.basis_.pop(best_removed_index)
            basis_matrix = np.delete(basis_matrix, best_removed_index, axis=1)
            gcv_score = best_gcv
            n_basis_functions -= 1

        # Recalculate coefficients for the final set of basis functions
        if self.basis_:
            self.coef_ = np.linalg.lstsq(
                basis_matrix, y - self.intercept_, rcond=None
            )[0]
        else:
            self.coef_ = None

    def _generate_candidate_splits(self, x_column):
        """
        Generate a possible knot (t) location where a
        hinge (h(x - t)_+ || h(t - x)_+) can be placed.
        """

        unique_values = np.unique(x_column)
        if len(unique_values) <= self.min_samples_split:
            return []

        return (unique_values[:-1] + unique_values[1:]) / 2

    def _add_basis_function(self, parent, variable_index, knot, direction):
        """
        Create a new hinge function for the model.
        """

        self.basis_.append({
            "parent": parent,
            "variable": variable_index,
            "knot": knot,
            "direction": direction,
        })

    def _compute_gcv(self, residuals, n_terms):
        """
        Caculate the Generalized Cross Validation (GCV) score.
        """

        n_samples = len(residuals)
        mse = np.mean(residuals ** 2)
        gcv_score = mse * (
            n_samples + n_terms + self.penalty
        ) / (
            n_samples - n_terms
        )
        return gcv_score

    def _evaluate_single_basis(self, X, basis_function):
        """
        Evaluates an added single basis function for a given (X) input;
        typically a hinge.
        """

        feature_index = basis_function["variable"]
        knot = basis_function["knot"]
        direction = basis_function["direction"]

        x_col = X[:, feature_index]
        if direction == 1:
            return np.maximum(0, x_col - knot)
        elif direction == -1:
            return np.maximum(0, knot - x_col)
        raise ValueError(f"Invalid direction: {direction}")

    def _evaluate_basis(self, X):
        """
        Evaluation all (X) basis functions for prediction and testing.
        """

        n_samples = X.shape[0]
        n_basis = len(self.basis_)

        basis_matrix = np.zeros((n_samples, n_basis))
        for index, basis_function in enumerate(self.basis_):
            basis_matrix[:, index] = self._evaluate_single_basis(
                X, basis_function
            )

        return basis_matrix

    def fit(self, X, y):
        """
        Train the model, running forward pass then backwards pass pruning.
        """

        # Reset any potential states from previous training
        self.basis_ = []
        self.coef_ = None
        self.intercept_ = 0.0
        self.mse_ = float('inf')
        self.r2_ = float('-inf')

        # Train and time the model
        start_time = time.time()
        self._forward_pass(X, y)
        end_time = time.time()
        print("Forward pass completed in:", end_time - start_time, "seconds")

        start_time = time.time()
        self._backward_pass(X, y)
        end_time = time.time()
        print("Backward pass completed in:", end_time - start_time, "seconds")
        return self

    def predict(self, X):
        """
        Predict the trained model using computations of the fitted
        basis functions and coefficients.
        """

        basis_matrix = self._evaluate_basis(X)
        return basis_matrix @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Caclulate the R^2 and MSE of the model.
        """
        y_pred = self.predict(X)
        self.mse_ = mean_squared_error(y, y_pred)
        self.r2_ = r2_score(y, y_pred)
        return None

    def permutation_importance(self, model, X, y, metric):
        """
        Calculate permutation feature importance for a model.

        Parameters:
        model (MARS): Trained MARS model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        metric (function): The metric used to evaluate the performance.

        Returns:
        list: List of tuples (feature_name, importance_score).
        """
        baseline_score = metric(y, model.predict(X.values))

        features = []
        for col in X.columns:
            X_shuffled = X.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col])

            # Ensure the shuffled DataFrame has the same structure
            X_shuffled = X_shuffled[X.columns]

            # Calculate the performance on the shuffled data
            score = metric(y, model.predict(X_shuffled.values))

            # Calculate feature importance - higher is better
            feature_importance = baseline_score - score
            features.append((col, feature_importance))

        features.sort(key=lambda x: x[1], reverse=True)
        return features

    def summary(self, X_test, y_test):
        """
        Generate a summary of the MARS model's performance.

        Parameters:
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test target vector.

        Returns:
        dict: A dictionary containing the model's summary metrics.
        """
        if not self.basis_:
            raise ValueError("The model has not been trained yet.")

        # Number of terms and predictors
        num_terms = len(self.basis_)
        predictors = {basis["variable"] for basis in self.basis_}
        num_predictors = len(predictors)

        # Importance of each predictor
        importance = {}
        for predictor in predictors:
            importance[predictor] = sum(
                1 for basis in self.basis_ if basis["variable"] == predictor
            )

        # Number of terms at each degree
        degree_counts = {}
        for basis in self.basis_:
            degree = basis.get("degree", 1)
            degree_counts[degree] = degree_counts.get(degree, 0) + 1

        # Calculate GCV and R2
        residuals = self.predict(X_test) - y_test
        gcv_value = self._compute_gcv(residuals, num_terms)
        r_squared = r2_score(y_test, self.predict(X_test))

        str_terms = f"{num_terms} of {self.max_terms} terms"
        str_predictors = f"{num_predictors} of {X_test.shape[1]} predictors"
        return {
            "selected_terms": str_terms,
            "selected_predictors": str_predictors,
            "importance": importance,
            "terms_per_degree": degree_counts,
            "gcv": gcv_value,
            "r2": r_squared,
        }
