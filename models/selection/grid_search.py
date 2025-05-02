"""
Custom grid search for hyperparameter tuning using GPU-based
parallel processing.


"""

import itertools
import multiprocessing
import cupy as cp
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score


class GridSearch:
    """
    Custom grid search for hyperparameter tuning using
    CPU-based parallel processing with GPU-accelerated
    calculations.

    This class provides methods to split data into chunks,
    predict using an ensemble of models, evaluate the ensemble,
    and offers insights into the best estimator.
    """

    def split_data(self, X, y, n_chunks):
        """
        Split the data into chunks (processors) for parallel processing.
        """
        X = cp.atleast_2d(X)
        y = cp.ravel(y)

        chunk_size = len(X) // n_chunks
        return [(X[i * chunk_size:(i + 1) * chunk_size],
                 y[i * chunk_size:(i + 1) * chunk_size])
                for i in range(n_chunks)]

    def predict(self, models, X):
        """
        Predict using the ensemble of models on GPU.
        """
        chunked_predictions = cp.array([model.predict(X) for model in models])
        return cp.mean(chunked_predictions, axis=0)

    def evaluate_ensemble(self, models, X_test, y_test):
        """
        Evaluates an ensemble of models and returns average MSE and R^2.
        """
        ensemble_mse = []
        ensemble_r2 = []

        for model in models:
            predictions = model.predict(X_test)
            ensemble_mse.append(mean_squared_error(y_test, predictions))
            ensemble_r2.append(r2_score(y_test, predictions))

        avg_mse = cp.mean(ensemble_mse)
        avg_r2 = cp.mean(ensemble_r2)

        return avg_mse, avg_r2

    def grid_search(
        self,
        model_class,
        fit_function,
        X_train,
        y_train,
        X_test,
        y_test,
        param_grid,
        n_jobs=-1,
        batch_size=5
    ):
        """
        Perform a grid search over the specified hyperparameters by
        batching models together and training them in parallel using the GPU.
        The best model is selected based on the average MSE and R^2

        Params:
        model_class: Class of the model to be trained.
        fit_function: Function to fit the model.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        param_grid: Dictionary of hyperparameters to search over.
        n_jobs: Number of parallel jobs to run.
            Default is -1 (all available cores).
        batch_size: Number of models to train in parallel. Default is 5.
        """
        keys, values = zip(*param_grid.items())
        param_combinations = [
            dict(zip(keys, v))
            for v in itertools.product(*values)
        ]

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Number of chunks for parallelism based on number of CPU cores
        n_chunks = min(n_jobs, batch_size)
        data_chunks = self.split_data(X_train, y_train, n_chunks)
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

            # Batch processing: Combine models in batches for GPU training
            models = []
            for start in range(0, len(data_chunks), batch_size):
                batch_data_chunks = data_chunks[start:start + batch_size]

                # Train models in parallel on the GPU (across multiple CPUs)
                models_batch = Parallel(n_jobs=n_jobs)(
                    delayed(fit_function)(
                        model_class,
                        X_chunk,
                        y_chunk,
                        **params
                    )
                    for X_chunk, y_chunk in batch_data_chunks
                )

                # Filter out invalid models and add them to the final list
                models.extend([
                    model for model in models_batch if model is not None
                ])

            if not models:
                print(f"No valid models for parameters: {params}. Skipping.")
                continue

            # Evaluate the ensemble of models (average MSE and R^2)
            avg_mse, avg_r2 = self.evaluate_ensemble(models, X_test, y_test)

            grid_searches.append({
                'params': params,
                'w_mse': avg_mse,
                'w_r2': avg_r2
            })

            # Track the best model based on R^2 and MSE
            if avg_r2 > avg_best_r2 and avg_mse < avg_best_mse:
                avg_best_r2 = avg_r2
                avg_best_mse = avg_mse
                best_params = params
                best_ensemble = models
                best_i = i

        print(
            f"\n[{best_i}] Best Params: {best_params}" +
            f"\nEnsemble Averages: \nMSE: {avg_best_mse}\nR^2: {avg_best_r2}"
        )

        return best_ensemble, cp.array(grid_searches)

    def weighted_mse(self, y_true, y_pred, weights):
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
        weighted_mse = cp.sum(weights * squared_errors) / cp.sum(weights)
        return weighted_mse

    def weighted_r2(self, y_true, y_pred, weights):
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

    def permutation_importance(self, model, X, y, metric):
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
            cp.random.shuffle(X_permuted[:, col])
            permuted_score = metric(y, model.predict(X_permuted))
            importance = baseline_score - permuted_score
            importances.append((col, importance))

        return sorted(importances, key=lambda x: x[1], reverse=True)
