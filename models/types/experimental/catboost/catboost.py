import numpy as np
from catboost import CatBoostRegressor
import pickle
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
    f_regression
)
from sklearn.inspection import partial_dependence
import shap
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

DEFAULT_SAVE_PATH = "catboost_model.pkl"


# Enhanced CatBoost Regressor with automatic feature selection, handling of
# outliers, cross-validated model building, and advanced optimisation
# techniques.
class CatBoost(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        save_path=DEFAULT_SAVE_PATH,
        random_state=None,
        # CatBoost specific parameters
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='R2',
        early_stopping_rounds=100,
        verbose=0,
        # Advanced parameters
        bootstrap_type='Bayesian',
        use_best_model=True,
        subsample=0.85,
        colsample_bylevel=0.8,
        # Feature engineering
        auto_feature_selection=True,
        feature_selection_method='mutual_info',
        feature_fraction=0.8,
        # Outlier handling
        handle_outliers=True,
        outlier_method='clip',
        # Cross-validation settings
        use_cv=True,
        cv_folds=5,
        # Hyperparameter optimisation
        optimise_hyperparams=False,
        n_iterations=30,
        # Learning rate scheduler
        use_lr_schedule=False,
        **kwargs
    ):
        self.save_path = save_path
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self.bootstrap_type = bootstrap_type
        self.use_best_model = use_best_model
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel

        self.auto_feature_selection = auto_feature_selection
        self.feature_selection_method = feature_selection_method
        self.feature_fraction = feature_fraction

        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method

        self.use_cv = use_cv
        self.cv_folds = cv_folds

        self.optimise_hyperparams = optimise_hyperparams
        self.n_iterations = n_iterations

        self.use_lr_schedule = use_lr_schedule

        self.kwargs = kwargs

        self.model_ = None
        self.cv_models_ = []
        self.feature_selector_ = None
        self.scaler_ = None
        self.selected_features_ = None
        self.feature_importances_ = None
        self.shap_values_ = None
        self.explainer_ = None
        self.feature_names_ = None
        self.best_params_ = None

        self._create_model()

    # Creates a CatBoostRegressor instance with the current parameters,
    # handling compatibility between bootstrap_type and subsample.
    def _create_model(self):
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'loss_function': self.loss_function,
            'eval_metric': self.eval_metric,
            'random_seed': self.random_state,
            'verbose': self.verbose,
            'bootstrap_type': self.bootstrap_type,
            'use_best_model': self.use_best_model,
            'colsample_bylevel': self.colsample_bylevel
        }

        if self.bootstrap_type != 'Bayesian':
            params['subsample'] = self.subsample

        params.update(self.kwargs)

        self.model_ = CatBoostRegressor(**params)

    # Processes outliers in the training data using either clipping or
    # removal based on robust scaling and configurable thresholds.
    def _handle_outliers(self, X, y):
        if not self.handle_outliers:
            return X, y

        print(
            f"[INFO] Handling outliers using {self.outlier_method} method...")

        if self.outlier_method == 'clip':
            if self.scaler_ is None:
                self.scaler_ = RobustScaler()
                self.scaler_.fit(X)

            X_scaled = self.scaler_.transform(X)

            mask = np.abs(X_scaled) > 3

            X_clipped = X.copy()

            for i in range(X.shape[1]):
                col_mask = mask[:, i]

                if np.any(col_mask):
                    q1, q3 = np.percentile(X[:, i][~col_mask], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    X_clipped[:, i] = np.clip(
                        X[:, i], lower_bound, upper_bound)

            return X_clipped, y

        elif self.outlier_method == 'remove':
            if self.scaler_ is None:
                self.scaler_ = RobustScaler()
                self.scaler_.fit(X)

            X_scaled = self.scaler_.transform(X)

            mask = np.any(np.abs(X_scaled) > 3, axis=1)

            if np.sum(~mask) < len(mask) * 0.5:
                warnings.warn(
                    "Too many outliers detected. " +
                    "Using clipping instead of removal.")
                return self._handle_outliers(X, y)

            return X[~mask], y[~mask]

        else:
            warnings.warn(
                f"Unknown outlier method: {self.outlier_method}. " +
                "Using original data.")
            return X, y

    # Performs feature selection using specified method
    # (mutual information or F-regression) to identify the most
    # informative features for the model.
    def _select_features(self, X, y):
        if not self.auto_feature_selection:
            return np.arange(X.shape[1])

        n_features = int(X.shape[1] * self.feature_fraction)

        if n_features < 1:
            n_features = 1
        elif n_features > X.shape[1]:
            n_features = X.shape[1]

        print(
            "[INFO] Performing automatic feature selection using" +
            f"{self.feature_selection_method}...")
        print(f"[INFO] Selecting {n_features} out of {X.shape[1]} features...")

        if self.feature_selection_method == 'mutual_info':
            self.feature_selector_ = SelectKBest(
                mutual_info_regression, k=n_features)
        elif self.feature_selection_method == 'f_regression':
            self.feature_selector_ = SelectKBest(f_regression, k=n_features)
        else:
            warnings.warn(
                "Unknown feature selection method: " +
                f"{self.feature_selection_method}. Using all features.")
            return np.arange(X.shape[1])

        self.feature_selector_.fit(X, y)
        selected_features = self.feature_selector_.get_support(indices=True)

        if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
            selected_names = [self.feature_names_[i]
                              for i in selected_features]

            features_list = list(zip(selected_features, selected_names))
            print(f"[INFO] Selected features: {features_list}")
        else:
            print(f"[INFO] Selected features: {selected_features}")

        return selected_features

    # Implements k-fold cross-validation to create an ensemble of
    # models trained on different data subsets for more robust predictions.
    def _cross_validate(self, X, y, **fit_params):
        if not self.use_cv:
            return False

        print(f"[INFO] Performing {self.cv_folds}-fold cross-validation...")

        kf = KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state)
        self.cv_models_ = []
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"[INFO] Training fold {fold+1}/{self.cv_folds}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            params = {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'l2_leaf_reg': self.l2_leaf_reg,
                'loss_function': self.loss_function,
                'eval_metric': self.eval_metric,
                'random_seed': (
                    self.random_state if self.random_state is not None
                    else 42 + fold
                ),
                'verbose': 0,
                'bootstrap_type': self.bootstrap_type,
                'use_best_model': self.use_best_model,
                'colsample_bylevel': self.colsample_bylevel,
            }

            if self.bootstrap_type != 'Bayesian':
                params['subsample'] = self.subsample

            params.update(self.kwargs)

            fold_model = CatBoostRegressor(**params)

            fold_fit_params = {k: v for k,
                               v in fit_params.items() if k != 'feature_names'}

            fold_model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False,
                **fold_fit_params
            )

            self.cv_models_.append(fold_model)
            fold_score = fold_model.score(X_val, y_val)
            cv_scores.append(fold_score)

            print(f"[INFO] Fold {fold+1} R² score: {fold_score:.4f}")

        avg_score = np.mean(cv_scores)
        print(f"[INFO] Cross-validation average R² score: {avg_score:.4f}")

        return True

    # Uses Bayesian optimisation to find optimal hyperparameters for the model,
    # evaluating candidates using customised validation.
    def _optimise_hyperparameters(self, X, y):
        if not self.optimise_hyperparams:
            return

        print("[INFO] Starting Bayesian hyperparameter optimisation...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state)

        search_spaces = {
            'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
            'depth': Integer(4, 10),
            'l2_leaf_reg': Real(0.1, 10, prior='log-uniform'),
            'colsample_bylevel': Real(0.5, 1.0)
        }

        if self.bootstrap_type != 'Bayesian':
            search_spaces['subsample'] = Real(0.5, 1.0)
        else:
            search_spaces['bootstrap_type'] = Categorical(
                ['Bayesian', 'Bernoulli'])

        base_model = CatBoostRegressor(
            iterations=self.iterations,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_state,
            verbose=0)

        def custom_scorer(estimator, X_eval, y_eval):
            estimator_copy = estimator.get_params()
            model = CatBoostRegressor(**estimator_copy)

            model.fit(
                X_eval,
                y_eval,
                eval_set=[
                    (X_val,
                     y_val)],
                use_best_model=True,
                early_stopping_rounds=50,
                verbose=False)

            return model.score(X_val, y_val)

        bayes_search = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_spaces,
            n_iter=self.n_iterations,
            cv=min(3, self.cv_folds),
            scoring=custom_scorer,
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )

        bayes_search.fit(X_train, y_train)

        self.best_params_ = bayes_search.best_params_

        print(f"[INFO] Best hyperparameters found: {self.best_params_}")
        print(f"[INFO] Best R² score: {bayes_search.best_score_:.4f}")

        for param, value in self.best_params_.items():
            setattr(self, param, value)

        self._create_model()

        return self.best_params_

    # Implements a step-decay learning rate schedule to reduce learning rate
    # as training progresses for better convergence.
    def _learning_rate_schedule(self, iteration, learning_rate):
        if iteration < self.iterations * 0.5:
            return learning_rate
        elif iteration < self.iterations * 0.75:
            return learning_rate * 0.5
        else:
            return learning_rate * 0.1

    # Executes the complete model training pipeline: outlier handling, feature
    # selection, hyperparameter optimisation, cross-validation,
    # and SHAP value calculation.
    # noqa: C901
    def fit(self, X, y, **fit_params):  # noqa: C901
        print("[INFO] Starting enhanced CatBoost training pipeline...")

        # Store feature names from DataFrame columns or fit_params
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            print(
                "[INFO] Using feature names from DataFrame: " +
                f"{self.feature_names_[:5]}...")
            self._create_model()  # Recreate model with feature names
        elif 'feature_names' in fit_params:
            self.feature_names_ = fit_params.pop('feature_names')
            print(
                "[INFO] Using feature names from parameters: " +
                f"{self.feature_names_[:5]}...")
            self._create_model()  # Recreate model with feature names

        # Convert to numpy arrays for consistent processing
        if hasattr(X, 'values'):
            X = X.values
        else:
            X = np.asarray(X)
        y = np.asarray(y)

        # Handle outliers if specified
        X_processed, y_processed = self._handle_outliers(X, y)
        if X_processed.shape[0] < X.shape[0]:
            print(
                f"[INFO] Removed {X.shape[0] - X_processed.shape[0]} " +
                f"outliers ({(X.shape[0] - X_processed.shape[0])/X.shape[0]*100:.1f}% of the data)")  # noqa: E501

        # Perform feature selection if specified
        self.selected_features_ = self._select_features(
            X_processed, y_processed)
        X_selected = X_processed[:, self.selected_features_]

        # Update feature names if feature selection was performed
        if self.feature_names_ is not None:
            original_names = self.feature_names_
            self.feature_names_ = [original_names[i]
                                   for i in self.selected_features_]
            print(
                "[INFO] Feature names after selection: " +
                f"{self.feature_names_[:5]}...")
            # Recreate model to update feature names after selection
            self._create_model()

        # Optimize hyperparameters if specified
        if self.optimise_hyperparams:
            self._optimise_hyperparameters(X_selected, y_processed)

        # Perform cross-validation if specified
        if self.use_cv:
            cv_params = {
                k: v for k,
                v in fit_params.items() if k != 'feature_names'}

            cv_performed = self._cross_validate(  # noqa
                X_selected, y_processed, **cv_params)  # noqa
        else:
            cv_performed = False  # noqa

        print("[INFO] Training final model on full dataset...")

        # Create validation set for early stopping if specified
        if (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
        ):
            print("[INFO] Creating validation set for early stopping...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y_processed, test_size=0.15,
                random_state=self.random_state)
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X_selected, y_processed
            eval_set = None

        # Remove feature_names from fit parameters - IMPORTANT FIX
        final_fit_params = {
            k: v for k,
            v in fit_params.items() if k != 'feature_names'}

        # Set up learning rate scheduling if specified
        if self.use_lr_schedule:
            class LRSchedulerCallback:
                def __init__(self, schedule_func, init_lr):
                    self.schedule_func = schedule_func
                    self.init_lr = init_lr

                def after_iteration(self, info):
                    iteration = info.iteration
                    new_lr = self.schedule_func(iteration, self.init_lr)
                    return {'learning_rate': new_lr}

            lr_callback = LRSchedulerCallback(
                self._learning_rate_schedule, self.learning_rate)
            if 'callbacks' in final_fit_params:
                final_fit_params['callbacks'].append(lr_callback)
            else:
                final_fit_params['callbacks'] = [lr_callback]
            print("[INFO] Using learning rate scheduling")

        # IMPORTANT: Removed passing feature_names to fit
        # method - it must be set during model creation

        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose > 0,
            **final_fit_params
        )

        self.best_model_ = self.model_

        if hasattr(self.model_, 'feature_importances_'):
            self.feature_importances_ = self.model_.feature_importances_

            if self.verbose > 0:
                importances = self.feature_importances_
                feature_indices = np.argsort(importances)[::-1]
                print("\n[INFO] Feature Importances:")

                for i, idx in enumerate(feature_indices):
                    if i >= 15:
                        break

                    feature_name = self.feature_names_[
                        idx] if self.feature_names_ is not None else \
                        f"Feature {self.selected_features_[idx]}"
                    print(f"  {feature_name}: {importances[idx]:.6f}")

        try:
            print("[INFO] Calculating SHAP values for model explainability...")
            shap_sample_size = min(500, X_selected.shape[0])
            X_shap = X_selected[:shap_sample_size]

            self.explainer_ = shap.TreeExplainer(self.model_)
            self.shap_values_ = self.explainer_.shap_values(X_shap)

            self.X_val = X_shap

            print("[INFO] SHAP values calculated successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to calculate SHAP values: {e}")
            self.shap_values_ = None
            self.explainer_ = None

        print("[INFO] Model training complete.")
        self.save_model()

        if self.verbose > 0:
            print("\n[INFO] Generating model visualisations...")

            plot_dir = os.path.dirname(self.save_path) or "."
            plot_dir = os.path.join(plot_dir, "model_plots")
            os.makedirs(plot_dir, exist_ok=True)

            if hasattr(self.model_, 'feature_importances_'):
                plt.figure(figsize=(10, 8))

                feature_names = (
                    self.feature_names_
                    if self.feature_names_ is not None
                    else [f"Feature {i}" for i in self.selected_features_]
                )

                indices = np.argsort(self.feature_importances_)[::-1]
                top_n = min(20, len(indices))

                plt.barh(range(top_n),
                         self.feature_importances_[indices][:top_n])
                plt.yticks(range(top_n), [feature_names[i]
                           for i in indices[:top_n]])
                plt.xlabel('Feature Importance')
                plt.title('Top Feature Importance')
                plt.tight_layout()

                plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
                plt.close()

            if self.shap_values_ is not None:
                try:
                    shap_bar = self.plot_shap_summary(
                        max_display=15, plot_type="bar", show=False)
                    shap_bar.savefig(os.path.join(plot_dir, "shap_bar.png"))
                    plt.close()

                    shap_dot = self.plot_shap_summary(
                        max_display=15, plot_type="dot", show=False)
                    shap_dot.savefig(os.path.join(plot_dir, "shap_dot.png"))
                    plt.close()

                    shap_importance = self.get_shap_feature_importance()
                    print("\n[INFO] Top 10 features by SHAP importance:")

                    for i, (feature, importance) in enumerate(
                            list(shap_importance.items())[:10]):
                        print(f"  {i+1}. {feature}: {importance:.6f}")

                    top_features = list(shap_importance.keys())[:3]

                    for feature in top_features:
                        dep_plot = self.plot_shap_dependence(
                            feature, show=False)
                        dep_plot.savefig(
                            os.path.join(
                                plot_dir,
                                f"shap_dependence_{feature}.png"))
                        plt.close()
                except Exception as e:
                    print(f"[WARNING] Error generating SHAP plots: {e}")

            try:
                curves = self.plot_learning_curves(show=False)

                if curves is not None:
                    curves.savefig(
                        os.path.join(
                            plot_dir,
                            "learning_curves.png"))
                    plt.close()
            except Exception as e:
                print(f"[WARNING] Error generating learning curves: {e}")

            if hasattr(
                    self.model_, 'feature_importances_') and \
                    self.feature_names_ is not None:
                try:
                    top_indices = np.argsort(
                        self.feature_importances_)[::-1][:5]
                    top_features = [self.feature_names_[i]
                                    for i in top_indices]

                    pdp = self.plot_partial_dependence(
                        top_features, n_cols=2, show=False)

                    if pdp is not None:
                        pdp.savefig(
                            os.path.join(
                                plot_dir,
                                "partial_dependence.png"))
                        plt.close()
                except Exception as e:
                    print(
                        "[WARNING] Error generating partial dependence " +
                        f"plots: {e}")

            print(f"\n[INFO] All visualisations saved to {plot_dir}")

        return self

    # Makes predictions using either the ensemble of cross-validated models
    # (if available) or the single trained model, applying
    # feature selection as needed.
    def predict(self, X):
        X = np.asarray(X)

        if self.selected_features_ is not None:
            X = X[:, self.selected_features_]

        if self.cv_models_:
            print("[INFO] Making predictions with CV ensemble...")
            predictions = np.zeros(X.shape[0])

            for model in self.cv_models_:
                predictions += model.predict(X)

            predictions /= len(self.cv_models_)
            return predictions

        if self.model_ is None or not hasattr(self.model_, 'predict'):
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first.")

        return self.model_.predict(X)

    # Saves both the complete wrapper class and the inner CatBoost model to
    # separate files for flexibility in deployment.
    def save_model(self):
        if self.model_ is None:
            print("[ERROR] No model to save. Fit the model first.")
            return

        try:
            save_dir = os.path.dirname(self.save_path)

            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f)

            print(f"[INFO] Complete CatBoost model saved to {self.save_path}")

            inner_model_path = self.save_path.replace(".pkl", "_inner.pkl")
            with open(inner_model_path, 'wb') as f:
                pickle.dump(self.model_, f)

            print(f"[INFO] Inner CatBoost model saved to {inner_model_path}")

        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")

    # Loads a trained model from file, supporting both the complete wrapper
    # class and standalone CatBoost models.
    @classmethod
    def load_model(cls, path=DEFAULT_SAVE_PATH):
        try:
            with open(path, 'rb') as f:
                loaded_object = pickle.load(f)

            if isinstance(loaded_object, cls):
                print(f"[INFO] CatBoost model loaded from {path}")
                return loaded_object
            elif hasattr(loaded_object, 'predict'):
                print(f"[INFO] CatBoost model loaded from {path}")
                return loaded_object
            else:
                print("[ERROR] Loaded object is not a valid model.")
                return None
        except FileNotFoundError:
            print(f"[ERROR] Model file not found at {path}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None

    def get_params(self, deep=True):
        params = {
            'save_path': self.save_path,
            'random_state': self.random_state,
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'loss_function': self.loss_function,
            'eval_metric': self.eval_metric,
            'early_stopping_rounds': self.early_stopping_rounds,
            'verbose': self.verbose,
            'bootstrap_type': self.bootstrap_type,
            'use_best_model': self.use_best_model,
            'subsample': self.subsample,
            'colsample_bylevel': self.colsample_bylevel,
            'auto_feature_selection': self.auto_feature_selection,
            'feature_selection_method': self.feature_selection_method,
            'feature_fraction': self.feature_fraction,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'use_cv': self.use_cv,
            'cv_folds': self.cv_folds,
            'optimise_hyperparams': self.optimise_hyperparams,
            'n_iterations': self.n_iterations,
            'use_lr_schedule': self.use_lr_schedule,
        }

        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        self._create_model()
        return self

    def get_feature_importances(self):
        if self.feature_importances_ is not None:
            return self.feature_importances_
        elif self.model_ and hasattr(self.model_, 'feature_importances_'):
            return self.model_.feature_importances_
        else:
            print("[WARNING] Feature importances not available.")
            return None

    def _compute_shap_values(self, X_val=None):
        print("[INFO] Computing SHAP values...")

        if X_val is None and hasattr(self, 'X_val'):
            X_val = self.X_val

        if X_val is None:
            print("[WARNING] No validation data provided for SHAP " +
                  "calculation.")
            return None

        try:
            self.explainer_ = shap.TreeExplainer(self.best_model_)

            shap_values = self.explainer_.shap_values(X_val)

            self.X_val = X_val

            return shap_values
        except Exception as e:
            print(f"[ERROR] Failed to compute SHAP values: {e}")
            return None

    # Visualises feature importance using SHAP values with different plot
    # types to understand feature impact on predictions.
    def plot_shap_summary(self, max_display=20, plot_type="bar", show=True):
        if self.shap_values_ is None or self.explainer_ is None:
            print("[ERROR] SHAP values not available. Run fit() first.")
            return None

        print("[INFO] Generating SHAP summary plot...")

        plt.figure(figsize=(10, 12))

        X_display = self.X_val if hasattr(self, 'X_val') else None

        if plot_type == "bar":
            shap.summary_plot(
                self.shap_values_,
                X_display,
                feature_names=self.feature_names_,
                max_display=max_display,
                plot_type="bar",
                show=False
            )
        elif plot_type == "dot":
            shap.summary_plot(
                self.shap_values_,
                X_display,
                feature_names=self.feature_names_,
                max_display=max_display,
                show=False
            )
        elif plot_type == "violin":
            shap.summary_plot(
                self.shap_values_,
                X_display,
                feature_names=self.feature_names_,
                plot_type="violin",
                max_display=max_display,
                show=False
            )

        plt.title(f"SHAP Feature Importance ({plot_type})")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        fig = plt.gcf()

        if show:
            plt.show()

        return fig

    # Creates dependence plots showing how a feature's effect on predictions
    # varies across its value range, optionally with feature interactions.
    def plot_shap_dependence(
            self, feature_idx, interaction_idx=None, show=True):
        if self.shap_values_ is None or self.explainer_ is None:
            print("[ERROR] SHAP values not available. Run fit() first.")
            return None

        if isinstance(feature_idx, str) and self.feature_names_ is not None:
            try:
                feature_idx = self.feature_names_.index(feature_idx)
            except ValueError:
                print(f"[ERROR] Feature name '{feature_idx}' not found.")
                return None

        if interaction_idx is not None and isinstance(
                interaction_idx, str) and self.feature_names_ is not None:
            try:
                interaction_idx = self.feature_names_.index(interaction_idx)
            except ValueError:
                print(f"[ERROR] Feature name '{interaction_idx}' not found.")
                return None

        print("[INFO] Generating SHAP dependence plot...")

        plt.figure(figsize=(10, 7))

        feature_names = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"Feature {i}" for i in self.selected_features_]
        )

        feature_name = feature_names[feature_idx]

        if interaction_idx is not None:
            interaction_name = feature_names[interaction_idx]
            shap.dependence_plot(
                feature_idx,
                self.shap_values_,
                feature_names=feature_names,
                interaction_index=interaction_idx,
                show=False)
            plt.title(
                f"SHAP Dependence Plot: {feature_name} " +
                f"(interaction with {interaction_name})")
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values_,
                feature_names=feature_names,
                show=False)
            plt.title(f"SHAP Dependence Plot: {feature_name}")

        fig = plt.gcf()

        if show:
            plt.show()

        return fig

    # Generates partial dependence plots showing marginal effects of features
    # on predicted outcomes, independent of other features.
    # noqa: C901
    def plot_partial_dependence(  # noqa: C901
            self, features, n_cols=3, grid_resolution=20, show=True):
        """Generates partial dependence plots for the specified features."""
        if self.model_ is None:
            print("[ERROR] Model not available. Run fit() first.")
            return None

        feature_indices = []
        feature_names = []

        for feature in features:
            if isinstance(feature, str) and self.feature_names_ is not None:
                try:
                    idx = self.feature_names_.index(feature)
                    feature_indices.append(idx)
                    feature_names.append(feature)
                except ValueError:
                    print(
                        f"[WARNING] Feature name '{feature}' not found, " +
                        "skipping.")
            else:
                feature_indices.append(feature)
                name = self.feature_names_[
                    feature] if self.feature_names_ is not None else \
                    f"Feature {feature}"
                feature_names.append(name)

        if not feature_indices:
            print("[ERROR] No valid features to plot.")
            return None

        print(
            "[INFO] Generating partial dependence plots for " +
            f"{len(feature_indices)} features...")

        X_sample = np.random.choice(
            np.arange(
                self.model_.feature_count_), size=min(
                500, self.model_.feature_count_), replace=False)

        n_rows = (len(feature_indices) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(
                n_cols * 4, n_rows * 3))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for i, (feature_idx, feature_name) in enumerate(
                zip(feature_indices, feature_names)):
            if i < len(axes):
                try:
                    pdp = partial_dependence(
                        self.model_.predict,
                        X_sample,
                        [feature_idx],
                        grid_resolution=grid_resolution)

                    axes[i].plot(pdp['values'][0], pdp['average'][0])
                    axes[i].set_title(f"PDP: {feature_name}")
                    axes[i].set_xlabel(feature_name)
                    axes[i].set_ylabel('Predicted Value')
                    axes[i].grid(True, linestyle='--', alpha=0.5)
                except Exception as e:
                    print(
                        "[WARNING] Error plotting PDP for " +
                        f"{feature_name}: {e}")
                    axes[i].text(0.5, 0.5, "Error plotting PDP",
                                 horizontalalignment='center')

        for i in range(len(feature_indices), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    # Visualises training and validation metrics over iterations to evaluate
    # convergence and potential overfitting.
    def plot_learning_curves(self, show=True):
        if not hasattr(self.model_, 'evals_result_'):
            print(
                "[ERROR] Learning curves not available. Model was" +
                " not trained with eval_set.")
            return None

        evals_result = self.model_.get_evals_result()

        if not evals_result:
            print("[ERROR] Learning curves not available.")
            return None

        train_metric = list(evals_result['learn'].keys())[0]
        val_metric = list(evals_result['validation'].keys())[0]

        train_scores = evals_result['learn'][train_metric]
        val_scores = evals_result['validation'][val_metric]

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)

        plt.plot(epochs, train_scores, 'b-', label=f'Training {train_metric}')
        plt.plot(epochs, val_scores, 'r-', label=f'Validation {val_metric}')

        plt.title('Learning Curves')
        plt.xlabel('Iterations')
        plt.ylabel(val_metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        fig = plt.gcf()

        if show:
            plt.show()

        return fig

    def get_shap_feature_importance(self, normalize=True):
        if self.shap_values_ is None:
            print("[ERROR] SHAP values not available. Run fit() first.")
            return None

        feature_importance = np.mean(np.abs(self.shap_values_), axis=0)

        if normalize and np.sum(feature_importance) > 0:
            feature_importance = feature_importance / \
                np.sum(feature_importance)

        feature_names = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"Feature {i}" for i in self.selected_features_]
        )

        importance_dict = {
            name: importance for name,
            importance in zip(
                feature_names,
                feature_importance)}

        importance_dict = dict(
            sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True))

        return importance_dict
