import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import optuna
import pickle # Changed from joblib to pickle
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split

np.random.seed(42)
tf.random.set_seed(42)

#######################
# Core Model Implementation
#######################

class FairEBMLayer(layers.Layer):
    
    def __init__(self, num_bins=32, interaction_terms=None, l2_regularization=0.001, 
                 sensitive_feature_indices=None, fairness_penalty=0.1):
        super(FairEBMLayer, self).__init__()
        self.num_bins = num_bins
        self.interaction_terms = interaction_terms
        self.l2_regularization = l2_regularization
        self.sensitive_feature_indices = sensitive_feature_indices or []
        self.fairness_penalty = fairness_penalty
        self.feature_bins = {}
        self.feature_mins = {}
        self.feature_maxs = {}
        
    def build(self, input_shape):
        self.num_features = input_shape[1]
        
        # Create weights for main effects
        self.main_effect_weights = []
        for i in range(self.num_features):
            self.main_effect_weights.append(
                self.add_weight(
                    name=f'feature_{i}_weights',
                    shape=(self.num_bins,),
                    initializer='zeros',
                    regularizer=tf.keras.regularizers.l2(self.l2_regularization),
                    trainable=True
                )
            )
            if i not in self.feature_bins:
                self.feature_bins[i] = np.linspace(0, 1, self.num_bins + 1)
        
        # Create weights for interaction effects
        self.interaction_weights = []
        if self.interaction_terms:
            for i, j in self.interaction_terms:
                self.interaction_weights.append(
                    self.add_weight(
                        name=f'interaction_{i}_{j}_weights',
                        shape=(self.num_bins, self.num_bins),
                        initializer='zeros',
                        regularizer=tf.keras.regularizers.l2(self.l2_regularization),
                        trainable=True
                    )
                )
        
        self.intercept = self.add_weight(
            name='intercept',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        
        super(FairEBMLayer, self).build(input_shape)
    
    def setup_feature_bins(self, X):
        """Setup feature bins for discretization"""
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            self.feature_mins[i] = np.min(feature_values)
            self.feature_maxs[i] = np.max(feature_values)
            
            # Add a small epsilon to max to ensure max value falls within bins
            max_with_eps = self.feature_maxs[i] + 1e-6
            
            # Create appropriate bins based on feature distribution
            if self.feature_mins[i] == self.feature_maxs[i]:
                bins = np.linspace(self.feature_mins[i] - 0.5, self.feature_maxs[i] + 0.5, self.num_bins + 1)
            else:
                try:
                    unique_values = np.unique(feature_values)
                    if len(unique_values) >= self.num_bins:
                        bins = np.percentile(feature_values, np.linspace(0, 100, self.num_bins + 1))
                    else:
                        bins = np.linspace(self.feature_mins[i], max_with_eps, self.num_bins + 1)
                except:
                    bins = np.linspace(self.feature_mins[i], max_with_eps, self.num_bins + 1)
            
            self.feature_bins[i] = np.array(bins, dtype=np.float32)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Initialize prediction with intercept
        predictions = tf.tile(self.intercept, [batch_size])
        
        # Track feature contributions for each sample (for fairness regularization)
        feature_contributions = {}
        
        # Add main effects for each feature
        for i in range(self.num_features):
            feature_values = inputs[:, i]
            bin_indices = self._get_bin_indices(feature_values, i)
            feature_contrib = tf.gather(self.main_effect_weights[i], bin_indices)
            predictions = predictions + feature_contrib
            
            # Store contributions for sensitive features
            if i in self.sensitive_feature_indices:
                feature_contributions[i] = feature_contrib
        
        # Add interaction effects if specified
        if self.interaction_terms:
            for idx, (i, j) in enumerate(self.interaction_terms):
                feat_i_indices = self._get_bin_indices(inputs[:, i], i)
                feat_j_indices = self._get_bin_indices(inputs[:, j], j)
                
                indices = tf.stack([feat_i_indices, feat_j_indices], axis=1)
                interaction_contrib = tf.gather_nd(self.interaction_weights[idx], indices)
                predictions = predictions + interaction_contrib
        
        # Calculate fairness regularization
        fairness_loss = self.calculate_fairness_loss(feature_contributions)
        
        # Reshape predictions to [batch_size, 1] for regression
        predictions = tf.reshape(predictions, [-1, 1])
        
        # Add fairness loss as an activity regularizer
        self.add_loss(fairness_loss)
        
        return predictions
    
    def _get_bin_indices(self, values, feature_idx):
        """Map feature values to bin indices"""
        bins = tf.convert_to_tensor(self.feature_bins[feature_idx], dtype=tf.float32)
        
        values = tf.expand_dims(values, axis=-1)
        bins = tf.expand_dims(bins, axis=0)
        
        comparisons = tf.greater_equal(values, bins)
        indices = tf.reduce_sum(tf.cast(comparisons, tf.int32), axis=-1) - 1
        indices = tf.clip_by_value(indices, 0, self.num_bins - 1)
        
        return indices
    
    def calculate_fairness_loss(self, feature_contributions):
        """Calculate fairness loss based on sensitive feature contributions"""
        if not self.sensitive_feature_indices or not feature_contributions:
            return tf.constant(0.0, dtype=tf.float32)
        
        fairness_loss = tf.constant(0.0, dtype=tf.float32)
        
        # Calculate variance of contribution for each sensitive feature
        for feature_idx in self.sensitive_feature_indices:
            if feature_idx in feature_contributions:
                # Get feature contribution
                contrib = feature_contributions[feature_idx]
                
                # Calculate variance - penalizes features with high variability in effect
                variance = tf.math.reduce_variance(contrib)
                fairness_loss += variance
        
        # Apply penalty weight
        return self.fairness_penalty * fairness_loss
    
    def get_feature_contribution(self, feature_idx, X=None):
        """Get the contribution pattern for a specific feature"""
        bins = self.feature_bins[feature_idx]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        weights = self.main_effect_weights[feature_idx].numpy()
        
        result = {
            'bin_centers': bin_centers,
            'weights': weights,
            'is_sensitive': feature_idx in self.sensitive_feature_indices
        }
        
        if X is not None:
            feature_values = X[:, feature_idx]
            indices = np.digitize(feature_values, bins) - 1
            indices = np.clip(indices, 0, self.num_bins - 1)
            counts = np.bincount(indices, minlength=self.num_bins)
            result['counts'] = counts
        
        return result
    
    def get_intercept(self):
        """Get the intercept term"""
        return self.intercept.numpy()[0]


class FairEnergyBasedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 num_bins=32,
                 interaction_terms=None,
                 l2_regularization=0.001,
                 learning_rate=0.01,
                 batch_size=128,
                 epochs=100,
                 validation_split=0.1,
                 patience=10,
                 verbose=1,
                 random_state=None,
                 sensitive_features=None,
                 fairness_penalty=0.1):
        self.num_bins = num_bins
        self.interaction_terms = interaction_terms
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.sensitive_features = sensitive_features or []
        self.fairness_penalty = fairness_penalty
        self.feature_names_ = None
        self.model_ = None
        self.history_ = None
        self.ebm_layer_ = None
        self.sensitive_indices_ = []
    
    def _get_sensitive_indices(self):
        """Convert sensitive feature names to indices"""
        indices = []
        if not self.feature_names_ or not self.sensitive_features:
            return indices
            
        for feature in self.sensitive_features:
            # Check for exact match
            if feature in self.feature_names_:
                indices.append(self.feature_names_.index(feature))
            else:
                # Check for partial matches
                for i, name in enumerate(self.feature_names_):
                    if feature.lower() in name.lower():
                        indices.append(i)
        
        return list(set(indices))  # Remove duplicates
    
    def _build_model(self, n_features):
        """Build the fair energy-based model"""
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Input layer
        inputs = layers.Input(shape=(n_features,))
        
        # Fair EBM layer with group-wise regularization
        self.ebm_layer_ = FairEBMLayer(
            num_bins=self.num_bins,
            interaction_terms=self.interaction_terms,
            l2_regularization=self.l2_regularization,
            sensitive_feature_indices=self.sensitive_indices_,
            fairness_penalty=self.fairness_penalty
        )
        outputs = self.ebm_layer_(inputs)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def get_feature_importances(self, X=None):
        """Calculate feature importances based on the range of effect"""
        importances = []
        
        for i in range(self.n_features_in_):
            contribution = self.ebm_layer_.get_feature_contribution(i, X)
            weights = contribution['weights']
            
            # Importance is the range of effect
            importance = np.max(weights) - np.min(weights)
            importances.append(importance)
        
        # Normalize
        importances = np.array(importances)
        if np.sum(np.abs(importances)) > 0:
            importances = importances / np.sum(np.abs(importances))
        
        # Create dictionary with feature names
        feature_names = self.feature_names_ or [f'feature_{i}' for i in range(self.n_features_in_)]
        return dict(zip(feature_names, importances))
    
    def fit(self, X, y):
        """Fit the model to training data with fairness constraints"""
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        
        # Update sensitive feature indices
        self.sensitive_indices_ = self._get_sensitive_indices()
        
        if self.verbose:
            if self.sensitive_indices_:
                if self.feature_names_:
                    sensitive_names = [self.feature_names_[i] for i in self.sensitive_indices_]
                    print(f"Applying fairness regularization to features: {sensitive_names}")
                else:
                    print(f"Applying fairness regularization to feature indices: {self.sensitive_indices_}")
                print(f"Fairness penalty weight: {self.fairness_penalty}")
            else:
                print("No sensitive features identified. Model will train without fairness constraints.")
        
        # Build the model
        self.model_ = self._build_model(self.n_features_in_)
        
        # Set up feature bins before training
        self.ebm_layer_.setup_feature_bins(X)
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Train the model
        self.history_ = self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping],
            validation_split=self.validation_split,
            verbose=1 if self.verbose else 0
        )
        
        return self
    
    def predict(self, X):
        """Predict regression target for X"""
        X = check_array(X)
        y_pred = self.model_.predict(X, verbose=0).flatten()
        return y_pred
    
    def set_feature_names(self, feature_names):
        """Set feature names for better interpretability"""
        self.feature_names_ = feature_names
        # Update sensitive indices based on feature names
        self.sensitive_indices_ = self._get_sensitive_indices()
    
    def plot_feature_effect(self, feature_idx_or_name, X=None, ax=None):
        """Plot the effect of a feature on the prediction"""
        # Convert feature name to index if needed
        feature_idx = feature_idx_or_name
        if isinstance(feature_idx_or_name, str) and self.feature_names_ is not None:
            if feature_idx_or_name in self.feature_names_:
                feature_idx = list(self.feature_names_).index(feature_idx_or_name)
            else:
                # Try partial matching
                matches = [i for i, name in enumerate(self.feature_names_) 
                          if feature_idx_or_name.lower() in name.lower()]
                if matches:
                    feature_idx = matches[0]
                else:
                    raise ValueError(f"Feature '{feature_idx_or_name}' not found")
        
        # Get feature name
        feature_name = self.feature_names_[feature_idx] if self.feature_names_ is not None else f"Feature {feature_idx}"
        
        # Get feature contribution
        contribution = self.ebm_layer_.get_feature_contribution(feature_idx, X)
        bin_centers = contribution['bin_centers']
        weights = contribution['weights']
        is_sensitive = feature_idx in self.sensitive_indices_
        
        # Create figure if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Plot effect with different color for sensitive features
        color = 'r' if is_sensitive else 'b'
        ax.plot(bin_centers, weights, color=color, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot data distribution if provided
        if X is not None and 'counts' in contribution:
            counts = contribution['counts']
            if counts.sum() > 0:
                ax2 = ax.twinx()
                ax2.bar(bin_centers, counts, width=(bin_centers[1]-bin_centers[0]), alpha=0.2, color='gray')
                ax2.set_ylabel('Count')
                ax2.grid(False)
        
        # Set labels
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Effect on prediction')
        title = f'Effect of {feature_name} on target'
        if is_sensitive:
            title += ' (Sensitive Feature)'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def evaluate_fairness(self, X, y_true, y_pred):
        """Evaluate fairness metrics across sensitive attributes"""
        fairness_metrics = {}
        
        # Convert to numpy if needed
        y_true_np = y_true.values if hasattr(y_true, 'values') else y_true
        y_pred_np = y_pred.values if hasattr(y_pred, 'values') else y_pred
        is_dataframe = hasattr(X, 'columns')
        X_np = X.values if is_dataframe else X
        
        # Skip if no sensitive features defined
        if not self.sensitive_features:
            fairness_metrics['overall_fairness'] = 1.0
            return fairness_metrics
        
        # Calculate error metrics
        errors = np.abs(y_true_np - y_pred_np)
        group_disparities = []
        
        for feature in self.sensitive_features:
            # Get the feature index
            feature_idx = None
            if is_dataframe:
                try:
                    feature_idx = list(X.columns).index(feature)
                except ValueError:
                    if self.verbose:
                        print(f"Warning: Feature '{feature}' not found in data. Skipping.")
                    continue
            elif self.feature_names_ is not None:
                try:
                    feature_idx = self.feature_names_.index(feature)
                except ValueError:
                    if self.verbose:
                        print(f"Warning: Feature '{feature}' not found in feature_names. Skipping.")
                    continue
            else:
                try:
                    feature_idx = int(feature.replace('feature_', ''))
                except:
                    if self.verbose:
                        print(f"Warning: Cannot determine index for feature '{feature}'. Skipping.")
                    continue
            
            # Get feature values
            feature_values = X_np[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) < 2:
                if self.verbose:
                    print(f"Warning: Feature '{feature}' has only one unique value. Skipping.")
                continue
            
            # Calculate group-wise metrics
            group_metrics = {}
            for value in unique_values:
                group_mask = (feature_values == value)
                if np.sum(group_mask) == 0:
                    continue
                    
                group_errors = errors[group_mask]
                group_metrics[value] = {
                    'mean_error': np.mean(group_errors),
                    'count': np.sum(group_mask)
                }
            
            if not group_metrics:
                continue
                
            # Calculate disparate impact (ratio of mean errors)
            error_values = [metrics['mean_error'] for metrics in group_metrics.values()]
            min_error = min(error_values)
            max_error = max(error_values)
            
            disparate_impact = min_error / max_error if min_error > 0 else 0
            stat_parity_diff = max_error - min_error
            
            # Group fairness metrics
            fairness_metrics[f"{feature}_disparate_impact"] = disparate_impact
            fairness_metrics[f"{feature}_stat_parity_diff"] = stat_parity_diff
            group_disparities.append(disparate_impact)
            
            # Store group metrics
            for value, metrics in group_metrics.items():
                value_str = str(value)
                fairness_metrics[f"{feature}_{value_str}_mean_error"] = metrics['mean_error']
                fairness_metrics[f"{feature}_{value_str}_count"] = metrics['count']
        
        # Calculate overall fairness
        fairness_metrics['overall_fairness'] = np.mean(group_disparities) if group_disparities else 1.0
        
        return fairness_metrics
    
    def score(self, X, y):
        """Return R^2 score on given test data and labels"""
        return r2_score(y, self.predict(X))

#######################
# Utility Functions
#######################

def calculate_metrics(y_true, y_pred, is_transformed=True):
    """Calculate evaluation metrics"""
    if is_transformed:
        # Calculate metrics on both original and transformed scales
        y_pred_original = np.expm1(y_pred)
        y_true_original = np.expm1(y_true)
        
        metrics = {
            'MSE (original)': mean_squared_error(y_true_original, y_pred_original),
            'RMSE (original)': np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
            'MAE (original)': mean_absolute_error(y_true_original, y_pred_original),
            'R² (original)': r2_score(y_true_original, y_pred_original),
            'MSE (transformed)': mean_squared_error(y_true, y_pred),
            'RMSE (transformed)': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE (transformed)': mean_absolute_error(y_true, y_pred),
            'R² (transformed)': r2_score(y_true, y_pred)
        }
    else:
        # Calculate metrics directly
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred)
        }
    
    return metrics

#######################
# Hyperparameter Optimization
#######################

def optimize_fair_ebm(X, y, sensitive_features, n_trials=10, cv=3, 
                   fairness_weight=0.3, random_state=42):
    """Run hyperparameter optimization for Fair EBM"""
    print(f"\n--- Starting Fair EBM Hyperparameter Optimization with {n_trials} trials ---")
    print(f"Fairness weight: {fairness_weight}, Sensitive features: {sensitive_features}")
    
    X_dim = X.shape[1]
    
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'num_bins': trial.suggest_int('num_bins', 16, 64, step=8),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 200, step=25),
            'fairness_penalty': trial.suggest_float('fairness_penalty', 0.01, 0.5, log=True)
        }
        
        # Determine interactions
        use_interactions = trial.suggest_categorical('use_interactions', [True, False])
        interaction_terms = None
        
        if use_interactions:
            num_interactions = trial.suggest_int('num_interactions', 1, min(10, (X_dim * (X_dim - 1)) // 2))
            potential_pairs = [(i, j) for i in range(X_dim) for j in range(i+1, X_dim)]
            np.random.seed(random_state)
            selected_indices = np.random.choice(len(potential_pairs), 
                                            size=min(num_interactions, len(potential_pairs)), 
                                            replace=False)
            interaction_terms = [potential_pairs[i] for i in selected_indices]
        
        # Cross-validation
        cv_scores = []
        fairness_scores = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(X)))):
            print(f"  CV Fold {fold_idx + 1}/{cv}", end="", flush=True)
            
            # Handle DataFrame or numpy array
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # Convert to numpy arrays if needed
            X_train_np = X_train_fold.values if hasattr(X_train_fold, 'values') else X_train_fold
            y_train_np = y_train_fold.values if hasattr(y_train_fold, 'values') else y_train_fold
            
            # Create and train model
            model = FairEnergyBasedRegressor(
                num_bins=params['num_bins'],
                interaction_terms=interaction_terms,
                l2_regularization=params['l2_regularization'],
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                fairness_penalty=params['fairness_penalty'],
                validation_split=0.1,
                patience=10,
                verbose=0,
                random_state=random_state,
                sensitive_features=sensitive_features
            )
            
            # Set feature names if available
            if hasattr(X_train_fold, 'columns'):
                model.set_feature_names(X_train_fold.columns.tolist())
            
            model.fit(X_train_np, y_train_np)
            
            # Evaluate performance
            y_val_pred = model.predict(X_val_fold)
            fold_score = r2_score(y_val_fold, y_val_pred)
            cv_scores.append(fold_score)
            
            # Evaluate fairness
            fairness_metrics = model.evaluate_fairness(X_val_fold, y_val_fold, y_val_pred)
            fairness_score = fairness_metrics.get('overall_fairness', 0.0)
            fairness_scores.append(fairness_score)
            
            print(f" - R²: {fold_score:.4f}, Fairness: {fairness_score:.4f}")
        
        # Calculate combined score
        mean_cv_score = np.mean(cv_scores)
        mean_fairness_score = np.mean(fairness_scores)
        scaled_r2 = max(0, mean_cv_score)
        combined_score = fairness_weight * mean_fairness_score + (1 - fairness_weight) * scaled_r2
        
        # Store trial metrics
        trial.set_user_attr('interaction_terms', interaction_terms)
        trial.set_user_attr('r2_score', mean_cv_score)
        trial.set_user_attr('fairness_score', mean_fairness_score)
        
        return combined_score
    
    # Create and run study
    study = optuna.create_study(direction='maximize', study_name='fair_ebm_optimization')
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_trial.params.copy()
    best_params['interaction_terms'] = study.best_trial.user_attrs['interaction_terms']
    
    # Summary information
    print(f"\nBest value: {study.best_trial.value:.4f}")
    print(f"Best R² score: {study.best_trial.user_attrs['r2_score']:.4f}")
    print(f"Best fairness score: {study.best_trial.user_attrs['fairness_score']:.4f}")
    
    # Create visualization if possible
    try:
        optuna.visualization.matplotlib.plot_optimization_history(study)
        optuna.visualization.matplotlib.plot_param_importances(study)
    except Exception as e:
        print(f"Could not create optimization plots: {e}")
    
    return best_params, study

#######################
# Main Pipeline
#######################

import re
import numpy as np
import pandas as pd

def aggregate_feature_importances(importances, feature_names):
    
    original_features = {}
    
    for feature_name in feature_names:
        original_name = extract_original_feature_name(feature_name)
        if original_name not in original_features:
            original_features[original_name] = []
        original_features[original_name].append(feature_name)
    
    # Aggregate importances by original feature
    aggregated_importances = {}
    for original_name, encoded_features in original_features.items():
        # Sum the importances of all encoded features from the same original feature
        total_importance = sum(importances.get(feature, 0) for feature in encoded_features)
        aggregated_importances[original_name] = total_importance
    
    # Sort importances
    sorted_importances = dict(sorted(aggregated_importances.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True))
    
    return sorted_importances

def extract_original_feature_name(feature_name):
    
    # Handle one-hot encoded features with pattern 'categorical__onehot__feature_value'
    if 'categorical__' in feature_name:
        # Extract feature name for categorical features
        match = re.search(r'categorical__([^_]+)_(.+)', feature_name)
        if match:
            # For categorical features
            return match.group(1)
    
    # Handle other transformation patterns like 'transformer__feature'
    elif '__' in feature_name:
        parts = feature_name.split('__')
        if len(parts) >= 2:
            return parts[1]
    
    # Return as is for features without transformation
    return feature_name

# Now update the get_feature_importances method in the FairEnergyBasedRegressor class
def get_feature_importances_with_aggregation(self, X=None, aggregate=True):
    
    importances = {}
    
    for i in range(self.n_features_in_):
        contribution = self.ebm_layer_.get_feature_contribution(i, X)
        weights = contribution['weights']
        
        # Importance is the range of effect
        importance = np.max(weights) - np.min(weights)
        
        # Use feature names if available, otherwise use indices
        feature_name = self.feature_names_[i] if self.feature_names_ else f'feature_{i}'
        importances[feature_name] = importance
    
    # Normalize importances
    total_importance = sum(abs(imp) for imp in importances.values())
    if total_importance > 0:
        importances = {feat: imp / total_importance for feat, imp in importances.items()}
    
    # Aggregate importances if requested
    if aggregate and self.feature_names_:
        return aggregate_feature_importances(importances, self.feature_names_)
    
    return importances

# Now update the plot_feature_importance function in the main pipeline
def plot_feature_importance(model, X_train, aggregate=True, output_dir=None, top_n=15):
    
    import matplotlib.pyplot as plt
    
    # Get feature importances (original method without aggregation)
    original_importances = model.get_feature_importances(X_train)
    sorted_original = dict(sorted(original_importances.items(), 
                                 key=lambda x: abs(x[1]), 
                                 reverse=True))
    
    # Get aggregated importances if requested
    if aggregate and model.feature_names_:
        aggregated_importances = aggregate_feature_importances(original_importances, 
                                                              model.feature_names_)
    else:
        aggregated_importances = sorted_original
    
    # Limit to top N features
    top_original = dict(list(sorted_original.items())[:top_n])
    top_aggregated = dict(list(aggregated_importances.items())[:top_n])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original importances
    bars1 = ax1.barh(list(top_original.keys()), list(top_original.values()))
    ax1.set_xlabel('Importance')
    ax1.set_ylabel('Feature')
    ax1.set_title('Original Feature Importances')
    
    # Highlight sensitive features in red
    if hasattr(model, 'sensitive_indices_') and model.feature_names_:
        sensitive_names = [model.feature_names_[i] for i in model.sensitive_indices_]
        for i, feature_name in enumerate(top_original.keys()):
            if any(s_name in feature_name for s_name in sensitive_names):
                bars1[i].set_color('red')
    
    # Plot aggregated importances
    bars2 = ax2.barh(list(top_aggregated.keys()), list(top_aggregated.values()))
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    ax2.set_title('Aggregated Feature Importances')
    
    # Highlight sensitive features in red (for aggregated plot)
    if hasattr(model, 'sensitive_indices_') and model.feature_names_:
        sensitive_names = [extract_original_feature_name(model.feature_names_[i]) 
                          for i in model.sensitive_indices_]
        for i, feature_name in enumerate(top_aggregated.keys()):
            if feature_name in sensitive_names or any(s_name in feature_name for s_name in sensitive_names):
                bars2[i].set_color('red')
    
    plt.tight_layout()
    
    # Save plot if output directory is specified
    if output_dir:
        import os
        plt.savefig(os.path.join(output_dir, "feature_importance_comparison.png"))
    
    return aggregated_importances

# Now update the train_fair_ebm_model function to use the new plotting function
def train_fair_ebm_model_with_aggregation(X_train, y_train, X_test, y_test, best_params, 
                                         sensitive_features=None, output_dir="fair_ebm_outputs", 
                                         random_state=42):
    """Train and evaluate the fair EBM model with aggregated feature importances"""
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pickle
    
    # Same implementation as before, but replace the feature importance plotting code
    print("\n--- Training Final Fair EBM Model with Best Parameters ---")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create the model
    model = FairEnergyBasedRegressor(
        num_bins=best_params['num_bins'],
        interaction_terms=best_params['interaction_terms'],
        l2_regularization=best_params['l2_regularization'],
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],
        validation_split=0.1,
        patience=15,
        verbose=1,
        random_state=random_state,
        sensitive_features=sensitive_features,
        fairness_penalty=best_params['fairness_penalty']
    )
    
    # Set feature names if available
    if hasattr(X_train, 'columns'):
        model.set_feature_names(X_train.columns.tolist())
    
    # Convert to numpy arrays if needed
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    
    # Train the model
    model.fit(X_train_np, y_train_np)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy_metrics = calculate_metrics(y_test, y_pred, is_transformed=True)
    fairness_metrics = model.evaluate_fairness(X_test, y_test, y_pred)
    
    # Combine metrics
    all_metrics = {**accuracy_metrics, **fairness_metrics}
    
    # Print metrics summary
    print("\nTest Set Metrics:")
    for metric_name, value in accuracy_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print("\nFairness Metrics:")
    for metric_name, value in fairness_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Use the new feature importance plot function (with aggregation)
    aggregated_importances = plot_feature_importance(
        model, X_train_np, aggregate=True, output_dir=output_dir, top_n=15
    )
    
    # Print top aggregated features
    print("\nTop 10 aggregated feature importances:")
    for i, (feature, importance) in enumerate(list(aggregated_importances.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Plot top feature effects (reduced from 6 to 4 plots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # First plot sensitive features
    sensitive_features_plotted = 0
    sensitive_features_to_plot = []
    
    if model.sensitive_indices_ and model.feature_names_:
        for idx in model.sensitive_indices_:
            if idx < len(model.feature_names_):
                feature_name = model.feature_names_[idx]
                sensitive_features_to_plot.append(feature_name)
                
                if sensitive_features_plotted < len(axes):
                    model.plot_feature_effect(feature_name, X_train_np, axes[sensitive_features_plotted])
                    sensitive_features_plotted += 1
    
    # Then plot other top features based on aggregated importance
    top_aggregated_features = list(aggregated_importances.keys())
    
    # For each top feature, find a representative feature from the original set
    for agg_feature in top_aggregated_features:
        # Skip if all axes are used
        if sensitive_features_plotted >= len(axes):
            break
            
        # Find an original feature that belongs to this aggregated feature
        for orig_feature in model.feature_names_:
            if extract_original_feature_name(orig_feature) == agg_feature and orig_feature not in sensitive_features_to_plot:
                # Plot this feature effect
                model.plot_feature_effect(orig_feature, X_train_np, axes[sensitive_features_plotted])
                sensitive_features_plotted += 1
                break
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/feature_effects.png")
    
    # Save model and results using pickle instead of joblib
    model_path = f"{output_dir}/ebm_model.pkl"
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save parameters and metrics
    results = {
        'best_params': best_params,
        'accuracy_metrics': accuracy_metrics,
        'fairness_metrics': fairness_metrics,
        'aggregated_importances': aggregated_importances
    }
    
    results_path = f"{output_dir}/results.pkl"
    print(f"Saving results to {results_path}...")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary report
    report = [
        "# Fair Energy-Based Model (EBM) Training Summary",
        "",
        "## Best Hyperparameters",
        "".join([f"- {param}: {value}\n" for param, value in best_params.items() 
               if param != 'interaction_terms']),
        f"- Number of interactions: {len(best_params['interaction_terms']) if best_params['interaction_terms'] else 0}",
        "",
        "## Fairness Configuration",
        f"- Sensitive features: {sensitive_features}",
        f"- Fairness penalty: {best_params['fairness_penalty']}",
        "",
        "## Accuracy Metrics",
        "".join([f"- {metric}: {value:.4f}\n" for metric, value in accuracy_metrics.items()]),
        "",
        "## Fairness Metrics",
        "".join([f"- {metric}: {value:.4f}\n" for metric, value in fairness_metrics.items()]),
        "",
        "## Top 10 Aggregated Feature Importances",
        "".join([f"- {feature}: {importance:.4f}\n" for feature, importance in list(aggregated_importances.items())[:10]]),
        "",
        "## Model Summary",
        "This model uses group-wise regularization to ensure that sensitive attributes",
        "have minimal or equal contributions to predictions, promoting fairness.",
        "",
        "Feature importances are aggregated from one-hot encoded features to their original categorical variables.",
        "",
        f"Model artifacts saved in: {output_dir}"
    ]
    
    with open(f"{output_dir}/summary.md", "w") as f:
        f.write("\n".join(report))
    
    # Instructions for loading the saved model
    print(f"\nTo load the saved model, use the following code:")
    print("```python")
    print("import pickle")
    print(f"with open('{model_path}', 'rb') as f:")
    print("    loaded_model = pickle.load(f)")
    print("```")
    
    return model, all_metrics

# Replace the original FairEnergyBasedRegressor.get_feature_importances with the updated version
# This is a monkey patch that should be applied after the class definition
FairEnergyBasedRegressor.get_feature_importances_original = FairEnergyBasedRegressor.get_feature_importances
FairEnergyBasedRegressor.get_feature_importances = get_feature_importances_with_aggregation

# In the main function, replace train_fair_ebm_model with train_fair_ebm_model_with_aggregation

def inverse_transform_target(y_pred, inverse_transform=True):
    """Inverse transform log-transformed target"""
    if inverse_transform:
        return np.expm1(y_pred)
    else:
        return y_pred

def main():
    # Configuration
    DATA_PATH = 'settlement_data_processed.csv'
    TARGET_COLUMN = 'SettlementValue'
    OUTPUT_DIR = 'fair_ebm_outputs'
    SENSITIVE_FEATURES = ['numeric__Driver Age', 'categorical__Gender_Female', 
                         'categorical__Gender_Male', 'categorical__Gender_Other']
    N_TRIALS = 20  
    CV_FOLDS = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n=== Fair Energy-Based Model Pipeline ===")
    print("This implementation uses pickle for model serialization")
    
    try:
        # 1. Load data
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded. Shape: {df.shape}")
        
        # Split features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        # Verify sensitive features
        valid_sensitive_features = [f for f in SENSITIVE_FEATURES if f in X.columns]
        
        if not valid_sensitive_features:
            print("No valid sensitive features found. Model will train without fairness constraints.")
        
        # 2. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # 3. Run hyperparameter optimization
        best_params, study = optimize_fair_ebm(
            X_train, y_train, 
            sensitive_features=valid_sensitive_features,
            n_trials=N_TRIALS, 
            cv=CV_FOLDS, 
            random_state=RANDOM_STATE
        )
        
        # 4. Create visualizations
        plt.figure()
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(f"{OUTPUT_DIR}/optimization_history.png")
        
        plt.figure()
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(f"{OUTPUT_DIR}/parameter_importances.png")
        
        # 5. Train final model with aggregated feature importances
        model, metrics = train_fair_ebm_model_with_aggregation(
            X_train, y_train, 
            X_test, y_test,
            best_params, 
            sensitive_features=valid_sensitive_features,
            output_dir=OUTPUT_DIR, 
            random_state=RANDOM_STATE
        )
        
        # 6. Save study results using pickle
        study_path = f"{OUTPUT_DIR}/optimization_study.pkl"
        print(f"\nSaving optimization study to {study_path}...")
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        # 7. Demonstrate prediction with inverse transformation
        print("\n=== Demonstration of Model Usage and Inverse Transformation ===")
        # Take a small sample for demonstration
        sample_size = min(5, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        
        if hasattr(X_test, 'iloc'):
            X_sample = X_test.iloc[sample_indices]
            y_sample = y_test.iloc[sample_indices]
        else:
            X_sample = X_test[sample_indices]
            y_sample = y_test[sample_indices]
        
        # Make predictions
        y_pred_transformed = model.predict(X_sample)
        
        # Transform predictions back to original scale
        y_pred_original = inverse_transform_target(y_pred_transformed)
        
        # Convert true values from log space to original
        y_sample_original = inverse_transform_target(y_sample)
        
        # Create comparison table
        comparison = pd.DataFrame({
            'True (log space)': y_sample,
            'Predicted (log space)': y_pred_transformed,
            'True (original)': y_sample_original,
            'Predicted (original)': y_pred_original
        })
        
        print("\nSample predictions comparison:")
        print(comparison)
        
        # Save the comparison to CSV
        comparison.to_csv(f"{OUTPUT_DIR}/prediction_samples.csv")
        
        usage_guide = [
            "# Fair EBM Model Usage Guide",
            "",
            "## Loading the model",
            "```python",
            "import pickle",
            "import numpy as np",
            "",
            "# Load the model",
            f"with open('{OUTPUT_DIR}/ebm_model.pkl', 'rb') as f:",
            "    model = pickle.load(f)",
            "```",
            "",
            "## Making predictions",
            "```python",
            "# Make predictions (in log-transformed space)",
            "predictions_log = model.predict(X_new)",
            "",
            "# Transform predictions back to original scale",
            "def inverse_transform_target(y_pred):",
            "    return np.expm1(y_pred)",
            "",
            "predictions_original = inverse_transform_target(predictions_log)",
            "```",
            "",
            "## Getting feature importances (aggregated)",
            "```python",
            "# Get feature importances with aggregation for one-hot encoded features",
            "importances = model.get_feature_importances(X_new, aggregate=True)",
            "print(importances)",
            "```",
            "",
            "## Analyzing fairness",
            "```python",
            "# Analyze fairness on new data",
            "fairness_metrics = model.evaluate_fairness(X_new, y_true, predictions_log)",
            "print(fairness_metrics)",
            "```"
        ]
        
        with open(f"{OUTPUT_DIR}/usage_guide.md", "w") as f:
            f.write("\n".join(usage_guide))
            
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print(f"See usage guide at: {OUTPUT_DIR}/usage_guide.md")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()