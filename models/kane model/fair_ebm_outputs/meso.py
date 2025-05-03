import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
import random
from typing import Dict, Tuple, List, Any, Union, Set
import os
import re

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

#######################
# Core Model Implementation with Fairness
#######################

class InterpretableMesomorphicRegressor(BaseEstimator, RegressorMixin):
    """
    An interpretable neural network regressor with fairness regularization.
    
    This model uses a mesomorphic architecture where a hypernetwork generates 
    instance-specific linear weights, making predictions both accurate and explainable.
    It incorporates fairness constraints by penalizing prediction differences across
    sensitive attribute groups (like gender or age) during training.
    
    The model is built on TensorFlow/Keras and follows scikit-learn's estimator API.
    It provides comprehensive interpretability features including instance-level explanations,
    feature importance analysis, and counterfactual generation to analyze fairness.
    
    """

    def __init__(self, 
                 hyper_hidden_units=(64, 32),
                 activation='relu',
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=100,
                 verbose=1,
                 validation_split=0.1,
                 fairness_weight=0.1,
                 random_state=None):
        self.hyper_hidden_units = hyper_hidden_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_split = validation_split
        self.fairness_weight = fairness_weight  # Weight for fairness regularization
        self.random_state = random_state
        self.feature_names_ = None
        self.model_ = None
        self.history_ = None
        self.sensitive_attributes_ = None
        self.sensitive_attribute_groups_ = None
    def _build_model(self, n_features):
        """Build the mesomorphic network architecture"""
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Input layer
        inputs = layers.Input(shape=(n_features,))
        
        # Hypernetwork: generates weights for the linear model
        x = inputs
        for units in self.hyper_hidden_units:
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Generate linear weights for each instance
        linear_weights = layers.Dense(n_features, activation='linear', name='linear_weights')(x)
        bias = layers.Dense(1, activation='linear', name='bias')(x)
        
        # Linear combination for final prediction
        weighted_inputs = layers.Multiply()([inputs, linear_weights])
        sum_weighted = layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(weighted_inputs)
        outputs = layers.Add()([sum_weighted, bias])
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def _fairness_loss(self, X, y, model, sensitive_attr_groups=None):
    
        if not sensitive_attr_groups or len(sensitive_attr_groups) == 0:
            return 0.0
        
        if self.feature_names_ is None:
            raise ValueError("Feature names must be set before calculating fairness loss")
        
        # Create list to store fairness loss values by attribute
        fairness_losses_by_attr = {}
        fairness_details = {}
        
        # Sample a subset of instances for efficiency
        sample_size = min(len(X), 100)  # Limit to 100 instances for computation efficiency
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        # Track the largest differences for each attribute group
        max_diffs = {}
        
        for idx in sample_indices:
            instance = X[idx:idx+1].copy()
            original_prediction = model.predict(instance, verbose=0)[0][0]
            
            # For each sensitive attribute group
            for attr_name, columns in sensitive_attr_groups.items():
                # Initialize tracking for this attribute if it doesn't exist
                if attr_name not in fairness_losses_by_attr:
                    fairness_losses_by_attr[attr_name] = []
                    fairness_details[attr_name] = []
                    max_diffs[attr_name] = {'diff': 0.0, 'original': 0.0, 'counterfactual': 0.0, 'attribute': ''}
                
                # Ensure all columns exist in the feature set
                valid_columns = [col for col in columns if col in self.feature_names_]
                if not valid_columns:
                    continue
                
                # Handle specific sensitive attribute types
                if attr_name == 'age':
                    # For age (continuous), create variations by adjusting the value
                    age_column = valid_columns[0]
                    age_idx = self.feature_names_.index(age_column)
                    
                    # Create age counterfactuals
                    current_age = instance[0, age_idx]
                    age_variations = [
                        max(18, current_age - 15),  # -15 years (min 18)
                        min(80, current_age + 15)   # +15 years (max 80)
                    ]
                    
                    for new_age in age_variations:
                        if abs(new_age - current_age) < 1:  # Skip if change is too small
                            continue
                            
                        counterfactual = instance.copy()
                        counterfactual[0, age_idx] = new_age
                        
                        # Get prediction
                        cf_prediction = model.predict(counterfactual, verbose=0)[0][0]
                        
                        # Calculate squared difference
                        diff = original_prediction - cf_prediction
                        fairness_loss = diff ** 2
                        fairness_losses_by_attr[attr_name].append(fairness_loss)
                        
                        fairness_details[attr_name].append({
                            'original_age': current_age,
                            'new_age': new_age,
                            'original_pred': original_prediction,
                            'counterfactual_pred': cf_prediction,
                            'difference': diff,
                            'diff_pct': abs(diff) / (abs(original_prediction) + 1e-8) * 100
                        })
                        
                        # Track largest difference
                        if abs(diff) > abs(max_diffs[attr_name]['diff']):
                            max_diffs[attr_name] = {
                                'diff': diff,
                                'original': original_prediction,
                                'counterfactual': cf_prediction,
                                'attribute': f"Age: {current_age} → {new_age}"
                            }
                
                elif attr_name == 'gender':
                    # For gender (categorical), create counterfactuals by switching active column
                    if len(valid_columns) < 2:
                        continue
                        
                    # Find which gender category is active (1) in the current instance
                    active_gender_idx = None
                    for i, col in enumerate(valid_columns):
                        col_idx = self.feature_names_.index(col)
                        if instance[0, col_idx] == 1:
                            active_gender_idx = i
                            break
                    
                    if active_gender_idx is None:
                        continue  # No active gender found
                    
                    # Get current gender name
                    current_gender = valid_columns[active_gender_idx].split('_')[-1] if '_' in valid_columns[active_gender_idx] else valid_columns[active_gender_idx]
                    
                    # Create counterfactuals by activating other genders
                    for i, col in enumerate(valid_columns):
                        if i == active_gender_idx:
                            continue  # Skip the active one
                        
                        # Get new gender name    
                        new_gender = col.split('_')[-1] if '_' in col else col
                            
                        counterfactual = instance.copy()
                        
                        # Set current active to 0
                        current_col_idx = self.feature_names_.index(valid_columns[active_gender_idx])
                        counterfactual[0, current_col_idx] = 0
                        
                        # Set new one to 1
                        new_col_idx = self.feature_names_.index(col)
                        counterfactual[0, new_col_idx] = 1
                        
                        # Get prediction
                        cf_prediction = model.predict(counterfactual, verbose=0)[0][0]
                        
                        # Calculate squared difference
                        diff = original_prediction - cf_prediction
                        fairness_loss = diff ** 2
                        fairness_losses_by_attr[attr_name].append(fairness_loss)
                        
                        fairness_details[attr_name].append({
                            'original_gender': current_gender,
                            'new_gender': new_gender,
                            'original_pred': original_prediction,
                            'counterfactual_pred': cf_prediction,
                            'difference': diff,
                            'diff_pct': abs(diff) / (abs(original_prediction) + 1e-8) * 100
                        })
                        
                        # Track largest difference
                        if abs(diff) > abs(max_diffs[attr_name]['diff']):
                            max_diffs[attr_name] = {
                                'diff': diff,
                                'original': original_prediction,
                                'counterfactual': cf_prediction,
                                'attribute': f"Gender: {current_gender} → {new_gender}"
                            }
        
        # Calculate fairness loss for each attribute and print stats
        overall_fairness_loss = 0.0
        if self.verbose:
            print("\n=== Fairness Analysis ===")
        
        for attr_name, losses in fairness_losses_by_attr.items():
            if not losses:
                continue
                
            avg_loss = np.mean(losses)
            overall_fairness_loss += avg_loss
            
            if self.verbose:
                # Calculate average percent differences
                details = fairness_details[attr_name]
                avg_pct_diff = np.mean([d['diff_pct'] for d in details])
                max_pct_diff = max([d['diff_pct'] for d in details])
                
                print(f"\n{attr_name.capitalize()} Fairness:")
                print(f"  Average difference: {avg_pct_diff:.2f}%")
                print(f"  Max difference: {max_pct_diff:.2f}%")
                print(f"  Fairness loss contribution: {avg_loss:.6f}")
                
                # Print the largest difference example
                max_diff = max_diffs[attr_name]
                print(f"  Largest difference example:")
                print(f"    {max_diff['attribute']}")
                print(f"    Original prediction: {max_diff['original']:.4f}")
                print(f"    Counterfactual prediction: {max_diff['counterfactual']:.4f}")
                print(f"    Difference: {max_diff['diff']:.4f} ({abs(max_diff['diff'])/(abs(max_diff['original'])+1e-8)*100:.2f}%)")
        
        # Weight the fairness loss by the fairness_weight parameter
        weighted_loss = self.fairness_weight * overall_fairness_loss
        
        if self.verbose:
            print("\nOverall Fairness:")
            print(f"  Raw fairness loss: {overall_fairness_loss:.6f}")
            print(f"  Fairness weight: {self.fairness_weight:.6f}")
            print(f"  Weighted fairness loss: {weighted_loss:.6f}")
        
        return weighted_loss
    
    def fit(self, X, y, sensitive_attr_groups=None):
        
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.sensitive_attribute_groups_ = sensitive_attr_groups or {}
        
        # Build the model
        self.model_ = self._build_model(self.n_features_in_)
        
        # Show fairness analysis before training
        if self.sensitive_attribute_groups_ and self.feature_names_:
            print("\n=== Initial Fairness Analysis (Before Training) ===")
            initial_fairness_loss = self._fairness_loss(X, y, self.model_, self.sensitive_attribute_groups_)
            print(f"Initial fairness loss: {initial_fairness_loss:.6f}")
        else:
            initial_fairness_loss = 0.0
        
        # Train the model
        self.history_ = self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_split=self.validation_split
        )
        
        # Calculate fairness regularization value after training
        if self.sensitive_attribute_groups_ and self.feature_names_:
            print("\n=== Final Fairness Analysis (After Training) ===")
            self.fairness_loss_ = self._fairness_loss(X, y, self.model_, self.sensitive_attribute_groups_)
            fairness_improvement = initial_fairness_loss - self.fairness_loss_
            print(f"Final fairness loss: {self.fairness_loss_:.6f}")
            print(f"Fairness improvement: {fairness_improvement:.6f} ({fairness_improvement/max(initial_fairness_loss, 1e-8)*100:.2f}%)")
        else:
            self.fairness_loss_ = 0.0
        
        return self
    
    def analyze_fairness(self, X_test, y_test=None):
    
        if not self.sensitive_attribute_groups_ or not self.feature_names_:
            print("No sensitive attributes defined. Cannot analyze fairness.")
            return {}
        
        print("\n=== Detailed Fairness Analysis ===")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Create dictionary to store results
        fairness_metrics = {}
        
        # For each attribute group
        for attr_name, columns in self.sensitive_attribute_groups_.items():
            valid_columns = [col for col in columns if col in self.feature_names_]
            if not valid_columns:
                continue
                
            print(f"\n{attr_name.capitalize()} Analysis:")
            
            if attr_name == 'age':
                # For continuous age, analyze by age groups
                age_col = valid_columns[0]
                age_idx = self.feature_names_.index(age_col)
                
                # Extract values
                if isinstance(X_test, np.ndarray):
                    ages = X_test[:, age_idx]
                else:
                    ages = X_test[age_col].values
                
                # Create age bins
                age_bins = [0, 25, 35, 45, 55, 65, 100]
                age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
                
                # Assign each instance to an age bin
                binned_ages = np.digitize(ages, age_bins) - 1
                binned_ages = np.clip(binned_ages, 0, len(age_labels) - 1)
                
                # Calculate metrics for each age group
                group_metrics = {}
                all_group_preds = []
                
                for i, label in enumerate(age_labels):
                    mask = binned_ages == i
                    count = np.sum(mask)
                    
                    if count > 0:
                        group_preds = y_pred[mask]
                        group_avg = np.mean(group_preds)
                        all_group_preds.append(group_avg)
                        
                        group_metrics[label] = {
                            'count': count,
                            'mean_prediction': group_avg,
                        }
                        
                        if y_test is not None:
                            group_errors = np.abs(y_test[mask] - group_preds)
                            group_metrics[label]['mean_error'] = np.mean(group_errors)
                
                # Calculate overall fairness metrics
                if len(group_metrics) >= 2:
                    # Statistical parity difference (max difference in predictions)
                    stat_parity_diff = max(all_group_preds) - min(all_group_preds)
                    
                    # Disparate impact (min group avg / max group avg)
                    disparate_impact = min(all_group_preds) / max(all_group_preds) if max(all_group_preds) > 0 else 1.0
                    
                    fairness_metrics[f"{attr_name}_stat_parity_diff"] = stat_parity_diff
                    fairness_metrics[f"{attr_name}_disparate_impact"] = disparate_impact
                    
                    print(f"  Statistical parity difference: {stat_parity_diff:.4f}")
                    print(f"  Disparate impact: {disparate_impact:.4f}")
                    
                    # Print group details
                    print("\n  Age Group Details:")
                    print("  {:10s} {:10s} {:15s} {:15s}".format("Age Group", "Count", "Mean Prediction", "Mean Error"))
                    for label, metrics in group_metrics.items():
                        error_str = f"{metrics.get('mean_error', 'N/A'):.4f}" if 'mean_error' in metrics else "N/A"
                        print("  {:10s} {:10.0f} {:15.4f} {:15s}".format(
                            label, metrics['count'], metrics['mean_prediction'], error_str
                        ))
                        
                        # Store in metrics dict
                        fairness_metrics[f"{attr_name}_{label}_mean_prediction"] = metrics['mean_prediction']
                        fairness_metrics[f"{attr_name}_{label}_count"] = metrics['count']
                        if 'mean_error' in metrics:
                            fairness_metrics[f"{attr_name}_{label}_mean_error"] = metrics['mean_error']
            
            elif attr_name == 'gender':
                # For gender, analyze by gender categories
                gender_metrics = {}
                all_gender_preds = []
                
                for col in valid_columns:
                    gender_name = col.split('_')[-1] if '_' in col else col
                    col_idx = self.feature_names_.index(col)
                    
                    # Get mask for this gender
                    if isinstance(X_test, np.ndarray):
                        mask = X_test[:, col_idx] == 1
                    else:
                        mask = X_test[col] == 1
                    
                    count = np.sum(mask)
                    
                    if count > 0:
                        group_preds = y_pred[mask]
                        group_avg = np.mean(group_preds)
                        all_gender_preds.append(group_avg)
                        
                        gender_metrics[gender_name] = {
                            'count': count,
                            'mean_prediction': group_avg,
                        }
                        
                        if y_test is not None:
                            group_errors = np.abs(y_test[mask] - group_preds)
                            gender_metrics[gender_name]['mean_error'] = np.mean(group_errors)
                
                # Calculate overall fairness metrics
                if len(gender_metrics) >= 2:
                    # Statistical parity difference (max difference in predictions)
                    stat_parity_diff = max(all_gender_preds) - min(all_gender_preds)
                    
                    # Disparate impact (min group avg / max group avg)
                    disparate_impact = min(all_gender_preds) / max(all_gender_preds) if max(all_gender_preds) > 0 else 1.0
                    
                    fairness_metrics[f"{attr_name}_stat_parity_diff"] = stat_parity_diff
                    fairness_metrics[f"{attr_name}_disparate_impact"] = disparate_impact
                    
                    print(f"  Statistical parity difference: {stat_parity_diff:.4f}")
                    print(f"  Disparate impact: {disparate_impact:.4f}")
                    
                    # Print gender details
                    print("\n  Gender Details:")
                    print("  {:10s} {:10s} {:15s} {:15s}".format("Gender", "Count", "Mean Prediction", "Mean Error"))
                    for gender, metrics in gender_metrics.items():
                        error_str = f"{metrics.get('mean_error', 'N/A'):.4f}" if 'mean_error' in metrics else "N/A"
                        print("  {:10s} {:10.0f} {:15.4f} {:15s}".format(
                            gender, metrics['count'], metrics['mean_prediction'], error_str
                        ))
                        
                        # Store in metrics dict
                        fairness_metrics[f"{attr_name}_{gender}_mean_prediction"] = metrics['mean_prediction']
                        fairness_metrics[f"{attr_name}_{gender}_count"] = metrics['count']
                        if 'mean_error' in metrics:
                            fairness_metrics[f"{attr_name}_{gender}_mean_error"] = metrics['mean_error']
        
        # Calculate overall fairness score
        if fairness_metrics:
            parity_diffs = [val for key, val in fairness_metrics.items() if "stat_parity_diff" in key]
            if parity_diffs:
                overall_fairness = 1 - np.mean(parity_diffs)
                fairness_metrics['overall_fairness'] = overall_fairness
                print(f"\nOverall fairness score: {overall_fairness:.4f} (higher is better)")
        
        return fairness_metrics
    
    def predict(self, X):
        """Predict regression target for X"""
        X = check_array(X)
        y_pred = self.model_.predict(X, verbose=0).flatten()
        return y_pred
    
    def get_instance_explanation(self, X_instance):
        """Get interpretable weights for a specific instance"""
        X_instance = check_array(X_instance, ensure_2d=True)
        if X_instance.shape[0] != 1:
            raise ValueError("Explanation can only be generated for a single instance")
        
        # Get intermediate layer model to extract weights
        weights_model = Model(
            inputs=self.model_.input,
            outputs=[
                self.model_.get_layer('linear_weights').output,
                self.model_.get_layer('bias').output
            ]
        )
        
        # Get instance-specific weights and bias
        linear_weights, bias = weights_model.predict(X_instance, verbose=0)
        linear_weights = linear_weights[0]
        bias = bias[0][0]
        
        # Calculate feature contributions
        feature_contributions = linear_weights * X_instance[0]
        total_contribution = np.sum(feature_contributions)
        prediction = total_contribution + bias
        
        # Format explanation
        features = self.feature_names_ or [f'feature_{i}' for i in range(self.n_features_in_)]
        
        return {
            'feature_weights': dict(zip(features, linear_weights)),
            'feature_contributions': dict(zip(features, feature_contributions)),
            'bias': bias,
            'prediction': prediction
        }
    
    def set_feature_names(self, feature_names):
        """Set feature names for better interpretability"""
        self.feature_names_ = feature_names
        
    def score(self, X, y):
        """Return R^2 score on given test data and labels"""
        return r2_score(y, self.predict(X))
    
    def generate_counterfactuals(self, X_instance, sensitive_attr_groups=None):
        
        if sensitive_attr_groups is None:
            sensitive_attr_groups = self.sensitive_attribute_groups_
            
        if not sensitive_attr_groups or len(sensitive_attr_groups) == 0:
            return {}, {}
            
        X_instance = check_array(X_instance, ensure_2d=True)
        if X_instance.shape[0] != 1:
            raise ValueError("Counterfactuals can only be generated for a single instance")
        
        if self.feature_names_ is None:
            raise ValueError("Feature names must be set before generating counterfactuals")
        
        # Create dictionaries to store counterfactuals and predictions
        counterfactuals = {}
        predictions = {
            'original': self.predict(X_instance)[0]
        }
        
        # For each sensitive attribute group
        for attr_name, columns in sensitive_attr_groups.items():
            # Ensure all columns exist in the feature set
            valid_columns = [col for col in columns if col in self.feature_names_]
            if not valid_columns:
                continue
                
            attr_counterfactuals = {}
            
            # Handle specific sensitive attribute types
            if attr_name == 'age':
                # For age (continuous), create variations by adjusting the value
                age_column = valid_columns[0]  # Assume the first one is the age column
                age_idx = self.feature_names_.index(age_column)
                
                # Create age counterfactuals
                current_age = X_instance[0, age_idx]
                age_variations = [
                    max(18, current_age - 15),  # -15 years (min 18)
                    min(80, current_age + 15)   # +15 years (max 80)
                ]
                
                for new_age in age_variations:
                    if abs(new_age - current_age) < 1:  # Skip if change is too small
                        continue
                        
                    counterfactual = X_instance.copy()
                    counterfactual[0, age_idx] = new_age
                    
                    # Store counterfactual
                    attr_counterfactuals[float(new_age)] = counterfactual
                    
                    # Calculate prediction
                    predictions[f"{attr_name}={new_age}"] = self.predict(counterfactual)[0]
            
            elif attr_name == 'gender':
                # For gender (categorical), create counterfactuals by switching active column
                if len(valid_columns) < 2:
                    continue
                    
                # Find which gender category is active (1) in the current instance
                active_gender_idx = None
                active_gender_name = None
                for i, col in enumerate(valid_columns):
                    col_idx = self.feature_names_.index(col)
                    if X_instance[0, col_idx] == 1:
                        active_gender_idx = i
                        active_gender_name = col.split('_')[-1] if '_' in col else col
                        break
                
                if active_gender_idx is None:
                    continue  # No active gender found
                
                # Create counterfactuals by activating other genders
                for i, col in enumerate(valid_columns):
                    if i == active_gender_idx:
                        continue  # Skip the active one
                        
                    gender_name = col.split('_')[-1] if '_' in col else col
                    counterfactual = X_instance.copy()
                    
                    # Set current active to 0
                    current_col_idx = self.feature_names_.index(valid_columns[active_gender_idx])
                    counterfactual[0, current_col_idx] = 0
                    
                    # Set new one to 1
                    new_col_idx = self.feature_names_.index(col)
                    counterfactual[0, new_col_idx] = 1
                    
                    # Store counterfactual
                    attr_counterfactuals[gender_name] = counterfactual
                    
                    # Calculate prediction
                    predictions[f"{attr_name}={gender_name}"] = self.predict(counterfactual)[0]
            
            # Store counterfactuals for this attribute
            if attr_counterfactuals:
                counterfactuals[attr_name] = attr_counterfactuals
        
        return counterfactuals, predictions
    
    def get_feature_importances(self, X=None, top_n=None, aggregate=True):
        
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self, 'feature_names_') or self.feature_names_ is None:
            raise ValueError("Feature names have not been set")
        
        # If X is provided, calculate average feature weights across instances
        if X is not None:
            n_samples = min(100, len(X))  # Limit to 100 samples for efficiency
            if hasattr(X, 'iloc'):
                # DataFrame
                X_sample = X.iloc[:n_samples].values
            else:
                # NumPy array
                X_sample = X[:n_samples]
            
            # Calculate feature importances for each instance and average
            importances = {}
            for i in range(n_samples):
                instance = X_sample[i:i+1]
                explanation = self.get_instance_explanation(instance)
                weights = explanation['feature_weights']
                
                # Add absolute weights to running totals
                for feature, weight in weights.items():
                    if feature not in importances:
                        importances[feature] = 0
                    importances[feature] += abs(weight)  # Use absolute values
            
            # Average the importances
            importances = {
                feature: importance / n_samples 
                for feature, importance in importances.items()
            }
        else:
            # Without data, we can only use a single calculation
            # Create a dummy instance with all features = 1
            dummy_instance = np.ones((1, len(self.feature_names_)))
            explanation = self.get_instance_explanation(dummy_instance)
            importances = {
                feature: abs(weight)  # Use absolute values
                for feature, weight in explanation['feature_weights'].items()
            }
        
        # Normalize importances to sum to 1
        total_importance = sum(importances.values())
        if total_importance > 0:
            importances = {
                feature: importance / total_importance 
                for feature, importance in importances.items()
            }
        
        # Aggregate importances if requested
        if aggregate:
            importances = self._aggregate_feature_importances(importances)
        
        # Sort by importance
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # Limit to top N if specified
        if top_n is not None:
            importances = dict(list(importances.items())[:top_n])
        
        return importances
    
    def _extract_original_feature_name(self, feature_name):
        
        # Handle categorical columns with pattern 'categorical__Feature_Value'
        if 'categorical__' in feature_name:
            match = re.search(r'categorical__([^_]+(?:\s[^_]+)*)_', feature_name)
            if match:
                return match.group(1)
        
        # Handle other transformation patterns like 'transformer__feature'
        elif '__' in feature_name:
            parts = feature_name.split('__')
            if len(parts) >= 2:
                return parts[1]
        
        # Return as is for features without transformation
        return feature_name
    
    def _aggregate_feature_importances(self, feature_weights):
       
        # Create a mapping from original feature name to encoded feature names
        original_features = {}
        
        for feature_name in self.feature_names_:
            original_name = self._extract_original_feature_name(feature_name)
            if original_name not in original_features:
                original_features[original_name] = []
            original_features[original_name].append(feature_name)
        
        # Aggregate importances by original feature
        aggregated_importances = {}
        for original_name, encoded_features in original_features.items():
            # Sum the absolute weights of all encoded features from the same original feature
            total_importance = sum(abs(feature_weights.get(feature, 0)) for feature in encoded_features)
            aggregated_importances[original_name] = total_importance
        
        # Normalize importances to sum to 1
        total_importance = sum(aggregated_importances.values())
        if total_importance > 0:
            aggregated_importances = {
                feature: importance / total_importance 
                for feature, importance in aggregated_importances.items()
            }
        
        # Sort importances by value in descending order
        return dict(sorted(aggregated_importances.items(), key=lambda x: x[1], reverse=True))
    
    def plot_feature_importance(self, X=None, aggregate=True, top_n=20, output_dir=None):
        
        # Get original importances (non-aggregated)
        original_importances = self.get_feature_importances(X, top_n=top_n, aggregate=False)
        
        # Get aggregated importances if requested
        if aggregate:
            aggregated_importances = self.get_feature_importances(X, top_n=top_n, aggregate=True)
        else:
            aggregated_importances = original_importances
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original importances
        orig_features = list(original_importances.keys())[:top_n]
        orig_values = [original_importances[f] for f in orig_features]
        
        bars1 = ax1.barh(orig_features, orig_values)
        ax1.set_xlabel('Importance')
        ax1.set_ylabel('Feature')
        ax1.set_title('Original Feature Importances')
        
        # Plot aggregated importances
        agg_features = list(aggregated_importances.keys())[:top_n]
        agg_values = [aggregated_importances[f] for f in agg_features]
        
        bars2 = ax2.barh(agg_features, agg_values)
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Feature')
        ax2.set_title('Aggregated Feature Importances')
        
        # Highlight sensitive features in both plots if they exist
        if hasattr(self, 'sensitive_attribute_groups_') and self.sensitive_attribute_groups_:
            # Get all sensitive features
            sensitive_features = []
            for group in self.sensitive_attribute_groups_.values():
                sensitive_features.extend(group)
            
            # Highlight in original plot
            for i, feature in enumerate(orig_features):
                if feature in sensitive_features:
                    bars1[i].set_color('red')
            
            # Highlight in aggregated plot - match by original feature name
            for i, feature in enumerate(agg_features):
                # Check if this aggregated feature contains any sensitive feature
                if any(self._extract_original_feature_name(sf) == feature for sf in sensitive_features):
                    bars2[i].set_color('red')
        
        plt.tight_layout()
        
        # Save plot if output directory is specified
        if output_dir:
            import os
            plt.savefig(os.path.join(output_dir, "mesomorphic_feature_importance_comparison.png"))
            print("Feature importance comparison saved to mesomorphic_feature_importance_comparison.png")
        
        return aggregated_importances if aggregate else original_importances

#######################
# Utility Functions
#######################

def inverse_transform_target(y_pred, inverse_transform=True):
    """Inverse transform log-transformed target"""
    if inverse_transform:
        # Clip to prevent overflow
        clipped_pred = np.clip(y_pred, -20, 20)
        return np.expm1(clipped_pred)
    else:
        return y_pred
    
def calculate_metrics(y_true, y_pred, is_transformed=True):
    """Calculate evaluation metrics"""
    if is_transformed:
        # Calculate metrics on both original and transformed scales
        y_pred_original = inverse_transform_target(y_pred, True)
        y_true_original = inverse_transform_target(y_true, True)
        
        metrics = {
            'MSE (original)': mean_squared_error(y_true_original, y_pred_original),
            'RMSE (original)': np.sqrt(mean_squared_error(y_true_original, y_pred_original)),
            'MAE (original)': mean_absolute_error(y_true_original, y_pred_original),
            'R²': r2_score(y_true, y_pred)
        }
    
    return metrics

def calculate_fairness_metrics(model, X_test, sensitive_attr_groups):
    
    if not sensitive_attr_groups or not model.feature_names_:
        return {}
        
    # Create empty dictionary to store fairness metrics
    fairness_metrics = {}
    
    # Calculate metrics for each attribute group
    for attr_name, columns in sensitive_attr_groups.items():
        valid_columns = [col for col in columns if col in model.feature_names_]
        if not valid_columns:
            continue
            
        if attr_name == 'age':
            # For continuous age, bin into age groups
            age_col = valid_columns[0]
            age_idx = model.feature_names_.index(age_col)
            
            # Extract ages
            ages = X_test[:, age_idx]
            
            # Create age bins
            age_bins = [0, 25, 35, 45, 55, 65, 100]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            
            # Assign each instance to an age bin
            binned_ages = np.digitize(ages, age_bins) - 1
            binned_ages = np.clip(binned_ages, 0, len(age_labels) - 1)
            
            # Calculate average prediction for each age group
            avg_predictions = {}
            for i, label in enumerate(age_labels):
                mask = binned_ages == i
                if np.sum(mask) > 0:
                    avg_predictions[label] = np.mean(model.predict(X_test[mask]))
            
            # Calculate max difference in average predictions (statistical parity)
            if len(avg_predictions) >= 2:
                values = list(avg_predictions.values())
                stat_parity_diff = max(values) - min(values)
                fairness_metrics[f'Statistical Parity Diff (Age)'] = stat_parity_diff
                
                # Store group averages
                for age_group, avg in avg_predictions.items():
                    fairness_metrics[f'Avg Prediction (Age={age_group})'] = avg
        
        elif attr_name == 'gender':
            # For gender, find which column represents which gender
            gender_predictions = {}
            
            for col in valid_columns:
                col_idx = model.feature_names_.index(col)
                gender_name = col.split('_')[-1] if '_' in col else col
                
                # Get instances where this gender is active (value=1)
                mask = X_test[:, col_idx] == 1
                if np.sum(mask) > 0:
                    gender_predictions[gender_name] = np.mean(model.predict(X_test[mask]))
            
            # Calculate difference between genders
            if len(gender_predictions) >= 2:
                values = list(gender_predictions.values())
                stat_parity_diff = max(values) - min(values)
                fairness_metrics[f'Statistical Parity Diff (Gender)'] = stat_parity_diff
                
                # Store gender-specific averages
                for gender, avg in gender_predictions.items():
                    fairness_metrics[f'Avg Prediction (Gender={gender})'] = avg
    
    return fairness_metrics

#######################
# Genetic Algorithm Functions
#######################

def get_hyperparameter_space() -> Dict[str, Any]:
    """Define the hyperparameter space for optimization"""
    return {
        'hyper_hidden_units': [
            (32, 16),
            (64, 32),
            (128, 64),
            (64, 32, 16),
            (128, 64, 32)
        ],
        'learning_rate': {
            'min': 1e-4,
            'max': 1e-1,
            'log': True
        },
        'batch_size': [32, 64, 128, 256],
        'epochs': {
            'min': 50,
            'max': 200,
            'step': 25
        },
        'fairness_weight': {  # Added fairness weight parameter
            'min': 0.0,
            'max': 1.0,
            'step': 0.1
        }
    }

def create_individual(hyperparameter_space: Dict[str, Any]) -> Dict[str, Any]:
    """Create a random individual (set of hyperparameters)"""
    individual = {}
    
    for param, space in hyperparameter_space.items():
        if isinstance(space, list):
            # For categorical parameters
            individual[param] = random.choice(space)
        elif isinstance(space, dict):
            # For numerical parameters
            if space.get('log', False):
                # Log-uniform sampling
                individual[param] = np.exp(
                    np.log(space['min']) + random.random() * 
                    (np.log(space['max']) - np.log(space['min'])))
            else:
                if 'step' in space:
                    # Discrete uniform sampling
                    steps = int((space['max'] - space['min']) / space['step']) + 1
                    value = space['min'] + random.randrange(steps) * space['step']
                    individual[param] = float(value)
                else:
                    # Continuous uniform sampling
                    individual[param] = space['min'] + random.random() * (space['max'] - space['min'])
    
    # Fix: Convert epochs to int
    if 'epochs' in individual:
        individual['epochs'] = int(individual['epochs'])
    
    return individual

def evaluate_individual(individual: Dict[str, Any], X: pd.DataFrame, y: pd.Series, 
                        sensitive_attr_groups: Dict[str, List[str]] = None, 
                        fairness_importance: float = 0.5,
                        cv: int = 5) -> Tuple[float, float, float]:
    
    print(f"\nEvaluating hyperparameters: {individual}")
    
    # Fix: Ensure epochs is an integer
    if 'epochs' in individual:
        individual['epochs'] = int(individual['epochs'])
    
    # Prepare sensitive attributes
    sensitive_attrs = sensitive_attr_groups or {}
    
    # Perform cross-validation
    cv_accuracy_scores = []
    cv_fairness_scores = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  CV Fold {fold_idx + 1}/{cv}", end="", flush=True)
        X_train_fold, X_val_fold = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train_fold, y_val_fold = y.iloc[train_idx].values, y.iloc[val_idx].values
        
        # Create and train model
        model = InterpretableMesomorphicRegressor(
            hyper_hidden_units=individual['hyper_hidden_units'],
            learning_rate=individual['learning_rate'],
            batch_size=individual['batch_size'],
            epochs=individual['epochs'],
            fairness_weight=individual['fairness_weight'],
            validation_split=0.1,
            verbose=0,
            random_state=42
        )
        model.set_feature_names(X.columns.tolist())
        model.fit(X_train_fold, y_train_fold, sensitive_attr_groups=sensitive_attrs)
        
        # Calculate accuracy score
        y_val_pred = model.predict(X_val_fold)
        accuracy_score = r2_score(y_val_fold, y_val_pred)
        cv_accuracy_scores.append(accuracy_score)
        
        # Calculate fairness score (1 - unfairness)
        if sensitive_attrs:
            fairness_score = 1.0 - model._fairness_loss(X_val_fold, y_val_fold, model.model_, sensitive_attrs)
        else:
            fairness_score = 1.0  # Perfect fairness if no sensitive attributes
        cv_fairness_scores.append(fairness_score)
        
        print(f" - R² Score: {accuracy_score:.4f}, Fairness: {fairness_score:.4f}")
    
    # Calculate average scores
    avg_accuracy = np.mean(cv_accuracy_scores)
    avg_fairness = np.mean(cv_fairness_scores)
    
    # Combined score with fairness importance
    combined_score = (1 - fairness_importance) * avg_accuracy + fairness_importance * avg_fairness
    
    print(f"Mean CV Scores - R²: {avg_accuracy:.4f}, Fairness: {avg_fairness:.4f}, Combined: {combined_score:.4f}")
    return combined_score, avg_accuracy, avg_fairness

def genetic_algorithm(X: pd.DataFrame, y: pd.Series, 
                     hyperparameter_space: Dict[str, Any],
                     sensitive_attr_groups: Dict[str, List[str]] = None,
                     fairness_importance: float = 0.3,
                     population_size: int = 10,
                     generations: int = 5,
                     elite_size: int = 2,
                     mutation_rate: float = 0.2,
                     cv: int = 3) -> Tuple[Dict[str, Any], Tuple[float, float, float], List[Tuple[float, float, float]]]:

    
    print(f"\n--- Starting Genetic Algorithm Optimization with Fairness ---")
    
    # Initialize population
    population = [create_individual(hyperparameter_space) for _ in range(population_size)]
    
    # Fix: Ensure epochs is an integer in all individuals
    for ind in population:
        if 'epochs' in ind:
            ind['epochs'] = int(ind['epochs'])
    
    # Evaluate initial population
    fitness_results = [evaluate_individual(ind, X, y, sensitive_attr_groups, fairness_importance, cv) 
                       for ind in population]
    fitness = [result[0] for result in fitness_results]  # Combined score
    accuracy = [result[1] for result in fitness_results]
    fairness = [result[2] for result in fitness_results]
    
    # Track best fitness over generations
    best_fitness_history = [(max(fitness), accuracy[np.argmax(fitness)], fairness[np.argmax(fitness)])]
    
    for generation in range(generations):
        print(f"\n=== Generation {generation + 1}/{generations} ===")
        
        # Select elite individuals
        elite_indices = np.argsort(fitness)[-elite_size:][::-1]
        elite_population = [population[i] for i in elite_indices]
        elite_fitness = [fitness[i] for i in elite_indices]
        elite_accuracy = [accuracy[i] for i in elite_indices]
        elite_fairness = [fairness[i] for i in elite_indices]
        
        # Create next generation
        next_population = elite_population.copy()
        next_fitness = elite_fitness.copy()
        next_accuracy = elite_accuracy.copy()
        next_fairness = elite_fairness.copy()
        
        # Fill the rest of the population with offspring
        while len(next_population) < population_size:
            # Selection - tournament selection
            selected_indices = random.sample(range(len(population)), 3)
            selected_indices.sort(key=lambda i: fitness[i], reverse=True)
            parent1 = population[selected_indices[0]]
            parent2 = population[selected_indices[1]]
            
            # Crossover
            child = {}
            for param in parent1.keys():
                child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
            
            # Mutation
            for param, space in hyperparameter_space.items():
                if random.random() < mutation_rate:
                    if param == 'hyper_hidden_units':
                        child[param] = random.choice(hyperparameter_space['hyper_hidden_units'])
                    elif param == 'batch_size':
                        child[param] = random.choice(hyperparameter_space['batch_size'])
                    elif param == 'epochs':
                        step = hyperparameter_space['epochs']['step']
                        min_val = hyperparameter_space['epochs']['min']
                        max_val = hyperparameter_space['epochs']['max']
                        steps = int((max_val - min_val) / step) + 1
                        current_steps = int((child[param] - min_val) / step)
                        new_steps = max(0, min(steps - 1, current_steps + random.choice([-1, 1])))
                        child[param] = int(min_val + new_steps * step)  # Fix: Convert to int
                    elif param == 'fairness_weight':
                        step = hyperparameter_space['fairness_weight']['step']
                        min_val = hyperparameter_space['fairness_weight']['min']
                        max_val = hyperparameter_space['fairness_weight']['max']
                        steps = int((max_val - min_val) / step) + 1
                        current_steps = int((child[param] - min_val) / step)
                        new_steps = max(0, min(steps - 1, current_steps + random.choice([-1, 1])))
                        child[param] = min_val + new_steps * step
                    else:  # learning_rate
                        if hyperparameter_space[param].get('log', False):
                            min_log = np.log(hyperparameter_space[param]['min'])
                            max_log = np.log(hyperparameter_space[param]['max'])
                            child[param] = np.exp(random.uniform(min_log, max_log))
            
            # Fix: Ensure epochs is an integer
            if 'epochs' in child:
                child['epochs'] = int(child['epochs'])
                
            next_population.append(child)
            
            # Evaluate new individual
            child_combined, child_accuracy, child_fairness = evaluate_individual(
                child, X, y, sensitive_attr_groups, fairness_importance, cv
            )
            next_fitness.append(child_combined)
            next_accuracy.append(child_accuracy)
            next_fairness.append(child_fairness)
        
        # Update population and fitness
        population = next_population
        fitness = next_fitness
        accuracy = next_accuracy
        fairness = next_fairness
        
        # Track best fitness
        best_idx = np.argmax(fitness)
        current_best = (fitness[best_idx], accuracy[best_idx], fairness[best_idx])
        best_fitness_history.append(current_best)
        
        # Print generation statistics
        print(f"Generation {generation + 1} - Best Combined: {current_best[0]:.4f}, "
              f"Accuracy: {current_best[1]:.4f}, Fairness: {current_best[2]:.4f}")
    
    # Find the best individual
    best_index = np.argmax(fitness)
    best_individual = population[best_index]
    best_scores = (fitness[best_index], accuracy[best_index], fairness[best_index])
    
    print("\nOptimization completed!")
    print(f"Best combined score: {best_scores[0]:.4f}")
    print(f"Best accuracy (R²): {best_scores[1]:.4f}")
    print(f"Best fairness score: {best_scores[2]:.4f}")
    print(f"Best hyperparameters: {best_individual}")
    
    return best_individual, best_scores, best_fitness_history

def identify_potential_sensitive_attributes(df):
    
    potential_sensitive = []
    
    # Look for columns with keywords sensitive attributes
    sensitive_keywords = [
        'numeric__Driver Age',
        'categorical__Gender_Female',
        'categorical__Gender_Male',
        'categorical__Gender_Other'
    ]
    
    # Check each column name for sensitive keywords
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in sensitive_keywords):
            potential_sensitive.append(col)
            continue
            
        # Also check for one-hot encoded columns that might represent sensitive attributes
        if '_' in col and any(keyword in col_lower.split('_')[-1].lower() for keyword in sensitive_keywords):
            potential_sensitive.append(col)
            
    # Also try to identify potentially sensitive categorical columns based on cardinality
    # (e.g., columns with 2-10 unique values might represent sensitive groups)
    for col in df.columns:
        if col in potential_sensitive:
            continue
            
        # Skip columns that are clearly numeric and not categorical
        if df[col].dtype in ['int64', 'float64'] and 'id' not in col.lower() and not col.endswith('_code'):
            # But include binary columns which might be flags for sensitive attributes
            if set(df[col].dropna().unique()) == {0, 1} and len(df[col].dropna().unique()) == 2:
                potential_sensitive.append(col)
            continue
                
        # Check cardinality for potential categorical sensitive attributes
        unique_values = df[col].nunique()
        if 2 <= unique_values <= 10:
            # Column has reasonable number of categories
            # Check if values suggest sensitive attributes
            values = df[col].dropna().astype(str).str.lower().unique()
            if any(keyword in ' '.join(values) for keyword in sensitive_keywords):
                potential_sensitive.append(col)
    
    return sorted(list(set(potential_sensitive)))

#######################
# Main Pipeline
#######################

def run_pipeline(data_path, target_column, test_data_path=None, sensitive_attr_groups=None, 
               fairness_importance=0.3, output_dir='meso_outputs', ga_population=10, 
               ga_generations=5, test_size=0.2, random_state=42, auto_detect_sensitive=True):
    """Run the complete pipeline with enhanced graphical explanations"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n=== Mesomorphic Model Pipeline with Graphical Explanations ===")
    
    try:
        # 1. Load training data
        print(f"\n[1/8] Loading training data from {data_path}...")
        train_df = pd.read_csv(data_path)
        print(f"Training data loaded. Shape: {train_df.shape}")
        
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        # 2. Load test data if provided
        if test_data_path:
            print(f"\n[1.5/8] Loading test data from {test_data_path}...")
            test_df = pd.read_csv(test_data_path)
            print(f"Test data loaded. Shape: {test_df.shape}")
            if target_column not in test_df.columns:
                print("Warning: Target column not in test data. Using features only.")
                X_test = test_df
                y_test = None
            else:
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
            # Split from training data if no test data provided
            X_train, _, y_train, _ = train_test_split(
                train_df.drop(columns=[target_column]), 
                train_df[target_column],
                test_size=0.0001,  # Minimal split to get training data
                random_state=random_state
            )
        else:
            print("\n[2/8] Splitting data into train/test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                train_df.drop(columns=[target_column]), 
                train_df[target_column],
                test_size=test_size,
                random_state=random_state
            )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # 3. Auto-detect sensitive attributes
        if auto_detect_sensitive and (sensitive_attr_groups is None or len(sensitive_attr_groups) == 0):
            print("\n[2.5/8] Auto-detecting potential sensitive attributes...")
            potential_sensitive = identify_potential_sensitive_attributes(train_df)
            print(f"Potential sensitive attributes found: {potential_sensitive}")
            
            if not sensitive_attr_groups:
                auto_groups = {}
                age_cols = [col for col in potential_sensitive if 'age' in col.lower()]
                if age_cols:
                    auto_groups['age'] = age_cols
                
                gender_cols = [col for col in potential_sensitive 
                             if any(term in col.lower() for term in ['gender', 'sex', 'female', 'male'])]
                if gender_cols:
                    auto_groups['gender'] = gender_cols
                
                sensitive_attr_groups = auto_groups
                print(f"Using auto-detected sensitive attribute groups: {sensitive_attr_groups}")
            
        # Validate sensitive attributes
        valid_sensitive_groups = {}
        if sensitive_attr_groups:
            for group_name, columns in sensitive_attr_groups.items():
                valid_columns = [col for col in columns if col in train_df.columns]
                if valid_columns:
                    valid_sensitive_groups[group_name] = valid_columns
                else:
                    print(f"Warning: No valid columns found for sensitive group '{group_name}'")
        
        if valid_sensitive_groups:
            print(f"\nUsing sensitive attribute groups: {valid_sensitive_groups}")
        else:
            print("\nNo valid sensitive attributes provided or detected. Running without fairness regularization.")

        # 4. Run genetic algorithm
        print("\n[3/8] Running genetic algorithm for hyperparameter optimization...")
        hyperparameter_space = get_hyperparameter_space()
        if 'epochs' in hyperparameter_space:
            hyperparameter_space['epochs']['min'] = int(hyperparameter_space['epochs']['min'])
            hyperparameter_space['epochs']['max'] = int(hyperparameter_space['epochs']['max'])
            hyperparameter_space['epochs']['step'] = int(hyperparameter_space['epochs']['step'])
            
        best_params, best_scores, fitness_history = genetic_algorithm(
            X_train, y_train, 
            hyperparameter_space,
            sensitive_attr_groups=valid_sensitive_groups,
            fairness_importance=fairness_importance,
            population_size=ga_population,
            generations=ga_generations,
            cv=3
        )
        
        if 'epochs' in best_params:
            best_params['epochs'] = int(best_params['epochs'])

        # 5. Train final model
        print("\n[4/8] Training final model with optimized hyperparameters...")
        final_model = InterpretableMesomorphicRegressor(
            hyper_hidden_units=best_params['hyper_hidden_units'],
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            fairness_weight=best_params['fairness_weight'],
            verbose=1,
            random_state=random_state
        )
        final_model.set_feature_names(X_train.columns.tolist())
        final_model.fit(X_train.values, y_train.values, sensitive_attr_groups=valid_sensitive_groups)

        # 6. Evaluate model
        print("\n[5/8] Evaluating model on test set...")
        y_pred = final_model.predict(X_test.values)
        if y_test is not None:
            accuracy_metrics = calculate_metrics(y_test, y_pred, is_transformed=True)
        else:
            accuracy_metrics = {}
            print("No target values in test set - skipping accuracy metrics")

        # 7. Analyze fairness
        print("\n[6/8] Analyzing fairness on test set...")
        if valid_sensitive_groups and y_test is not None:
            fairness_metrics = final_model.analyze_fairness(X_test.values, y_test.values)
        else:
            fairness_metrics = {}
            print("No sensitive attributes or target values - skipping fairness analysis")

        # 8. Feature importance analysis
        print("\n[7/8] Calculating feature importances...")
        feature_importances = final_model.plot_feature_importance(
            X=X_train.values, 
            aggregate=True, 
            top_n=15,
            output_dir=output_dir
        )
        
        print("\nTop 10 aggregated feature importances:")
        for i, (feature, importance) in enumerate(list(feature_importances.items())[:10]):
            print(f"{i+1}. {feature}: {importance:.4f}")

        # Combine all metrics
        metrics = {**accuracy_metrics, **fairness_metrics}
        
        if metrics:
            print("\nTest Set Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # 9. Generate comprehensive graphical explanations
        print("\n[8/8] Generating comprehensive graphical explanations...")
        
        # Create explanation subdirectory
        explanation_dir = os.path.join(output_dir, 'explanations')
        Path(explanation_dir).mkdir(exist_ok=True)
        
        # Select representative instances
        if len(X_test) > 10:
            sample_indices = [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4, -1, 
                            random.randint(0, len(X_test)-1)]
        else:
            sample_indices = range(min(5, len(X_test)))
        
        explanations = {}
        for idx in sample_indices:
            X_instance = X_test.values[idx:idx+1]
            instance_features = X_test.iloc[idx].to_dict()
            
            # Get all explanations
            explanation = final_model.get_instance_explanation(X_instance)
            counterfactuals, cf_predictions = final_model.generate_counterfactuals(X_instance)
            
            # Store explanations
            explanations[f"instance_{idx}"] = {
                'features': instance_features,
                'prediction': float(explanation['prediction']),
                'bias': float(explanation['bias']),
                'feature_weights': {k: float(v) for k, v in explanation['feature_weights'].items()},
                'feature_contributions': {k: float(v) for k, v in explanation['feature_contributions'].items()},
                'counterfactuals': {
                    attr: {str(k): float(cf_predictions[f"{attr}={k}"]) 
                          for k in cf.keys()}
                    for attr, cf in counterfactuals.items()
                }
            }
            
            # Generate visualizations for each instance
            self.plot_detailed_instance_explanation(
                instance=instance_features,
                explanation=explanation,
                counterfactuals=counterfactuals,
                cf_predictions=cf_predictions,
                output_dir=explanation_dir,
                idx=idx
            )
            
            # Print to console
            print(f"\nExplanation for instance {idx}:")
            if y_test is not None:
                print(f"True value: {y_test.values[idx]:.2f}", end=", ")
            print(f"Predicted: {explanation['prediction']:.2f}")
            print("Top 5 contributing features:")
            for feature, contrib in sorted(explanation['feature_contributions'].items(), 
                                         key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"  {feature}: {contrib:.4f} (weight: {explanation['feature_weights'][feature]:.4f})")
        
        # Save all artifacts
        print("\nSaving all artifacts...")
        
        # Save model
        model_path = os.path.join(output_dir, 'mesomorphic_fair_model.pkl')
        joblib.dump(final_model, model_path)
        
        # Save metrics
        if metrics:
            metrics_path = os.path.join(output_dir, 'model_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in metrics.items()}, f, indent=4)
        
        # Save hyperparameters
        hyperparams_path = os.path.join(output_dir, 'best_hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump({k: list(v) if isinstance(v, tuple) else 
                     (int(v) if isinstance(v, np.integer) else 
                      float(v) if isinstance(v, np.floating) else v)
                     for k, v in best_params.items()}, f, indent=4)
        
        # Save feature importances
        importances_path = os.path.join(output_dir, 'feature_importances.json')
        with open(importances_path, 'w') as f:
            json.dump({k: float(v) for k, v in feature_importances.items()}, f, indent=4)
        
        # Save explanations
        explanation_path = os.path.join(output_dir, 'explanations.json')
        with open(explanation_path, 'w') as f:
            json.dump(explanations, f, indent=4)
        
        print("\nPipeline completed successfully!")
        print(f"All artifacts saved to: {output_dir}")
        print(f"Detailed explanations saved to: {explanation_dir}")
        
        return {
            'model': final_model,
            'metrics': metrics,
            'feature_importances': feature_importances,
            'explanations': explanations,
            'best_params': best_params
        }
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_detailed_instance_explanation(self, instance, explanation, counterfactuals, 
                                     cf_predictions, output_dir, idx):
    """Generate comprehensive visual explanations for an instance"""
    plt.style.use('seaborn')
    
    # 1. Feature Contribution Plot
    plt.figure(figsize=(12, 8))
    contribs = explanation['feature_contributions']
    sorted_features = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features, values = zip(*sorted_features)
    
    colors = ['#1f77b4' if v > 0 else '#d62728' for v in values]
    plt.barh(features[::-1], values[::-1], color=colors[::-1])
    plt.title(f'Feature Contributions for Instance {idx}\nPrediction: {explanation["prediction"]:.2f}', pad=20)
    plt.xlabel('Contribution to Prediction')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'instance_{idx}_contributions.png'), dpi=300)
    plt.close()
    
    # 2. Counterfactual Analysis Plot (if available)
    if counterfactuals:
        plt.figure(figsize=(10, 6))
        original_pred = cf_predictions['original']
        diffs = []
        labels = []
        
        for attr, values in counterfactuals.items():
            for val in values:
                pred_key = f"{attr}={val}"
                diff = cf_predictions[pred_key] - original_pred
                diffs.append(diff)
                labels.append(f"{attr}\n{val}")
        
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in diffs]
        plt.bar(labels, diffs, color=colors)
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title(f'Counterfactual Analysis for Instance {idx}\nOriginal Prediction: {original_pred:.2f}')
        plt.ylabel('Prediction Difference')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'instance_{idx}_counterfactuals.png'), dpi=300)
        plt.close()
    
    # 3. Feature Importance Comparison Plot
    plt.figure(figsize=(12, 6))
    weights = explanation['feature_weights']
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features, weights = zip(*sorted_weights)
    
    plt.bar(features, weights, color='#9467bd')
    plt.title(f'Instance-Specific Feature Weights for Instance {idx}')
    plt.ylabel('Weight Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'instance_{idx}_weights.png'), dpi=300)
    plt.close()
    
    # 4. Combined Waterfall Plot
    if explanation['feature_contributions']:
        plt.figure(figsize=(14, 8))
        contribs = explanation['feature_contributions']
        sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate cumulative contributions
        cumulative = explanation['bias']
        positions = range(len(sorted_contribs))
        
        for i, (feature, contrib) in enumerate(sorted_contribs):
            plt.barh(i, contrib, left=cumulative, 
                    color='#1f77b4' if contrib > 0 else '#d62728')
            cumulative += contrib
        
        # Add bias and prediction lines
        plt.axvline(x=explanation['bias'], color='#7f7f7f', linestyle='--', label='Bias')
        plt.axvline(x=explanation['prediction'], color='#ff7f0e', linestyle='-', label='Final Prediction')
        
        plt.yticks(positions, [f[0] for f in sorted_contribs])
        plt.title(f'Waterfall Plot for Instance {idx}\nBias: {explanation["bias"]:.2f}, Prediction: {explanation["prediction"]:.2f}')
        plt.xlabel('Contribution to Prediction')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'instance_{idx}_waterfall.png'), dpi=300)
        plt.close()

# Add the plotting method to the class
InterpretableMesomorphicRegressor.plot_detailed_instance_explanation = plot_detailed_instance_explanation

if __name__ == "__main__":
    # Configure pipeline with the specified protected characteristics
    config = {
        'data_path': 'settlement_data_processed.csv',
        'target_column': 'SettlementValue',
        'sensitive_attr_groups': {
            'age': ['numeric__Driver Age'],
            'gender': ['categorical__Gender_Female', 'categorical__Gender_Male', 'categorical__Gender_Other']
        },
        'fairness_importance': 0.3,    # Moderate emphasis on fairness
        'output_dir': 'meso_fair_outputs',
        'ga_population': 10, 
        'ga_generations': 5,
        'test_size': 0.2,
        'random_state': 42,
        'auto_detect_sensitive': False 
    }
    
    # Run pipeline
    model, metrics = run_pipeline(**config)