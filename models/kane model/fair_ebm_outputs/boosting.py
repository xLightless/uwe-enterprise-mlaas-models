# XGBoost Model Training Script with Aggregated Feature Importance

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score  # for regression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re

# File path to the processed CSV
DATA_PATH = 'settlement_data_processed.csv'

def load_data(filepath):
    """Load data from CSV file"""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully with shape: {df.shape}")
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print(f"Splitting data with target column: {target_column}")
    
    # Assuming the last column is the target variable if not specified
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train, random_state=42, hyperparameter_tuning=False):
    """Train an XGBoost model"""
    print("Training XGBoost model...")
    
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=random_state)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state
        )
        model.fit(X_train, y_train)
    
    return model

def inverse_transform_target(y_pred, inverse_transform=True):
    """Inverse transform log-transformed target"""
    if inverse_transform:
        return np.expm1(y_pred)
    else:
        return y_pred

def evaluate_model(model, X_test, y_test, inverse_transform=True):
    """Evaluate the model on test data"""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics on transformed data
    mse_transformed = mean_squared_error(y_test, y_pred)
    rmse_transformed = np.sqrt(mse_transformed)
    r2_transformed = r2_score(y_test, y_pred)
    
    # Inverse transform predictions and actual values if needed
    y_pred_original = inverse_transform_target(y_pred, inverse_transform)
    y_test_original = inverse_transform_target(y_test.values, inverse_transform)
    
    # Calculate metrics on original scale
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    rmse_original = np.sqrt(mse_original)
    r2_original = r2_score(y_test_original, y_pred_original)
    
    if inverse_transform:
        print("--- Metrics on transformed scale ---")
        print(f"Mean Squared Error: {mse_transformed:.4f}")
        print(f"Root Mean Squared Error: {rmse_transformed:.4f}")
        print(f"R² Score: {r2_transformed:.4f}")
        
        print("\n--- Metrics on original scale (after inverse transform) ---")
        print(f"Mean Squared Error: {mse_original:.4f}")
        print(f"Root Mean Squared Error: {rmse_original:.4f}")
        print(f"R² Score: {r2_original:.4f}")
    else:
        print("--- Model metrics ---")
        print(f"Mean Squared Error: {mse_transformed:.4f}")
        print(f"Root Mean Squared Error: {rmse_transformed:.4f}")
        print(f"R² Score: {r2_transformed:.4f}")
    
    return {
        'mse_transformed': mse_transformed,
        'rmse_transformed': rmse_transformed,
        'r2_transformed': r2_transformed,
        'mse_original': mse_original,
        'rmse_original': rmse_original,
        'r2_original': r2_original,
        'y_pred': y_pred,
        'y_pred_original': y_pred_original,
        'y_test_original': y_test_original
    }

def extract_original_feature_name(column_name):
    """
    Extracts the original feature name from an encoded feature name.
    
    Args:
        column_name: The encoded feature name
        
    Returns:
        The original feature name
    """
    # Handle one-hot encoded features with pattern 'categorical__onehot__feature_value'
    if 'categorical__' in column_name:
        # Handle feature names with spaces (e.g. 'Weather Conditions')
        match = re.search(r'categorical__([^_]+(?:\s[^_]+)*)_', column_name)
        if match:
            return match.group(1)
    
    # Handle other transformation patterns like 'transformer__feature'
    elif '__' in column_name:
        parts = column_name.split('__')
        if len(parts) >= 2:
            return parts[1]
    
    # Return as is for features without transformation
    return column_name

def aggregate_feature_importances(importances, feature_names):
    """
    Aggregates feature importances from one-hot encoded columns back to their original features.
    
    Args:
        importances: pandas DataFrame with feature importances
        feature_names: List of feature names
        
    Returns:
        DataFrame with aggregated feature importances
    """
    # Create a mapping from original feature name to encoded feature names
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
        total_importance = sum(importances.loc[feature, 'importance'] for feature in encoded_features 
                              if feature in importances.index)
        aggregated_importances[original_name] = total_importance
    
    # Create a DataFrame from the aggregated importances
    agg_df = pd.DataFrame({
        'importance': aggregated_importances
    }).sort_values('importance', ascending=False)
    
    return agg_df

def plot_feature_importance(model, X_train, aggregate=True, top_n=20):
    """
    Plot feature importance with option to aggregate one-hot encoded features.
    
    Args:
        model: Trained XGBoost model
        X_train: Training data features
        aggregate: Whether to aggregate one-hot encoded features
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importances
    """
    print("Plotting feature importance...")
    
    # Extract feature importances from the model
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    if aggregate:
        # Aggregate one-hot encoded feature importances
        aggregated_importances = aggregate_feature_importances(feature_importances, X_train.columns)
        
        # Create a side-by-side comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original feature importances
        top_original = feature_importances.head(top_n)
        sns.barplot(x='importance', y=top_original.index, data=top_original, ax=ax1)
        ax1.set_title('Original Feature Importance (One-Hot Encoded)')
        ax1.set_xlabel('Importance')
        ax1.set_ylabel('Feature')
        
        # Plot aggregated feature importances
        top_aggregated = aggregated_importances.head(top_n)
        sns.barplot(x='importance', y=top_aggregated.index, data=top_aggregated, ax=ax2)
        ax2.set_title('Aggregated Feature Importance')
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance_comparison.png')
        print("Feature importance comparison saved as 'xgboost_feature_importance_comparison.png'")
        
        # Also create a separate plot for just the aggregated importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y=aggregated_importances.head(15).index, data=aggregated_importances.head(15))
        plt.title('XGBoost Aggregated Feature Importance (Top 15)')
        plt.tight_layout()
        plt.savefig('xgboost_aggregated_feature_importance.png')
        print("Aggregated feature importance saved as 'xgboost_aggregated_feature_importance.png'")
        
        return aggregated_importances
    else:
        # Plot original feature importances without aggregation
        plt.figure(figsize=(10, 8))
        top_features = feature_importances.head(top_n)
        sns.barplot(x='importance', y=top_features.index, data=top_features)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('xgboost_feature_importance.png')
        print("Feature importance plot saved as 'xgboost_feature_importance.png'")
        
        return feature_importances

def save_model(model, filename='xgboost_model.pkl'):
    """Save the trained model to disk"""
    print(f"Saving model to {filename}")
    joblib.dump(model, filename)
    print("Model saved successfully")

def main():
    # Assuming the target column name - you should replace this with your actual target column
    target_column = 'SettlementValue'  # Replace with your actual target column name
    
    # Flag to indicate if the target was log-transformed during preprocessing
    target_was_log_transformed = True
    
    # Load data (already preprocessed)
    df = load_data(DATA_PATH)
    
    # Display data information
    print("\nData Information:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Train XGBoost model
    model = train_xgboost(X_train, y_train, hyperparameter_tuning=False)  # Set to True for hyperparameter tuning
    
    # Evaluate model with inverse transformation
    eval_results = evaluate_model(model, X_test, y_test, inverse_transform=target_was_log_transformed)
    
    # Plot feature importance with aggregation
    feature_importance = plot_feature_importance(model, X_train, aggregate=True)
    print("\nTop 10 most important features (aggregated):")
    print(feature_importance.head(10))
    
    # Save model
    save_model(model)
    
    # Optional: Plot actual vs predicted values (on original scale)
    plt.figure(figsize=(10, 6))
    plt.scatter(eval_results['y_test_original'], eval_results['y_pred_original'], alpha=0.5)
    plt.plot(
        [eval_results['y_test_original'].min(), eval_results['y_test_original'].max()], 
        [eval_results['y_test_original'].min(), eval_results['y_test_original'].max()], 
        'r--'
    )
    plt.xlabel('Actual Settlement Value')
    plt.ylabel('Predicted Settlement Value')
    plt.title('XGBoost: Actual vs Predicted (Original Scale)')
    plt.tight_layout()
    plt.savefig('xgboost_predictions_original_scale.png')
    print("Predictions plot saved as 'xgboost_predictions_original_scale.png'")
    
    # Also plot on transformed scale for comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, eval_results['y_pred'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual (Log-Transformed)')
    plt.ylabel('Predicted (Log-Transformed)')
    plt.title('XGBoost: Actual vs Predicted (Log-Transformed Scale)')
    plt.tight_layout()
    plt.savefig('xgboost_predictions_log_scale.png')
    print("Predictions plot saved as 'xgboost_predictions_log_scale.png'")

if __name__ == "__main__":
    main()