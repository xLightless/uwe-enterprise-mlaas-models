import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re

DATA_PATH = 'settlement_data_processed.csv'

def aggregate_feature_importance(model, X_train, n_top=20):
    """
    Aggregates feature importances from one-hot encoded columns back to their original features.
    
    """
    # Get feature importances from the model
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    # Extract original feature names from the encoded column names
    def extract_original_feature(column_name):
        # For one-hot encoded features (categorical__onehot__Feature_Value)
        if '__onehot__' in column_name:
            # Handle feature names with spaces (e.g. 'Weather Conditions')
            match = re.search(r'categorical__onehot__([^_]+(?:\s[^_]+)*)_', column_name)
            if match:
                return match.group(1)
        # For other transformed features (transformer__Feature)
        elif '__' in column_name:
            parts = column_name.split('__')
            if len(parts) >= 2:
                return parts[1]
        # For features without transformation
        return column_name
    
    # Add original feature names to the DataFrame
    feature_importances['original_feature'] = feature_importances.index.map(extract_original_feature)
    
    # Aggregate importances by original feature
    aggregated_importance = feature_importances.groupby('original_feature')['importance'].sum().reset_index()
    aggregated_importance = aggregated_importance.sort_values('importance', ascending=False)
    
    # Plot the top N features for both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original feature importances (not aggregated)
    # Get top N features to plot
    top_features = feature_importances.head(n_top).copy()
    
    # Create the plot with a direct mapping for both x and y
    sns.barplot(
        x='importance', 
        y='index',
        data=top_features.reset_index().head(n_top), 
        ax=ax1
    )
    ax1.set_title('Original Feature Importance (One-Hot Encoded)')
    ax1.set_xlabel('Importance')
    ax1.set_ylabel('Feature')
    
    # Plot aggregated feature importances
    # Get top N aggregated features
    top_agg_features = aggregated_importance.head(n_top).copy()
    
    sns.barplot(
        x='importance', 
        y='original_feature',
        data=top_agg_features, 
        ax=ax2
    )
    ax2.set_title('Aggregated Feature Importance')
    ax2.set_xlabel('Importance')
    ax2.set_ylabel('Feature')
    
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    print("Feature importance comparison saved as 'feature_importance_comparison.png'")
    
    return aggregated_importance

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
    if target_column not in df.columns:
        print(f"Warning: Target column '{target_column}' not found in dataframe.")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42, hyperparameter_tuning=False):
    """Train a Random Forest model"""
    print("Training Random Forest model...")
    
    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=random_state)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
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

def plot_feature_importance(model, X_train, aggregate=True):
    """Plot feature importance with option to aggregate one-hot encoded features"""
    print("Plotting feature importance...")
    
    if aggregate:
        # Use the aggregation function
        feature_importance = aggregate_feature_importance(model, X_train)
        print("Feature importance aggregated and plotted")
        
        # Also create a separate regular feature importance plot
        standard_importance = pd.DataFrame(
            model.feature_importances_,
            index=X_train.columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        
        # Get top 20 features for the plot
        top_features = standard_importance.head(20).reset_index()
        
        # Create the plot with explicit x and y mappings
        sns.barplot(
            x='importance', 
            y='index',
            data=top_features
        )
        plt.title('Random Forest Feature Importance (Non-Aggregated)')
        plt.tight_layout()
        plt.savefig('random_forest_feature_importance_original.png')
        print("Original feature importance plot saved as 'random_forest_feature_importance_original.png'")
        
        # Return the top 10 aggregated features for display
        return feature_importance.head(10)
    else:
        # Original implementation without aggregation
        feature_importance = pd.DataFrame(
            model.feature_importances_,
            index=X_train.columns,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        
        # Get top 20 features for the plot 
        top_features = feature_importance.head(20).reset_index()
        
        # Create the plot with explicit x and y mappings
        sns.barplot(
            x='importance', 
            y='index',
            data=top_features
        )
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('random_forest_feature_importance.png')
        print("Feature importance plot saved as 'random_forest_feature_importance.png'")
        
        return feature_importance.head(10)

def save_model(model, filename='random_forest_model.pkl'):
    """Save the trained model to disk"""
    print(f"Saving model to {filename}")
    joblib.dump(model, filename)
    print("Model saved successfully")

def main():
   
    target_column = 'SettlementValue' 
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
    
    # Train model - set hyperparameter_tuning=True if you want to perform grid search
    model = train_random_forest(X_train, y_train, hyperparameter_tuning=False)
    
    # Evaluate model with inverse transformation
    eval_results = evaluate_model(model, X_test, y_test, inverse_transform=target_was_log_transformed)
    
    # Plot feature importance with aggregation
    feature_importance = plot_feature_importance(model, X_train, aggregate=True)
    print("\nTop 10 most important features (aggregated):")
    print(feature_importance)
    
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
    plt.title('Random Forest: Actual vs Predicted (Original Scale)')
    plt.tight_layout()
    plt.savefig('random_forest_predictions_original_scale.png')
    print("Predictions plot saved as 'random_forest_predictions_original_scale.png'")
    
    # Also plot on transformed scale for comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, eval_results['y_pred'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual (Log-Transformed)')
    plt.ylabel('Predicted (Log-Transformed)')
    plt.title('Random Forest: Actual vs Predicted (Log-Transformed Scale)')
    plt.tight_layout()
    plt.savefig('random_forest_predictions_log_scale.png')
    print("Predictions plot saved as 'random_forest_predictions_log_scale.png'")

if __name__ == "__main__":
    main()