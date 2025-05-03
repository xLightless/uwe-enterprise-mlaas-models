# Fair EBM Model Usage Guide

## Loading the model
```python
import pickle
import numpy as np

# Load the model
with open('fair_ebm_outputs/ebm_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Making predictions
```python
# Make predictions (in log-transformed space)
predictions_log = model.predict(X_new)

# Transform predictions back to original scale
def inverse_transform_target(y_pred):
    return np.expm1(y_pred)

predictions_original = inverse_transform_target(predictions_log)
```

## Getting feature importances (aggregated)
```python
# Get feature importances with aggregation for one-hot encoded features
importances = model.get_feature_importances(X_new, aggregate=True)
print(importances)
```

## Analyzing fairness
```python
# Analyze fairness on new data
fairness_metrics = model.evaluate_fairness(X_new, y_true, predictions_log)
print(fairness_metrics)
```