from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import pandas as pd
import joblib

# Ensure the project works for both Windows and Linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')
#Best Hyperparameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
# Define a grid of hyperparameters to search over
param_grid = {
    'n_estimators': [100, 150, 200],              # Number of trees
    'max_depth': [10, 15],               # Maximum depth of each tree
    'min_samples_split': [2, 5],         # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 4],           # Minimum samples at a leaf node

}

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(
    class_weight='balanced',   # Handle class imbalance
    random_state=42,           # Ensure reproducibility
    n_jobs=-1                  # Use all available CPU cores
)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(
    estimator=rf_clf, 
    param_grid=param_grid, 
    cv=3,
    scoring='accuracy',
    verbose=2
)

# Train the model using GridSearchCV on the training data
grid_search.fit(X_train, y_train.values.ravel())

# Get the best model and its hyperparameters
best_rf_clf = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the best model on the validation data
y_val_pred = best_rf_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with Best Model: {val_accuracy:.2f}")

# Generate a classification report and confusion matrix for the validation data
print("Best Model Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Best Model Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Save the best model to a file
model_save_path = current_dir / 'best_rf_model.joblib'
joblib.dump(best_rf_clf, model_save_path)
print(f"Best model saved to: {model_save_path}")
