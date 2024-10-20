import pandas as pd
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Load data
current_dir = Path(__file__).resolve().parent
X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')

# Define the parameter distribution for alpha (L2 regularization strength)
param_dist = {
    'alpha': uniform(1e-2, 10),  # Random search over alpha values in range [1e-6, 1e-2]
    'hidden_layer_sizes' : [randint.rvs(40,80),],
    'learning_rate_init' : uniform(0.0001, 0.001),
    'validation_fraction' : uniform(0.01, 0.01)
}

# Initialize the MLPClassifier with fixed parameters, except alpha
mlp_clf = MLPClassifier(
    batch_size='auto',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    verbose=True,
    tol=1e-4,
    n_iter_no_change=3
)

# Set up RandomizedSearchCV to tune the alpha parameter
random_search = RandomizedSearchCV(
    estimator=mlp_clf,
    param_distributions=param_dist,
    n_iter=20,  # Try 20 different alpha values
    cv=2,  # 3-fold cross-validation
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train.values.ravel())

# Get the best estimator
best_mlp_clf = random_search.best_estimator_
print('Best parameters found:\n', random_search.best_params_)

# Save the best model
model_save_path = current_dir / 'best_trained_mlp_model_alpha.joblib'
joblib.dump(best_mlp_clf, model_save_path)
print(f"Best trained model saved to: {model_save_path}")

# Load validation data
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')

# Make predictions on validation data
y_val_pred = best_mlp_clf.predict(X_val)

# Evaluate the model
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Print classification report
print("Classification Report on Validation Data:")
print(classification_report(y_val, y_val_pred))

