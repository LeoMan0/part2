import pandas as pd
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load data
current_dir = Path(__file__).resolve().parent
X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')


# Define the MLPClassifier with adjusted parameters to reduce overfitting
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(20,),  # Simplified architecture
    alpha=4,                # Increased regularization strength
    batch_size='auto',
    learning_rate_init=0.0001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

# Fit the model
mlp_clf.fit(X_train, y_train.values.ravel())

# Save the trained model and scaler
model_save_path = current_dir / '1trained_mlp_model.joblib'
joblib.dump(mlp_clf, model_save_path)
print(f"Trained model saved to: {model_save_path}")


# Evaluate on validation data
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')


# Make predictions on validation data
y_val_pred = mlp_clf.predict(X_val)

# Evaluate the model
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Print classification report
print("Classification Report on Validation Data:")
print(classification_report(y_val, y_val_pred))

