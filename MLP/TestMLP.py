import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

# Load the trained model
current_dir = Path(__file__).resolve().parent
X_test_path = current_dir.parent / 'X_test.csv'
y_test_path = current_dir.parent / 'y_test.csv'

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# Load the trained MLP model
mlp_clf = joblib.load('1trained_mlp_model.joblib')
# Use the model to make predictions on the test set
y_pred = mlp_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

