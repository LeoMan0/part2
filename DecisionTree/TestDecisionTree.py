import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path


current_dir = Path(__file__).resolve().parent

# Build paths for the data files in the parent directory
#X_train_path = current_dir.parent / 'X_train.csv'
#y_train_path = current_dir.parent / 'y_train.csv'

X_test = pd.read_csv(current_dir.parent / 'X_test.csv')
y_test = pd.read_csv(current_dir.parent / 'y_test.csv')

#X_test = pd.read_csv(current_dir.parent / 'X_val.csv')
#y_test = pd.read_csv(current_dir.parent / 'y_val.csv')

clf_loaded = joblib.load('best_trained_model.joblib')

# Use the loaded model to make predictions on the test set
y_pred = clf_loaded.predict(X_test)

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




