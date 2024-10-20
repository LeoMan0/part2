import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path


current_dir = Path(__file__).resolve().parent

X_test = pd.read_csv(current_dir.parent / "X_test.csv")
y_test = pd.read_csv(current_dir.parent / "y_test.csv")

#load the trained model
clf_loaded = joblib.load("trained_vot_model.joblib")

#use model to make predictions on the test set
y_pred = clf_loaded.predict(X_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

#print the metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
