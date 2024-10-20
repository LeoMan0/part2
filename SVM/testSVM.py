import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

#load trained model
current_dir = Path(__file__).resolve().parent
X_test_path = current_dir.parent / "X_train.csv"
y_test_path = current_dir.parent / "y_train.csv"

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

#load the trained svm model
svm_clf = joblib.load("ltrained_svm_model.joblib")

#use the model to make a prediction on the test set
y_pred = svm_clf.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# print evaluation metrics
print(f"Accuracy: {accuracy * 100:2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
