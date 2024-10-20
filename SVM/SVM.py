import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


# load data
current_dir = Path(__file__).resolve().parent
X_train = pd.read_csv(current_dir.parent / "X_train.csv")
y_train = pd.read_csv(current_dir.parent / "y_train.csv")

# Define the SVMClassifire
svm_clf = SGDClassifier()

# fit the model
svm_clf.fit(X_train, y_train.values.ravel())

#save the trained model and scaler
model_save_path = current_dir / "ltrained_svm_model.joblib"
joblib.dump(svm_clf, model_save_path)
print(f"Trained model saved to: {model_save_path}")
