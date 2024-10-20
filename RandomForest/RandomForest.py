from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import pandas as pd
import joblib

# To ensure the project works for both Windows and Linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')

#{'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}

rf_clf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=15, 
    min_samples_leaf=1,
    min_samples_split=2,
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)
# Train the model on the training data
rf_clf.fit(X_train, y_train.values.ravel())

# Evaluate the model on the validation data
y_val_pred = rf_clf.predict(X_train)
val_accuracy = accuracy_score(y_train, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Generate a classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_train, y_val_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_train, y_val_pred))

model_save_path = current_dir / 'rf_model.joblib'
joblib.dump(rf_clf, model_save_path)
