import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#ensure the project works for both windows and linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / "X_train.csv")
y_train = pd.read_csv(current_dir.parent / "y_train.csv")
X_val = pd.read_csv(current_dir.parent / "X_val.csv")
y_val = pd.read_csv(current_dir.parent / "y_val.csv")

# define a grid of hyperparameters to search over
param_grid = {
    "max_iter": [1000, 2000, 3000],
    "alpha": [0.0001, 0.0005, 0.001],
    "power_t": [0.5, 0.3, 0.7]
}

#inifialize the SGDClassifier
svm_clf = SGDClassifier(class_weight="balanced",random_state=42)

#use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid,
                           cv=3, scoring="accuracy", verbose=2)

#train the model using GridSearchCV on the training data
grid_search.fit(X_train, y_train.values.ravel())

#get the best model and its hyperparameters
best_svm_clf = grid_search.best_estimator_
print(f"Best Hyperpatameters: {grid_search.best_params_}")

#evaluate the best model on the validation data
y_val_pred = best_svm_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with Best Model: {val_accuracy:.2f}")

#generate a classicfication report and confusion matrix for the validation data
print("Best Model Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Best Model Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

#save the best model to a file
model_save_path = current_dir / "best_svm_model.joblib"
joblib.dump(best_svm_clf, model_save_path)
print(f"Best model saved to: {model_save_path}")
