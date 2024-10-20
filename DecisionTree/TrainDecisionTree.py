from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
# To ensure that the project works for both Windows and linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')

#Best Hyperparameters: {'criterion': 'gini', 'max_depth': 15, 'min_samples_leaf': 4, 'min_samples_split': 10}
# Define the hyperparameters grid to search over
#param_grid = {
#    'max_depth': [5, 10, 15, 20],               # Try different depths
#    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
#    'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required to be at a leaf node
#    'criterion': ['gini', 'entropy']            # Use Gini or Entropy as the split criterion
#}
#Best Hyperparameters: {'criterion': 'gini', 'max_depth': 12, 'min_samples_leaf': 3, 'min_samples_split': 9}

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 10, 20, 50],
    'max_features': [None, 'sqrt', 'log2', 0.5],
    'class_weight': [None, 'balanced'],
    'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_leaf_nodes': [None, 10, 20, 50, 100],
    'min_impurity_decrease': [0.0, 0.0001, 0.001, 0.01],
}

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV to tune hyperparameters based on validation accuracy
#grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_grid,
    n_iter=100,  # Number of parameter settings that are sampled
    cv=3,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
# Fit the model using grid search
#grid_search.fit(X_train, y_train.values.ravel())
random_search.fit(X_train, y_train.values.ravel())
# Get the best estimator from grid search
#best_clf = grid_search.best_estimator_
best_clf = random_search.best_estimator_
# Check the best hyperparameters
#print("Best Hyperparameters:", grid_search.best_params_)
print("Best Hyperparameters:", random_search.best_params_)
# Evaluate the tuned model on validation data
y_val_pred = best_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the best model
joblib.dump(best_clf, 'best_trained_model.joblib')

