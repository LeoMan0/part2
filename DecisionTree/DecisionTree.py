from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pandas as pd
import joblib

# To ensure that the project works for both Windows and linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')

# Train the model
clf = DecisionTreeClassifier('max_depth'= 13, 'min_samples_leaf' = 3, 'min_samples_split' = 9)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'trained_model.joblib')


