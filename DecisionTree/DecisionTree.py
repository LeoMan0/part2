from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import joblib


# To ensure that the project works for both Windows and linux
current_dir = Path(__file__).resolve().parent

# Build paths for the data files in the parent directory
X_train_path = current_dir.parent / 'X_train.csv'
y_train_path = current_dir.parent / 'y_train.csv'

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

# Train the model
clf = DecisionTreeClassifier(max_depth = 10)
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'trained_model.joblib')


