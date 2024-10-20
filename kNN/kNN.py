from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import joblib

# To ensure that the project works for both Windows and Linux
current_dir = Path(__file__).resolve().parent

# Build paths for the data files in the parent directory
X_train_path = current_dir.parent / 'X_train.csv'
y_train_path = current_dir.parent / 'y_train.csv'

# Load the training data
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)


# Initialize the kNN model
k = 250  # Choose the number of neighbors, can tune this parameter
knn_clf = KNeighborsClassifier(n_neighbors=250)

knn_clf.fit(X_train, y_train.values.ravel())

# Save the trained mode
joblib.dump(knn_clf, 'trained_knn_model.joblib')


