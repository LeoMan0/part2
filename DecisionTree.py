from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your preprocessed data
data = pd.read_csv("preprocessed_data.csv")

# Split the data into features and labels
X = data.iloc[:, 1:]  # Features
y = data.iloc[:, 0]   # Labels

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'trained_model.joblib')


