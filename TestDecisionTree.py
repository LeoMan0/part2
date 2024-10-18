import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your preprocessed data (same as in the training file)
data = pd.read_csv('preprocessed_data.csv')

# Split the data into features and labels
X = data.iloc[:, 1:]  # Features (same as training script)
y = data.iloc[:, 0]   # Labels (same as training script)

# Split the data again, ensuring the same split with the same random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
clf_loaded = joblib.load('trained_model.joblib')

# Use the loaded model to make predictions on the test set
y_pred = clf_loaded.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

corr_matrix = pd.concat([X, y], axis=1).corr()
print(corr_matrix.iloc[0, 1:])  # Print correlation of the label with all features

