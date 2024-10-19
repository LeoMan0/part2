from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your preprocessed data
data = pd.read_csv('preprocessed_data.csv')
data = data.reset_index(drop=True)

# Split the data into features and labels
X = data.iloc[:, 1:]  # Features
y = data.iloc[:, 0]   # Labels (Activity ID)
seed = 40

# Split into training and (testing + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
#y
#Split temp into test and validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)


X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)
#X_val.to_csv('X_val.csv', index = False)

y_train.to_csv('y_train.csv', index = False)
y_test.to_csv('y_test.csv', index = False)
#y_val.to_csv('y_val.csv', index = False)


