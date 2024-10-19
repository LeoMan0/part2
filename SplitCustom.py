import pandas as pd
from sklearn.model_selection import train_test_split

# Load your preprocessed data
data = pd.read_csv('preprocessed_data.csv')
data = data.reset_index(drop=True)

# Initialize empty dataframes for storing the overall train and test sets
X_train_all = pd.DataFrame()
X_test_all = pd.DataFrame()
y_train_all = pd.DataFrame()
y_test_all = pd.DataFrame()

# Function to split each label-specific dataframe into train and test sets
def split_and_append(group):
    X = group.iloc[:, 1:]  # Features (everything except the label column)
    y = group.iloc[:, [0]]  # Labels (first column as dataframe for concat later)
    # Split 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

# Group the data by the label column (first column)
for label, group in data.groupby(data.columns[0]):
    X_train, X_test, y_train, y_test = split_and_append(group)
    
    # Append the chunks to the overall train/test sets
    X_train_all = pd.concat([X_train_all, X_train], ignore_index=True)
    X_test_all = pd.concat([X_test_all, X_test], ignore_index=True)
    y_train_all = pd.concat([y_train_all, y_train], ignore_index=True)
    y_test_all = pd.concat([y_test_all, y_test], ignore_index=True)

# Save the final train and test sets to CSV
X_train_all.to_csv('X_train.csv', index=False)
X_test_all.to_csv('X_test.csv', index=False)
y_train_all.to_csv('y_train.csv', index=False)
y_test_all.to_csv('y_test.csv', index=False)

print("Data split completed and saved to CSV files.")
