import pandas as pd
from sklearn.utils import shuffle

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('preprocessed_data.csv')

# Add a sequential "time" column to simulate time-based order
data['time'] = range(1, len(data) + 1)

X = data.iloc[:, 1:-1]  # Features (excluding the label and the new time column)
y = data.iloc[:, 0]     # Labels (Activity ID)
train_size = 0.7  # 70% for training
val_test_size = 0.15  # 15% for validation and 15% for test

# Create empty DataFrames to hold the split data
X_train = pd.DataFrame()
X_val = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.Series(dtype='int')
y_val = pd.Series(dtype='int')
y_test = pd.Series(dtype='int')

# Loop through each unique label and perform time-based split
for label in y.unique():
    # Extract rows corresponding to the current label
    label_indices = y[y == label].index
    X_label = X.loc[label_indices]
    y_label = y.loc[label_indices]

    # Sort the data for this label by the time column
    X_label['time'] = range(1, len(X_label) + 1)  # Add sequential time to each label
    X_label_sorted = X_label.sort_values(by='time')
    y_label_sorted = y_label.loc[X_label_sorted.index]

    # First, perform the train (70%) and temporary (30%) split
    train_split_index = int(len(X_label_sorted) * train_size)
    
    X_label_train = X_label_sorted.iloc[:train_split_index].drop(columns=['time'])
    X_label_temp = X_label_sorted.iloc[train_split_index:].drop(columns=['time'])
    y_label_train = y_label_sorted.iloc[:train_split_index]
    y_label_temp = y_label_sorted.iloc[train_split_index:]

    # Now split the temporary set (30%) into validation (15%) and test (15%)
    val_split_index = int(len(X_label_temp) * 0.5)  # Half of the temp set for validation, half for test

    X_label_val = X_label_temp.iloc[:val_split_index]
    X_label_test = X_label_temp.iloc[val_split_index:]
    y_label_val = y_label_temp.iloc[:val_split_index]
    y_label_test = y_label_temp.iloc[val_split_index:]

    # Append the split data to the respective train, validation, and test sets
    X_train = pd.concat([X_train, X_label_train])
    X_val = pd.concat([X_val, X_label_val])
    X_test = pd.concat([X_test, X_label_test])
    y_train = pd.concat([y_train, y_label_train])
    y_val = pd.concat([y_val, y_label_val])
    y_test = pd.concat([y_test, y_label_test])

# Shuffle the training, validation, and test sets after splitting
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_val, y_val = shuffle(X_val, y_val, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print("Training label distribution:")
print(y_train.value_counts())

print("Test label distribution:")
print(y_test.value_counts())
# Save the train, validation, and test sets
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

