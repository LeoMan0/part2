import pandas as pd

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('preprocessed_data.csv')

# Add a sequential "time" column to simulate time-based order
data['time'] = range(1, len(data) + 1)

X = data.iloc[:, 1:-1]  # Features (excluding the label and the new time column)
y = data.iloc[:, 0]     # Labels (Activity ID)
test_size = 0.3  # 30% for testing, 70% for training

# Create empty DataFrames to hold the split data
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.Series(dtype='int')
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

    # Perform the time-based split (train on earlier data, test on later data)
    split_index = int(len(X_label_sorted) * (1 - test_size))
    
    X_label_train = X_label_sorted.iloc[:split_index].drop(columns=['time'])
    X_label_test = X_label_sorted.iloc[split_index:].drop(columns=['time'])
    y_label_train = y_label_sorted.iloc[:split_index]
    y_label_test = y_label_sorted.iloc[split_index:]

    # Append the split data to the respective train and test sets
    X_train = pd.concat([X_train, X_label_train])
    X_test = pd.concat([X_test, X_label_test])
    y_train = pd.concat([y_train, y_label_train])
    y_test = pd.concat([y_test, y_label_test])

# Check that both train and test contain all labels
unique_labels_train = y_train.unique()
unique_labels_test = y_test.unique()

print("Final Labels in y_train: ", unique_labels_train)
print("Final Labels in y_test: ", unique_labels_test)


# Check the number of rows for each label in the train set
label_counts_train = y_train.value_counts()
print("Label counts in the training set:")
print(label_counts_train)

# Check the number of rows for each label in the test set
label_counts_test = y_test.value_counts()
print("\nLabel counts in the test set:")
print(label_counts_test)

# Optionally, you can create a summary table comparing both
label_distribution = pd.DataFrame({
    'Train': label_counts_train,
    'Test': label_counts_test
}).fillna(0)  # Fill NaN with 0 for labels not present in either set
# Calculate the total count (Train + Test) for each label
label_distribution['Total'] = label_distribution['Train'] + label_distribution['Test']

# Calculate the percentage of rows in the Test set
label_distribution['Test (%)'] = (label_distribution['Test'] / label_distribution['Total']) * 100

# Round the percentages for readability
label_distribution['Test (%)'] = label_distribution['Test (%)'].round(2)

# Print the final table
print("\nLabel distribution across train and test sets with percentages:")
print(label_distribution[['Train', 'Test', 'Test (%)']])
# Save the train and test sets
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

