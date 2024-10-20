import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path


current_dir = Path(__file__).resolve().parent


X_test = pd.read_csv(current_dir.parent / 'X_test.csv')
y_test = pd.read_csv(current_dir.parent / 'y_test.csv')

#X_test = pd.read_csv(current_dir.parent / 'X_val.csv')
#y_test = pd.read_csv(current_dir.parent / 'y_val.csv')

#X_test = pd.read_csv(current_dir.parent / 'X_train.csv')
#y_test = pd.read_csv(current_dir.parent / 'y_train.csv')
#print(X_test.head())
# Load the trained model
clf_loaded = joblib.load('rf_model.joblib')

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

import matplotlib.pyplot as plt
import seaborn as sns
class_report = classification_report(y_test, y_pred,output_dict=True)

# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
#print(conf_matrix)
#print('Classification Report:')
#print(class_report)
# Create classification report DataFrame
df_report = pd.DataFrame(class_report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])

# Plot the classification report as a table using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))  # Set the size of the table
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_report.round(2).values,  # Rounded to 2 decimal places
                 colLabels=df_report.columns,
                 rowLabels=df_report.index,
                 loc='center',
                 cellLoc='center',
                 colColours=['#f2f2f2']*len(df_report.columns))  # Set column background color

# Adjust the font size for the table
table.scale(1, 1.5)  # Scale the table for more vertical space in cells
table.auto_set_font_size(False)
table.set_fontsize(8)

# Adjust the column width to fit the content
table.auto_set_column_width(col=list(range(len(df_report.columns))))
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
# Show the table
plt.show()



# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24],  # Class names for x-axis
            yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])  # Class names for y-axis
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


