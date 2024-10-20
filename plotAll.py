import matplotlib.pyplot as plt

# Array of different model names and their corresponding accuracies
model_names = ['Decision Tree', 'Random Forest','kNN', 'MLP', 'SVM']
accuracies = [0.37, 0.64, 0.60, 0.57, 0.51]  # Example accuracy values for each model

# Create a bar plot
plt.figure(figsize=(10, 5))
plt.bar(model_names, accuracies, color='skyblue')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracies Comparison')

# Display accuracy values on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', fontsize=12)

# Add gridlines for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

