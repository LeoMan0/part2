import pandas as pd
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
# Load data
current_dir = Path(__file__).resolve().parent
X_train = pd.read_csv(current_dir.parent / 'X_train.csv')
y_train = pd.read_csv(current_dir.parent / 'y_train.csv')

#Best parameters found:
 #{'activation': 'tanh', 'alpha': np.float64(0.0004131329245555858), 'batch_size': 128, 'hidden_layer_sizes': (50, 67), 'learning_rate': 'adaptive', 'learning_rate_init': np.float64(0.009494989415641891), 'n_iter_no_change': 8, 'solver': 'sgd', 'tol': 0.0001}
# Define the MLPClassifier with adjusted parameters to reduce overfitting
#{'alpha': np.float64(0.0003834619912671628), 'hidden_layer_sizes': (17,), 'learning_rate_init': np.float64(0.00029664617716135765), 'max_iter': 300, 'n_iter_no_change': 18, 'solver': 'adam'}
 #{'alpha': np.float64(0.061763482887407344), 'hidden_layer_sizes': 51}
# {'alpha': np.float64(0.35388521115218396), 'hidden_layer_sizes': 47, 'learning_rate_init': np.float64(0.0010093204020787821), 'validation_fraction': np.float64(0.01258779981600017)}
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(47,),  # Simplified architecture
    alpha=0.35,                # Increased regularization strength
    batch_size='auto',
    learning_rate_init = 0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.013,
    verbose=True,
    tol=1e-4,            
    n_iter_no_change=30
)

# Fit the model
mlp_clf.fit(X_train, y_train.values.ravel())

# Save the trained model and scaler
model_save_path = current_dir / 'trained_mlp_model.joblib'
joblib.dump(mlp_clf, model_save_path)
print(f"Trained model saved to: {model_save_path}")


# Evaluate on validation data
X_val = pd.read_csv(current_dir.parent / 'X_val.csv')
y_val = pd.read_csv(current_dir.parent / 'y_val.csv')


# Make predictions on validation data
y_val_pred = mlp_clf.predict(X_val)

# Evaluate the model
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Print classification report
print("Classification Report on Validation Data:")
print(classification_report(y_val, y_val_pred))
train_loss = mlp_clf.loss_curve_


y_val_pred_proba = mlp_clf.predict_proba(X_val)
val_loss = log_loss(y_val, y_val_pred_proba)

#plt.plot(train_loss, label='train loss')
#plt.axhline(y=val_loss, color='r', linestyle='--', label='val loss')
#plt.xlabel('n')
#plt.ylabel('loss')
#plt.legend()
#plt.show()
