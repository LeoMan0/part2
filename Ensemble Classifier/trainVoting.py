import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path

#ensure the project works for both windows and linux
current_dir = Path(__file__).resolve().parent

X_train = pd.read_csv(current_dir.parent / "X_train.csv")
y_train = pd.read_csv(current_dir.parent / "y_train.csv")

#initialize all classifiers
rf_clf = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight="balanced", random_state=42)
bg_clf = BaggingClassifier(random_state=42)
hgb_clf = HistGradientBoostingClassifier(random_state=42)
vot_clf = VotingClassifier(estimators=[('rf', rf_clf),('bg', bg_clf),('hgb', hgb_clf)])


#fit the models
rf_clf.fit(X_train, y_train.values.ravel())
bg_clf.fit(X_train, y_train.values.ravel())
hgb_clf.fit(X_train, y_train.values.ravel())
vot_clf.fit(X_train, y_train.values.ravel())

#save the trained model
model_save_path = current_dir / "trained_vot_model.joblib"
joblib.dump(vot_clf, model_save_path)
print(f"Trained model saved to: {model_save_path}")
