import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


DATA_PATH = "library_lost_item_training_data.csv"

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset: {len(df)} samples")


df["item_types"] = df["item_types"].apply(
    lambda s: ",".join(sorted(s.split(",")))
)

categorical = ["item_types", "weekday"]
numeric = [
    "time_since_person",
    "time_of_day",
    "current_sit_minutes",
    "total_session_minutes",
    "num_previous_returns",
    "seat_now_occupied",
    "new_person_present",
]

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("[INFO] Train/Test Split:")
print(" Train:", len(X_train))
print(" Test :", len(X_test))

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("pre", preprocessor),
    ("rf", rf)
])

print("\n[INFO] Training Random Forest...")
model.fit(X_train, y_train)
print("[INFO] Training complete!")

print("\n[INFO] Evaluating model...")

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\n===== ACCURACY =====")
print(acc)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, pred))

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, pred)
print(cm)

encoder = model.named_steps["pre"].named_transformers_["cat"]
encoded_cat_features = encoder.get_feature_names_out(categorical)

feature_names = list(encoded_cat_features) + numeric
importances = model.named_steps["rf"].feature_i
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df.to_csv("feature_importances.csv", index=False)
print("\n[INFO] Saved feature importances → feature_importances.csv")

top = importance_df.head(20)
plt.figure(figsize=(10, 6))
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("[INFO] Saved feature importances plot → feature_importances.png")

MODEL_PATH = "lost_item_rf_model.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\n[INFO] Model saved → {MODEL_PATH}")

print("\n==============================")
print(" Training Completed Successfully")
print("==============================")
