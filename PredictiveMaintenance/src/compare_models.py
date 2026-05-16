import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score


# Load dataset
df = pd.read_csv("data/SENSOR IOTT.csv")

# Drop non-numeric / identifier column
df = df.drop("timestamp", axis=1)

# Features and target
X = df.drop("machine_failure", axis=1)
y = df["machine_failure"]

# Train-test split - same split used in train.py/tune.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load saved pipelines
baseline_pipeline = joblib.load("models/best_pipeline.joblib")
tuned_rf_pipeline = joblib.load("models/tuned_random_forest_pipeline.joblib")

models = {
    "Baseline Best Pipeline": baseline_pipeline,
    "Tuned Random Forest Pipeline": tuned_rf_pipeline
}

for name, pipeline in models.items():

    print(f"\n===== {name} =====")

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))