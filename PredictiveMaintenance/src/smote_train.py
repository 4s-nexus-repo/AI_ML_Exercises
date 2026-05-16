import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score

from imblearn.over_sampling import SMOTE


# Load dataset
df = pd.read_csv("data/SENSOR IOTT.csv")

# Drop identifier column
df = df.drop("timestamp", axis=1)

# Features and target
X = df.drop("machine_failure", axis=1)
y = df["machine_failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Before SMOTE:")
print(y_train.value_counts())

# Apply SMOTE ONLY on training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train on SMOTE-balanced data
pipeline.fit(X_train_smote, y_train_smote)

# Predict on ORIGINAL untouched test data
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nSMOTE Random Forest Results")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/smote_random_forest_pipeline.joblib")

print("\nSMOTE pipeline saved as: models/smote_random_forest_pipeline.joblib")