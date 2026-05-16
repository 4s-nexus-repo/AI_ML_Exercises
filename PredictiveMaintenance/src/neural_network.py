import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(y_train_smote.value_counts())

# Neural network pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42
    ))
])

# Train neural network
pipeline.fit(X_train_smote, y_train_smote)

# Predict
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nNeural Network Results")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "models/neural_network_pipeline.joblib")

print("\nNeural network pipeline saved as: models/neural_network_pipeline.joblib")