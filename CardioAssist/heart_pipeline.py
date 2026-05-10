import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, "heart_pipeline.joblib")

print("Model saved!")

# Test prediction
sample = pd.DataFrame({
    'age': [52],
    'sex': [1],
    'cp': [0],
    'trestbps': [125],
    'chol': [212],
    'fbs': [0],
    'restecg': [1],
    'thalach': [168],
    'exang': [0],
    'oldpeak': [1],
    'slope': [2],
    'ca': [2],
    'thal': [3]
})

pred = pipeline.predict(sample)
print("Prediction:", "Disease" if pred[0] == 1 else "No Disease")