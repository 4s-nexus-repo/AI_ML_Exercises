import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


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

# Load trained Random Forest pipeline
pipeline = joblib.load("models/smote_random_forest_pipeline.joblib")

# Extract actual trained model from pipeline
model = pipeline.named_steps["model"]

# Scale test data using pipeline scaler
X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns
)