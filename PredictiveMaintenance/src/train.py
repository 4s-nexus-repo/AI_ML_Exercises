import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from evaluation import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)
# Load dataset
df = pd.read_csv("C:\\Users\\karth\\Desktop\\AI\\ML_Project\\AI_ML_Exercises\\PredictiveMaintenance\\data\\SENSOR IOTT.csv")


print("Dataset loaded successfully")
print(df.head())
print(df.shape)

# Drop non-numeric / identifier columns
df = df.drop("timestamp", axis=1)
# Features and target
X = df.drop("machine_failure", axis=1)
print(X.columns)
y = df["machine_failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-test split complete")
print(X_train.shape, X_test.shape)
# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling complete")
print(X_train_scaled.shape)
# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}
# Train and evaluate models

results = {}
pipelines = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Accuracy
    accuracy = evaluate_model(y_test, y_pred, name)

    # Store result
    results[name] = accuracy
    pipelines[name] = pipeline

    
    #best model by accuracy
    print("\nModel Accuracy Summary:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

best_model = max(results, key=results.get)
best_pipeline = pipelines[best_model]
print(f"\nBest Model: {best_model} with accuracy {results[best_model]:.4f}")

# Save best model
# Save best pipeline
joblib.dump(best_pipeline, "models/best_pipeline.joblib")

print("\nBest pipeline saved as: models/best_pipeline.joblib")

