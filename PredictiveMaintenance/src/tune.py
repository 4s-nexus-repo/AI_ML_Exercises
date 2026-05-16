import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("data/SENSOR IOTT.csv")

# Drop non-numeric / identifier column
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

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="recall",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest CV Recall:")
print(grid_search.best_score_)

# Evaluate best model on test data
best_pipeline = grid_search.best_estimator_

y_pred = best_pipeline.predict(X_test)

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred))

# Save tuned pipeline
joblib.dump(best_pipeline, "models/tuned_random_forest_pipeline.joblib")

print("\nTuned Random Forest pipeline saved as: models/tuned_random_forest_pipeline.joblib")