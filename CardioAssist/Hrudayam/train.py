import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "heart.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "model_name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Remove duplicates
    df = df.drop_duplicates()

    # 3. Split features and target
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    X = df[feature_cols]
    y = df["target"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5. Different scalers
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    best_score = 0
    best_pipeline = None
    best_name = None

    # 6. Try each scaler with each model
    for scaler_name, scaler in scalers.items():
        for model_name, model in models.items():

            pipeline = Pipeline([
                ("scaler", scaler),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(
                f"{model_name} + {scaler_name}",
                pipeline,
                X_test,
                y_test
            )

            results.append(metrics)

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_pipeline = pipeline
                best_name = f"{model_name} + {scaler_name}"

    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.sort_values(by="f1", ascending=False))

    # 7. Hyperparameter tuning for Random Forest
    print("\nRunning hyperparameter tuning...")

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    tuned_model = grid_search.best_estimator_
    tuned_metrics = evaluate_model(
        "Tuned RandomForest",
        tuned_model,
        X_test,
        y_test
    )

    print("\nBest Random Forest Parameters:")
    print(grid_search.best_params_)

    print("\nTuned Random Forest Results:")
    print(tuned_metrics)

    if tuned_metrics["f1"] > best_score:
        best_pipeline = tuned_model
        best_name = "Tuned RandomForest"
        best_score = tuned_metrics["f1"]

    # 8. Save best model
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.joblib"))

    print(f"\nBest model saved: {best_name}")
    print(f"Best F1 score: {best_score:.4f}")

    # 9. Feature importance from Random Forest / Gradient Boosting / Decision Tree
    final_model = best_pipeline.named_steps["model"]

    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_

        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        print("\nFeature Importance:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
        plt.show()
    else:
        print("\nBest model does not support feature_importances_ directly.")


if __name__ == "__main__":
    main()