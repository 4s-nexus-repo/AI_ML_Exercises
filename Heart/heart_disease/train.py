import os
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler,
    RobustScaler, MaxAbsScaler, Normalizer
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from config import (
    DATA_PATH, OUTPUTS_DIR, MODELS_DIR,
    CONTINUOUS_FEATURES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE
)
from evaluate import evaluate_model, save_comparison_chart


def load_and_prepare():
    """Load dataset, remove duplicates, split into train and test."""
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    print(f"Dataset loaded  : {df.shape[0]} rows x {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples    : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def apply_scaling(X_train, X_test):
    """Apply all 5 scaling techniques. Fit on train only, transform both."""
    scalers = {
        '1. Min-Max Scaler':     MinMaxScaler(),
        '2. Z-score (Standard)': StandardScaler(),
        '3. Robust Scaler':      RobustScaler(),
        '4. Max Abs Scaler':     MaxAbsScaler(),
        '5. Unit Vector (L2)':   Normalizer(norm='l2'),
    }

    scaled = {}

    for name, scaler in scalers.items():
        X_tr = X_train.copy()
        X_te = X_test.copy()

        if name == '5. Unit Vector (L2)':
            # Normalizer is row-wise — apply to ALL features
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns, index=X_train.index
            )
            X_te = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns, index=X_test.index
            )
        else:
            # Others — scale continuous columns only
            X_tr[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
            X_te[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])

        scaled[name] = (X_tr, X_te)

    print("All 5 scalers applied ✔")
    return scaled


def train_and_save(X_train, X_test, y_train, y_test, scaled):
    """Train all 6 models, evaluate each, save to disk."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Z-score scaled data for linear models
    X_train_scaled, X_test_scaled = scaled['2. Z-score (Standard)']

    results = []

    # ── Linear models — need scaling ──────────────────────────
    linear_models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000),
        'LogReg L2 Ridge': LogisticRegression(
            penalty='l2', solver='lbfgs',
            max_iter=2000, random_state=RANDOM_STATE),
        'LogReg L1 Lasso': LogisticRegression(
            penalty='l1', solver='liblinear',
            max_iter=2000, random_state=RANDOM_STATE),
    }

    print("\n--- Training Linear Models (uses X_train_scaled) ---")
    for name, model in linear_models.items():
        model.fit(X_train_scaled, y_train)
        results.append(evaluate_model(model, X_test_scaled, y_test, name))
        save_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, save_path)
        print(f"Saved model: {save_path}")

    # ── Tree-based models — no scaling needed ─────────────────
    tree_models = {
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=RANDOM_STATE),
    }

    print("\n--- Training Tree Models (uses raw X_train) ---")
    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(model, X_test, y_test, name))
        save_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.joblib")
        joblib.dump(model, save_path)
        print(f"Saved model: {save_path}")

    # Save the fitted StandardScaler so predict.py can use it
    scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.joblib')
    joblib.dump(scaled['2. Z-score (Standard)'], scaler_path)
    print(f"Saved scaler: {scaler_path}")

    # Save comparison chart
    save_comparison_chart(results)


def main():
    print("========================================")
    print("  Heart Disease Prediction — Training")
    print("========================================")
    X_train, X_test, y_train, y_test = load_and_prepare()
    scaled = apply_scaling(X_train, X_test)
    train_and_save(X_train, X_test, y_train, y_test, scaled)
    print("\nTraining complete! Models saved to outputs/models/")


if __name__ == '__main__':
    main()