import os
import joblib
import numpy as np
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
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    print(f"Dataset loaded  : {df.shape[0]} rows x {df.shape[1]} columns")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples    : {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def apply_scaling(X_train, X_test):
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
            X_tr = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns, index=X_train.index
            )
            X_te = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns, index=X_test.index
            )
        else:
            X_tr[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
            X_te[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])  # transform, NOT fit_transform

        scaled[name] = (X_tr, X_te)

    print("All 5 scalers applied")
    return scaled


def train_and_save(X_train, X_test, y_train, y_test, scaled):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train_scaled, X_test_scaled = scaled['2. Z-score (Standard)']

    results = []

    # ── Linear models ─────────────────────────────────────────
    linear_models = {
        'Logistic_Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'LogReg_L2_Ridge':     LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE),
        'LogReg_L1_Lasso':     LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=RANDOM_STATE),
    }
    print("\n--- Training Linear Models ---")
    for name, model in linear_models.items():
        model.fit(X_train_scaled, y_train)
        results.append(evaluate_model(model, X_test_scaled, y_test, name))
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.joblib'))
        print(f"Saved: {name}.joblib")

    # ── Tree models ───────────────────────────────────────────
    tree_models = {
        'Decision_Tree':    DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random_Forest':    RandomForestClassifier(random_state=RANDOM_STATE),
        'Gradient_Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    print("\n--- Training Tree Models ---")
    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(model, X_test, y_test, name))
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.joblib'))
        print(f"Saved: {name}.joblib")

    # ── Save scaler fitted on NUMPY ARRAYS (not DataFrame) ────
    # Key fix: fit a fresh scaler on .values (numpy) so it has
    # no feature_names_in_ — predict.py can then call transform()
    # without pandas routing causing unhashable type errors.
    print("\n--- Saving scaler (numpy-fitted) ---")
    numpy_scaler = StandardScaler()
    cont_idx     = [list(X_train.columns).index(c) for c in CONTINUOUS_FEATURES]
    cont_train_np = X_train.values[:, cont_idx].astype(np.float64)  # pure numpy
    numpy_scaler.fit(cont_train_np)                                  # fit on numpy, no DataFrame

    joblib.dump(numpy_scaler,        os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(CONTINUOUS_FEATURES, os.path.join(MODELS_DIR, 'scaler_columns.joblib'))
    joblib.dump(list(X_train.columns), os.path.join(MODELS_DIR, 'train_columns.joblib'))
    print("Saved: scaler.joblib (numpy-fitted — no pandas dependency)")
    print("Saved: scaler_columns.joblib")
    print("Saved: train_columns.joblib")

    save_comparison_chart(results)


def main():
    print("=" * 45)
    print("  Heart Disease Prediction — Training")
    print("=" * 45)
    X_train, X_test, y_train, y_test = load_and_prepare()
    scaled = apply_scaling(X_train, X_test)
    train_and_save(X_train, X_test, y_train, y_test, scaled)
    print("\nTraining complete!")


if __name__ == '__main__':
    main()