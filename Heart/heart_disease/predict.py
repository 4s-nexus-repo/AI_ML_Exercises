import joblib
import pandas as pd
import os

from config import MODELS_DIR

# Models that were trained on scaled data
LINEAR_MODELS = [
    'Logistic_Regression',
    'LogReg_L2_Ridge',
    'LogReg_L1_Lasso'
]

# Models that were trained on raw data
TREE_MODELS = [
    'Decision_Tree',
    'Random_Forest',
    'Gradient_Boosting'
]


def predict(patient_data: dict, model_name: str = 'Logistic_Regression'):
    """
    Load a saved model and predict for a new patient.

    Parameters
    ----------
    patient_data : dict  — raw feature values, no pre-scaling needed
    model_name   : str   — name of the saved .joblib file without extension

    Returns
    -------
    prediction  : int   — 0 = No Disease, 1 = Heart Disease
    probability : array — [prob_no_disease, prob_disease]
    """
    # Load model
    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Run train.py first to generate model files.")
        return None, None

    model = joblib.load(model_path)

    # Build input DataFrame from raw patient values
    patient_df = pd.DataFrame([patient_data])

    # Apply scaling for linear models
    if model_name in LINEAR_MODELS:
        scaler_path = os.path.join(MODELS_DIR, 'standard_scaler.joblib')
        if not os.path.exists(scaler_path):
            print("Scaler file not found. Run train.py first.")
            return None, None
        X_train_scaled, _ = joblib.load(scaler_path)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_scaled)
        patient_df = pd.DataFrame(
            scaler.transform(patient_df),
            columns=patient_df.columns
        )

    # Predict
    prediction  = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0]

    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    print(f"\nModel      : {model_name}")
    print(f"Prediction : {result}")
    print(f"Confidence : {probability[0]*100:.1f}% No Disease | {probability[1]*100:.1f}% Disease")

    return prediction, probability


def predict_all_models(patient_data: dict):
    """Run the same patient through all 6 saved models."""
    print("\n========================================")
    print("  Predictions from all 6 models")
    print("========================================")

    all_models = LINEAR_MODELS + TREE_MODELS
    for model_name in all_models:
        predict(patient_data, model_name=model_name)


def main():
    print("========================================")
    print("  Heart Disease Prediction — Predict")
    print("========================================")

    # Example new patient — raw values, no scaling needed
    new_patient = {
        'age':      55,
        'sex':       1,
        'cp':        2,
        'trestbps': 140,
        'chol':     250,
        'fbs':        0,
        'restecg':    1,
        'thalach':  160,
        'exang':      0,
        'oldpeak':  1.5,
        'slope':      2,
        'ca':         0,
        'thal':       2
    }

    print("\nPatient data:")
    for key, val in new_patient.items():
        print(f"  {key:10}: {val}")

    # Predict using best model (Logistic Regression)
    print("\n--- Best model prediction ---")
    predict(new_patient, model_name='Logistic_Regression')

    # Predict using all models
    predict_all_models(new_patient)


if __name__ == '__main__':
    main()