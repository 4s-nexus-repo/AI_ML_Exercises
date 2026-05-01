import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR

LINEAR_MODELS = ['Logistic_Regression', 'LogReg_L2_Ridge', 'LogReg_L1_Lasso']
TREE_MODELS   = ['Decision_Tree', 'Random_Forest', 'Gradient_Boosting']


def predict(patient_data: dict, model_name: str = 'Logistic_Regression'):
    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Run train.py first to generate model files.")
        return None, None

    model = joblib.load(model_path)

    train_columns = joblib.load(os.path.join(MODELS_DIR, 'train_columns.joblib'))

    # Build pure numpy row in training column order
    input_row = np.array(
        [float(patient_data[col]) for col in train_columns],
        dtype=np.float64
    ).reshape(1, -1)

    if model_name in LINEAR_MODELS:
        scaler_path  = os.path.join(MODELS_DIR, 'scaler.joblib')
        columns_path = os.path.join(MODELS_DIR, 'scaler_columns.joblib')

        if not os.path.exists(scaler_path):
            print("Scaler file not found. Run train.py first.")
            return None, None

        scaler     = joblib.load(scaler_path)
        CONTINUOUS = list(joblib.load(columns_path))

        # Remove feature_names_in_ so sklearn uses numpy path, not pandas path
        if hasattr(scaler, 'feature_names_in_'):
            del scaler.feature_names_in_

        cont_idx    = [list(train_columns).index(c) for c in CONTINUOUS]
        cont_values = input_row[:, cont_idx].copy()   # pure numpy (1, 5)
        scaled      = scaler.transform(cont_values)   # pure numpy out

        for i, idx in enumerate(cont_idx):
            input_row[0, idx] = scaled[0, i]

    prediction  = model.predict(input_row)[0]
    probability = model.predict_proba(input_row)[0]

    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    print(f"\nModel      : {model_name}")
    print(f"Prediction : {result}")
    print(f"Confidence : {probability[0]*100:.1f}% No Disease | {probability[1]*100:.1f}% Disease")
    return prediction, probability


def predict_all_models(patient_data: dict):
    print("\n========================================")
    print("  Predictions from all 6 models")
    print("========================================")
    for model_name in LINEAR_MODELS + TREE_MODELS:
        predict(patient_data, model_name=model_name)


def main():
    print("========================================")
    print("  Heart Disease Prediction — Predict")
    print("========================================")

    new_patient = {
        'age':      55, 'sex':       1,  'cp':        2,
        'trestbps': 140,'chol':     250, 'fbs':        0,
        'restecg':    1,'thalach':  160, 'exang':      0,
        'oldpeak':  1.5,'slope':      2, 'ca':         0,
        'thal':       2
    }

    print("\nPatient data:")
    for key, val in new_patient.items():
        print(f"  {key:10}: {val}")

    print("\n--- Best model prediction ---")
    predict(new_patient, model_name='Logistic_Regression')
    predict_all_models(new_patient)


if __name__ == '__main__':
    main()