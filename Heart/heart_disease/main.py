import os
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import MODELS_DIR

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease using trained ML models (Logistic Regression, Random Forest, Gradient Boosting)",
    version="1.0.0"
)

# ── Models that need scaled input ─────────────────────────────

LINEAR_MODELS = ['Logistic_Regression', 'LogReg_L2_Ridge', 'LogReg_L1_Lasso']
TREE_MODELS   = ['Decision_Tree', 'Random_Forest', 'Gradient_Boosting']


# ── Request body — patient features ───────────────────────────

class PatientData(BaseModel):
    age:      float = Field(..., example=55,  description="Age in years")
    sex:      float = Field(..., example=1,   description="1 = male, 0 = female")
    cp:       float = Field(..., example=2,   description="Chest pain type (0-3)")
    trestbps: float = Field(..., example=140, description="Resting blood pressure (mm Hg)")
    chol:     float = Field(..., example=250, description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., example=0,   description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg:  float = Field(..., example=1,   description="Resting ECG results (0-2)")
    thalach:  float = Field(..., example=160, description="Maximum heart rate achieved")
    exang:    float = Field(..., example=0,   description="Exercise induced angina (1=yes, 0=no)")
    oldpeak:  float = Field(..., example=1.5, description="ST depression induced by exercise")
    slope:    float = Field(..., example=2,   description="Slope of peak exercise ST segment (0-2)")
    ca:       float = Field(..., example=0,   description="Number of major vessels (0-3)")
    thal:     float = Field(..., example=2,   description="Thalassemia (0-3)")


# ── Response body ─────────────────────────────────────────────

class PredictionResult(BaseModel):
    model:           str
    prediction:      int
    result:          str

class AllPredictions(BaseModel):
    patient:     dict
    predictions: list[PredictionResult]
    summary:     str


# ── Helper: load and predict ───────────────────────────────────

def run_prediction(patient_data: dict, model_name: str) -> PredictionResult:
    """Load a model and return prediction for the given patient."""

    model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    model      = joblib.load(model_path)

    # Build input as ordered numpy array
    train_columns = joblib.load(os.path.join(MODELS_DIR, 'train_columns.joblib'))
    input_row = np.array(
        [float(patient_data[col]) for col in train_columns],
        dtype=np.float64
    ).reshape(1, -1)

    # Scale if linear model
    if model_name in LINEAR_MODELS:
        scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        CONTINUOUS = list(joblib.load(os.path.join(MODELS_DIR, 'scaler_columns.joblib')))

        # Remove pandas metadata so sklearn uses numpy path
        if hasattr(scaler, 'feature_names_in_'):
            del scaler.feature_names_in_

        cont_idx    = [list(train_columns).index(c) for c in CONTINUOUS]
        cont_values = input_row[:, cont_idx].copy()
        scaled      = scaler.transform(cont_values)
        for i, idx in enumerate(cont_idx):
            input_row[0, idx] = scaled[0, i]

    # Predict
    prediction  = int(model.predict(input_row)[0])
    probability = model.predict_proba(input_row)[0]

    return PredictionResult(
        model                  = model_name,
        prediction             = prediction,
        result                 = "Heart Disease" if prediction == 1 else "No Heart Disease",
    )



# ── Ngrok public URL (optional) ──────────────────────────────

def start_ngrok(port: int = 8000):
    """Start ngrok tunnel and print the public URL."""
    try:
        from pyngrok import ngrok, conf

        # Kill any existing tunnels first
        ngrok.kill()

        # Open tunnel
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url

        print("\n" + "=" * 55)
        print("  Ngrok tunnel is LIVE")
        print("=" * 55)
        print(f"  Local URL  : http://127.0.0.1:{port}")
        print(f"  Public URL : {public_url}")
        print(f"  Swagger UI : {public_url}/docs")
        print(f"  Share this with your instructor: {public_url}/docs")
        print("=" * 55 + "\n")

        return public_url

    except ImportError:
        print("pyngrok not installed. Run: poetry add pyngrok")
        return None
    except Exception as e:
        print(f"Ngrok error: {e}")
        print("Make sure your authtoken is set:")
        print("  ngrok config add-authtoken YOUR_TOKEN")
        return None



# ── Auto-start ngrok on app startup ──────────────────────────

@app.on_event("startup")
async def startup_event():
    """Start ngrok tunnel when the app starts."""
    import os
    if os.environ.get("START_NGROK", "false").lower() == "true":
        start_ngrok(port=8000)

# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    """Welcome endpoint — confirms the API is running."""
    return {
        "message": "Heart Disease Prediction API is running",
        "usage":   "POST /predict with patient data to get a prediction",
        "docs":    "Visit /docs for interactive Swagger UI"
    }


@app.get("/models")
def list_models():
    """Returns all available models."""
    return {
        "linear_models":    LINEAR_MODELS,
        "tree_models":      TREE_MODELS,
        "all_models":       LINEAR_MODELS + TREE_MODELS
    }


@app.post("/predict", response_model=AllPredictions)
def predict(patient: PatientData):
    """
    Predict heart disease for a patient using all 6 models.

    Send patient features in the request body.
    Returns predictions from all models with confidence percentages.
    """
    patient_dict = patient.model_dump()
    predictions  = []

    for model_name in LINEAR_MODELS + TREE_MODELS:
        result = run_prediction(patient_dict, model_name)
        predictions.append(result)

    # Summary — majority vote across all 6 models
    disease_count = sum(1 for p in predictions if p.prediction == 1)
    summary = (
        f"{disease_count}/6 models predict Heart Disease"
        if disease_count > 3
        else f"{6 - disease_count}/6 models predict No Heart Disease"
    )

    return AllPredictions(
        patient     = patient_dict,
        predictions = predictions,
        summary     = summary
    )


@app.post("/predict/{model_name}", response_model=PredictionResult)
def predict_single(model_name: str, patient: PatientData):
    """
    Predict using a specific model.

    model_name options: Logistic_Regression, Decision_Tree,
                        Random_Forest, Gradient_Boosting,
                        LogReg_L2_Ridge, LogReg_L1_Lasso
    """
    patient_dict = patient.model_dump()
    return run_prediction(patient_dict, model_name)