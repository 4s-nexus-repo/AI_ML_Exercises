from fastapi import FastAPI
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("models/smote_random_forest_pipeline.joblib")

# Create FastAPI app
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API Running"}


@app.post("/predict")
def predict():

    # Example sensor input
    sample = pd.DataFrame({
        "temperature": [77.48],
        "vibration": [0.64],
        "pressure": [27.97],
        "humidity": [30.46],
        "rotation_speed": [1500.0],
        "voltage": [220.0],
        "current": [5.0],
        "oil_level": [75.0],
        "load": [60.0],
        "motor_temperature": [65.0],
        "gearbox_temperature": [58.0],
        "sound_level": [71.47],
        "fan_speed": [1258.74],
        "reactive_power": [50.06],
        "active_power": [97.76]
    })

    prediction = pipeline.predict(sample)

    result = "Machine Failure" if prediction[0] == 1 else "No Machine Failure"

    return {"prediction": result}