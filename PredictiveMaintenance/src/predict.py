import pandas as pd
import joblib

# Load saved model
pipeline = joblib.load("models/best_pipeline.joblib")

# Sample machine/sensor input
# Load dataset to get correct feature names
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

print("Sample used for prediction:")
print(sample)
feature_order = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rotation_speed",
    "voltage",
    "current",
    "oil_level",
    "load",
    "motor_temperature",
    "gearbox_temperature",
    "sound_level",
    "fan_speed",
    "reactive_power",
    "active_power"
]

sample = sample[feature_order]

prediction = pipeline.predict(sample)

print("Prediction:", "Machine Failure" if prediction[0] == 1 else "No Machine Failure")