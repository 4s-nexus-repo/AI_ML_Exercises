## Predictive Maintenance ML Project

## Overview

This project explores machine failure prediction using sensor-based predictive maintenance data.

The aim is to compare multiple machine learning approaches for detecting machine failures based on operational sensor readings such as:

- temperature
- vibration
- pressure
- humidity
- rotation speed
- motor temperature
- oil level
- power metrics

The project evolved from an exploratory notebook workflow into a modular ML engineering structure using reusable Python scripts, sklearn pipelines, model tuning, imbalance handling, explainability, and FastAPI deployment.

---

# Project Structure

PredictiveMaintenance/

├── data/
│ └── SENSOR IOTT.csv
│
├── models/
│ └── saved trained pipelines (.joblib)
│
├── notebooks/
│ └── Predictive maintenance.ipynb
│
├── src/
│ ├── train.py
│ ├── predict.py
│ ├── evaluation.py
│ ├── tune.py
│ ├── compare_models.py
│ ├── smote_train.py
│ ├── neural_network.py
│ ├── shap_analysis.py
│ └── api.py
│
└── README.md

---

# Workflow Summary

## 1. Baseline Training

Compared several ML models:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Gradient Boosting

Initial best baseline model:
- Random Forest

---

## 2. Pipeline Architecture

Implemented sklearn Pipeline to combine:

- StandardScaler
- ML model

This simplified preprocessing and prediction workflows.

---

## 3. Hyperparameter Tuning

Used GridSearchCV with cross-validation to tune Random Forest hyperparameters.

Primary optimization metric:
- Recall

Reason:
Predictive maintenance problems are highly sensitive to missed failures.

---

## 4. Class Imbalance Handling

Observed that high accuracy did not correspond to good failure detection.

Implemented SMOTE to oversample minority failure cases.

Result:
- Recall improved significantly compared to baseline model.

---

## 5. Neural Network Comparison

Implemented MLPClassifier neural network with SMOTE-balanced data.

Compared:
- baseline Random Forest
- tuned Random Forest
- SMOTE Random Forest
- Neural Network

Observed:
- Neural network achieved improved recall for machine failure detection.

---

# Key Findings

| Model | Accuracy | Recall (Failure Class) |
|---|---|---|
| Baseline Random Forest | ~0.84 | ~0.06 |
| SMOTE Random Forest | ~0.81 | ~0.22 |
| Neural Network + SMOTE | ~0.79 | ~0.28 |

Key learning:
- Accuracy alone is misleading for imbalanced predictive maintenance datasets.
- Recall is more important when detecting critical machine failures.

---

# Explainability

Implemented SHAP analysis for feature importance and interaction analysis.

Observed important interactions between:
- vibration
- temperature

This improved interpretability of model predictions.

---

# FastAPI Deployment

Created a FastAPI endpoint to expose the trained ML pipeline as a local prediction service.

Endpoints:
- GET /
- POST /predict

Run locally using:

```bash
poetry run uvicorn src.api:app --reload