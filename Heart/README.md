# Heart Disease Prediction

> Python · scikit-learn · FastAPI · Poetry · Rohith Bellapu

---

## Table of Contents

- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the project](#running-the-project)
  - [Run 1 — EDA](#run-1--eda-exploratory-data-analysis)
  - [Run 2 — Training](#run-2--training)
  - [Run 3 — Tuning & SHAP](#run-3--tuning--shap)
  - [Run 4 — Neural Network](#run-4--neural-network)
  - [Run 5 — Prediction](#run-5--prediction)
  - [Run 6 — FastAPI](#run-6--fastapi)
  - [Run 7 — Ngrok](#run-7--ngrok-share-public-url)
- [Predicting for a different patient](#predicting-for-a-different-patient)
- [Models and results](#models-and-results)
- [Quick reference](#quick-reference--all-commands)

---

## Project Structure

```
Heart/
├── pyproject.toml               ← Poetry config and dependencies
├── poetry.lock                  ← exact versions locked
├── heart_disease/               ← all Python source files
│   ├── __init__.py
│   ├── heart.csv                ← dataset
│   ├── config.py                ← paths and constants
│   ├── eda.py                   ← data exploration and visualisation
│   ├── evaluate.py              ← evaluation helper (used by train.py)
│   ├── train.py                 ← trains and saves all 6 models
│   ├── tune.py                  ← hyperparameter tuning (GridSearch + RandomizedSearch)
│   ├── shap_explain.py          ← SHAP explainability for tree models
│   ├── neural_network.py        ← Linear vs Tree vs MLP comparison (homework)
│   ├── predict.py               ← loads a saved model, predicts new patient
│   └── main.py                  ← FastAPI endpoints + ngrok public URL
└── outputs/                     ← auto-created when you run the project
    ├── models/                  ← saved .joblib model files
    ├── tuning/                  ← tuning comparison plots
    ├── shap/                    ← SHAP explanation plots
    └── neural_network/          ← MLP comparison plots
```

---

## Prerequisites

| Requirement | Check |
|-------------|-------|
| Python 3.11+ | `python3 --version` |
| pip3 | `pip3 --version` |
| Poetry | `poetry --version` |

---

## Setup

**Step 1 — Clone or download the project**

```bash
git clone git@github.com:4s-nexus-repo/AI_ML_Exercises.git
```

**Step 2 — Install dependencies**

```bash
pip3 install poetry
```

**Step 3 — Verify setup**

```bash
poetry --version
```

Expected output:
```
Poetry (version 2.3.4)
```

```bash
poetry run python config.py
```

Expected output:
```
=== Project Configuration ===
Base directory  : /your/path/to/Heart
Data path       : /your/path/to/Heart/heart.csv
heart.csv found ✔
```

---

## Running the Project

Run all commands from inside the `heart_disease/` folder:

```bash
cd /path/to/Heart/heart_disease
```

---

### Run 1 — EDA (Exploratory Data Analysis)

```bash
poetry run python eda.py
```

**What this does:**

- Loads `heart.csv` and removes duplicate rows
- Prints dataset shape, data types, missing values, and target distribution
- Saves 3 plots to the `outputs/` folder

| Plot saved | What it shows |
|------------|---------------|
| `outputs/target_distribution.png` | Bar chart of class balance — No Disease vs Disease |
| `outputs/categorical_vs_target.png` | Count plots for all 8 categorical features split by target |
| `outputs/correlation_heatmap.png` | Lower-triangle heatmap showing correlations between features |

---

### Run 2 — Training

```bash
poetry run python train.py
```

**What this does:**

- Loads and splits the data into 80% train and 20% test
- Applies all 5 scaling techniques
- Trains 6 machine learning models
- Evaluates each model and prints metrics to the terminal
- Saves all trained models as `.joblib` files to `outputs/models/`
- Saves confusion matrix plots and a comparison chart to `outputs/`

**Expected terminal output per model:**

```
=== Logistic Regression ===
Accuracy:  0.8689
Precision: 0.8750
Recall:    0.8750
F1 Score:  0.8750

Classification Report:
              precision    recall  f1-score   support
  No Disease       0.86      0.86      0.86        29
Heart Disease       0.88      0.88      0.88        32
    accuracy                           0.87        61
```

**Files saved after training:**

| File | Description |
|------|-------------|
| `outputs/models/Logistic_Regression.joblib` | Trained logistic regression model |
| `outputs/models/LogReg_L2_Ridge.joblib` | Trained ridge (L2) model |
| `outputs/models/LogReg_L1_Lasso.joblib` | Trained lasso (L1) model |
| `outputs/models/Decision_Tree.joblib` | Trained decision tree |
| `outputs/models/Random_Forest.joblib` | Trained random forest |
| `outputs/models/Gradient_Boosting.joblib` | Trained gradient boosting |
| `outputs/models/scaler.joblib` | Fitted StandardScaler (numpy-based, no pandas metadata) |
| `outputs/models/scaler_columns.joblib` | Continuous feature names the scaler was fit on |
| `outputs/models/train_columns.joblib` | Full column order used during training |
| `outputs/model_comparison.png` | Bar chart comparing all 6 models |
| `outputs/cm_*.png` | Confusion matrix for each model |

---

### Run 3 — Tuning & SHAP

```bash
poetry run python tune.py          # 3a. hyperparameter tuning + save tuned models
poetry run python shap_explain.py  # 3b. SHAP explainability plots
```

**What this does:**

- Runs the 3 tree models with default settings to get a baseline accuracy
- Runs `GridSearchCV` and `RandomizedSearchCV` to find best hyperparameters
- Saves tuned models to `outputs/models/` as `.joblib` files
- Runs SHAP `TreeExplainer` on Random Forest, Gradient Boosting, and Decision Tree
- Saves beeswarm, bar, waterfall, dependence, force, and comparison plots to `outputs/shap/`

> **Important:** `tune.py` must run before `shap_explain.py`.
> Running `shap_explain.py` without first running `tune.py` will throw a `FileNotFoundError`.

**Files saved after this step:**

| File | Description |
|------|-------------|
| `outputs/models/Random_Forest_Tuned.joblib` | Best RF from RandomizedSearchCV |
| `outputs/models/Gradient_Boosting_Tuned.joblib` | Best GB from RandomizedSearchCV |
| `outputs/models/Decision_Tree_Tuned.joblib` | Best DT from GridSearchCV |
| `outputs/tuning/tuning_comparison.png` | Baseline vs tuned accuracy bar chart |
| `outputs/tuning/gridsearch_exploration.png` | GridSearch parameter exploration chart |
| `outputs/shap/shap_*_beeswarm.png` | SHAP beeswarm — one dot per patient per feature |
| `outputs/shap/shap_*_waterfall.png` | SHAP waterfall — single patient explanation |
| `outputs/shap/shap_comparison_all_models.png` | SHAP importance across all 3 tree models |

---

### Run 4 — Neural Network

```bash
poetry run python neural_network.py
```

**What this does:**

- Compares Linear vs Tree-based vs Neural Network (MLPClassifier) models
- All MLP configurations use `activation=relu` and `solver=sgd`
- Only the architecture (hidden layer sizes) changes across the 4 MLP configs
- Runs 5-fold cross validation for each model
- Saves a comparison bar chart and MLP training loss curves

**MLP architectures compared:**

| Config | Architecture | Neurons |
|--------|-------------|---------|
| MLP (50) | 13 → 50 → 1 | 751 |
| MLP (100) | 13 → 100 → 1 | 1,501 |
| MLP (64,32) | 13 → 64 → 32 → 1 | 2,945 |
| MLP (128,64,32) | 13 → 128 → 64 → 32 → 1 | 12,353 |

**Files saved:**

| File | Description |
|------|-------------|
| `outputs/neural_network/model_comparison.png` | CV vs test accuracy bar chart for all 8 models |
| `outputs/neural_network/mlp_loss_curves.png` | Training loss per epoch for each MLP config |

> **Note:** Neural network models use scaled data. Tree models use raw data.
> `learning_rate=adaptive` reduces the learning rate when training loss stops improving.

---

### Run 5 — Prediction

```bash
poetry run python predict.py
```

**What this does:**

- Loads all 6 saved `.joblib` model files from `outputs/models/`
- Enforces the exact column order used during training
- Scales continuous features for linear models using the saved scaler
- Runs all 6 models on the same patient and prints predictions

**Expected output:**

```
========================================
  Heart Disease Prediction — Predict
========================================

Patient data:
  age       : 55
  sex       : 1
  ...

--- Best model prediction ---
Model      : Logistic_Regression
Prediction : Heart Disease

========================================
  Predictions from all 6 models
========================================

Model      : Logistic_Regression    →  Heart Disease
Model      : LogReg_L2_Ridge        →  Heart Disease
Model      : Decision_Tree          →  Heart Disease
Model      : Random_Forest          →  Heart Disease
Model      : Gradient_Boosting      →  Heart Disease
```

---

### Run 6 — FastAPI

```bash
poetry run uvicorn main:app --reload
```

**What this does:**

- Starts a local HTTP server at `http://127.0.0.1:8000`
- Loads the same trained `.joblib` models — no retraining
- Exposes 4 endpoints for making predictions via HTTP

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — confirms API is running |
| GET | `/models` | Lists all 6 available model names |
| POST | `/predict` | Runs all 6 models, returns results + majority vote |
| POST | `/predict/{model_name}` | Runs a specific model only |

**Test via Swagger UI:**

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

This opens an interactive page where you can:
1. Click any endpoint
2. Click **Try it out**
3. Enter patient values
4. Click **Execute**
5. See the prediction result in the browser

**Example request body for POST /predict:**

```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 160,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

**Example response:**

```json
{
  "predictions": [
    { "model": "Logistic_Regression",  "result": "Heart Disease" },
    { "model": "LogReg_L2_Ridge",      "result": "Heart Disease" },
    { "model": "LogReg_L1_Lasso",      "result": "Heart Disease" },
    { "model": "Decision_Tree",        "result": "Heart Disease" },
    { "model": "Random_Forest",        "result": "Heart Disease" },
    { "model": "Gradient_Boosting",    "result": "Heart Disease" }
  ],
  "summary": "6/6 models predict Heart Disease"
}
```

---

### Run 7 — Ngrok (share public URL)

Ngrok wraps your local API with a public HTTPS URL that anyone can access from anywhere.

**One-time setup:**

```bash
# Step 1 — install pyngrok (already in pyproject.toml)
poetry add pyngrok

# Step 2 — get a free token from ngrok.com and set it
poetry run python -c "from pyngrok import ngrok; ngrok.set_auth_token('YOUR_TOKEN_HERE')"
```

**Start the API with ngrok:**

```bash
START_NGROK=true poetry run uvicorn main:app --reload
```

**Expected terminal output:**

```
=======================================================
  Ngrok tunnel is LIVE
=======================================================
  Local URL  : http://127.0.0.1:8000
  Public URL : https://abc123.ngrok-free.app
  Swagger UI : https://abc123.ngrok-free.app/docs
  Share this with your instructor: https://abc123.ngrok-free.app/docs
=======================================================
```

Share the Swagger UI link with anyone, and they can test all endpoints in their browser without installing anything.

---

## Predicting for a Different Patient

Open `heart_disease/predict.py` and edit the `new_patient` dictionary inside `main()`:

```python
new_patient = {
    'age':      55,    # patient age in years
    'sex':       1,    # 1 = male, 0 = female
    'cp':        2,    # chest pain type (0-3)
    'trestbps': 140,   # resting blood pressure (mm Hg)
    'chol':     250,   # serum cholesterol (mg/dl)
    'fbs':        0,   # fasting blood sugar > 120 mg/dl (1=true, 0=false)
    'restecg':    1,   # resting ECG results (0-2)
    'thalach':  160,   # maximum heart rate achieved
    'exang':      0,   # exercise induced angina (1=yes, 0=no)
    'oldpeak':  1.5,   # ST depression induced by exercise
    'slope':      2,   # slope of peak exercise ST segment (0-2)
    'ca':         0,   # number of major vessels (0-3)
    'thal':       2,   # thalassemia (0-3)
}
```

Then run:

```bash
poetry run python predict.py
```

Or send a POST request to the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":55,"sex":1,"cp":2,"trestbps":140,"chol":250,"fbs":0,"restecg":1,"thalach":160,"exang":0,"oldpeak":1.5,"slope":2,"ca":0,"thal":2}'
```

---

## Scaling Techniques Used

Five scaling methods are applied to the 5 continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`). Categorical features are left unscaled.

| # | Technique | Formula | Output range |
|---|-----------|---------|--------------|
| 1 | Min-Max Scaler | `(x - min) / (max - min)` | `[0, 1]` |
| 2 | Z-score (Standard) | `(x - mean) / std` | `mean=0, std=1` |
| 3 | Robust Scaler | `(x - median) / IQR` | `median=0` |
| 4 | Max Abs Scaler | `x / |x_max|` | `[-1, 1]` |
| 5 | Unit Vector (L2) | `x / L2 norm (row-wise)` | `L2 norm = 1` |

The Z-score (`StandardScaler`) is used as the default scaler for all linear models as it achieved the highest accuracy on this dataset.

---

## Models and Results

| Model | Category | Data used | Approx. accuracy |
|-------|----------|-----------|-----------------|
| Logistic Regression | Linear | `X_train_scaled` | 86–87% |
| Ridge (L2 regularisation) | Linear | `X_train_scaled` | 86–87% |
| Lasso (L1 regularisation) | Linear | `X_train_scaled` | 86–87% |
| Decision Tree | Tree-based | `X_train` (raw) | 80–85% |
| Random Forest | Tree-based | `X_train` (raw) | 82–87% |
| Gradient Boosting | Tree-based | `X_train` (raw) | 80–85% |
| MLP (50) relu+sgd | Neural Network | `X_train_scaled` | 78–85% |
| MLP (64,32) relu+sgd | Neural Network | `X_train_scaled` | 80–87% |

> **Why linear models use scaled data and tree models use raw data:**
>
> Linear models compute dot products and distances — features with larger magnitudes (e.g. `chol ~250`) would dominate smaller ones (e.g. `fbs = 0 or 1`) without scaling.
>
> Tree models split on thresholds — `is chol > 240?` works the same whether `chol` is `240` or `0.48`. Scale is irrelevant.
>
> Neural networks (MLP) are sensitive to feature scale like linear models, so scaled data is used.

---

## Quick Reference — All Commands

```bash
# ── Setup (run once) ──────────────────────────────────────────
pip3 install poetry
cd /path/to/Heart
poetry install

# ── Navigate to source folder ─────────────────────────────────
cd heart_disease

# ── Verify setup ──────────────────────────────────────────────
poetry run python config.py

# ── Run the project (in order) ────────────────────────────────
poetry run python eda.py                  # 1. generate EDA plots
poetry run python train.py                # 2. train and save models
poetry run python tune.py                 # 3a. hyperparameter tuning
poetry run python shap_explain.py         # 3b. SHAP explainability
poetry run python neural_network.py       # 4. neural network comparison
poetry run python predict.py              # 5. predict via terminal

# ── FastAPI (local) ───────────────────────────────────────────
poetry run uvicorn main:app --reload      # 6. start API at localhost:8000

# ── FastAPI + Ngrok (public URL for instructor) ───────────────
START_NGROK=true poetry run uvicorn main:app --reload  # 7. get public URL
```

---

*Heart Disease Prediction · Python / scikit-learn / FastAPI / Poetry · Rohith Bellapu*