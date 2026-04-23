# Heart Disease Prediction

> Python · scikit-learn · Poetry · Rohith Bellapu

---

## Table of Contents

- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the project](#running-the-project)
    - [Run 1 — EDA](#run-1--eda-exploratory-data-analysis)
    - [Run 2 — Training](#run-2--training)
    - [Run 3 — Tuning and Shap](#run-3--tuning--shap)
    - [Run 4 — Prediction](#run-4--prediction)
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
│   └── predict.py               ← loads a saved model, predicts new patient
└── outputs/                     ← auto-created when you run the project
    ├── models/                  ← saved .joblib model files
    ├── tuning/                  ← tuning comparison plots
    └── shap/                    ← SHAP explanation plots
```
---

## Prerequisites

| Requirement | Check |
|-------------|-------|
| Python 3.11+ | `python3 --version` |
| pip3 | `pip3 --version` |
| Poetry | `poetry --version` |

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
---
```bash
poetry run python config.py
```

Expected output:
```
=== Project Configuration ===
Base directory  : /your/path/to/Heart
Data path       : /your/path/to/Heart/heart.csv
Outputs dir     : /your/path/to/Heart/heart_disease/outputs
Models dir      : /your/path/to/AI_ML_Exercises/Heart/heart_disease/outputs/models
Continuous feats: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
Categorical feats: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
Target column   : target
Test size       : 0.2
Random state    : 42
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
| `outputs/models/standard_scaler.joblib` | Fitted StandardScaler for linear models |
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
> `shap_explain.py` loads the tuned `.joblib` files saved by `tune.py`.  
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

### Run 4 — Prediction

```bash
poetry run python predict.py
```

**What this does:**

- Loads all 6 saved `.joblib` model files from `outputs/models/`
- Runs a sample patient through every model
- Prints the prediction and confidence percentage for each model

**Expected output:**

```
========================================
  Heart Disease Prediction — Predict
========================================

Patient data:
  age       : 55
  sex       : 1
  cp        : 2
  trestbps  : 140
  chol      : 250
  ...

--- Best model prediction ---
Model      : Logistic_Regression
Prediction : Heart Disease
Confidence : 5.5% No Disease | 94.5% Disease

========================================
  Predictions from all 6 models
========================================
Model : Logistic_Regression    →  Heart Disease  (94.5%)
Model : LogReg_L2_Ridge        →  Heart Disease  (93.8%)
Model : Decision_Tree          →  Heart Disease  (100.0%)
Model : Random_Forest          →  Heart Disease  (87.0%)
Model : Gradient_Boosting      →  Heart Disease  (91.3%)
```

---

## Predicting for a Different Patient

Open `heart_disease/predict.py` and edit the `new_patient` dictionary inside `main()`:

```python
def main():
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

| Model | Data used | Approx. accuracy |
|-------|-----------|-----------------|
| Logistic Regression | `X_train_scaled` | 86–87% |
| Ridge (L2 regularisation) | `X_train_scaled` | 86–87% |
| Lasso (L1 regularisation) | `X_train_scaled` | 86–87% |
| Decision Tree | `X_train` (raw) | 80–85% |
| Random Forest | `X_train` (raw) | 82–87% |
| Gradient Boosting | `X_train` (raw) | 80–85% |

> **Why linear models use scaled data and tree models use raw data:**
>
> Linear models (Logistic Regression, Ridge, Lasso) compute dot products and distances — features with larger magnitudes (e.g. `chol ~250`) would dominate features with small magnitudes (e.g. `fbs = 0 or 1`) without scaling.
>
> Tree models (Decision Tree, Random Forest, Gradient Boosting) split on thresholds — `is chol > 240?` works the same whether `chol` is `240` or `0.48`. Scale is completely irrelevant to the split decision.

---

*Heart Disease Prediction · Python / scikit-learn / Poetry · Rohith Bellapu*