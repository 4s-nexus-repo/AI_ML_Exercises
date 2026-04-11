import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')

# Create output folder for all saved plots
os.makedirs('outputs', exist_ok=True)

print("All libraries imported successfully!")


# ── Load Data ─────────────────────────────────────────────────

df = pd.read_csv('/Users/rohithreddy/ML Classes/heart.csv')

print("First 5 rows:")
print(df.head())  # print() needed outside Jupyter

print("\nSex value counts:")
print(df["sex"].value_counts())

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())


# ── Target Variable Distribution ──────────────────────────────

print("\nTarget variable distribution:")
print(df['target'].value_counts())
print("\nProportions:")
print(df['target'].value_counts(normalize=True).round(4))


# ── Data Visualization ────────────────────────────────────────

plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.tight_layout()
plt.savefig('outputs/target_distribution.png', dpi=130)  # savefig instead of show()
plt.close()                                               # release memory
print("Saved: outputs/target_distribution.png")

cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, hue='target', ax=axes[i])
    axes[i].set_title(f"{col} vs target")
    axes[i].legend(title='target', loc='upper right')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/categorical_vs_target.png', dpi=130)
plt.close()
print("Saved: outputs/categorical_vs_target.png")

numeric_cols = df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 6))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=130)
plt.close()
print("Saved: outputs/correlation_heatmap.png")


# ── Remove Duplicates ─────────────────────────────────────────

df = df.drop_duplicates()
print(f"\nAfter removing duplicates: {df.shape}")


# ── Train/Test Split ──────────────────────────────────────────

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")


# ── Feature Scaling — 5 Techniques ───────────────────────────

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scalers = {
    '1. Min-Max Scaler':     MinMaxScaler(),
    '2. Z-score (Standard)': StandardScaler(),
    '3. Robust Scaler':      RobustScaler(),
    '4. Max Abs Scaler':     MaxAbsScaler(),
    '5. Unit Vector (L2)':   Normalizer(norm='l2'),
}

scaled = {}  # stores (X_train_scaled, X_test_scaled) per scaler

for name, scaler in scalers.items():
    X_tr = X_train.copy()
    X_te = X_test.copy()

    if name == '5. Unit Vector (L2)':
        # Normalizer acts row-wise — applied to ALL features
        X_tr = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index
        )
        X_te = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index
        )
    else:
        # Others: scale continuous columns only
        X_tr[continuous_features] = scaler.fit_transform(X_train[continuous_features])
        X_te[continuous_features] = scaler.transform(X_test[continuous_features])

    scaled[name] = (X_tr, X_te)

# Extract Z-score scaled data for linear models
X_train_scaled, X_test_scaled = scaled['2. Z-score (Standard)']

print("\nScaled statistics per technique:")
for name, (X_tr, _) in scaled.items():
    stats = X_tr[continuous_features].describe().loc[['mean', 'std', 'min', 'max']]
    print(f'\n── {name} ' + '─' * 40)
    print(stats.round(3).to_string())

print("\nAll 5 scalers fitted and applied ✔")


# ── Evaluation Helper ─────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name):
    """Predict on test set, print metrics, save confusion matrix plot."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()

    filename = model_name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(f'outputs/cm_{filename}.png', dpi=130)
    plt.close()
    print(f"Saved: outputs/cm_{filename}.png")

    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}


results = []
print("\nHelper function defined.")


# ── Model 1 — Logistic Regression (scaled) ───────────────────

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr_model, X_test_scaled, y_test, 'Logistic Regression'))


# ── Model 2 — Decision Tree (raw) ────────────────────────────

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
results.append(evaluate_model(dt_model, X_test, y_test, 'Decision Tree'))


# ── Model 3 — Random Forest (raw) ────────────────────────────
# Issue 3 fixed: was incorrectly trained on X_train_scaled
# Tree-based models don't need scaling — use raw X_train / X_test

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)                                   # raw
results.append(evaluate_model(rf_model, X_test, y_test, 'Random Forest'))  # raw


# ── Model 4 — Gradient Boosting (raw) ────────────────────────

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
results.append(evaluate_model(gb_model, X_test, y_test, 'Gradient Boosting'))


# ── Model 5 — Ridge / LogReg L2 (scaled) ─────────────────────

lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=42)
lr_l2.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr_l2, X_test_scaled, y_test, 'LogReg (L2/Ridge)'))


# ── Model 6 — Lasso / LogReg L1 (scaled) ─────────────────────

lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=42)
lr_l1.fit(X_train_scaled, y_train)
results.append(evaluate_model(lr_l1, X_test_scaled, y_test, 'LogReg (L1/Lasso)'))


# ── Final Summary ─────────────────────────────────────────────

results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
print("\n" + "="*60)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 5))
x = range(len(results_df))
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
colors  = ['#3498db', '#27ae60', '#e67e22', '#e74c3c']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    plt.bar([xi + i * width for xi in x], results_df[metric],
            width=width, label=metric, color=color, alpha=0.85)

plt.xticks([xi + width * 1.5 for xi in x], results_df['Model'], rotation=15, ha='right')
plt.ylim(0.5, 1.05)
plt.ylabel('Score')
plt.title('Model Comparison — All Metrics')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=130)
plt.close()
print("\nSaved: outputs/model_comparison.png")
print("\nDone! All plots saved to the outputs/ folder.")
