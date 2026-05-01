import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import OUTPUTS_DIR
from train import load_and_prepare, apply_scaling

OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'neural_network')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: Load and scale data ───────────────────────────────

X_train, X_test, y_train, y_test = load_and_prepare()
scaled = apply_scaling(X_train, X_test)
X_train_sc, X_test_sc = scaled['2. Z-score (Standard)']


# ── Step 2: Define models ─────────────────────────────────────
# Using SGD optimizer and ReLU activation for all MLP models
# Only changing the hidden layer sizes to see how depth affects results

models = {
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=42),
        True
    ),
    'Decision Tree': (
        DecisionTreeClassifier(max_depth=5, random_state=42),
        False
    ),
    'Random Forest': (
        RandomForestClassifier(n_estimators=100, random_state=42),
        False
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        False
    ),
    'MLP (50) relu+sgd': (
        MLPClassifier(
            hidden_layer_sizes=(50,),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        ),
        True
    ),
    'MLP (100) relu+sgd': (
        MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        ),
        True
    ),
    'MLP (64,32) relu+sgd': (
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        ),
        True
    ),
    'MLP (128,64,32) relu+sgd': (
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='sgd',
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        ),
        True
    ),
}


# ── Step 3: Train and evaluate each model ─────────────────────

print("\n" + "=" * 55)
print("  Training and Evaluating All Models")
print("=" * 55)

names       = []
accuracies  = []
cv_scores   = []

for name, (model, needs_scaling) in models.items():
    if needs_scaling:
        Xtr, Xte = X_train_sc, X_test_sc
    else:
        Xtr, Xte = X_train, X_test

    # cross validation
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='accuracy')

    # train
    model.fit(Xtr, y_train)

    # test accuracy
    preds = model.predict(Xte)
    acc   = accuracy_score(y_test, preds)

    names.append(name)
    cv_scores.append(cv.mean())
    accuracies.append(acc)

    print(f"\n{name}")
    print(f"  CV accuracy  : {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"  Test accuracy: {acc:.4f}")
    print(classification_report(
        y_test, preds,
        target_names=['No Disease', 'Disease']
    ))


# ── Step 4: Bar chart comparing all models ────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

x     = range(len(names))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], cv_scores,
               width, label='CV Accuracy', color='steelblue', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], accuracies,
               width, label='Test Accuracy', color='coral', alpha=0.8)

# add value labels on top of each bar
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.1)
ax.set_title('Model Comparison: Linear vs Tree vs Neural Network\n'
             '(MLP: activation=relu, solver=sgd)',
             fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
plt.savefig(path, dpi=130)
plt.close()
print(f"\nSaved: {path}")


# ── Step 5: MLP loss curves ───────────────────────────────────
# Shows how each MLP learned over epochs

mlp_names = [n for n in names if 'MLP' in n]
mlp_models = {n: m for n, (m, _) in models.items() if 'MLP' in n}

plt.figure(figsize=(10, 5))
for name, model in mlp_models.items():
    if hasattr(model, 'loss_curve_'):
        plt.plot(model.loss_curve_, label=name, linewidth=1.5)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Loss Curves (solver=sgd)')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'mlp_loss_curves.png')
plt.savefig(path, dpi=130)
plt.close()
print(f"Saved: {path}")


# ── Step 6: Summary ───────────────────────────────────────────

print("\n" + "=" * 55)
print("  Final Results (sorted by test accuracy)")
print("=" * 55)

results_df = pd.DataFrame({
    'Model':        names,
    'CV Accuracy':  [round(s, 4) for s in cv_scores],
    'Test Accuracy': [round(a, 4) for a in accuracies]
}).sort_values('Test Accuracy', ascending=False)

print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\nBest model: {best['Model']} with test accuracy {best['Test Accuracy']}")


def main():
    pass  # everything runs at module level above


if __name__ == '__main__':
    pass