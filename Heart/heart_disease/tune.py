import os
import warnings

import joblib

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import uniform, randint

from config import (
    OUTPUTS_DIR, MODELS_DIR,
    RANDOM_STATE
)

TUNE_DIR = os.path.join(OUTPUTS_DIR, 'tuning')


# ── Baseline ──────────────────────────────────────────────────

def run_baseline(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        'Decision Tree':     DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    results = {}
    print(f"\n{'Model':<25} {'CV Acc':>10} {'Test Acc':>10} {'AUC':>8}")
    print('=' * 58)

    for name, model in models.items():
        cv_sc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            'model':    model,
            'cv_mean':  cv_sc.mean(),
            'test_acc': accuracy_score(y_test, preds),
            'auc':      roc_auc_score(y_test, proba),
        }
        print(f"{name:<25} {cv_sc.mean():>10.4f} "
              f"{accuracy_score(y_test, preds):>10.4f} "
              f"{roc_auc_score(y_test, proba):>8.4f}")

    return results


# ── GridSearchCV ──────────────────────────────────────────────

def run_gridsearch(X_train, y_train):
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [50, 100, 200],
         'max_depth': [3, 5, 10, None],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4]},
        cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    print(f"\nRF GridSearchCV ({3*4*3*3} combos x 5 = {3*4*3*3*5} fits)...")
    rf_grid.fit(X_train, y_train)
    print(f"Best: {rf_grid.best_params_}  CV={rf_grid.best_score_:.4f}")

    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {'max_depth': [3, 5, 7, 10, None],
         'min_samples_split': [2, 5, 10, 20],
         'min_samples_leaf': [1, 2, 4, 8],
         'criterion': ['gini', 'entropy']},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    print(f"\nDT GridSearchCV ({5*4*4*2} combos x 5 = {5*4*4*2*5} fits)...")
    dt_grid.fit(X_train, y_train)
    print(f"Best: {dt_grid.best_params_}  CV={dt_grid.best_score_:.4f}")

    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [50, 100, 200],
         'learning_rate': [0.01, 0.05, 0.1, 0.2],
         'max_depth': [2, 3, 5],
         'subsample': [0.7, 0.8, 1.0]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    print(f"\nGB GridSearchCV ({3*4*3*3} combos x 5 = {3*4*3*3*5} fits)...")
    gb_grid.fit(X_train, y_train)
    print(f"Best: {gb_grid.best_params_}  CV={gb_grid.best_score_:.4f}")

    return rf_grid, dt_grid, gb_grid


# ── RandomizedSearchCV ────────────────────────────────────────

def run_randomsearch(X_train, y_train):
    rf_random = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        {'n_estimators': randint(50, 500),
         'max_depth': randint(3, 20),
         'min_samples_split': randint(2, 20),
         'min_samples_leaf': randint(1, 10),
         'max_features': uniform(0.3, 0.7)},
        n_iter=100, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    print("\nRF RandomizedSearchCV (100 iterations)...")
    rf_random.fit(X_train, y_train)
    print(f"Best: {rf_random.best_params_}  CV={rf_random.best_score_:.4f}")

    gb_random = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'n_estimators': randint(50, 500),
         'learning_rate': uniform(0.005, 0.295),
         'max_depth': randint(2, 8),
         'subsample': uniform(0.6, 0.4),
         'min_samples_split': randint(2, 20),
         'min_samples_leaf': randint(1, 10)},
        n_iter=100, cv=5, scoring='accuracy',
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    print("\nGB RandomizedSearchCV (100 iterations)...")
    gb_random.fit(X_train, y_train)
    print(f"Best: {gb_random.best_params_}  CV={gb_random.best_score_:.4f}")

    return rf_random, gb_random


# ── Compare and save plots ────────────────────────────────────

def compare_and_save(X_train, X_test, y_train, y_test,
                     baseline, rf_grid, dt_grid, gb_grid,
                     rf_random, gb_random):
    os.makedirs(TUNE_DIR, exist_ok=True)

    rows = []
    for name, r in baseline.items():
        rows.append({'Model': f'{name} (Baseline)',
                     'CV Acc': round(r['cv_mean'], 4),
                     'Test Acc': round(r['test_acc'], 4),
                     'AUC': round(r['auc'], 4), 'Tag': 'Baseline'})

    for name, model in [('RF GridSearch', rf_grid.best_estimator_),
                         ('RF RandomSearch', rf_random.best_estimator_),
                         ('GB GridSearch', gb_grid.best_estimator_),
                         ('GB RandomSearch', gb_random.best_estimator_),
                         ('DT GridSearch', dt_grid.best_estimator_)]:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        cv    = cross_val_score(model, X_train, y_train, cv=5).mean()
        rows.append({'Model': name,
                     'CV Acc': round(cv, 4),
                     'Test Acc': round(accuracy_score(y_test, preds), 4),
                     'AUC': round(roc_auc_score(y_test, proba), 4),
                     'Tag': 'Tuned'})

    comp_df = pd.DataFrame(rows).sort_values('AUC', ascending=False)
    print("\n" + comp_df.drop('Tag', axis=1).to_string(index=False))

    # Comparison bar chart
    colors = ['#e74c3c' if t == 'Baseline' else '#27ae60' for t in comp_df['Tag']]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Hyperparameter Tuning — Baseline vs Tuned',
                 fontsize=14, fontweight='bold')
    for ax, metric in zip(axes, ['Test Acc', 'AUC']):
        bars = ax.barh(comp_df['Model'], comp_df[metric],
                       color=colors, edgecolor='white', height=0.6)
        for bar, val in zip(bars, comp_df[metric]):
            ax.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel(metric)
        ax.set_xlim(0.6, 1.02)
    axes[0].legend(handles=[Patch(color='#e74c3c', label='Baseline'),
                             Patch(color='#27ae60', label='Tuned')])
    plt.tight_layout()
    plt.savefig(os.path.join(TUNE_DIR, 'tuning_comparison.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {TUNE_DIR}/tuning_comparison.png")

    # GridSearch exploration chart
    rf_cv_res = pd.DataFrame(rf_grid.cv_results_)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RF GridSearchCV — Parameter Exploration', fontsize=13, fontweight='bold')
    for depth in [3, 5, 10]:
        mask = rf_cv_res['param_max_depth'] == depth
        grp  = rf_cv_res[mask].groupby('param_n_estimators')['mean_test_score'].mean()
        axes[0].plot(grp.index, grp.values, marker='o', label=f'max_depth={depth}')
    axes[0].set_xlabel('n_estimators')
    axes[0].set_ylabel('Mean CV Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.4)
    axes[1].hist(rf_cv_res['mean_test_score'], bins=20,
                 color='#3498db', edgecolor='white', alpha=0.85)
    axes[1].axvline(rf_grid.best_score_, color='#e74c3c', lw=2, ls='--',
                    label=f'Best: {rf_grid.best_score_:.4f}')
    axes[1].legend()
    axes[1].grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(TUNE_DIR, 'gridsearch_exploration.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {TUNE_DIR}/gridsearch_exploration.png")


# ── Save tuned models ─────────────────────────────────────────

def save_tuned_models(rf_random, gb_random, dt_grid):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in [('Random_Forest_Tuned', rf_random.best_estimator_),
                         ('Gradient_Boosting_Tuned', gb_random.best_estimator_),
                         ('Decision_Tree_Tuned', dt_grid.best_estimator_)]:
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        joblib.dump(model, path)
        print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────

def main():
    from train import load_and_prepare, apply_scaling

    print("=" * 60)
    print("  Hyperparameter Tuning")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_and_prepare()
    scaled = apply_scaling(X_train, X_test)
    X_train_scaled, X_test_scaled = scaled['2. Z-score (Standard)']

    print("\n[1/4] Baseline...")
    baseline = run_baseline(X_train, X_test, y_train, y_test)

    print("\n[2/4] GridSearchCV...")
    rf_grid, dt_grid, gb_grid = run_gridsearch(X_train, y_train)

    print("\n[3/4] RandomizedSearchCV...")
    rf_random, gb_random = run_randomsearch(X_train, y_train)

    print("\n[4/4] Comparing and saving...")
    compare_and_save(X_train, X_test, y_train, y_test,
                     baseline, rf_grid, dt_grid, gb_grid,
                     rf_random, gb_random)
    save_tuned_models(rf_random, gb_random, dt_grid)

    print("\nTuning complete!")
    return rf_random, gb_random, dt_grid


if __name__ == '__main__':
    main()
