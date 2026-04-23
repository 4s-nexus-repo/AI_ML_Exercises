import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from config import OUTPUTS_DIR, MODELS_DIR

SHAP_DIR = os.path.join(OUTPUTS_DIR, 'shap')


# ── Core SHAP plots for one model ────────────────────────────

def explain_model(model_name, model, X_test):
    """Run TreeExplainer and save beeswarm, bar, waterfall, dependence."""
    os.makedirs(SHAP_DIR, exist_ok=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Normalise — RF returns list [class_0, class_1], GB/DT return array
    if isinstance(shap_values, list):
        sv   = shap_values[1]
        base = explainer.expected_value[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        # new SHAP format: 3D array shape (n_samples, n_features, n_classes)
        sv   = shap_values[:, :, 1]
        base = explainer.expected_value[1] \
            if hasattr(explainer.expected_value, '__len__') \
            else explainer.expected_value
    else:
        sv   = shap_values
        base = explainer.expected_value
    base = float(np.atleast_1d(base)[0])
    tag = model_name.replace(' ', '_').lower()
    print(f"\nSHAP shape : {sv.shape}  (patients x features)")
    print(f"Base value : {base:.4f}")

    # Feature importance ranking
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X_test.columns)
    print(importance.sort_values(ascending=False).round(4).to_string())

    # Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_test, feature_names=list(X_test.columns),
                      show=False, plot_type='dot')
    plt.title(f'{model_name} — SHAP Beeswarm',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f'shap_{tag}_beeswarm.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    # Bar (global importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, feature_names=list(X_test.columns),
                      show=False, plot_type='bar')
    plt.title(f'{model_name} — SHAP Global Importance',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f'shap_{tag}_bar.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    # Waterfall — first Disease patient
    preds       = model.predict(X_test)
    dis_indices = np.where(preds == 1)[0]
    idx         = dis_indices[0] if len(dis_indices) > 0 else 0

    explanation = shap.Explanation(
        values=sv[idx],
        base_values=base,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns)
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f'{model_name} — Waterfall (Patient {idx})',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f'shap_{tag}_waterfall.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    # Dependence — top 2 features
    top2 = [X_test.columns[i]
            for i in np.argsort(np.abs(sv).mean(axis=0))[-2:][::-1]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} — SHAP Dependence (Top 2 Features)',
                 fontsize=13, fontweight='bold')
    for ax, feat in zip(axes, top2):
        shap.dependence_plot(feat, sv, X_test, ax=ax, show=False)
        ax.set_title(feat, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f'shap_{tag}_dependence.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    print(f"Saved 4 plots → {SHAP_DIR}/shap_{tag}_*.png")
    return sv, base


# ── Force plot (GB only) ──────────────────────────────────────

def force_plot(model, shap_values, base_value, X_test):
    """Save horizontal force plot for the first Disease patient."""
    preds = model.predict(X_test)
    idx   = np.where(preds == 1)[0][0] if np.any(preds == 1) else 0

    plt.figure()
    shap.force_plot(
        base_value, shap_values[idx], X_test.iloc[idx],
        feature_names=list(X_test.columns),
        matplotlib=True, show=False
    )
    plt.tight_layout()
    path = os.path.join(SHAP_DIR, 'shap_gradient_boosting_force.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Cross-model comparison ────────────────────────────────────

def compare_all(rf_sv, gb_sv, dt_sv, X_test):
    """Bar chart comparing SHAP importance across all 3 models."""
    shap_imp = pd.DataFrame({
        'Feature':           list(X_test.columns),
        'Random Forest':     np.abs(rf_sv).mean(axis=0),
        'Gradient Boosting': np.abs(gb_sv).mean(axis=0),
        'Decision Tree':     np.abs(dt_sv).mean(axis=0),
    }).set_index('Feature')

    print("\nSHAP rankings (1 = most important):")
    print(shap_imp.rank(ascending=False).astype(int)
          .sort_values('Random Forest').to_string())

    shap_norm = shap_imp.div(shap_imp.sum())
    fig, ax   = plt.subplots(figsize=(12, 7))
    shap_norm.sort_values('Random Forest', ascending=True).plot(
        kind='barh', ax=ax,
        color=['#3498db', '#e74c3c', '#27ae60'],
        alpha=0.85, width=0.7
    )
    ax.set_xlabel('Normalised mean |SHAP value|', fontsize=11)
    ax.set_title('SHAP Feature Importance — All 3 Tree Models',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, 'shap_comparison_all_models.png'),
                dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {SHAP_DIR}/shap_comparison_all_models.png")

    # SHAP vs built-in feature_importances_
    rf_path = os.path.join(MODELS_DIR, 'Random_Forest_Tuned.joblib')
    if os.path.exists(rf_path):
        best_rf = joblib.load(rf_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('SHAP vs Built-in Feature Importance (Random Forest)',
                     fontsize=13, fontweight='bold')

        builtin = pd.Series(best_rf.feature_importances_,
                            index=X_test.columns).sort_values()
        axes[0].barh(builtin.index, builtin.values,
                     color='#3498db', alpha=0.85)
        axes[0].set_title('Built-in feature_importances_\n(mean impurity decrease)',
                          fontsize=10)
        axes[0].set_xlabel('Importance')

        shap_mean = pd.Series(np.abs(rf_sv).mean(axis=0),
                              index=X_test.columns).sort_values()
        axes[1].barh(shap_mean.index, shap_mean.values,
                     color='#e74c3c', alpha=0.85)
        axes[1].set_title('SHAP mean |value|\n(contribution per prediction)',
                          fontsize=10)
        axes[1].set_xlabel('Mean |SHAP value|')

        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'shap_vs_builtin.png'),
                    dpi=130, bbox_inches='tight')
        plt.close()
        print(f"Saved: {SHAP_DIR}/shap_vs_builtin.png")


# ── main ──────────────────────────────────────────────────────

def main():
    from train import load_and_prepare

    print("=" * 60)
    print("  SHAP Explainability — Tree Models")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare()

    # Load tuned models saved by tune.py
    def load(name):
        path = os.path.join(MODELS_DIR, f'{name}.joblib')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run tune.py first:\n"
                f"  poetry run python tune.py"
            )
        return joblib.load(path)

    best_rf = load('Random_Forest_Tuned')
    best_gb = load('Gradient_Boosting_Tuned')
    best_dt = load('Decision_Tree_Tuned')

    print("\n[1/4] SHAP for Random Forest...")
    rf_sv, _ = explain_model('Random Forest', best_rf, X_test)

    print("\n[2/4] SHAP for Gradient Boosting...")
    gb_sv, gb_base = explain_model('Gradient Boosting', best_gb, X_test)
    force_plot(best_gb, gb_sv, gb_base, X_test)

    print("\n[3/4] SHAP for Decision Tree...")
    dt_sv, _ = explain_model('Decision Tree', best_dt, X_test)

    print("\n[4/4] Cross-model comparison...")
    compare_all(rf_sv, gb_sv, dt_sv, X_test)

    print("\nSHAP complete!")


if __name__ == '__main__':
    main()