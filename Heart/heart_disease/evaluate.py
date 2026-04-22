import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from config import OUTPUTS_DIR


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model — print metrics and save confusion matrix."""
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
    print(classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Disease"]
    ))

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()

    filename = model_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    path = os.path.join(OUTPUTS_DIR, f'cm_{filename}.png')
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved: {path}")

    return {
        'Model':     model_name,
        'Accuracy':  acc,
        'Precision': prec,
        'Recall':    rec,
        'F1 Score':  f1
    }


def save_comparison_chart(results):
    """Save a grouped bar chart comparing all models."""
    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)

    plt.figure(figsize=(12, 5))
    x       = range(len(results_df))
    width   = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors  = ['#3498db', '#27ae60', '#e67e22', '#e74c3c']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        plt.bar([xi + i * width for xi in x],
                results_df[metric], width=width,
                label=metric, color=color, alpha=0.85)

    plt.xticks([xi + width * 1.5 for xi in x],
               results_df['Model'], rotation=15, ha='right')
    plt.ylim(0.5, 1.05)
    plt.ylabel('Score')
    plt.title('Model Comparison — All Metrics')
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUTS_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"\nSaved: {path}")

    print("\n=== Final Model Comparison ===")
    print(results_df.to_string(index=False))


def main():
    """
    evaluate.py does not run standalone —
    evaluate_model() is imported and called by train.py.
    This main() just confirms the functions loaded correctly.
    """
    print("evaluate.py loaded successfully ✔")
    print("Available functions:")
    print("  evaluate_model(model, X_test, y_test, model_name)")
    print("  save_comparison_chart(results)")


if __name__ == '__main__':
    main()