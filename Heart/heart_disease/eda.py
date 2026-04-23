import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

from config import DATA_PATH, OUTPUTS_DIR


def load_data():
    """Load and deduplicate the dataset."""
    df = pd.read_csv(DATA_PATH)
    df.drop_duplicates(inplace=True)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def print_stats(df):
    """Print basic dataset statistics."""
    print("\n=== First 5 rows ===")
    print(df.head())

    print(f"\n=== Dataset shape ===")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")

    print("\n=== Data types and nulls ===")
    df.info()

    print("\n=== Descriptive statistics ===")
    print(df.describe())

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    print("\n=== Target variable distribution ===")
    print(df['target'].value_counts())
    print("\nProportions:")
    print(df['target'].value_counts(normalize=True).round(4))

    print("\n=== Sex value counts ===")
    print(df["sex"].value_counts())


def plot_target_distribution(df):
    """Plot 1 — target class balance."""
    plt.figure(figsize=(5, 4))
    sns.countplot(x='target', data=df)
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'target_distribution.png')
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved: {path}")


def plot_categorical_vs_target(df):
    """Plot 2 — all categorical features vs target."""
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
    path = os.path.join(OUTPUTS_DIR, 'categorical_vs_target.png')
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df):
    """Plot 3 — correlation heatmap of numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, mask=mask, annot=True,
                cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'correlation_heatmap.png')
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = load_data()
    print_stats(df)
    plot_target_distribution(df)
    plot_categorical_vs_target(df)
    plot_correlation_heatmap(df)

    print("\nEDA complete! All plots saved to outputs/")


if __name__ == '__main__':
    main()