
#matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# 1. Load dataset
df = pd.read_csv("heart.csv")   # make sure heart.csv is in same folder

# 2. Basic inspection
print("\nFIRST 5 ROWS:")
print(df.head())

print("\nSHAPE:")
print(df.shape)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nTARGET VALUE COUNTS:")
print(df["target"].value_counts())

print("\nBASIC STATISTICS:")
print(df.describe())

"""
#checking the below:
df.head()          → see actual data
df.shape           → rows and columns
df.dtypes          → numeric/categorical types
isnull().sum()     → missing values
target counts      → class balance
describe()         → min, max, mean, std
"""


duplicates = df[df.duplicated()]
print(duplicates)

#duplicated() checks COMPLETE row duplicates

print("Before removing duplicates:", df.shape)

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

df = df.drop_duplicates()

print(df.shape)


#EDA
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load data
# ==============================

df = pd.read_csv("heart.csv")

print("Original shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Create output folder
OUTPUT_DIR = "outputs/eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 2. Target distribution
# ==============================

plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"))
plt.close()


# ==============================
# 3. Correlation heatmap
# ==============================

plt.figure(figsize=(14, 10))
corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()


# ==============================
# 4. Histograms
# ==============================

df.hist(figsize=(15, 12))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"))
plt.close()


# ==============================
# 5. Boxplots for numeric features
# ==============================

features = ["age", "chol", "trestbps", "thalach", "oldpeak"]

for feature in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="target", y=feature, data=df)
    plt.title(f"{feature} vs Target")
    plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feature}_vs_target.png"))
    plt.close()


# ==============================
# 6. Correlation with target
# ==============================

target_corr = df.corr()["target"].sort_values(ascending=False)

print("\nCorrelation with target:")
print(target_corr)


# ==============================
# 7. Save cleaned dataset
# ==============================

cleaned_path = "heart_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print("\nEDA completed successfully.")
print(f"Plots saved in: {OUTPUT_DIR}")
print(f"Cleaned dataset saved as: {cleaned_path}")