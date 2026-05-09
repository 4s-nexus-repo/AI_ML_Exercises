# %% [markdown]
# # Heart Disease Prediction
# 
# **Author**: Siva
# 
# This notebook builds a machine learning model to predict whether a patient has heart disease based on clinical parameters.
# 
# **Dataset**: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) — a cleaned version of the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.
# 
# **Objective**: Predict the `target` column (1 = heart disease, 0 = no heart disease) using patient features such as age, cholesterol, blood pressure, and more.
# 
# **Libraries used**: pandas, numpy, matplotlib, seaborn, scikit-learn, kagglehub

# %% [markdown]
# ## 1. Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

import kagglehub
import shap
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 2. Download and Load the Dataset

# %%
# Download latest version from Kaggle
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
print("Path to dataset files:", path)

# Load the CSV
csv_file = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_file)
print(f"\nDataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

# %% [markdown]
# ## 3. Data Overview
# 
# Let's understand the structure, data types, and basic statistics of the dataset.

# %%
df.head(10)

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ### Observations - Statistical Summary (`df.describe()`)
# 
# **Dataset Size:**
# - The dataset has **1025 records** across all 14 columns with no missing values (count = 1025 for every column).
# 
# **Age:**
# - Patients range from **29 to 77 years** old, with a mean age of ~54 years.
# - 50% of patients are between 48 and 61 years (IQR), indicating the dataset mostly represents middle-aged to older adults.
# 
# **Sex:**
# - Mean of 0.70 indicates **~70% of patients are male** (sex=1), so the dataset is male-dominated.
# 
# **Chest Pain Type (cp):**
# - Values range from 0 to 3, with a mean of ~0.94, suggesting most patients have lower chest pain types (0 or 1).
# 
# **Resting Blood Pressure (trestbps):**
# - Ranges from **94 to 200 mm Hg**, mean ~132 mm Hg.
# - Normal resting BP is around 120 mm Hg — the mean being slightly above suggests some patients have mild hypertension.
# - The 75th percentile is 140 mm Hg, which is the threshold for Stage 1 hypertension.
# 
# **Cholesterol (chol):**
# - Ranges from **126 to 564 mg/dl**, mean ~246 mg/dl.
# - Desirable cholesterol is below 200 mg/dl — the mean of 246 indicates many patients have **borderline high to high cholesterol**.
# - Max of 564 is an extreme outlier worth noting.
# 
# **Fasting Blood Sugar (fbs):**
# - Mean of 0.15 means only **~15% of patients** have fasting blood sugar > 120 mg/dl.
# - This is a highly imbalanced binary feature.
# 
# **Resting ECG (restecg):**
# - Values 0, 1, 2 with mean ~0.53, indicating roughly equal distribution between normal (0) and abnormal (1) results.
# 
# **Max Heart Rate (thalach):**
# - Ranges from **71 to 202 bpm**, mean ~149 bpm.
# - A wide range (std ~23) suggests significant variation in cardiac fitness across patients.
# - The max of 202 is near the theoretical maximum heart rate for younger patients (220 - age).
# 
# **Exercise Induced Angina (exang):**
# - Mean of 0.34 means **~34% of patients** experience exercise-induced angina.
# 
# **ST Depression - Oldpeak:**
# - Ranges from **0 to 6.2**, mean ~1.07.
# - 50th percentile (median) is 0.8 and 75th percentile is 1.8, indicating the distribution is **right-skewed** — most patients have low ST depression.
# - Higher values indicate more severe exercise-induced cardiac stress.
# 
# **Slope:**
# - Values 0, 1, 2 with mean ~1.39, suggesting most patients fall in slope categories 1 and 2.
# 
# **Major Vessels (ca):**
# - Ranges from 0 to 4, with mean ~0.75.
# - 75th percentile is 1.0, meaning **75% of patients have 0 or 1 major vessels** colored by fluoroscopy.
# 
# **Thalassemia (thal):**
# - Mean of ~2.32 indicates most patients have value 2 (reversible defect) or 3.
# 
# **Target:**
# - Mean of **0.51** indicates a nearly balanced dataset — ~51% have heart disease and ~49% don't. This is ideal for classification without needing resampling techniques.

# %%
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# %%
# Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print(f"Dataset size before removing duplicates: {df.shape[0]}")

df = df.drop_duplicates()
print(f"Dataset size after removing duplicates: {df.shape[0]}")

# %% [markdown]
# ### Observations - Data Overview
# 
# - The dataset contains **14 columns**: 13 features and 1 target variable.
# - All columns are numeric (int64 or float64) — no text/categorical encoding needed.
# - **No missing values** are present, which means we can skip imputation.
# - **723 duplicate rows** found out of 1025 (**70.5% duplicates!**). After removal, only **302 unique rows** remain.
#   - Removing duplicates is critical here. If kept, the model would memorize repeated rows rather than learning patterns, leading to **inflated accuracy** and **data leakage** (same rows appearing in both train and test sets).
# 
# **Column descriptions:**
# 
# | Feature | Description | Type |
# |---------|-------------|------|
# | `age` | Age of the patient in years | Continuous |
# | `sex` | Sex (1 = male, 0 = female) | Binary |
# | `cp` | Chest pain type (4 values: 0, 1, 2, 3) | Categorical |
# | `trestbps` | Resting blood pressure (in mm Hg on admission to the hospital) | Continuous |
# | `chol` | Serum cholesterol in mg/dl | Continuous |
# | `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Binary |
# | `restecg` | Resting electrocardiographic results (values 0, 1, 2) | Categorical |
# | `thalach` | Maximum heart rate achieved | Continuous |
# | `exang` | Exercise induced angina (1 = yes, 0 = no) | Binary |
# | `oldpeak` | ST depression induced by exercise relative to rest | Continuous |
# | `slope` | The slope of the peak exercise ST segment (0, 1, 2) | Categorical |
# | `ca` | Number of major vessels (0-3) colored by fluoroscopy | Categorical |
# | `thal` | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) | Categorical |
# | `target` | Heart disease diagnosis (1 = disease, 0 = no disease) | Binary (Target) |

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)
# 
# ### 4.1 Target Variable Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Count plot
target_counts = df['target'].value_counts()
sns.countplot(x='target', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Heart Disease Distribution')
axes[0].set_xlabel('Target (0 = No Disease, 1 = Disease)')
axes[0].set_ylabel('Count')
for i, count in enumerate(target_counts):
    axes[0].text(i, count + 5, str(count), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(target_counts, labels=['Disease (1)', 'No Disease (0)'],
            autopct='%1.1f%%', colors=sns.color_palette('Set2'), startangle=90)
axes[1].set_title('Heart Disease Proportion')

plt.tight_layout()
plt.show()

print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"\nClass balance ratio: {target_counts.min() / target_counts.max():.2f}")

# %% [markdown]
# ### Observation - Target Distribution
# 
# - **164 patients** have heart disease (target=1) and **138 patients** don't (target=0).
# - The class balance ratio is **0.84**, which is reasonably balanced.
# - This means we don't need to apply oversampling/undersampling techniques like SMOTE.
# - A balanced dataset ensures our model won't be biased toward predicting one class over the other.

# %% [markdown]
# ### 4.2 Distribution of Continuous Features

# %%
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    sns.histplot(data=df, x=col, hue='target', kde=True, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend(title='Target', labels=['Disease', 'No Disease'])

axes[-1].set_visible(False)
plt.suptitle('Distribution of Continuous Features by Target', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation - Continuous Features
# 
# - **age**: Both classes have a similar spread (30-70 years). The Disease group (orange) peaks around age 50-55, while the No Disease group (green) peaks slightly later around 55-60. Age alone is not a strong separator but older patients show a slight lean toward no disease.
# 
# - **trestbps** (resting blood pressure): Both distributions are nearly overlapping with peaks around 120-130 mm Hg. This suggests resting blood pressure has **weak discriminating power** between the two classes.
# 
# - **chol** (cholesterol): Both classes show similar right-skewed distributions centered around 200-275 mg/dl. There is significant overlap, making cholesterol a **poor standalone predictor**. A few extreme outliers exist (>400 mg/dl).
# 
# - **thalach** (max heart rate): This is the **most visually separable** feature. Disease patients (orange) tend to achieve **higher max heart rates** (peak ~150-170 bpm), while no-disease patients (green) have a lower, wider distribution (peak ~130-150 bpm). This is likely a **strong predictor**.
# 
# - **oldpeak** (ST depression): Both distributions are **heavily right-skewed** with most values near 0. However, no-disease patients (green) tend to have **higher oldpeak values** (longer tail extending to 4-6), while disease patients cluster more tightly near 0. This makes oldpeak a useful discriminating feature.

# %% [markdown]
# ### 4.3 Distribution of Categorical Features

# %%
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='target', data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Target')
    axes[i].legend(title='Target', labels=['No Disease', 'Disease'])

plt.suptitle('Categorical Features vs Target', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation - Categorical Features
# 
# - **sex**: The dataset is male-dominated (sex=1). Among females (sex=0), ~75% have heart disease vs ~45% of males — **females in this dataset are at higher risk**. However, the small female sample size means this should be interpreted cautiously.
# 
# - **cp** (chest pain type): Type 0 (typical angina) is the most common and is dominated by **no-disease** patients (~100 vs ~40). Types 1, 2, and 3 show a **higher proportion of disease** cases, especially type 2 where disease patients significantly outnumber no-disease. Chest pain type is a **strong predictor**.
# 
# - **fbs** (fasting blood sugar): Overwhelmingly most patients have fbs=0 (~120 no-disease, ~140 disease). The fbs=1 group is small (~20 each) with nearly equal split. **Very weak discriminating power** — not useful on its own.
# 
# - **restecg**: Values 0 and 1 are common. Both show a slight lean toward disease cases but the distributions are similar. **Limited predictive value**.
# 
# - **exang** (exercise-induced angina): Clear separation here. Among exang=0 (no angina), disease patients dominate (~140 vs ~60). Among exang=1, no-disease patients dominate (~75 vs ~20). **Strong predictor** — absence of exercise angina is associated with disease.
# 
# - **slope**: Slope=1 is most common with no-disease patients leading (~115 vs ~65). Slope=2 shows the **reverse pattern** — disease patients outnumber no-disease. Slope=0 has very few cases. **Good discriminating feature**.
# 
# - **ca** (major vessels): ca=0 has the highest count with disease patients clearly dominating (~120 vs ~50). As ca increases (1, 2, 3, 4), no-disease patients tend to dominate. **Strong predictor** — fewer visible vessels correlates with disease.
# 
# - **thal**: Value 2 (reversible defect) is the most common and shows a strong lean toward **disease** (~95 vs ~40). Value 3 shows the opposite with no-disease dominating. Values 0 and 1 have very few cases. **Strong predictor**.

# %% [markdown]
# ### 4.4 Correlation Heatmap

# %%
plt.figure(figsize=(10, 8))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, square=True)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Top features correlated with target
target_corr = df.corr()['target'].drop('target').sort_values(ascending=False)
print("Feature correlations with target (sorted):\n")
print(target_corr.to_string())

# %% [markdown]
# ### Observation - Correlation Analysis
# 
# **Correlations with Target (last row):**
# - **Strongly positively correlated**: `cp` (0.43), `thalach` (0.42), `slope` (0.34) — higher values of these features are associated with heart disease.
# - **Strongly negatively correlated**: `exang` (-0.44), `oldpeak` (-0.43), `ca` (-0.41), `thal` (-0.34), `sex` (-0.28) — higher values reduce the likelihood of disease.
# - **Weakly correlated**: `fbs` (-0.03), `chol` (-0.08), `restecg` (0.13), `trestbps` (-0.15) — these features have minimal linear relationship with the target and may contribute little to prediction.
# 
# **Notable inter-feature correlations:**
# - `slope` & `oldpeak` have the strongest inter-feature correlation at **-0.58** — as oldpeak increases, slope tends to decrease. This makes clinical sense as both relate to ST segment behavior during exercise.
# - `thalach` & `age` show a correlation of **-0.40** — older patients tend to have lower max heart rates, which is physiologically expected.
# - `cp` & `exang` are correlated at **-0.39** — patients with certain chest pain types are less likely to have exercise-induced angina.
# - `thalach` & `exang` show **-0.38** — patients with exercise-induced angina tend to achieve lower max heart rates.
# - `ca` & `age` have a correlation of **0.30** — older patients tend to have more major vessels visible.
# 
# **Multicollinearity check:**
# - No pair of features exceeds |0.60| correlation, so **multicollinearity is not a concern**. All features can be safely included in the model without redundancy issues.

# %%
fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for i, col in enumerate(continuous_features):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Target')

plt.suptitle('Box Plots of Continuous Features by Target', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# # # **Observation - Box Plots & Outliers**
# Median comparison between classes (No Disease=0 vs Disease=1):
# 
# age: No-disease patients have a higher median (~58) compared to disease patients (~52). The no-disease group also has a wider spread (IQR ~45-63) vs disease (~48-58). One outlier exists around age 30 for no-disease.
# 
# trestbps: Very similar medians (~130) and IQR for both classes — confirms that resting blood pressure is not a strong differentiator. Several outliers exist above 170-200 mm Hg in both groups.
# 
# chol: Nearly identical box plots for both classes — medians around 250 mg/dl with similar IQR. Multiple outliers above 350-500 mg/dl, especially one extreme value near 564 in the no-disease group. Cholesterol does not distinguish the two classes well.
# 
# thalach: The clearest separation among all features. Disease patients have a noticeably higher median (~160 bpm) compared to no-disease (~138 bpm). The entire IQR of disease patients sits higher than no-disease. This confirms thalach as a top predictor.
# 
# oldpeak: No-disease patients have a higher median (~1.5) and much wider spread (IQR ~0.5-2.8) with outliers reaching 5-6. Disease patients cluster near lower values (median ~0.5, IQR ~0-1.2). The clear difference in medians and spread makes this a strong feature.
# 
# **Outlier summary:**
# 
# trestbps: outliers above ~170 mm Hg (both classes)
# chol: extreme outliers above 400 mg/dl (especially no-disease)
# oldpeak: outliers above 4.0 (no-disease class)
# These outliers are medically plausible (real patients can have extreme values), so we retain them rather than removing.
# ## # 4.6 Pair Plot — Feature Relationships
# Let's visualize pairwise relationships between the top 5 most important continuous features, colored by target.

# %%
pair_features = ['age', 'trestbps', 'thalach', 'oldpeak', 'chol', 'target']
g = sns.pairplot(df[pair_features], hue='target', palette='Set2',
                 diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
g.figure.suptitle('Pair Plot of Continuous Features by Target', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# # Observation - Pair Plot
# Diagonal (KDE plots): Shows the distribution of each feature by target — confirms thalach and oldpeak have the best class separation.
# thalach vs oldpeak: The most interesting scatter — disease patients (orange) cluster in the top-left (high heart rate, low oldpeak) while no-disease patients (green) spread toward the bottom-right (lower heart rate, higher oldpeak). These two features together could form a strong decision boundary.
# age vs thalach: Older patients tend to have lower max heart rates (negative correlation). Disease patients generally sit higher on the thalach axis regardless of age.
# trestbps and chol: Heavy overlap in all scatter plots involving these features — confirms they are weak predictors that don't separate classes well in any pairwise combination either.
# Key insight: Pair plots help identify non-linear relationships and feature interactions that correlation heatmaps (which only measure linear relationships) might miss.

# %% [markdown]
# # 4.7 Statistical Tests
# So far, our observations are based on visual inspection. Let's use statistical tests to confirm whether the differences we observed are statistically significant.
# 
# T-test (for continuous features): Tests if the means of two groups are significantly different.
# Chi-square test (for categorical features): Tests if there is a significant association between two categorical variables.
# A p-value < 0.05 means the difference is statistically significant (not due to random chance).

# %%
# T-tests for continuous features
print("=" * 60)
print("T-TESTS FOR CONTINUOUS FEATURES")
print("=" * 60)
print(f"{'Feature':<12} {'t-statistic':>12} {'p-value':>12} {'Significant?':>14}")
print("-" * 60)

disease = df[df['target'] == 1]
no_disease = df[df['target'] == 0]

for col in continuous_features:
    t_stat, p_value = stats.ttest_ind(disease[col], no_disease[col])
    sig = "Yes ***" if p_value < 0.001 else "Yes **" if p_value < 0.01 else "Yes *" if p_value < 0.05 else "No"
    print(f"{col:<12} {t_stat:>12.4f} {p_value:>12.6f} {sig:>14}")

# %%
# Chi-square tests for categorical features
print("=" * 60)
print("CHI-SQUARE TESTS FOR CATEGORICAL FEATURES")
print("=" * 60)
print(f"{'Feature':<12} {'chi2-stat':>12} {'p-value':>12} {'Significant?':>14}")
print("-" * 60)

for col in categorical_features:
    contingency = pd.crosstab(df[col], df['target'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    sig = "Yes ***" if p_value < 0.001 else "Yes **" if p_value < 0.01 else "Yes *" if p_value < 0.05 else "No"
    print(f"{col:<12} {chi2:>12.4f} {p_value:>12.6f} {sig:>14}")

# %% [markdown]
# # Observation - Statistical Tests
# Significance levels: *** p < 0.001 (highly significant), ** p < 0.01, * p < 0.05, No = not significant.
# 
# **What the tests confirm:**
# 
# Features that showed clear visual separation in our EDA (like thalach, oldpeak, cp, exang, ca, thal) should have highly significant p-values (p < 0.001), statistically proving the differences are real.
# Features that showed overlapping distributions (fbs, chol, trestbps) should have higher p-values, confirming they are weak predictors.
# **Why this matters:**
# 
# Visual inspection can be misleading — what looks like a difference in a chart might just be noise. Statistical tests give us mathematical confidence.
# This helps us make informed decisions about feature selection — we could potentially drop non-significant features to simplify the model.

# %% [markdown]
# # 5. Data Preprocessing
# 5.1 Feature and Target Separation

# %%
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# %% [markdown]
# # 5.2 Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\nTest target distribution:\n{y_test.value_counts(normalize=True).round(3)}")

# %% [markdown]
# # Observation - Train-Test Split
# After duplicate removal, we have 302 samples with 13 features each.
# Training set: 241 samples (80%) and Test set: 61 samples (20%).
# stratify=y ensures both sets maintain the same class distribution:
# Training: 54.4% disease, 45.6% no disease
# Test: 54.1% disease, 45.9% no disease
# The proportions are nearly identical, confirming stratification is working correctly.
# With only 302 samples, this is a relatively small dataset — cross-validation will be important later to get reliable performance estimates.

# %% [markdown]
# # 5.3 Feature Scaling
# We scale the continuous features using StandardScaler so that features with larger magnitudes (like chol) don't dominate the model.

# %%
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale only the continuous features
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

print("Scaled training data sample:")
X_train_scaled.head()

# %% [markdown]
# # Observation - Feature Scaling
# We fit the scaler on training data only and then transform both train and test sets. This prevents data leakage — the test set must not influence preprocessing.
# Only continuous features (age, trestbps, chol, thalach, oldpeak) are scaled; categorical/binary features (sex, cp, fbs, etc.) are left as-is since scaling them would distort their meaning.
# After scaling, the continuous features now have values centered around 0 with positive and negative values:
# e.g., age: values like 1.42, -0.47, 0.42 (originally 29-77 years)
# e.g., chol: values like 5.88, -0.89, -0.59 (originally 126-564 mg/dl)
# e.g., trestbps: values like -0.97, 0.76, -0.20 (originally 94-200 mm Hg)
# Why StandardScaler? It transforms each feature to have mean=0 and std=1. This is important for algorithms like Logistic Regression that are sensitive to feature magnitudes. Without scaling, chol (range 126-564) would dominate over oldpeak (range 0-6.2).
# Note that sex, cp, fbs, restecg, exang, slope, ca, thal remain unchanged (original integer values) — these are already on small, meaningful scales.

# %% [markdown]
# 5.4 Does Scaling Actually Help? — Before vs After Comparison
# Let's verify the impact of scaling by comparing model performance with and without scaling. This is especially important for algorithms sensitive to feature magnitudes.

# %%
from sklearn.base import clone

compare_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print(f"{'Model':<25} {'Without Scaling':>17} {'With Scaling':>14} {'Difference':>12}")
print('=' * 70)

for name, model in compare_models.items():
    # Without scaling
    cv_no_scale = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # With scaling
    model_scaled = clone(model)
    cv_scaled = cross_val_score(model_scaled, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    diff = cv_scaled.mean() - cv_no_scale.mean()
    arrow = '\u2191' if diff > 0.005 else ('\u2193' if diff < -0.005 else '\u2192')
    print(f"{name:<25} {cv_no_scale.mean():.4f} \u00b1 {cv_no_scale.std():.3f}   {cv_scaled.mean():.4f} \u00b1 {cv_scaled.std():.3f}   {arrow} {diff:+.4f}")

# %% [markdown]
# Observation - Scaling Impact
# Logistic Regression: scikit-learn's lbfgs solver handles unscaled features reasonably well internally. However, scaling is still best practice because: (a) other solvers like liblinear are more sensitive, (b) the regularization parameter C behaves consistently across features only when they're on the same scale, and (c) convergence is faster with scaled data.
# Decision Tree and Random Forest are scale-invariant — they split on thresholds, so the actual magnitude doesn't matter. Their scores should be nearly identical with or without scaling.
# Takeaway: Even when results look similar on this dataset, scaling is essential for reproducibility and portability — a different solver, dataset, or algorithm (SVM, KNN, Neural Networks) would show significant differences. Always scale for distance-based and gradient-based algorithms.

# %% [markdown]
# 6. Model Building
# We will train six models and compare their performance:
# 
# Logistic Regression — a simple, interpretable baseline
# Decision Tree — captures non-linear relationships
# Random Forest — an ensemble of decision trees for better generalization
# Support Vector Machine (SVM) — finds the optimal separating hyperplane
# K-Nearest Neighbors (KNN) — classifies based on closest training examples
# Gradient Boosting — sequentially builds trees to correct previous errors

# %%
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print()

# %% [markdown]
# Observation - Model Training
# Results summary:
# 
# 
# Model	Test Accuracy	CV Accuracy	CV Std
# Logistic Regression	0.8033	0.8216 (+/- 0.0210)	Low variance
# Decision Tree	0.8033	0.7346 (+/- 0.0523)	High variance
# Random Forest	0.7541	0.8173 (+/- 0.0340)	Moderate variance
# 
# 
# Key observations:
# 
# Logistic Regression performs best overall — highest CV accuracy (0.8216) with the lowest variance (0.0210), indicating it generalizes well and is the most stable model.
# 
# Decision Tree has the same test accuracy as Logistic Regression (0.8033) but a much lower CV accuracy (0.7346) with high variance (0.0523). This gap suggests the Decision Tree is overfitting — it memorizes the training data well but doesn't generalize consistently across folds.
# 
# Random Forest has the lowest test accuracy (0.7541) but a strong CV accuracy (0.8173). The discrepancy between test and CV accuracy could be due to the small test set size (only 61 samples) — a few wrong predictions significantly impact the score.
# 
# Cross-validation is more trustworthy than the single test accuracy here because with only 61 test samples, the test accuracy is sensitive to which specific samples ended up in the test set.
# 
# All models achieve ~75-82% accuracy, which is reasonable for a small dataset of 302 samples with 13 features.

# %% [markdown]
# 6.1 Hyperparameter Tuning (GridSearchCV)
# The models above used default hyperparameters. Let's tune the top models using GridSearchCV to find the best combination of parameters. GridSearchCV tries every combination and picks the one with the best cross-validation score.

# %%
# Define parameter grids for top models
param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
}

tuned_results = {}

for name, config in param_grids.items():
    grid = GridSearchCV(
        config['model'], config['params'],
        cv=5, scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    tuned_results[name] = {
        'model': best_model,
        'best_params': grid.best_params_,
        'best_cv_score': grid.best_score_,
        'test_accuracy': accuracy
    }

    print(f"{'='*50}")
    print(f"{name} (Tuned)")
    print(f"{'='*50}")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"Test Accuracy:    {accuracy:.4f}")
    print()

# %% [markdown]
# Observation - Hyperparameter Tuning
# What is GridSearchCV?
# 
# It exhaustively tries every combination of hyperparameters we specify.
# For each combination, it runs 5-fold cross-validation and picks the one with the highest mean CV accuracy.
# n_jobs=-1 uses all CPU cores for parallel computation.
# What was tuned:
# 
# Logistic Regression: C (regularization strength) and solver (optimization algorithm). A smaller C means stronger regularization (simpler model).
# SVM: C (margin penalty), kernel (rbf vs linear decision boundary), gamma (kernel coefficient).
# Gradient Boosting: n_estimators (number of trees), learning_rate (step size), max_depth (tree depth).
# Key takeaway: Tuning can improve or maintain performance. Compare the tuned CV scores with the default ones from Section 6 to see the impact. Even small improvements in CV accuracy matter on a small dataset.
# 
# 6.2 Beyond GridSearchCV — Smarter Tuning Alternatives
# GridSearchCV tries every combination, which becomes slow as the parameter grid grows. Here are three smarter alternatives:
# 
# 

# %% [markdown]
# 1. RandomizedSearchCV — Random Sampling
# Instead of trying all combinations, it randomly samples a fixed number of combinations from the parameter space.
# 
# GridSearchCV	RandomizedSearchCV
# Strategy	Try ALL combinations	Try N random combinations
# Speed	Slow (exponential with params)	Fast (you control N)
# Guarantee	Finds the best in the grid	May miss the best, but often finds a good one
# When to use	Small grid (<100 combos)	Large grid, continuous params
# 

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter DISTRIBUTIONS (not fixed lists)
param_distributions = {
    'C': uniform(0.01, 100),              # continuous range
    'solver': ['lbfgs', 'liblinear'],     # categorical
    'max_iter': randint(100, 2000)        # random integers
}

random_search = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions,
    n_iter=50,           # only try 50 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_scaled, y_train)
print(f"Best params: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
#Key difference: GridSearchCV uses lists ([0.01, 0.1, 1, 10]), RandomizedSearchCV uses distributions (uniform(0.01, 100)) — so it can explore values you never explicitly listed.

# %% [markdown]
# 2. BayesSearchCV (scikit-optimize) — Bayesian Optimization
# Instead of random guessing, it learns from previous results to decide where to search next. It builds a probabilistic model of the objective function.
# 
# RandomizedSearchCV	BayesSearchCV
# Strategy	Random sampling	Informed sampling (learns from results)
# Efficiency	May waste tries on bad regions	Focuses on promising regions
# Overhead	None	Builds a surrogate model
# When to use	Quick exploration	Expensive models, need best performance
# 

# %%
# pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Define search spaces with types
search_spaces = {
    'C': Real(0.01, 100, prior='log-uniform'),   # log scale — explores 0.01-1 as much as 1-100
    'solver': Categorical(['lbfgs', 'liblinear']),
    'max_iter': Integer(100, 2000)
}

bayes_search = BayesSearchCV(
    LogisticRegression(),
    search_spaces,
    n_iter=30,           # 30 informed tries often beats 100 random tries
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
bayes_search.fit(X_train_scaled, y_train)
print(f"Best params: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_:.4f}")

# %% [markdown]
# Key insight: After each trial, Bayesian optimization asks: "Given what I've seen so far, where should I look next?" — balancing exploration (trying new regions) and exploitation (refining promising regions).

# %% [markdown]
# 3. Optuna — Modern, Flexible, and Powerful
# Optuna is a dedicated hyperparameter optimization framework. It's more flexible than scikit-learn's built-in tools and supports pruning (early stopping of bad trials).
# 
# BayesSearchCV	Optuna
# Integration	scikit-learn API only	Any ML framework (sklearn, PyTorch, XGBoost, etc.)
# Pruning	No	Yes — stops bad trials early, saving time
# Visualization	Limited	Built-in dashboards
# Parallelism	Basic	Distributed across machines
# When to use	Quick sklearn tuning	Serious optimization, deep learning
# 

# %%
# pip install optuna
import optuna

def objective(trial):
    # Optuna SUGGESTS values — you define the search space inline
    C = trial.suggest_float('C', 0.01, 100, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    
    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean()

# Create a study and optimize
study = optuna.create_study(direction='maximize')   # maximize accuracy
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best params: {study.best_params}")
print(f"Best CV score: {study.best_value:.4f}")

# Optuna also provides visualization
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)

# %% [markdown]
# Key advantages of Optuna:
# 
# Pruning: If a trial looks bad after 2 folds, it stops early instead of completing all 5 folds
# Define-by-run: Search space is defined inside the function — supports conditional params (e.g., gamma only when kernel='rbf')
# Visualization: Built-in plots for optimization history, parameter importance, and parallel coordinate plots

# %% [markdown]
# Summary — Which to Use When?
# Method	Best For	Trials Needed	Intelligence
# GridSearchCV	Small grids (<100 combos)	All combos	None (brute force)
# RandomizedSearchCV	Quick baseline, large grids	You choose N	Random
# BayesSearchCV	Moderate tuning, sklearn only	20-50 usually enough	Learns from history
# Optuna	Serious tuning, any framework, deep learning	30-100+	Learns + prunes
# Rule of thumb: Start with RandomizedSearchCV for quick results. Switch to Optuna when you need the best performance or are tuning deep learning models.

# %% [markdown]
# 7. Model Evaluation
# 7.1 Classification Reports

# %%
for name, res in results.items():
    print(f"{'='*50}")
    print(f"Classification Report — {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, res['y_pred'], target_names=['No Disease', 'Disease']))
    print()

# %% [markdown]
# Observation - Classification Reports
# Quick reference:
# 
# Precision: Of all patients predicted as X, how many actually were X?
# Recall: Of all patients who actually were X, how many did the model catch?
# F1-score: Harmonic mean of precision and recall (balances both).
# Support: Number of actual samples in each class (28 no-disease, 33 disease).
# Logistic Regression (Accuracy: 0.80):
# 
# No Disease: precision=0.81, recall=0.75, f1=0.78
# Disease: precision=0.80, recall=0.85, f1=0.82
# Disease recall is 0.85 — it correctly identifies 85% of sick patients. This is good for a medical setting where missing a disease case (false negative) is dangerous.
# Decision Tree (Accuracy: 0.80):
# 
# No Disease: precision=0.79, recall=0.79, f1=0.79
# Disease: precision=0.82, recall=0.82, f1=0.82
# More balanced precision and recall across both classes compared to Logistic Regression. Equal recall for both classes (0.79 and 0.82).
# Random Forest (Accuracy: 0.75):
# 
# Macro avg precision=0.75, recall=0.75, f1=0.75
# Lower across all metrics compared to the other two models.
# The drop in performance may be due to the small dataset — Random Forest typically shines with larger datasets where its ensemble averaging has more impact.
# Overall takeaway:
# 
# In medical diagnosis, recall for the Disease class is most critical — we'd rather have a false alarm (low precision) than miss a sick patient (low recall).
# Logistic Regression has the highest disease recall (0.85), making it the preferred model from a clinical safety perspective.

# %% [markdown]
# 7.2 Confusion Matrices

# %%
n_models = len(results)
n_cols = min(n_models, 3)
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[i].set_title(f'{name}\nAccuracy: {res["accuracy"]:.4f}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

# Hide unused subplots
for j in range(n_models, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Observation - Confusion Matrices
# How to read: Rows = Actual class, Columns = Predicted class.
# 
# Logistic Regression	Decision Tree	Random Forest
# True Negatives (TN) — correctly predicted No Disease	21	22	20
# False Positives (FP) — healthy but predicted Disease	7	6	8
# False Negatives (FN) — sick but predicted No Disease	5	6	7
# True Positives (TP) — correctly predicted Disease	28	27	26
# Total correct	49/61	49/61	46/61
# Key observations:
# 
# Logistic Regression has the fewest false negatives (5) — it misses only 5 sick patients out of 33. This is the most important metric in medical diagnosis. It correctly identifies 28 out of 33 disease cases (recall = 85%).
# 
# Decision Tree is balanced with 6 FP and 6 FN — it makes equal errors in both directions. Slightly better at identifying healthy patients (TN=22) than Logistic Regression.
# 
# Random Forest has the most errors overall (15 wrong out of 61) with the highest false negatives (7) and false positives (8). It misses 7 sick patients, making it the least safe model for clinical use.
# 
# In a medical context, false negatives (bottom-left) are the most dangerous — a missed disease diagnosis could delay treatment. Logistic Regression minimizes this risk with only 5 missed cases.
# 
# All three models have more false positives than false negatives, meaning they lean slightly toward over-diagnosing rather than under-diagnosing — which is the safer bias in healthcare.

# %% [markdown]
# 7.3 ROC Curves

# %%
plt.figure(figsize=(10, 7))

for name, res in results.items():
    y_prob = res['model'].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Observation - ROC Curves
# Why do we need ROC curves?
# 
# Accuracy and confusion matrices evaluate a model at a single threshold (default 0.5 — if predicted probability > 0.5, classify as Disease). But what if 0.5 isn't the best threshold? In medical diagnosis, we might want to lower the threshold to catch more disease cases (higher recall) at the cost of more false alarms.
# 
# The ROC curve answers: "How well does the model separate the two classes across ALL possible thresholds?"
# 
# X-axis (False Positive Rate): How many healthy patients are wrongly flagged as sick?
# Y-axis (True Positive Rate): How many sick patients are correctly identified?
# A perfect model hugs the top-left corner (100% TPR, 0% FPR).
# The dashed diagonal line represents a random classifier (AUC = 0.5) — no better than flipping a coin.
# AUC Scores (Area Under Curve):
# 
# Model	AUC
# Logistic Regression	0.8701
# Random Forest	0.8582
# Decision Tree	0.8019
# Key observations:
# 
# Logistic Regression has the highest AUC (0.8701) — its curve stays closest to the top-left corner, meaning it has the best overall ability to distinguish disease from no-disease across all thresholds. It is the most reliable model.
# 
# Random Forest (AUC = 0.8582) is close behind. Despite having lower test accuracy (0.75), its AUC is strong — meaning with a tuned threshold, it could perform much better. The jagged/step-like shape of its curve is typical for tree-based models.
# 
# Decision Tree has the lowest AUC (0.8019) with a noticeably angular curve. Tree models produce fewer distinct probability levels (since predictions come from leaf nodes), resulting in fewer threshold steps and a less smooth curve. This confirms it generalizes the worst.
# 
# All models have AUC well above 0.5 (random), confirming they are all learning meaningful patterns from the data.
# 
# AUC vs Accuracy: Random Forest had the lowest accuracy (0.75) but the second-highest AUC (0.86). This shows that AUC is a more robust metric — it evaluates the model's ranking ability rather than its performance at a single arbitrary threshold.

# %% [markdown]
# 7.4 Learning Curves
# Learning curves show how model performance changes as the training set size increases. They help diagnose:
# 
# Overfitting: Training score is high but validation score is low (big gap).
# Underfitting: Both scores are low (model is too simple).
# Good fit: Both scores converge at a high value.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

learning_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for i, (name, model) in enumerate(learning_models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    axes[i].plot(train_sizes, train_mean, 'o-', label='Training Score', color='#66c2a5')
    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#66c2a5')
    axes[i].plot(train_sizes, val_mean, 'o-', label='Validation Score', color='#fc8d62')
    axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#fc8d62')

    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Training Set Size')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend(loc='lower right')
    axes[i].set_ylim(0.5, 1.05)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Learning Curves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# Observation - Learning Curves
# How to read learning curves:
# 
# Green line = Training accuracy (how well the model fits the data it has seen)
# Orange line = Validation accuracy (how well it generalizes to unseen data)
# Shaded area = Standard deviation across 5 folds (uncertainty)
# What to look for:
# 
# Decision Tree: Training score stays at ~1.0 (perfect!) while validation score is much lower → classic overfitting. The model memorizes training data instead of learning patterns.
# Logistic Regression: Both curves converge and the gap is small → good fit. The model generalizes well without memorizing.
# Random Forest: Training score near 1.0 but validation gap is smaller than Decision Tree → moderate overfitting, better than a single tree but not as stable as Logistic Regression.
# Key insight: If both curves are still increasing at the right edge, the model could benefit from more data. If they've plateaued, more data won't help — you need a better model or better features.

# %% [markdown]
# 7.5 Precision-Recall Curves
# While ROC curves show TPR vs FPR, Precision-Recall curves focus specifically on the positive class (Disease). This is more informative when we care most about correctly identifying disease cases.

# %%
plt.figure(figsize=(10, 7))

for name, res in results.items():
    y_prob = res['model'].predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap:.4f})')

plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves — Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Observation - Precision-Recall Curves
# ROC vs Precision-Recall — When to use which?
# 
# ROC Curve	Precision-Recall Curve
# Metric	AUC (Area Under ROC)	AP (Average Precision)
# Best for	Balanced datasets	Imbalanced datasets
# Focus	Overall classification ability	Performance on the positive class
# Baseline	Diagonal line (AUC = 0.5)	Horizontal line at class prevalence
# How to read:
# 
# X-axis (Recall): Of all actual disease patients, how many were caught?
# Y-axis (Precision): Of all patients predicted as disease, how many actually had it?
# A perfect model hugs the top-right corner (precision=1, recall=1).
# AP (Average Precision) summarizes the curve into a single number — higher is better.
# Why this matters for healthcare:
# 
# A doctor wants high recall (catch all sick patients) AND high precision (don't alarm too many healthy patients).
# The PR curve shows the trade-off — as you increase recall (catch more cases), precision typically drops (more false alarms).
# The model with the highest AP maintains the best precision across all recall levels.

# %% [markdown]
# 7.6 Model Comparison Summary

# %%
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [res['accuracy'] for res in results.values()],
    'CV Mean Accuracy': [res['cv_mean'] for res in results.values()],
    'CV Std': [res['cv_std'] for res in results.values()],
    'AUC': [
        auc(*roc_curve(y_test, res['model'].predict_proba(X_test_scaled)[:, 1])[:2])
        for res in results.values()
    ]
}).sort_values('AUC', ascending=False)

comparison = comparison.reset_index(drop=True)
print("Model Comparison (sorted by AUC):\n")
comparison

# %%
# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].barh(comparison['Model'], comparison['Test Accuracy'], color=sns.color_palette('Set2'))
axes[0].set_xlabel('Test Accuracy')
axes[0].set_title('Test Accuracy by Model')
axes[0].set_xlim(0.7, 1.0)
for i, v in enumerate(comparison['Test Accuracy']):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

# AUC comparison
axes[1].barh(comparison['Model'], comparison['AUC'], color=sns.color_palette('Set2'))
axes[1].set_xlabel('AUC Score')
axes[1].set_title('AUC Score by Model')
axes[1].set_xlim(0.7, 1.0)
for i, v in enumerate(comparison['AUC']):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# Observation - Model Comparison Summary
# Rank	Model	Test Accuracy	CV Mean Accuracy	CV Std	AUC
# 1	Logistic Regression	0.8033	0.8216	0.0210	0.8701
# 2	Random Forest	0.7541	0.8173	0.0340	0.8582
# 3	Decision Tree	0.8033	0.7346	0.0523	0.8019
# Why do we use Cross-Validation (CV)?
# 
# A single train-test split gives us only one accuracy number based on one specific random split. With a small dataset like ours (302 samples, 61 test samples), this number is unreliable — a few lucky or unlucky samples in the test set can swing accuracy significantly. For example, Decision Tree and Logistic Regression both show 0.8033 test accuracy, making them look equally good — but are they really?
# 
# 5-Fold Cross-Validation solves this by:
# 
# Splitting the training data into 5 equal parts (folds)
# Training on 4 folds and testing on the remaining 1 fold
# Repeating this 5 times, each time using a different fold as the test set
# Averaging the 5 accuracy scores
# This gives us:
# 
# CV Mean: A more reliable estimate of true model performance (tested on all data, not just one split)
# CV Std: How much performance varies across folds — lower = more stable and trustworthy
# Thanks to CV, we can see that Decision Tree (CV: 0.7346, Std: 0.0523) is actually much worse than Logistic Regression (CV: 0.8216, Std: 0.0210) — something the identical test accuracy of 0.8033 completely hid.
# 
# Logistic Regression is the clear winner across all reliable metrics:
# 
# Highest CV Accuracy (0.8216): Most consistent performance across 5 folds.
# Lowest CV Std (0.0210): Most stable — predictions don't fluctuate much depending on which data it sees.
# Highest AUC (0.8701): Best ability to rank disease vs no-disease patients across all thresholds.
# Why not Decision Tree?
# 
# Despite matching Logistic Regression on test accuracy (0.8033), its CV accuracy is much lower (0.7346) with the highest variance (0.0523). This is a classic sign of overfitting — it performs well on one specific test split but poorly across different data splits.
# Why not Random Forest?
# 
# Strong CV accuracy (0.8173) and AUC (0.8582), but the lowest test accuracy (0.7541). With only 302 samples, the ensemble of 100 trees doesn't get enough data to fully leverage its power. Random Forest would likely improve with a larger dataset.
# Key takeaway: For small, structured datasets like this one, simpler models (Logistic Regression) often outperform complex ones (Random Forest, Decision Tree). Simplicity also means better interpretability — important in healthcare where doctors need to understand why a prediction was made.

# %% [markdown]
# 8. Feature Importance
# Let's examine which features are most important for the best-performing model.

# %%
# Random Forest feature importance
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=sns.color_palette('viridis', len(feature_importance)))
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nTop 5 most important features:")
for _, row in feature_importance.tail(5).iloc[::-1].iterrows():
    print(f"  {row['Feature']:12s} : {row['Importance']:.4f}")

# %% [markdown]
# Observation - Feature Importance
# The Random Forest model ranks features by Gini importance — how much each feature contributes to reducing impurity (misclassification) across all decision trees in the ensemble.
# 
# Top 5 features (importance > 0.075):
# 
# Rank	Feature	Importance	Description
# 1	cp	~0.175	Chest pain type — by far the most important feature, nearly double the second
# 2	thalach	~0.130	Max heart rate — strong separator as seen in EDA
# 3	ca	~0.095	Major vessels by fluoroscopy — fewer vessels = more disease
# 4	oldpeak	~0.088	ST depression — higher values in no-disease patients
# 5	thal	~0.080	Thalassemia type — reversible defect linked to disease
# Middle tier (importance 0.04-0.075):
# 
# age (~0.075), exang (~0.068), trestbps (~0.065), chol (~0.060) — moderate contributors.
# Bottom 3 (importance < 0.04):
# 
# sex (~0.035), slope (~0.028), restecg (~0.022), fbs (~0.012)
# fbs is nearly useless at ~0.012 — confirming our earlier finding that fasting blood sugar has almost no predictive value.
# Consistency with EDA findings:
# 
# The top features here (cp, thalach, ca, oldpeak, thal) are the same ones that showed the strongest correlations with the target and the clearest visual separation in our distribution plots and box plots. This validates that our EDA observations were correct.
# chol and trestbps, which showed weak separation in EDA, are ranked in the middle — they contribute some information in combination with other features but are not strong on their own.
# fbs ranked last in both correlation analysis and feature importance — it could potentially be dropped without affecting model performance.

# %% [markdown]
# 9. Conclusion
# Key Findings
# Data Quality: The dataset had no missing values but contained 723 duplicate rows (70.5%). After removal, only 302 unique records remained — the original Cleveland Heart Disease dataset.
# 
# EDA Insights:
# 
# cp (chest pain type) is the single most important predictor of heart disease.
# thalach (max heart rate), ca (major vessels), oldpeak (ST depression), and thal are also strong predictors.
# fbs (fasting blood sugar) and restecg have almost no predictive value.
# The target classes are reasonably balanced (54% disease, 46% no disease).
# Model Performance:
# 
# Logistic Regression is the best model — highest CV accuracy (0.8216), lowest variance (0.0210), highest AUC (0.8701), and highest disease recall (85%).
# Decision Tree overfits (good test accuracy but poor CV accuracy).
# Random Forest underperforms due to the small dataset size.
# Feature Importance: The top features identified by Random Forest are consistent with the EDA findings and known medical risk factors for heart disease.
# 
# Key Lesson: For small, structured datasets, simpler models (Logistic Regression) often outperform complex ones — and they're more interpretable, which matters in healthcare.
# 
# Next Steps
# Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) to improve model performance.
# Try additional models like SVM, KNN, or Gradient Boosting.
# Use SHAP values for more interpretable feature importance.
# Deploy the best model as an API for real-time predictions.

# %% [markdown]
# 10. Predict on New Data
# Let's use our best model (Logistic Regression) to predict heart disease for new patients. We'll create sample patient data and see how the model classifies them.

# %%
# Select the best model
best_model = results['Logistic Regression']['model']

# Create new patient data
# Columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
new_patients = pd.DataFrame({
    'age':      [63,   45,   55],
    'sex':      [1,    0,    1],
    'cp':       [0,    2,    1],
    'trestbps': [145,  110,  130],
    'chol':     [233,  204,  260],
    'fbs':      [1,    0,    0],
    'restecg':  [0,    1,    0],
    'thalach':  [150,  172,  140],
    'exang':    [0,    0,    1],
    'oldpeak':  [2.3,  0.0,  1.4],
    'slope':    [0,    2,    1],
    'ca':       [0,    0,    2],
    'thal':     [1,    2,    3]
})

# Add descriptive labels for display
patient_labels = ['Patient A (63M, chest pain type 0, high BP)',
                  'Patient B (45F, chest pain type 2, normal BP)',
                  'Patient C (55M, chest pain type 1, exercise angina)']

print("New Patient Data:")
new_patients

# %%
# Scale the new data (same scaler used on training data)
new_patients_scaled = new_patients.copy()
new_patients_scaled[continuous_features] = scaler.transform(new_patients[continuous_features])

# Predict class and probability
predictions = best_model.predict(new_patients_scaled)
probabilities = best_model.predict_proba(new_patients_scaled)

# Display results
print("=" * 65)
print("PREDICTION RESULTS (Logistic Regression)")
print("=" * 65)

for i, label in enumerate(patient_labels):
    pred = "Heart Disease" if predictions[i] == 1 else "No Heart Disease"
    prob_no = probabilities[i][0] * 100
    prob_yes = probabilities[i][1] * 100

    print(f"\n{label}")
    print(f"  Prediction : {pred}")
    print(f"  Confidence : {prob_no:.1f}% No Disease | {prob_yes:.1f}% Disease")

# %% [markdown]
# Observation - New Predictions
# How prediction works step-by-step:
# 
# Create a DataFrame with the same 13 columns as the training data (same column names and order).
# Scale the continuous features using the same scaler that was fit on training data — this is critical! Using a new scaler would produce wrong scaled values.
# Call predict() to get the class label (0 or 1).
# Call predict_proba() to get the probability for each class — this tells us how confident the model is.
# Prediction Results:
# 
# Patient	Profile	Prediction	Confidence
# Patient A	63M, cp=0, BP=145, chol=233, thalach=150, oldpeak=2.3	No Heart Disease	76.1% No Disease, 23.9% Disease
# Patient B	45F, cp=2, BP=110, chol=204, thalach=172, oldpeak=0.0	Heart Disease	0.5% No Disease, 99.5% Disease
# Patient C	55M, cp=1, BP=130, chol=260, thalach=140, exang=1, ca=2	No Heart Disease	96.7% No Disease, 3.3% Disease
# Why these predictions make sense (linking back to EDA):
# 
# Patient A → No Disease (76.1%): Despite being 63 with high BP and cholesterol, this patient has cp=0 (typical angina) which our EDA showed is dominated by no-disease patients. The high oldpeak=2.3 also points to no disease. The model is moderately confident.
# 
# Patient B → Disease (99.5%): The model is extremely confident. This patient has cp=2 (strongly linked to disease in EDA), very high thalach=172 (disease patients had higher heart rates), oldpeak=0.0 (disease patients cluster near 0), and ca=0 (fewer vessels = more disease). Every key feature aligns with the disease profile.
# 
# Patient C → No Disease (96.7%): Very high confidence. This patient has exang=1 (exercise angina — linked to no disease), ca=2 (more major vessels = no disease), and thal=3 (no-disease dominant in EDA). These are all strong no-disease indicators.
# 
# Key takeaway: The model doesn't just look at one feature — it combines all 13 features to make a probability-based decision. The confidence score shows how strongly the features align with one class. Patient B is a textbook disease case (99.5%), while Patient A has mixed signals (only 76.1% confident).

# %% [markdown]
# 11. SHAP Values — Model Explainability
# Feature importance (Section 8) tells us which features are important overall, but SHAP (SHapley Additive exPlanations) tells us how each feature influenced each individual prediction.
# 
# SHAP answers: "For this specific patient, which features pushed the prediction toward disease and which pushed it toward no disease?"

# %%
# Create SHAP explainer for Logistic Regression
best_lr = results['Logistic Regression']['model']
explainer = shap.LinearExplainer(best_lr, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# %% [markdown]
# 11.1 SHAP Summary Plot (Global Importance)
# This shows the impact of each feature across all test predictions. Each dot is one patient. The color represents the feature value (red = high, blue = low), and the x-axis shows the SHAP value (positive = pushes toward disease, negative = pushes toward no disease).

# %%
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=True)

# %% [markdown]
# 11.2 SHAP Bar Plot (Mean Absolute Impact)

# %%
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type='bar', show=True)

# %% [markdown]
# 11.3 SHAP Waterfall Plot — Explaining Individual Predictions
# Let's see exactly how the model made its prediction for specific patients from our test set.

# %%
# Explain the first test patient
shap_explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_scaled.iloc[0].values,
    feature_names=list(X.columns)
)

print(f"Actual label: {'Disease' if y_test.iloc[0] == 1 else 'No Disease'}")
print(f"Predicted: {'Disease' if best_lr.predict(X_test_scaled.iloc[[0]])[0] == 1 else 'No Disease'}")
print(f"Prediction probability: {best_lr.predict_proba(X_test_scaled.iloc[[0]])[0]}")
print()
shap.waterfall_plot(shap_explanation, show=True)

# %% [markdown]
# Observation - SHAP Values
# What SHAP tells us that Feature Importance doesn't:
# 
# Aspect	Gini Feature Importance (Section 8)	SHAP Values
# Scope	Global only (one number per feature)	Global AND per-patient
# Direction	Only tells importance magnitude	Tells if feature pushes toward disease or no-disease
# Interaction	Ignores feature interactions	Captures how features work together
# Consistency	Can be biased toward high-cardinality features	Mathematically grounded (game theory)
# How to read the Waterfall Plot:
# 
# Starts from the base value (average model output across all training data).
# Each feature pushes the prediction higher (toward disease, in red) or lower (toward no disease, in blue).
# The final value at the top is the model's output for this specific patient.
# Longer bars = more influential features for this specific prediction.
# Why SHAP matters in healthcare:
# 
# A doctor won't trust a model that just says "Disease" with no explanation.
# SHAP provides a transparent, per-patient explanation: "This patient was predicted as Disease because their chest pain type (cp=2) strongly pushed toward disease, while their age slightly pushed away from it."
# This builds trust and allows doctors to validate the model's reasoning against their clinical knowledge.

# %% [markdown]
# 12. Pipeline — Combining Preprocessing and Model
# So far, we've been manually scaling data before passing it to the model. This is error-prone — if we forget to scale, or use the wrong scaler, predictions will be wrong.
# 
# A Pipeline bundles preprocessing and model into a single object that handles everything automatically.

# %%
# Create a pipeline: StandardScaler → Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Train the pipeline on RAW (unscaled) data
pipeline.fit(X_train, y_train)

# Predict using RAW data — pipeline handles scaling internally
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline Test Accuracy: {pipeline_accuracy:.4f}")

# Cross-validate the pipeline
pipeline_cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Pipeline CV Accuracy:   {pipeline_cv.mean():.4f} (+/- {pipeline_cv.std():.4f})")

print(f"\nPipeline steps:")
for step_name, step_obj in pipeline.steps:
    print(f"  {step_name}: {step_obj.__class__.__name__}")

# %%
# Predict on new patients using pipeline — NO manual scaling needed!
new_patients_pipeline = pd.DataFrame({
    'age':      [50],
    'sex':      [1],
    'cp':       [2],
    'trestbps': [120],
    'chol':     [240],
    'fbs':      [0],
    'restecg':  [1],
    'thalach':  [165],
    'exang':    [0],
    'oldpeak':  [0.5],
    'slope':    [2],
    'ca':       [0],
    'thal':     [2]
})

pred = pipeline.predict(new_patients_pipeline)
prob = pipeline.predict_proba(new_patients_pipeline)

print("Prediction with Pipeline (no manual scaling needed):")
print(f"  Result: {'Heart Disease' if pred[0] == 1 else 'No Heart Disease'}")
print(f"  Confidence: {prob[0][0]*100:.1f}% No Disease | {prob[0][1]*100:.1f}% Disease")

# %% [markdown]
# Observation - Pipeline
# Without Pipeline (manual approach):
# 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# model.fit(X_train_scaled, y_train)
# # For new data: must remember to scale first!
# new_data_scaled = scaler.transform(new_data)
# model.predict(new_data_scaled)
# With Pipeline (automated):
# 
# pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
# pipeline.fit(X_train, y_train)  # scales + trains automatically
# pipeline.predict(new_data)       # scales + predicts automatically
# Benefits:
# 
# No data leakage: Pipeline ensures fit_transform is only called on training data during cross-validation — each fold gets its own scaler fit.
# Simpler code: One object to train, predict, and save instead of managing scaler and model separately.
# Production-ready: When you deploy the model, you ship one pipeline object — no risk of forgetting to scale or using the wrong scaler.
# Works with GridSearchCV: You can tune both preprocessing and model parameters together.

# %% [markdown]
# 13. Model Saving — Persist the Trained Model
# Training a model takes time and compute. We don't want to retrain every time we need a prediction. Saving the model lets us:
# 
# Load it later for predictions without retraining
# Deploy it to a web server or API
# Share it with others

# %%
# Save the pipeline (includes scaler + model)
model_path = os.path.join(os.path.dirname(csv_file), '..', 'heart_disease_pipeline.joblib')
model_path = os.path.abspath(model_path)

joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")
print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")

# %%
# Load the saved model and verify it works
loaded_pipeline = joblib.load(model_path)

# Predict with the loaded model
loaded_pred = loaded_pipeline.predict(new_patients_pipeline)
loaded_prob = loaded_pipeline.predict_proba(new_patients_pipeline)

print("Prediction with LOADED model (verifying save/load works):")
print(f"  Result: {'Heart Disease' if loaded_pred[0] == 1 else 'No Heart Disease'}")
print(f"  Confidence: {loaded_prob[0][0]*100:.1f}% No Disease | {loaded_prob[0][1]*100:.1f}% Disease")
print(f"\n  Matches original prediction: {np.array_equal(pred, loaded_pred)}")

# %% [markdown]
# Observation - Model Saving
# Why joblib over pickle?
# 
# joblib is optimized for objects containing large numpy arrays (like model weights and scaler parameters), making it faster and producing smaller files.
# pickle works too, but joblib is the recommended choice for scikit-learn models.
# What gets saved?
# 
# The entire Pipeline object, including:
# The fitted StandardScaler (with learned mean and std from training data)
# The trained LogisticRegression model (with learned coefficients)
# This means you can load the file and immediately make predictions on raw, unscaled data.

# %% [markdown]
# How to use the saved model in another script:

# %%
import joblib
import pandas as pd

joblib.dump(pipeline, 'heart_disease_pipeline.joblib')


# %%

# Load the model
pipeline = joblib.load('heart_disease_pipeline.joblib')

# Create new patient data (raw values, no scaling needed)
patient = pd.DataFrame({
    'age': [55],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [1.2],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})


# Predict
result = pipeline.predict(patient)
probability = pipeline.predict_proba(patient)

# %% [markdown]
# Important considerations:
# 
# The saved model is tied to the scikit-learn version used to create it. Loading with a different version may fail.
# Never load .joblib files from untrusted sources — they can execute arbitrary code (same as pickle).
# In production, consider versioning your models with timestamps or git hashes.


