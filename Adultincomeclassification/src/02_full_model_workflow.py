from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import joblib

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "adult.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n--- Dataset loaded ---")
print(df.head())
print("\nShape:", df.shape)


# ---------------------------------------------------------
# 3. Clean hidden missing values
# ---------------------------------------------------------

# In this dataset, missing values are shown as "?"
df = df.replace("?", np.nan)

print("\n--- Missing values after replacing '?' with NaN ---")
print(df.isnull().sum())


# ---------------------------------------------------------
# 4. Separate features and target
# ---------------------------------------------------------

# Target:
# <=50K becomes 0
# >50K becomes 1
df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

X = df.drop("income", axis=1)
y = df["income"]

print("\n--- Target distribution ---")
print(y.value_counts())


# ---------------------------------------------------------
# 5. Identify numerical and categorical columns
# ---------------------------------------------------------

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)


# ---------------------------------------------------------
# 6. Train-test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Train/test split ---")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ---------------------------------------------------------
# 7. Preprocessing
# ---------------------------------------------------------

# Numerical pipeline:
# missing values -> median
# scaling -> StandardScaler

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline:
# missing values -> most frequent
# one-hot encoding

try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", one_hot_encoder)
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Fit only on training data, then transform both train and test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n--- After preprocessing ---")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)

# Save preprocessor
joblib.dump(preprocessor, MODELS_DIR / "adult_income_preprocessor.joblib")


# ---------------------------------------------------------
# 8. Evaluation helper function
# ---------------------------------------------------------

def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Results ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


# ---------------------------------------------------------
# 9. TensorFlow / Keras model
# ---------------------------------------------------------

tf.random.set_seed(42)

input_dim = X_train_processed.shape[1]

tensorflow_model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

tensorflow_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n--- Training TensorFlow/Keras model ---")

tensorflow_history = tensorflow_model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

# Predict probabilities
tf_probabilities = tensorflow_model.predict(X_test_processed).ravel()

# Convert probabilities to 0/1 using 0.5 threshold
tf_predictions = (tf_probabilities >= 0.5).astype(int)

tensorflow_results = evaluate_model(
    "TensorFlow/Keras Neural Network",
    y_test,
    tf_predictions
)

# Save TensorFlow model
tensorflow_model.save(MODELS_DIR / "tensorflow_adult_income_model.keras")


# ---------------------------------------------------------
# 10. PyTorch model
# ---------------------------------------------------------

torch.manual_seed(42)

X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class PyTorchAdultIncomeModel(nn.Module):
    def __init__(self, input_size):
        super(PyTorchAdultIncomeModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


pytorch_model = PyTorchAdultIncomeModel(input_dim)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

print("\n--- Training PyTorch model ---")

epochs = 20

for epoch in range(epochs):
    pytorch_model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        logits = pytorch_model(batch_X)

        # 2. Calculate loss
        loss = loss_function(logits, batch_y)

        # 3. Clear old gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")


# PyTorch evaluation
pytorch_model.eval()

with torch.no_grad():
    test_logits = pytorch_model(X_test_tensor)
    test_probabilities = torch.sigmoid(test_logits)
    pytorch_predictions = (test_probabilities >= 0.5).int().numpy().ravel()

pytorch_results = evaluate_model(
    "PyTorch Neural Network",
    y_test,
    pytorch_predictions
)

# Save PyTorch model
torch.save(
    pytorch_model.state_dict(),
    MODELS_DIR / "pytorch_adult_income_model.pth"
)


# ---------------------------------------------------------
# 11. Compare both models
# ---------------------------------------------------------

comparison = pd.DataFrame([
    {
        "Model": tensorflow_results["model"],
        "Accuracy": tensorflow_results["accuracy"],
        "Precision": tensorflow_results["precision"],
        "Recall": tensorflow_results["recall"],
        "F1 Score": tensorflow_results["f1_score"]
    },
    {
        "Model": pytorch_results["model"],
        "Accuracy": pytorch_results["accuracy"],
        "Precision": pytorch_results["precision"],
        "Recall": pytorch_results["recall"],
        "F1 Score": pytorch_results["f1_score"]
    }
])

print("\n--- Model Comparison ---")
print(comparison)

comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

with open(REPORTS_DIR / "model_comparison.txt", "w") as file:
    file.write("ML Project 02 - Adult Income Classification\n")
    file.write("TensorFlow/Keras vs PyTorch Comparison\n\n")
    file.write(comparison.to_string(index=False))

print("\nWorkflow completed successfully.")
print("Models saved in:", MODELS_DIR)
print("Reports saved in:", REPORTS_DIR)