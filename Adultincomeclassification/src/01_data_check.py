from pathlib import Path
import pandas as pd

# Project root folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = BASE_DIR / "data" / "adult.csv"

print("Dataset path:", DATA_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Dataset info ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Target column check ---")
if "income" in df.columns:
    print(df["income"].value_counts())
else:
    print("No column named 'income' found.")
    print("Available columns:")
    print(df.columns.tolist())