import os

# Goes up from current file location to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH   = os.path.join(BASE_DIR, 'heart.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR  = os.path.join(BASE_DIR, 'outputs', 'models')

# Features
CONTINUOUS_FEATURES  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
TARGET_COLUMN        = 'target'

# Model settings
TEST_SIZE    = 0.2
RANDOM_STATE = 42


def main():
    print("=== Project Configuration ===")
    print(f"Base directory  : {BASE_DIR}")
    print(f"Data path       : {DATA_PATH}")
    print(f"Outputs dir     : {OUTPUTS_DIR}")
    print(f"Models dir      : {MODELS_DIR}")
    print(f"Continuous feats: {CONTINUOUS_FEATURES}")
    print(f"Categorical feats: {CATEGORICAL_FEATURES}")
    print(f"Target column   : {TARGET_COLUMN}")
    print(f"Test size       : {TEST_SIZE}")
    print(f"Random state    : {RANDOM_STATE}")
    print()
    # Check heart.csv exists
    if os.path.exists(DATA_PATH):
        print(f"heart.csv found ✔")
    else:
        print(f"heart.csv NOT found ✘ — expected at {DATA_PATH}")


if __name__ == '__main__':
    main()