from data_loader import load_data, clean_missing_values
from preprocessing import (
    split_features_and_target,
    identify_column_types,
    split_train_test,
    preprocess_data
)


df = load_data()
df = clean_missing_values(df)

X, y = split_features_and_target(df)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

numerical_cols, categorical_cols = identify_column_types(X)

print("\nNumerical columns:")
print(numerical_cols)

print("\nCategorical columns:")
print(categorical_cols)

X_train, X_test, y_train, y_test = split_train_test(X, y)

print("\nTrain/test split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

X_train_processed, X_test_processed, preprocessor = preprocess_data(
    X_train,
    X_test,
    numerical_cols,
    categorical_cols
)

print("\nAfter preprocessing:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)