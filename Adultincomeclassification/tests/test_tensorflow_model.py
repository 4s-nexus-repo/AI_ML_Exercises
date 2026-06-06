from data_loader import load_data, clean_missing_values
from preprocessing import (
    split_features_and_target,
    identify_column_types,
    split_train_test,
    preprocess_data
)
from tensorflow_model import train_tensorflow_model, predict_tensorflow_model
from evaluate import evaluate_model


df = load_data()
df = clean_missing_values(df)

X, y = split_features_and_target(df)

numerical_cols, categorical_cols = identify_column_types(X)

X_train, X_test, y_train, y_test = split_train_test(X, y)

X_train_processed, X_test_processed, preprocessor = preprocess_data(
    X_train,
    X_test,
    numerical_cols,
    categorical_cols
)

input_dim = X_train_processed.shape[1]

tensorflow_model, tensorflow_history = train_tensorflow_model(
    X_train_processed,
    y_train,
    input_dim
)

tf_predictions, tf_probabilities = predict_tensorflow_model(
    tensorflow_model,
    X_test_processed
)

tensorflow_results = evaluate_model(
    "TensorFlow/Keras Neural Network",
    y_test,
    tf_predictions
)
