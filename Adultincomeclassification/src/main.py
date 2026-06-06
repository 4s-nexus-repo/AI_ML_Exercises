import pandas as pd

from data_loader import load_data, clean_missing_values
from preprocessing import (
    split_features_and_target,
    identify_column_types,
    split_train_test,
    preprocess_data
)
from tensorflow_model import train_tensorflow_model, predict_tensorflow_model
from pytorch_model import (
    prepare_pytorch_data,
    train_pytorch_model,
    predict_pytorch_model
)
from evaluate import evaluate_model


def main():
    print("\n--- ML Project 02: Adult Income Classification ---")

    # 1. Load and clean data
    df = load_data()
    df = clean_missing_values(df)

    print("\nDataset loaded and cleaned.")
    print("Shape:", df.shape)

    # 2. Split features and target
    X, y = split_features_and_target(df)

    print("\nFeatures and target created.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 3. Identify column types
    numerical_cols, categorical_cols = identify_column_types(X)

    print("\nNumerical columns:")
    print(numerical_cols)

    print("\nCategorical columns:")
    print(categorical_cols)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("\nTrain/test split completed.")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 5. Preprocessing
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train,
        X_test,
        numerical_cols,
        categorical_cols
    )

    input_dim = X_train_processed.shape[1]

    print("\nPreprocessing completed.")
    print("X_train_processed:", X_train_processed.shape)
    print("X_test_processed:", X_test_processed.shape)
    print("Input dimension:", input_dim)

    # 6. TensorFlow/Keras model
    print("\n--- Training TensorFlow/Keras model ---")

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

    # 7. PyTorch model
    print("\n--- Training PyTorch model ---")

    train_loader, X_test_tensor, y_test_tensor = prepare_pytorch_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )

    pytorch_model, pytorch_train_losses = train_pytorch_model(
        train_loader,
        input_dim,
        epochs=20
    )

    pytorch_predictions, pytorch_probabilities = predict_pytorch_model(
        pytorch_model,
        X_test_tensor
    )

    pytorch_results = evaluate_model(
        "PyTorch Neural Network",
        y_test,
        pytorch_predictions
    )

    # 8. Compare results
    comparison_df = pd.DataFrame([
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

    print("\n--- Final Model Comparison ---")
    print(comparison_df)

    print("\nModular workflow completed successfully.")


if __name__ == "__main__":
    main()