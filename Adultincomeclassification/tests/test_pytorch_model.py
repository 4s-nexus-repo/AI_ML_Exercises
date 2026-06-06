from data_loader import load_data, clean_missing_values
from preprocessing import (
    split_features_and_target,
    identify_column_types,
    split_train_test,
    preprocess_data
)
from pytorch_model import (
    prepare_pytorch_data,
    train_pytorch_model,
    predict_pytorch_model
)
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