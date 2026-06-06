import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_tensorflow_model(input_dim):
    """
    Build a TensorFlow/Keras neural network for binary classification.
    """

    tf.random.set_seed(42)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_tensorflow_model(X_train_processed, y_train, input_dim):
    """
    Train the TensorFlow/Keras model.
    """

    model = build_tensorflow_model(input_dim)

    history = model.fit(
        X_train_processed,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        verbose=1
    )

    return model, history


def predict_tensorflow_model(model, X_test_processed):
    """
    Predict class labels using the trained TensorFlow/Keras model.
    """

    probabilities = model.predict(X_test_processed).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    return predictions, probabilities