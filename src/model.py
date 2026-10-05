"""
Defines the Multilayer Perceptron (MLP) architecture.
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from config import IMG_SIZE, NUM_CLASSES, HIDDEN_LAYERS, DROPOUT_RATE, LEARNING_RATE


def build_mlp():
    """
    Builds a fully connected feed-forward neural network.

    Architecture:
        Input(784) -> Dense(512, relu) -> Dropout
                   -> Dense(256, relu) -> Dropout
                   -> Dense(10, softmax)
    """
    model = Sequential(name="MLP_MNIST")
    model.add(Input(shape=(IMG_SIZE * IMG_SIZE,)))

    for units in HIDDEN_LAYERS:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


if __name__ == "__main__":
    build_mlp()