"""
Loads and preprocesses the MNIST dataset.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from config import DATA_DIR, IMG_SIZE, NUM_CLASSES


def load_data():
    """
    Loads MNIST, normalizes pixel values, flattens images,
    and one-hot encodes labels.

    Returns:
        x_train, y_train, x_test, y_test
    """
    # Keras will cache the dataset locally
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Flatten 28x28 -> 784 for MLP
    x_train = x_train.reshape(-1, IMG_SIZE * IMG_SIZE)
    x_test  = x_test.reshape(-1, IMG_SIZE * IMG_SIZE)

    # One-hot encode labels
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test  = to_categorical(y_test,  NUM_CLASSES)

    print(f"[INFO] Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    load_data()