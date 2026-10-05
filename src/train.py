"""
Trains the MLP on MNIST and saves the model + training curves.
"""
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Allow `python train.py` from src/ to resolve the project config.py first.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MODEL_PATH, OUTPUT_DIR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, RANDOM_SEED
from src.data_loader import load_data
from src.model import build_mlp


def plot_history(history):
    """Saves accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def train():
    # 1. Load data
    x_train, y_train, x_test, y_test = load_data()

    # 2. Build model
    model = build_mlp()

    # 3. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # 4. Train
    history = model.fit(
        x_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # 5. Save curves
    plot_history(history)

    # 6. Final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[RESULT] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    return model, history


if __name__ == "__main__":
    train()