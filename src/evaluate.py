"""
Evaluates the saved model: accuracy, classification report, confusion matrix.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from config import MODEL_PATH, OUTPUT_DIR, NUM_CLASSES
from src.data_loader import load_data


def evaluate():
    # 1. Load data & model
    _, _, x_test, y_test = load_data()
    model = load_model(MODEL_PATH)

    # 2. Predict
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 3. Metrics
    acc = np.mean(y_pred == y_true)
    print(f"[RESULT] Test Accuracy: {acc:.4f}\n")
    print("[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, digits=4))

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.title("Confusion Matrix - MNIST MLP")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    evaluate()