"""
Central configuration: paths and hyperparameters.
"""
import os

# ---------- Paths ----------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")

MODEL_PATH    = os.path.join(MODEL_DIR, "mlp_mnist.h5")

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- Hyperparameters ----------
IMG_SIZE      = 28          # MNIST images are 28x28
NUM_CLASSES   = 10          # Digits 0-9
HIDDEN_LAYERS = [512, 256]  # Neurons in each hidden layer
DROPOUT_RATE  = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE    = 128
EPOCHS        = 15
VALIDATION_SPLIT = 0.1
RANDOM_SEED   = 42