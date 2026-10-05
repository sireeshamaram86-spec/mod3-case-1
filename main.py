"""
End-to-end pipeline:
    1. Train the MLP on MNIST
    2. Evaluate on the test set
"""
from src.train import train
from src.evaluate import evaluate


def main():
    print("=" * 60)
    print(" STEP 1: TRAINING ")
    print("=" * 60)
    train()

    print("\n" + "=" * 60)
    print(" STEP 2: EVALUATION ")
    print("=" * 60)
    evaluate()


if __name__ == "__main__":
    main()