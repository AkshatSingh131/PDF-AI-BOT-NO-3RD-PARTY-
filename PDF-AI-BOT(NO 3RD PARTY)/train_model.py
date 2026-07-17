"""
train_model.py

Step 3 of the "own trained model" PDF AI bot pipeline.

Trains a logistic regression classifier FROM SCRATCH (gradient descent
written by hand with numpy - no sklearn, no pretrained anything) on the
feature vectors produced by feature_extraction.py.

Input:  features.csv  (cols: 7 features + label)
Output: model_weights.json  (weights + bias + feature means/stds for
        normalization, so inference.py can reload the exact same model)
"""

import csv
import json
import numpy as np

FEATURE_NAMES = [
    "cosine_sim",
    "overlap_count",
    "overlap_ratio",
    "rare_overlap_score",
    "length_ratio",
    "max_overlap_idf",
    "cap_matches",
    "question_marker_score",
    "definitional_score",
]


def load_features(path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[name]) for name in FEATURE_NAMES])
            y.append(int(r["label"]))
    return np.array(X), np.array(y)


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.1, epochs=2000, l2=0.01):
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0

    for epoch in range(epochs):
        z = X @ weights + bias
        preds = sigmoid(z)
        error = preds - y

        grad_w = (X.T @ error) / n + l2 * weights
        grad_b = error.mean()

        weights -= lr * grad_w
        bias -= lr * grad_b

        if epoch % 500 == 0 or epoch == epochs - 1:
            eps = 1e-9
            loss = -np.mean(
                y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps)
            )
            acc = ((preds >= 0.5).astype(int) == y).mean()
            print(f"epoch {epoch:5d}  loss={loss:.4f}  train_acc={acc:.3f}")

    return weights, bias


def main(features_csv="features.csv", out_path="model_weights.json"):
    X, y = load_features(features_csv)
    print(f"Loaded {len(y)} samples, {X.shape[1]} features")

    X_norm, mean, std = normalize(X)
    weights, bias = train_logistic_regression(X_norm, y)

    model = {
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Saved trained model to {out_path}")
    print("Learned weights:")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  {name:20s} {w:+.4f}")


if __name__ == "__main__":
    import sys
    feats = sys.argv[1] if len(sys.argv) > 1 else "features.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "model_weights.json"
    main(feats, out)