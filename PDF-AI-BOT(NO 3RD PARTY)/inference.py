"""
inference.py

Step 4 (final) of the "own trained model" PDF AI bot pipeline.

Loads:
  - idf_model.json      (vocabulary weights learned from training corpus)
  - model_weights.json  (logistic regression weights trained from scratch)

Given a user's question and a PDF, this:
  1. Extracts chunks from the PDF (same chunker as generate_training_data.py)
  2. Computes the 7 features for (question, chunk) for every chunk
  3. Normalizes features using the training mean/std
  4. Scores every chunk with the trained logistic regression
  5. Returns the highest-scoring chunk as the answer

No API calls, no pretrained embeddings/LLM anywhere in this file.
"""

import sys
import json
import numpy as np

sys.path.insert(0, "..")
from generate_training_data import extract_chunks  # noqa: E402
from feature_extraction import extract_features, load_idf_model, FEATURE_NAMES  # noqa: E402


def load_model(path="model_weights.json"):
    with open(path) as f:
        return json.load(f)


def score_chunk(question, chunk, idf_model, model):
    from feature_extraction import extract_features as _ef
    raw_feats = _ef(question, chunk, idf_model)
    feats = np.array(raw_feats)
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    feats_norm = (feats - mean) / std

    weights = np.array(model["weights"])
    bias = model["bias"]
    z = feats_norm @ weights + bias
    prob = 1 / (1 + np.exp(-z))

    # Deterministic adjustments on top of the learned score:
    #  - exercise-style chunks (question_marker_score) get pushed down,
    #    since single/few-word queries make the learned weight alone
    #    too weak to overcome their repeated-keyword advantage.
    #  - definitional-phrasing chunks (definitional_score) get boosted,
    #    since a true definition should outrank a passing mention.
    # Both are rule-based adjustments layered on the trained model's
    # output, not a replacement for it.
    question_marker_score = raw_feats[FEATURE_NAMES.index("question_marker_score")]
    definitional_score = raw_feats[FEATURE_NAMES.index("definitional_score")]
    penalty = 0.35 * question_marker_score
    boost = 0.15 * definitional_score
    adjusted = float(prob) - penalty + boost
    return adjusted


def answer_question(question, pdf_path, idf_model, model, top_k=1):
    chunks = extract_chunks(pdf_path)
    scored = [(score_chunk(question, c, idf_model, model), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def main(pdf_path, question):
    idf_model = load_idf_model("idf_model.json")
    model = load_model("model_weights.json")

    results = answer_question(question, pdf_path, idf_model, model, top_k=3)

    print(f"Question: {question}\n")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"#{rank}  score={score:.3f}")
        print(f"    {chunk}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <path_to_pdf> \"<question>\"")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])