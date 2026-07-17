"""
feature_extraction.py

Step 2 of the "own trained model" PDF AI bot pipeline.

Converts (question, chunk_text, label) rows into small numeric feature
vectors. NO pretrained embeddings, NO API calls - just hand-rolled
TF-IDF built from the training corpus itself, plus a few overlap stats.

Feature vector per pair (8 numbers):
    1. cosine similarity between TF-IDF(question) and TF-IDF(chunk)
    2. raw word-overlap count
    3. raw word-overlap ratio (overlap / len(question words))
    4. rare-word overlap score (sum of IDF of overlapping words)
    5. length ratio (len(chunk) / len(question))
    6. max single-word IDF among overlapping words
    7. number of overlapping words that are capitalized in the chunk
       (proxy for proper-noun / key-term matches)
    8. question_marker_score - how "exercise-like" the chunk looks
       (ends with '?', contains multiple '?', or starts with a lettered
       sub-part like "a." / "b)") - used to DOWNWEIGHT textbook exercise
       prompts, which are questions, not answers
"""

import re
import csv
import json
import math
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "with", "this", "that", "it", "as", "by", "be",
    "will", "can", "has", "have", "had", "at", "from", "which", "these",
    "those", "also", "such", "into", "than", "then", "what", "does",
    "do", "say", "about",
}


def tokenize(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


# ---------------------------------------------------------------------
# Build IDF table from the training corpus (this is the "fit" step -
# same role as sklearn's TfidfVectorizer.fit, but hand-written)
# ---------------------------------------------------------------------
class IDFModel:
    def __init__(self):
        self.idf = {}
        self.n_docs = 0

    def fit(self, documents):
        """documents: list of raw text strings (chunks + questions)."""
        self.n_docs = len(documents)
        doc_freq = Counter()
        for doc in documents:
            unique_words = set(tokenize(doc))
            for w in unique_words:
                doc_freq[w] += 1

        for w, df in doc_freq.items():
            # smoothed idf, always >= 0
            self.idf[w] = math.log((1 + self.n_docs) / (1 + df)) + 1

    def word_idf(self, word):
        return self.idf.get(word, 1.0)  # unseen words get neutral weight

    def tfidf_vector(self, text):
        words = tokenize(text)
        tf = Counter(words)
        # log-scaled TF (1 + log(count)) instead of raw count - stops
        # chunks that repeat a word several times (e.g. multi-part
        # exercises restating "probability" in each sub-question) from
        # getting an inflated similarity score purely from repetition
        return {w: (1 + math.log(tf[w])) * self.word_idf(w) for w in tf}


def cosine_sim(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_features(question, chunk_text, idf_model):
    q_words = tokenize(question)
    c_words = tokenize(chunk_text)
    q_set, c_set = set(q_words), set(c_words)
    overlap = q_set & c_set

    q_vec = idf_model.tfidf_vector(question)
    c_vec = idf_model.tfidf_vector(chunk_text)
    cos = cosine_sim(q_vec, c_vec)

    overlap_count = len(overlap)
    overlap_ratio = overlap_count / len(q_set) if q_set else 0.0
    rare_overlap_score = sum(idf_model.word_idf(w) for w in overlap)
    max_overlap_idf = max((idf_model.word_idf(w) for w in overlap), default=0.0)
    length_ratio = len(c_words) / len(q_words) if q_words else 0.0

    # capitalization proxy: how many overlap words appear capitalized
    # in the original (untokenized) chunk text
    cap_matches = 0
    for w in overlap:
        if re.search(r"\b" + re.escape(w) + r"\b", chunk_text, flags=re.IGNORECASE):
            found = re.search(r"\b" + re.escape(w) + r"\b", chunk_text, flags=re.IGNORECASE)
            if found and found.group(0)[0].isupper():
                cap_matches += 1

    # question-likeness: exercise prompts end in '?', often contain
    # multiple '?' (multi-part questions), or start with a lettered
    # sub-part like "a." / "b)" - all signal "this is a prompt, not
    # an answer" and should be penalized, not preferred
    question_marker_score = 0.0
    stripped = chunk_text.strip()
    if stripped.endswith("?"):
        question_marker_score += 1.0
    question_marker_score += 0.5 * chunk_text.count("?")
    if re.match(r"^[a-e][\.\)]\s", stripped):
        question_marker_score += 1.0

    # definitional-phrasing score: textbook definitions have a fairly
    # recognizable structural signature ("X is defined as...", "we
    # define X as...", "refers to...", "is called...", "the objective
    # of X is to..."). Reward chunks that both overlap with the
    # question AND contain this pattern near an overlapping word.
    definitional_score = 0.0
    definitional_patterns = [
        r"\bis defined as\b", r"\bwe define\b", r"\bdefined to be\b",
        r"\brefers to\b", r"\bis called\b", r"\bis known as\b",
        r"\bthe objective of\b", r"\bis a measure of\b",
    ]
    lowered_chunk = chunk_text.lower()
    for pattern in definitional_patterns:
        if re.search(pattern, lowered_chunk):
            definitional_score += 1.0
    # only counts if the chunk also actually overlaps with the question -
    # otherwise a definitional sentence about an unrelated topic would
    # score high for no reason
    if overlap_count == 0:
        definitional_score = 0.0

    return [
        cos,
        overlap_count,
        overlap_ratio,
        rare_overlap_score,
        length_ratio,
        max_overlap_idf,
        cap_matches,
        question_marker_score,
        definitional_score,
    ]


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


def save_idf_model(idf_model, path="idf_model.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"idf": idf_model.idf, "n_docs": idf_model.n_docs}, f)


def load_idf_model(path="idf_model.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = IDFModel()
    model.idf = data["idf"]
    model.n_docs = data["n_docs"]
    return model


def build_feature_dataset(pairs_csv, out_csv="features.csv"):
    rows = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # fit IDF on every question + chunk seen in training data
    corpus = []
    for r in rows:
        corpus.append(r["question"])
        corpus.append(r["chunk_text"])

    idf_model = IDFModel()
    idf_model.fit(corpus)
    save_idf_model(idf_model, "idf_model.json")

    out_rows = []
    for r in rows:
        feats = extract_features(r["question"], r["chunk_text"], idf_model)
        out_rows.append(feats + [int(r["label"])])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} feature rows to {out_csv}")
    return idf_model


if __name__ == "__main__":
    import sys
    pairs_csv = sys.argv[1] if len(sys.argv) > 1 else "../training_pairs.csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "features.csv"
    build_feature_dataset(pairs_csv, out_csv)