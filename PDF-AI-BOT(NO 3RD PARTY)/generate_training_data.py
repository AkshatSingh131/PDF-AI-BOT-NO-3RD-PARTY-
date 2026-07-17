"""
generate_training_data.py

Step 1 of the "own trained model" PDF AI bot pipeline.

Takes a PDF, splits it into chunks, and auto-generates pseudo
(question, chunk_text, label) pairs WITHOUT any third-party API or
pretrained LLM. Everything here is rule-based / heuristic.

Positive pairs (label=1): question generated FROM a chunk, paired with
                           that same chunk.
Negative pairs (label=0): same question, paired with a few OTHER
                           random chunks.

Output: training_pairs.csv  with columns: question, chunk_text, label
"""

import re
import csv
import random
import fitz  # PyMuPDF

random.seed(42)


# ---------------------------------------------------------------------
# 1. PDF -> chunks
# ---------------------------------------------------------------------
def extract_chunks(pdf_path, min_words=15, max_words=60):
    """
    Extracts text from a PDF and splits it into chunks of roughly
    min_words-max_words each.

    NOTE: real PDFs frequently lose paragraph/blank-line structure during
    text extraction (a well-known PyMuPDF/PDF quirk - paragraph breaks
    are a visual layout thing, not a semantic marker in the file). So
    instead of relying on blank lines, we pull the full text, split it
    into sentences, then greedily group sentences into chunks by word
    count. This works whether or not blank lines survive extraction.
    """
    doc = fitz.open(pdf_path)
    full_text_parts = []

    for page in doc:
        text = page.get_text("text")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            full_text_parts.append(text)

    full_text = " ".join(full_text_parts)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?]) +", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Greedily group sentences into chunks
    chunks = []
    buffer = ""
    for sentence in sentences:
        candidate = (buffer + " " + sentence).strip() if buffer else sentence
        if len(candidate.split()) <= max_words:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence

        if len(buffer.split()) >= min_words and len(buffer.split()) >= max_words * 0.6:
            chunks.append(buffer)
            buffer = ""

    if buffer and len(buffer.split()) >= 3:
        chunks.append(buffer)

    return chunks


# ---------------------------------------------------------------------
# 2. chunk -> pseudo question  (rule-based, no API/model)
# ---------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "for", "and", "or", "with", "this", "that", "it", "as", "by", "be",
    "will", "can", "has", "have", "had", "at", "from", "which", "these",
    "those", "also", "such", "into", "than", "then",
    # interrogative words - must be excluded or exercise-style chunks
    # ("What is the probability...") get mis-parsed as if the question
    # word itself is the subject, producing tautological pseudo-questions
    "what", "why", "how", "when", "where", "who", "whom", "whose",
}


def first_sentence(chunk_text):
    match = re.split(r"(?<=[.!?]) ", chunk_text, maxsplit=1)
    return match[0]


def pick_subject(sentence):
    """
    Very simple heuristic subject picker: first run of capitalized
    words, else the first noun-ish (longest) word outside stopwords.
    """
    words = sentence.split()

    # Try to find a capitalized phrase (likely a proper noun / key term)
    cap_phrase = []
    for w in words:
        clean = re.sub(r"[^A-Za-z0-9]", "", w)
        if clean and clean[0].isupper() and clean.lower() not in STOPWORDS:
            cap_phrase.append(clean)
        elif cap_phrase:
            break
    if cap_phrase:
        return " ".join(cap_phrase)

    # Fallback: longest non-stopword word
    candidates = [
        re.sub(r"[^A-Za-z0-9]", "", w)
        for w in words
        if re.sub(r"[^A-Za-z0-9]", "", w).lower() not in STOPWORDS
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        return None
    return max(candidates, key=len)


def generate_question(chunk_text):
    """
    Templates:
      "X is Y ..."       -> "What is X?"
      "X can Y ..."      -> "What can X do?"
      default            -> "What does the document say about X?"
    """
    sentence = first_sentence(chunk_text)
    subject = pick_subject(sentence)

    if not subject:
        return None

    lowered = sentence.lower()
    if re.search(r"\b" + re.escape(subject.lower()) + r"\b\s+is\b", lowered):
        return f"What is {subject}?"
    if re.search(r"\b" + re.escape(subject.lower()) + r"\b\s+(can|allows|enables)\b", lowered):
        return f"What can {subject} do?"
    if re.search(r"\b" + re.escape(subject.lower()) + r"\b\s+(uses|use)\b", lowered):
        return f"What does {subject} use?"

    return f"What does the document say about {subject}?"


# ---------------------------------------------------------------------
# 3. Build positive + negative pairs
# ---------------------------------------------------------------------
def build_pairs(chunks, num_negatives=4):
    pairs = []
    questions = []

    for chunk in chunks:
        q = generate_question(chunk)
        if q is None:
            questions.append(None)
            continue
        questions.append(q)
        pairs.append((q, chunk, 1))

    n = len(chunks)
    for i, q in enumerate(questions):
        if q is None:
            continue
        # sample negatives from other chunks
        other_indices = [j for j in range(n) if j != i]
        random.shuffle(other_indices)
        neg_count = 0
        for j in other_indices:
            if neg_count >= num_negatives:
                break
            pairs.append((q, chunks[j], 0))
            neg_count += 1

    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------
def main(pdf_path, out_csv="training_pairs.csv"):
    chunks = extract_chunks(pdf_path)
    print(f"Extracted {len(chunks)} chunks from {pdf_path}")

    pairs = build_pairs(chunks)
    print(f"Generated {len(pairs)} (question, chunk, label) pairs")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "chunk_text", "label"])
        writer.writerows(pairs)

    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_training_data.py <path_to_pdf> [out_csv]")
        sys.exit(1)
    pdf_arg = sys.argv[1]
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "training_pairs.csv"
    main(pdf_arg, out_arg)