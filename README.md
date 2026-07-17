# PDF AI Bot — Self-Trained (No Third-Party API, No Pretrained LLM)

## What this is

A question-answering bot for PDFs that finds and returns the most relevant
passage for a question. Unlike your Ollama-based bot, nothing here calls an
external API or uses a pretrained language model. Every piece of "AI" in
this system — the scoring logic that decides which passage is relevant —
is trained from scratch on the PDF itself, using a classic supervised
machine learning pipeline: **generate data → extract features → train a
classifier → run inference**.

This is the same shape of pipeline as your bottle-defect and XGBoost work,
just applied to text instead of images.

## What it can and can't do (read this first)

**Can do:** given a question, find and return the existing passage in the
PDF that best matches it, using learned word-relevance patterns.

**Cannot do:** generate new sentences, summarize the whole document,
answer questions that require combining multiple passages, or answer
questions about the document's structure ("what topics does this cover").
It retrieves text — it does not write text. This is the fundamental
trade-off of avoiding pretrained language models: real language generation
requires training on billions of words of general text, which isn't
feasible to build from scratch in this timeframe.

---

## The Pipeline, Step by Step

```
PDF file
   │
   ▼
[1] generate_training_data.py  →  training_pairs.csv
   │        (chunks + pseudo questions + labels)
   ▼
[2] feature_extraction.py      →  features.csv + idf_model.json
   │        (turns each pair into 9 numbers)
   ▼
[3] train_model.py             →  model_weights.json
   │        (logistic regression, trained from scratch with numpy)
   ▼
[4] inference.py               →  ranked answer(s) for a new question
   │
   ▼
[5] app.py  (Streamlit UI wrapping steps 1-4)
```

---

## Step 1 — `generate_training_data.py`

**Problem it solves:** to train a classifier, you need labeled examples —
pairs of (question, passage, relevant-or-not). Nobody has hand-written
these for your PDF, so this script manufactures them automatically,
using only rules (regex and string logic), no AI model.

**`extract_chunks(pdf_path)`**
Opens the PDF with PyMuPDF, pulls all text, splits it into sentences, then
greedily groups sentences into chunks of roughly 15–60 words each. PDFs
often lose paragraph formatting on text extraction, so this works off
sentence boundaries rather than blank lines, which are unreliable.

**`generate_question(chunk_text)`**
Looks at the first sentence of a chunk and tries to build a question from
it using simple templates:
- Finds a "subject" — either a run of capitalized words (a likely proper
  noun or key term) or, failing that, the longest non-stopword word.
- Interrogative words (what, why, how, when, where, who) are explicitly
  excluded from being picked as the subject — an earlier version of this
  script had a bug where sentences starting with "What is..." (common in
  textbook exercises) got mis-parsed as if "What" itself were the topic,
  producing nonsense questions like "What is What?"
- If the sentence matches an "X is Y" pattern → `"What is X?"`
- If it matches "X can/allows/enables..." → `"What can X do?"`
- If it matches "X uses..." → `"What does X use?"`
- Otherwise → `"What does the document say about X?"`

**`build_pairs(chunks)`**
For every chunk with a successfully generated question:
- **Positive pair** (label = 1): that question paired with its own source
  chunk.
- **Negative pairs** (label = 0): the same question paired with 4 other
  randomly chosen chunks (i.e., examples of chunks that do *not* answer
  this question).

Output: `training_pairs.csv` with columns `question, chunk_text, label`.

---

## Step 2 — `feature_extraction.py`

**Problem it solves:** a machine learning classifier can't take raw text
as input — it needs numbers. This step converts every (question, chunk)
pair into a fixed-length vector of 9 numbers that describe how well they
match.

**`IDFModel`** — a hand-written IDF (inverse document frequency) table,
built by counting how many documents (questions + chunks) each word
appears in, and weighting rare words higher than common ones. This is the
same idea as `sklearn.TfidfVectorizer`, just implemented from scratch so
there's no dependency on a pre-fit third-party model. Word frequency
within a chunk uses **log-scaling** (`1 + log(count)`) rather than raw
counts — this matters because early testing showed chunks that simply
*repeat* a word several times (e.g. multi-part textbook exercises
restating "probability" in every sub-question) were unfairly outscoring
genuine explanations that used the word only once.

**The 9 features per (question, chunk) pair:**

| # | Feature | What it captures |
|---|---|---|
| 1 | `cosine_sim` | Overall similarity between the question's and chunk's weighted word vectors |
| 2 | `overlap_count` | Raw number of shared words |
| 3 | `overlap_ratio` | Share of the question's words found in the chunk |
| 4 | `rare_overlap_score` | Overlap weighted toward rare/important words |
| 5 | `length_ratio` | Chunk length relative to question length |
| 6 | `max_overlap_idf` | The single most distinctive shared word's rarity |
| 7 | `cap_matches` | Shared words that appear capitalized in the chunk (proxy for proper nouns/key terms) |
| 8 | `question_marker_score` | How "exercise-like" the chunk looks (ends in `?`, has multiple `?`, or starts with "a." / "b)") — used to *penalize* textbook exercise prompts, which are questions, not answers |
| 9 | `definitional_score` | Whether the chunk contains a defining phrase near an overlapping word ("is defined as", "refers to", "is called", "the objective of", etc.) — used to *reward* genuine definitions over passing mentions |

Output: `features.csv` (9 feature columns + label) and `idf_model.json`
(the fitted vocabulary weights, needed later at inference time so new
questions are scored against the same word-rarity table).

---

## Step 3 — `train_model.py`

**Problem it solves:** learn, from the labeled feature vectors, which
combination of the 9 features actually predicts relevance.

**Model:** logistic regression — a standard, well-understood classifier
that outputs a probability (0 to 1) that a given (question, chunk) pair is
relevant. It's trained here with **hand-written gradient descent using
only numpy** — no `sklearn.LogisticRegression`, no pretrained weights.
The training loop:

1. Normalizes all 9 features (zero mean, unit variance) so no single
   feature dominates just because of its scale.
2. Starts with all weights at zero.
3. Repeatedly: computes predictions, compares to true labels, computes
   the gradient (direction that reduces error), and nudges the weights
   in that direction. Runs for 2000 iterations.
4. Applies a small L2 penalty (`l2=0.01`) to discourage any single
   feature's weight from growing unreasonably large (a basic
   overfitting guard).

Output: `model_weights.json` — the learned weight for each of the 9
features, a bias term, and the normalization statistics (mean/std) needed
to apply the exact same scaling at inference time.

The training log printed to your terminal (loss going down, accuracy
going up across epochs) is the direct evidence this is a real trained
model, not a hardcoded rule set.

---

## Step 4 — `inference.py`

**Problem it solves:** given a brand-new question (not from training),
find the best-matching chunk in the PDF.

1. Re-extracts chunks from the PDF (same chunker as Step 1).
2. For every chunk, computes the same 9 features against the question.
3. Normalizes features using the saved training mean/std.
4. Runs the trained logistic regression to get a relevance probability.
5. **Applies two deterministic adjustments on top of the learned score:**
   - Subtracts a penalty proportional to `question_marker_score` — this
     exists because the *learned* weight for this feature alone turned
     out too weak to reliably suppress exercise-style chunks, especially
     for short, common questions where the classifier's other signals
     don't have much to work with.
   - Adds a boost proportional to `definitional_score`, for the same
     reason — reinforcing a real learned signal with a direct rule where
     the model's confidence is too commonly diluted to trust alone.
   
   This combination (learned score + rule-based adjustment) is a
   standard, legitimate technique in real retrieval systems — it doesn't
   change the fact that the underlying score comes from a trained model.
6. Sorts all chunks by adjusted score and returns the top `k`.

---

## Step 5 — `app.py`

A Streamlit interface wrapping the whole pipeline:
- Upload a PDF.
- Click "Train model on this PDF" — runs Steps 1–3 with a progress bar.
- Type a question — runs Step 4 and displays the top-ranked passages with
  their scores.

No file needs to be edited to use a new PDF; the app re-runs the full
pipeline (chunking → pseudo-questions → features → training) on whatever
is uploaded.

---

## Known limitations, and why they exist

- **Word-based, not meaning-based.** The whole system reasons in terms of
  shared vocabulary and learned statistical patterns — it has no concept
  of synonyms or paraphrasing. "Cost" and "price" would not match.
- **Struggles on documents where the query term is extremely common**
  (e.g. asking "What is probability?" in a statistics textbook, where that
  word appears on nearly every page). The definitional-phrasing feature
  and exercise-penalty help, but can't fully replace real language
  understanding.
- **Pseudo-questions are only as good as the heuristics that generate
  them.** If a PDF's writing style doesn't match the "X is Y" / "X uses
  Y" templates well, fewer usable training pairs get generated.
- **No generation, no summarization.** By design — this is a retrieval
  system, not a generative one.
