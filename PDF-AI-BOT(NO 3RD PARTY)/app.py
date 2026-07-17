"""
app.py

Streamlit UI for the "own trained model" PDF AI bot.

Wraps the existing pipeline (generate_training_data.py, feature_extraction.py,
train_model.py, inference.py) with zero changes to their logic - this file
only handles the interface: upload a PDF, train on it, then ask questions.

No third-party API, no pretrained embeddings/LLM anywhere in this stack.
"""

import os
import streamlit as st

from generate_training_data import extract_chunks, build_pairs
from feature_extraction import build_feature_dataset, load_idf_model
from train_model import main as train_main
from inference import answer_question, load_model

st.set_page_config(page_title="PDF AI Bot - Self-Trained", page_icon="📄", layout="centered")

DATA_DIR = "bot_data"
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS_CSV = os.path.join(DATA_DIR, "training_pairs.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")
IDF_PATH = os.path.join(DATA_DIR, "idf_model.json")
WEIGHTS_PATH = os.path.join(DATA_DIR, "model_weights.json")
PDF_PATH = os.path.join(DATA_DIR, "current.pdf")


def save_uploaded_pdf(uploaded_file):
    with open(PDF_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())


def run_training_pipeline(pdf_path, progress_callback=None):
    import csv

    chunks = extract_chunks(pdf_path)
    if progress_callback:
        progress_callback(f"Extracted {len(chunks)} chunks", 0.2)

    pairs = build_pairs(chunks)
    with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "chunk_text", "label"])
        writer.writerows(pairs)
    if progress_callback:
        progress_callback(f"Generated {len(pairs)} training pairs", 0.45)

    # feature_extraction and train_model write idf_model.json /
    # model_weights.json into the CURRENT working directory by default,
    # so we temporarily cd into DATA_DIR to keep everything together
    cwd = os.getcwd()
    try:
        os.chdir(DATA_DIR)
        build_feature_dataset("training_pairs.csv", "features.csv")
        if progress_callback:
            progress_callback("Built feature vectors", 0.65)

        train_main("features.csv", "model_weights.json")
        if progress_callback:
            progress_callback("Model trained", 1.0)
    finally:
        os.chdir(cwd)


def is_trained():
    return os.path.exists(WEIGHTS_PATH) and os.path.exists(IDF_PATH)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("📄 PDF AI Bot")
st.caption("Fully self-trained - no third-party API, no pretrained LLM, no Ollama.")

with st.expander("How this works", expanded=False):
    st.markdown(
        "This bot extracts passages from your PDF, auto-generates practice "
        "questions from the text itself, and trains a logistic regression "
        "classifier **from scratch** (hand-written gradient descent, no "
        "sklearn/pretrained weights) to rank which passage best answers a "
        "question. It retrieves and returns existing text from the PDF - "
        "it does not generate new sentences, so it cannot summarize the "
        "whole document or answer questions that need combining multiple "
        "passages."
    )

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None:
    save_uploaded_pdf(uploaded_pdf)
    st.success(f"Loaded: {uploaded_pdf.name}")

    if st.button("Train model on this PDF", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update(msg, pct):
            status_text.write(msg)
            progress_bar.progress(pct)

        with st.spinner("Training..."):
            run_training_pipeline(PDF_PATH, progress_callback=update)
        st.success("Training complete. You can ask questions now.")

st.divider()

if is_trained():
    question = st.text_input("Ask a question about the PDF")
    top_k = st.slider("Number of results", min_value=1, max_value=5, value=3)

    if question:
        idf_model = load_idf_model(IDF_PATH)
        model = load_model(WEIGHTS_PATH)
        results = answer_question(question, PDF_PATH, idf_model, model, top_k=top_k)

        for rank, (score, chunk) in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"**#{rank}** &nbsp; score: `{score:.3f}`")
                st.write(chunk)
else:
    st.info("Upload a PDF and click **Train model on this PDF** to get started.")
