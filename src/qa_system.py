import json
import re
import os
import torch
from transformers import pipeline


# Configuration

MODEL_NAME   = "deepset/roberta-base-squad2"
WINDOW_SIZE  = 5    # sentences per chunk
STRIDE       = 3    # sentences to advance per step
MAX_ANSWER_LEN = 50


# Device

device = 0 if torch.cuda.is_available() else -1


# Chunking

def split_sentences(text):
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def chunk_document(text, window=WINDOW_SIZE, stride=STRIDE):
    """Split document into overlapping sentence-window chunks."""
    sentences = split_sentences(text)

    if not sentences:
        return [text]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + window])
        if chunk.strip():
            chunks.append(chunk)
        if i + window >= len(sentences):
            break
        i += stride

    return chunks if chunks else [text]


# Question Answering

def answer_question(qa_pipe, question, document):
    """Run QA model over all chunks, return the answer with the highest score."""
    chunks = chunk_document(document)

    best_answer = ""
    best_score  = -1.0

    for chunk in chunks:
        result = qa_pipe(
            question=question,
            context=chunk,
            max_answer_len=MAX_ANSWER_LEN,
        )
        if result["score"] > best_score:
            best_score  = result["score"]
            best_answer = result["answer"]

    answer = best_answer.strip().replace('\n', ' ')
    return answer if answer else "unknown"


# Main

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if os.path.exists(os.path.join(script_dir, "input.json")):
        input_path  = os.path.join(script_dir, "input.json")
        output_path = os.path.join(script_dir, "predictions.json")
    elif os.path.exists(os.path.join(project_root, "data", "input.json")):
        input_path  = os.path.join(project_root, "data", "input.json")
        output_path = os.path.join(project_root, "data", "predictions.json")
    else:
        raise FileNotFoundError("Could not find input.json.")

    # Load model
    print("Loading model:", MODEL_NAME)
    qa_pipe = pipeline(
        "question-answering",
        model=MODEL_NAME,
        device=device,
    )
    print("Model ready. Device:", "CUDA" if device == 0 else "CPU")
    print()

    # Load input
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Loaded", len(data), "questions from", input_path)
    print()

    # Run pipeline
    predictions = []
    for item in data:
        qid      = item["question_id"]
        question = item["question"]
        document = item["document"]

        answer = answer_question(qa_pipe, question, document)

        predictions.append({
            "question_id": qid,
            "answer":      answer,
        })

        print("[" + qid + "]", question[:70])
        print("  ->", answer)
        print()

    # Save predictions
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print("Predictions saved to:", output_path)
    print("Total:", len(predictions))


if __name__ == "__main__":
    main()
