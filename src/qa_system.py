import json
import re
import torch
from transformers import pipeline


MODEL_NAME = "deepset/roberta-base-squad2"
WINDOW_SIZE = 5
STRIDE = 3
MAX_ANSWER_LEN = 50


device = 0 if torch.cuda.is_available() else -1



def split_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:
            cleaned_sentences.append(sentence)

    return cleaned_sentences



def chunk_document(text):
    """Split a document into overlapping chunks."""
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return [text]

    chunks = []
    start = 0

    while start < len(sentences):
        chunk = " ".join(sentences[start:start + WINDOW_SIZE])
        if chunk.strip() != "":
            chunks.append(chunk)

        if start + WINDOW_SIZE >= len(sentences):
            break

        start = start + STRIDE

    return chunks



def answer_question(qa_pipe, question, document):
    """Find the best answer from all document chunks."""
    chunks = chunk_document(document)
    best_answer = ""
    best_score = -1

    for chunk in chunks:
        result = qa_pipe(
            question=question,
            context=chunk,
            max_answer_len=MAX_ANSWER_LEN
        )

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    best_answer = best_answer.strip().replace("\n", " ")

    if best_answer == "":
        return "unknown"

    return best_answer



def main():
    print("Loading model:", MODEL_NAME)
    qa_pipe = pipeline(
        "question-answering",
        model=MODEL_NAME,
        device=device
    )

    if device == 0:
        print("Model ready. Device: CUDA")
    else:
        print("Model ready. Device: CPU")

    print()

    with open("data/input.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loaded", len(data), "questions from data/input.json")
    print()

    predictions = []

    for item in data:
        qid = item["question_id"]
        question = item["question"]
        document = item["document"]

        answer = answer_question(qa_pipe, question, document)

        predictions.append({
            "question_id": qid,
            "answer": answer
        })

        print("[" + qid + "]", question)
        print("->", answer)
        print()

    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print("Predictions saved to: data/predictions.json")
    print("Total:", len(predictions))


if __name__ == "__main__":
    main()
