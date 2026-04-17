import json
import evaluate


# Load metric

squad_metric = evaluate.load("squad")


# Main

def main():
    # Load predictions
    with open("data/predictions.json", "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Load gold answers
    with open("data/gold_answers.json", "r", encoding="utf-8") as f:
        gold = json.load(f)

    # Lookup for gold answers by question_id
    gold_lookup = {item["question_id"]: item["answer"] for item in gold}

    # Format for squad metric
    # predictions: list of {"id": ..., "prediction_text": ...}
    # references:  list of {"id": ..., "answers": {"text": [...], "answer_start": [0]}}
    formatted_predictions = []
    formatted_references = []

    for item in predictions:
        qid = item["question_id"]
        pred_text = item["answer"]
        gold_text = gold_lookup.get(qid, "")

        formatted_predictions.append({
            "id": qid,
            "prediction_text": pred_text,
        })

        formatted_references.append({
            "id": qid,
            "answers": {
                "text": [gold_text],
                "answer_start": [0],
            },
        })

    # Compute EM and F1
    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references,
    )

    exact_match = results["exact_match"]
    f1 = results["f1"]

    print("Evaluation Results")
    print("------------------")
    print("Exact Match:", round(exact_match, 2))
    print("F1 Score:   ", round(f1, 2))
    print()

    # Per-question breakdown
    print("Per-question breakdown:")
    print()
    for pred, ref in zip(formatted_predictions, formatted_references):
        qid = pred["id"]
        pred_text = pred["prediction_text"]
        gold_text = ref["answers"]["text"][0]

        per = squad_metric.compute(
            predictions=[pred],
            references=[ref],
        )

        em = int(per["exact_match"])
        f1_per = round(per["f1"], 1)

        print("[" + qid + "] EM=" + str(em) + "  F1=" + str(f1_per))
        print("  Gold:", gold_text)
        print("  Pred:", pred_text)
        print()


if __name__ == "__main__":
    main()
