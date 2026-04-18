import json
import evaluate


squad_metric = evaluate.load("squad")



def main():
    with open("data/predictions.json", "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open("data/gold_answers.json", "r", encoding="utf-8") as f:
        gold_answers = json.load(f)

    gold_lookup = {}
    for item in gold_answers:
        gold_lookup[item["question_id"]] = item["answer"]

    formatted_predictions = []
    formatted_references = []

    for item in predictions:
        qid = item["question_id"]
        pred_text = item["answer"]
        gold_text = gold_lookup[qid]

        formatted_predictions.append({
            "id": qid,
            "prediction_text": pred_text
        })

        formatted_references.append({
            "id": qid,
            "answers": {
                "text": [gold_text],
                "answer_start": [0]
            }
        })

    results = squad_metric.compute(
        predictions=formatted_predictions,
        references=formatted_references
    )

    print("Evaluation Results")
    print("------------------")
    print("Exact Match:", round(results["exact_match"], 2))
    print("F1 Score:", round(results["f1"], 2))
    print()
    print("Per-question breakdown:")
    print()

    for i in range(len(formatted_predictions)):
        pred = formatted_predictions[i]
        ref = formatted_references[i]

        per_question = squad_metric.compute(
            predictions=[pred],
            references=[ref]
        )

        print("[" + pred["id"] + "] EM=" + str(int(per_question["exact_match"])) + " F1=" + str(round(per_question["f1"], 1)))
        print("Gold:", ref["answers"]["text"][0])
        print("Pred:", pred["prediction_text"])
        print()


if __name__ == "__main__":
    main()
