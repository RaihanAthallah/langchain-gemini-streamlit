import json
from pathlib import Path

from rouge_score import rouge_scorer

from agents.faq_agent import invoke_faq_agent


def run_evaluation(dataset_path: Path) -> dict[str, float]:
    with dataset_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_total = 0.0
    rougeL_total = 0.0

    for idx, row in enumerate(rows):
        question = row["question"]
        reference = row["answer"]
        prediction = invoke_faq_agent(question, thread_id=f"eval-{idx}")["answer"]
        score = scorer.score(reference, prediction)
        rouge1_total += score["rouge1"].fmeasure
        rougeL_total += score["rougeL"].fmeasure

    count = max(len(rows), 1)
    return {
        "samples": float(len(rows)),
        "rouge1_f1": rouge1_total / count,
        "rougeL_f1": rougeL_total / count,
    }


if __name__ == "__main__":
    dataset = Path("base-knowledge/eval_dataset.json")
    if not dataset.exists():
        raise FileNotFoundError("Create base-knowledge/eval_dataset.json first.")
    results = run_evaluation(dataset)
    print(json.dumps(results, indent=2))

