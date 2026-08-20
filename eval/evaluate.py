"""
Evaluation harness for the Phase 1 pipeline.

Reproduces the 88%-accuracy result from the original notebook on a
50-claim SciFact held-out set, and collects wrong_cases for error
analysis. Before citing an accuracy number in the paper, bump
N_EVAL_CLAIMS up substantially -- 50 is enough for iteration, not
enough for a headline figure.
"""

from src.data_loader import (
    load_scifact,
    get_valid_claims,
    get_unique_claims,
    build_document_corpus,
)
from src.pipeline import JurisVerifyPipeline
from src.verification import normalize_for_scifact_eval

N_EVAL_CLAIMS = 50  # bump this before reporting results in the paper


def run_eval():
    claims_dataset, corpus_dataset = load_scifact()
    valid_claims = get_valid_claims(claims_dataset)
    unique_claims = get_unique_claims(valid_claims)
    documents = build_document_corpus(corpus_dataset)

    pipeline = JurisVerifyPipeline(documents)

    eval_claims = unique_claims[:N_EVAL_CLAIMS]

    correct = 0
    wrong_cases = []

    for item in eval_claims:
        claim_text = item["claim"]
        gold = item["evidence_label"]

        result = pipeline.predict_claim(claim_text)
        prediction = normalize_for_scifact_eval(result["verdict"])

        if prediction == gold:
            correct += 1
        else:
            wrong_cases.append({
                "claim": claim_text,
                "gold": gold,
                "prediction": prediction,
                "evidence": result["evidence"],
            })

    accuracy = round(correct / len(eval_claims) * 100, 2)

    print(f"Correct: {correct}")
    print(f"Total: {len(eval_claims)}")
    print(f"Accuracy: {accuracy}%")

    return {"accuracy": accuracy, "wrong_cases": wrong_cases}


if __name__ == "__main__":
    run_eval()
