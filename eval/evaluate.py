import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_scifact,
    get_valid_claims,
    get_unique_claims,
    build_document_corpus,
)
from src.pipeline import JurisVerifyPipeline
from src.verification import NLIVerifier, normalize_for_scifact_eval

N_EVAL_CLAIMS = 50  # bump this before reporting results in the paper
PRONOUNS_PATTERN = r"^(?:it|this|that|they|these|those)\b"


def split_claim_into_statements(claim: str) -> list:
    """Split a claim into atomic sub-statements, avoiding splits before bare pronouns.

    Splits on:
    - Semicolons (';')
    - Explicit clause conjunctions with punctuation: ', while ', ', whereas ', ', although ', ', and ', ', but '

    If the subsequent clause begins with a bare pronoun (it/this/that/they/these/those)
    or is a short list fragment, the split is suppressed to avoid dangling anaphoric
    references without an antecedent.
    """
    claim = claim.strip()
    if not claim:
        return []

    raw_text = claim.rstrip(".")
    pattern = r"(;\s*|,\s+(?:while|whereas|although)\s+|,\s+(?:and|but)\s+)"

    tokens = re.split(pattern, raw_text, flags=re.IGNORECASE)

    merged_parts = []
    current = tokens[0].strip()

    i = 1
    while i < len(tokens):
        delimiter = tokens[i]
        next_part = tokens[i + 1].strip() if i + 1 < len(tokens) else ""

        # Suppress split if next clause starts with a bare pronoun or is a short list item
        starts_with_pronoun = bool(re.search(PRONOUNS_PATTERN, next_part, flags=re.IGNORECASE))
        is_short_list_item = len(next_part.split()) <= 2

        if starts_with_pronoun or is_short_list_item:
            current = current + delimiter + next_part
        else:
            if current:
                merged_parts.append(current)
            current = next_part
        i += 2

    if current:
        merged_parts.append(current)

    statements = []
    for part in merged_parts:
        cleaned = part.strip()
        if len(cleaned) > 10:
            if not cleaned.endswith("."):
                cleaned += "."
            statements.append(cleaned)

    return statements if statements else [claim if claim.endswith(".") else claim + "."]


def compute_nli_faithfulness(claim: str, evidence: str, verifier: NLIVerifier, return_details: bool = False):
    """Compute NLI-based factual consistency / faithfulness score (strict, entailment-only).

    Methodology (SummaC/FactCC family):
    1. Decompose the claim into atomic sub-statements.
    2. Check each sub-statement against the extracted evidence sentence using NLIVerifier
       (premise=evidence, hypothesis=sub_statement).
    3. Faithfulness = (number of SUPPORTED statements) / (total statements).

    LIMITATION: all atomic sub-statements from a single claim are checked against
    the same evidence sentence (extract_best_evidence returns one sentence per claim,
    not one per sub-statement) -- for genuinely multi-part claims where each part is
    supported by different source sentences, this may understate faithfulness.
    Acceptable for now since most SciFact claims are single-assertion.

    NOTE: Deliberately reuses the existing NLIVerifier (DeBERTa-v3-large-mnli) rather
    than an LLM judge, keeping the eval harness fully local/reproducible with no
    external API dependency.
    """
    if not evidence or not evidence.strip():
        if return_details:
            return 0.0, []
        return 0.0

    statements = split_claim_into_statements(claim)
    if not statements:
        if return_details:
            return 0.0, []
        return 0.0

    supported_count = 0
    details = []
    for stmt in statements:
        verdict = verifier.verify(claim=stmt, evidence=evidence)
        if verdict == "SUPPORTED":
            supported_count += 1
        details.append({"statement": stmt, "verdict": verdict})

    score = round(supported_count / len(statements), 4)
    if return_details:
        return score, details
    return score


def compute_weighted_nli_faithfulness(claim: str, evidence: str, verifier: NLIVerifier, return_details: bool = False):
    """Same as compute_nli_faithfulness but gives partial credit:
    SUPPORTED = 1.0, PARTIALLY_SUPPORTED = 0.5, CONTRADICTED = 0.0.
    Reflects the project's 3-way AGREE/SCOPE-MISMATCH/CONTRADICT
    distinction rather than collapsing to binary supported/unsupported.

    LIMITATION: all atomic sub-statements from a single claim are checked against
    the same evidence sentence (extract_best_evidence returns one sentence per claim,
    not one per sub-statement) -- for genuinely multi-part claims where each part is
    supported by different source sentences, this may understate faithfulness.
    Acceptable for now since most SciFact claims are single-assertion.

    NOTE: Deliberately reuses the existing NLIVerifier (DeBERTa-v3-large-mnli) rather
    than an LLM judge, keeping the eval harness fully local/reproducible with no
    external API dependency.
    """
    if not evidence or not evidence.strip():
        if return_details:
            return 0.0, []
        return 0.0

    statements = split_claim_into_statements(claim)
    if not statements:
        if return_details:
            return 0.0, []
        return 0.0

    weights = {"SUPPORTED": 1.0, "PARTIALLY_SUPPORTED": 0.5, "CONTRADICTED": 0.0}
    total_credit = 0.0
    details = []
    for stmt in statements:
        verdict = verifier.verify(claim=stmt, evidence=evidence)
        credit = weights.get(verdict, 0.0)
        total_credit += credit
        details.append({"statement": stmt, "verdict": verdict, "credit": credit})

    score = round(total_credit / len(statements), 4)
    if return_details:
        return score, details
    return score


def run_eval():
    claims_dataset, corpus_dataset = load_scifact()
    valid_claims = get_valid_claims(claims_dataset)
    unique_claims = get_unique_claims(valid_claims)
    documents = build_document_corpus(corpus_dataset)

    pipeline = JurisVerifyPipeline(documents)

    eval_claims = unique_claims[:N_EVAL_CLAIMS]

    correct = 0
    wrong_cases = []
    strict_faithfulness_scores = []
    weighted_faithfulness_scores = []
    strict_by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    weighted_by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    all_eval_details = []

    for item in eval_claims:
        claim_text = item["claim"]
        gold = item["evidence_label"]

        result = pipeline.predict_claim(claim_text)
        prediction = normalize_for_scifact_eval(result["verdict"])

        # Compute both Strict and Weighted NLI Faithfulness scores reusing pipeline.verifier
        strict_score, strict_details = compute_nli_faithfulness(
            claim_text, result["evidence"], pipeline.verifier, return_details=True
        )
        weighted_score, weighted_details = compute_weighted_nli_faithfulness(
            claim_text, result["evidence"], pipeline.verifier, return_details=True
        )

        strict_faithfulness_scores.append(strict_score)
        weighted_faithfulness_scores.append(weighted_score)
        if gold in strict_by_label:
            strict_by_label[gold].append(strict_score)
            weighted_by_label[gold].append(weighted_score)

        all_eval_details.append({
            "claim": claim_text,
            "gold": gold,
            "prediction": prediction,
            "verdict": result["verdict"],
            "evidence": result["evidence"],
            "evidence_doc_id": result.get("evidence_doc_id"),
            "strict_faithfulness": strict_score,
            "weighted_faithfulness": weighted_score,
            "statements": weighted_details,
        })

        if prediction == gold:
            correct += 1
        else:
            wrong_cases.append({
                "claim": claim_text,
                "gold": gold,
                "prediction": prediction,
                "evidence": result["evidence"],
                "evidence_doc_id": result.get("evidence_doc_id"),
                "strict_faithfulness": strict_score,
                "weighted_faithfulness": weighted_score,
            })

    accuracy = round(correct / len(eval_claims) * 100, 2)
    avg_strict_faithfulness = (
        round(sum(strict_faithfulness_scores) / len(strict_faithfulness_scores), 4)
        if strict_faithfulness_scores
        else 0.0
    )
    avg_weighted_faithfulness = (
        round(sum(weighted_faithfulness_scores) / len(weighted_faithfulness_scores), 4)
        if weighted_faithfulness_scores
        else 0.0
    )

    print(f"Correct: {correct}")
    print(f"Total: {len(eval_claims)}")
    print(f"Accuracy: {accuracy}%")

    print(f"\n============================ NLI FAITHFULNESS REPORT ============================")
    print(f"Overall Metrics:")
    print(f"  - Strict Faithfulness (Entailment Only):    {avg_strict_faithfulness:.4f}")
    print(f"  - Weighted Faithfulness (Partial Credit):   {avg_weighted_faithfulness:.4f}")
    print(f"\nBreakdown by Gold Label:")
    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        s_scores = strict_by_label[label]
        w_scores = weighted_by_label[label]
        s_avg = sum(s_scores) / len(s_scores) if s_scores else 0.0
        w_avg = sum(w_scores) / len(w_scores) if w_scores else 0.0
        print(f"  - {label:<12} ({len(s_scores):>2} claims): Strict = {s_avg:.4f} | Weighted = {w_avg:.4f}")
    print(f"=================================================================================\n")

    return {
        "accuracy": accuracy,
        "strict_faithfulness": avg_strict_faithfulness,
        "weighted_faithfulness": avg_weighted_faithfulness,
        "faithfulness": avg_strict_faithfulness,
        "strict_faithfulness_by_label": {
            label: round(sum(scores) / len(scores), 4) if scores else 0.0
            for label, scores in strict_by_label.items()
        },
        "weighted_faithfulness_by_label": {
            label: round(sum(scores) / len(scores), 4) if scores else 0.0
            for label, scores in weighted_by_label.items()
        },
        "wrong_cases": wrong_cases,
        "eval_details": all_eval_details,
    }


if __name__ == "__main__":
    run_eval()
