import sys
import os
import re
import json
import time
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_scifact,
    get_valid_claims,
    get_unique_claims,
    build_document_corpus,
)
from src.pipeline import JurisVerifyPipeline
from src.extraction import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_SECTION_WEIGHT,
    EVIDENCE_MODES,
    MAX_BUNDLE_SENTENCES,
)
from src.prosecution import (
    ADJUDICATION_RESULT_KEYS,
    ADJUDICATION_STATUSES,
    DEFAULT_PROSECUTION_MODE,
    PROSECUTION_MODES,
    PROSECUTION_RESULT_KEYS,
)
from src.verification import NLIVerifier, normalize_for_scifact_eval

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


def _histogram(values, width=0.1):
    """Bin values in [0, 1] into fixed-width buckets, lowest bucket first."""
    bins = {}
    n_bins = int(round(1.0 / width))
    for i in range(n_bins):
        lo = round(i * width, 4)
        bins[f"{lo:.1f}-{min(lo + width, 1.0):.1f}"] = 0
    for value in values:
        if value is None:
            continue
        index = min(int(max(0.0, min(1.0, value)) / width), n_bins - 1)
        lo = round(index * width, 4)
        bins[f"{lo:.1f}-{min(lo + width, 1.0):.1f}"] += 1
    return bins


def compute_prosecution_diagnostics(records: list) -> dict:
    """Aggregate arm E's per-claim prosecution records.

    Diagnostics only. Nothing here feeds a prediction, and none of the
    existing accuracy/faithfulness metrics are recomputed or replaced.
    """
    attempted = [r for r in records if r.get("prosecution_attempted")]
    found = [r for r in attempted if r.get("counter_evidence")]

    def by_gold(label, rows):
        return [r for r in rows if r.get("gold") == label]

    verdict_distribution = {}
    for record in attempted:
        key = record.get("counter_verdict") or "NONE_FOUND"
        verdict_distribution[key] = verdict_distribution.get(key, 0) + 1

    section_distribution = {}
    for record in found:
        key = record.get("counter_section") or "unknown"
        section_distribution[key] = section_distribution.get(key, 0) + 1

    pool_sizes = [r.get("candidate_pool_size", 0) for r in attempted]
    counter_ranks = [r["counter_rank"] for r in found if r.get("counter_rank") is not None]
    in_selection = [r for r in found if r.get("counter_in_d_selection")]

    def mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    def rate(numerator, denominator):
        return round(numerator / denominator * 100, 2) if denominator else 0.0

    diagnostics = {
        "claims_prosecuted": len(attempted),
        "claims_with_counter_evidence": len(found),
        "counter_evidence_rate": rate(len(found), len(attempted)),
        "counter_evidence_by_gold_label": {},
        "counter_verdict_distribution": verdict_distribution,
        "counter_section_distribution": section_distribution,
        "mean_candidate_pool_size": mean(pool_sizes),
        "min_candidate_pool_size": min(pool_sizes) if pool_sizes else 0,
        "max_candidate_pool_size": max(pool_sizes) if pool_sizes else 0,
        "mean_counter_rank": mean(counter_ranks),
        "counter_rank_zero_count": sum(1 for r in counter_ranks if r == 0),
        "counter_already_in_baseline_bundle": len(in_selection),
        "counter_confidence_histogram": _histogram(
            [r.get("counter_confidence") for r in found]
        ),
        "counter_p_contradiction_histogram": _histogram(
            [r.get("counter_p_contradiction") for r in found]
        ),
        "mean_counter_confidence": mean(
            [r["counter_confidence"] for r in found if r.get("counter_confidence") is not None]
        ),
        "mean_counter_score_margin": mean(
            [r["counter_score"] for r in found if r.get("counter_score") is not None]
        ),
    }

    for label in ["SUPPORT", "CONTRADICT", "NEI"]:
        label_attempted = by_gold(label, attempted)
        label_found = by_gold(label, found)
        if not label_attempted:
            continue
        diagnostics["counter_evidence_by_gold_label"][label] = {
            "prosecuted": len(label_attempted),
            "with_counter_evidence": len(label_found),
            "rate": rate(len(label_found), len(label_attempted)),
        }

    return diagnostics


def print_prosecution_diagnostics(diagnostics: dict) -> None:
    """Print arm E's diagnostics as a block separate from the eval metrics."""
    print("\n============================ PASSIVE PROSECUTION DIAGNOSTICS ============================")
    print("(arm E: counter-evidence is recorded only; no prediction was modified)")
    print(f"  Claims prosecuted:            {diagnostics['claims_prosecuted']}")
    print(f"  With counter-evidence:        {diagnostics['claims_with_counter_evidence']}"
          f" ({diagnostics['counter_evidence_rate']}%)")
    print(f"  Candidate pool size:          mean {diagnostics['mean_candidate_pool_size']}"
          f", range {diagnostics['min_candidate_pool_size']}-{diagnostics['max_candidate_pool_size']}")
    print(f"  Mean counter rank:            {diagnostics['mean_counter_rank']}"
          f" (rank 0 on {diagnostics['counter_rank_zero_count']} claims)")
    print(f"  Already in baseline bundle:   {diagnostics['counter_already_in_baseline_bundle']}")
    print(f"  Mean counter confidence:      {diagnostics['mean_counter_confidence']}")
    print(f"  Mean counter margin:          {diagnostics['mean_counter_score_margin']}")

    print("\n  Counter-evidence rate by gold label:")
    for label, stats in diagnostics["counter_evidence_by_gold_label"].items():
        print(f"    - {label:<12} {stats['with_counter_evidence']:>4}/{stats['prosecuted']:<4}"
              f" ({stats['rate']}%)")

    print("\n  Counter-verdict distribution:")
    for verdict, count in sorted(
        diagnostics["counter_verdict_distribution"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    - {verdict:<22} {count}")

    if diagnostics["counter_section_distribution"]:
        print("\n  Counter-evidence section distribution:")
        for section, count in sorted(
            diagnostics["counter_section_distribution"].items(), key=lambda kv: -kv[1]
        ):
            print(f"    - {section:<22} {count}")

    print("\n  P(contradiction) histogram for found counter-evidence:")
    for bucket, count in diagnostics["counter_p_contradiction_histogram"].items():
        if count:
            print(f"    - {bucket:<12} {count}")
    print("=========================================================================================\n")


def compute_adjudication_diagnostics(records: list) -> dict:
    """Aggregate arm F's adjudication decisions and the accuracy tradeoff.

    Diagnostics only. The accuracy metrics computed elsewhere are not touched;
    this block re-reads the recorded `baseline_prediction` so a single arm-F
    run yields both arm D's predictions and arm F's.

    The headline numbers are deliberately the tradeoff, not peak CONTRADICT
    accuracy: `contradict_rescued` against `support_regressions`.
    """
    adjudicated = [r for r in records if r.get("adjudication_status")]
    overrides = [r for r in adjudicated if r.get("override_applied")]

    def rate(numerator, denominator):
        return round(numerator / denominator * 100, 2) if denominator else 0.0

    status_counts = {status: 0 for status in ADJUDICATION_STATUSES}
    for record in adjudicated:
        status = record["adjudication_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    reason_counts = {}
    for record in adjudicated:
        reason = record.get("adjudication_reason") or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    gate_failures = {}
    for record in adjudicated:
        for gate, passed in (record.get("adjudication_gates") or {}).items():
            if passed is False:
                gate_failures[gate] = gate_failures.get(gate, 0) + 1

    # The tradeoff. An override is a rescue when it turns a wrong SUPPORT into
    # a right CONTRADICT, and a regression when it turns a right SUPPORT wrong.
    rescued = [
        r for r in overrides
        if r.get("gold") == "CONTRADICT" and r.get("baseline_prediction") == "SUPPORT"
    ]
    regressions = [
        r for r in overrides
        if r.get("gold") == "SUPPORT" and r.get("baseline_prediction") == "SUPPORT"
    ]
    neutral_overrides = [
        r for r in overrides if r not in rescued and r not in regressions
    ]

    # Baseline vs adjudicated accuracy, from the same records.
    def accuracy(field):
        scored = [r for r in records if r.get("gold") in ("SUPPORT", "CONTRADICT")]
        by_label = {}
        correct = 0
        for record in scored:
            gold = record["gold"]
            predicted = record.get(field)
            hit = predicted == gold
            correct += hit
            bucket = by_label.setdefault(gold, {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += hit
        return {
            "overall": rate(correct, len(scored)),
            "correct": correct,
            "total": len(scored),
            "by_label": {
                label: {
                    "n": stats["n"],
                    "correct": stats["correct"],
                    "accuracy": rate(stats["correct"], stats["n"]),
                }
                for label, stats in sorted(by_label.items())
            },
        }

    has_baseline = any("baseline_prediction" in r for r in records)

    return {
        "claims_adjudicated": len(adjudicated),
        "overrides_applied": len(overrides),
        "override_rate": rate(len(overrides), len(adjudicated)),
        "adjudication_status_counts": status_counts,
        "adjudication_reason_counts": reason_counts,
        "gate_failure_counts": gate_failures,
        "contradict_rescued": len(rescued),
        "support_regressions": len(regressions),
        "other_overrides": len(neutral_overrides),
        "net_claims_gained": len(rescued) - len(regressions),
        "override_precision": round(
            len(rescued) / len(overrides), 4
        ) if overrides else 0.0,
        "arm_h_recoverable_misses": 44,
        "share_of_arm_h_recoverable_rescued": rate(len(rescued), 44),
        "baseline_accuracy": accuracy("baseline_prediction") if has_baseline else None,
        "adjudicated_accuracy": accuracy("prediction"),
        "overridden_claims": [
            {
                "claim_index": r.get("claim_index"),
                "gold": r.get("gold"),
                "baseline_prediction": r.get("baseline_prediction"),
                "prediction": r.get("prediction"),
                "adjudication_reason": r.get("adjudication_reason"),
                "counter_section": r.get("counter_section"),
                "counter_evidence_doc_id": r.get("counter_evidence_doc_id"),
                "evidence_doc_id": r.get("evidence_doc_id"),
                "counter_evidence": r.get("counter_evidence"),
            }
            for r in overrides
        ],
    }


def print_adjudication_diagnostics(diagnostics: dict) -> None:
    """Print arm F's adjudication block, tradeoff first."""
    print("\n============================ ADJUDICATION DIAGNOSTICS (ARM F) ============================")
    print(f"  Claims adjudicated:           {diagnostics['claims_adjudicated']}")
    print(f"  Overrides applied:            {diagnostics['overrides_applied']}"
          f" ({diagnostics['override_rate']}% of adjudicated)")

    print("\n  THE TRADEOFF (the metric that matters):")
    print(f"    CONTRADICT rescued:         +{diagnostics['contradict_rescued']}")
    print(f"    SUPPORT regressions:        -{diagnostics['support_regressions']}")
    if diagnostics["other_overrides"]:
        print(f"    other overrides:             {diagnostics['other_overrides']}")
    print(f"    net claims gained:          {diagnostics['net_claims_gained']:+d}")
    print(f"    override precision:         {diagnostics['override_precision']}")
    print(f"    share of arm H's 44 recoverable misses rescued: "
          f"{diagnostics['share_of_arm_h_recoverable_rescued']}%")

    baseline = diagnostics.get("baseline_accuracy")
    adjudicated = diagnostics["adjudicated_accuracy"]
    if baseline:
        print("\n  ARM D (baseline_verdict) vs ARM F (adjudicated):")
        header = f"    {'metric':<14}{'arm D':>10}{'arm F':>10}{'delta':>10}"
        print(header)
        print(f"    {'overall':<14}{baseline['overall']:>9}%{adjudicated['overall']:>9}%"
              f"{adjudicated['overall'] - baseline['overall']:>+9.2f}")
        for label in sorted(set(baseline["by_label"]) | set(adjudicated["by_label"])):
            b = baseline["by_label"].get(label, {"accuracy": 0.0, "correct": 0, "n": 0})
            a = adjudicated["by_label"].get(label, {"accuracy": 0.0, "correct": 0, "n": 0})
            print(f"    {label:<14}{b['accuracy']:>9}%{a['accuracy']:>9}%"
                  f"{a['accuracy'] - b['accuracy']:>+9.2f}"
                  f"   ({b['correct']}/{b['n']} -> {a['correct']}/{a['n']})")

    print("\n  Adjudication statuses:")
    for status, count in sorted(
        diagnostics["adjudication_status_counts"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    - {status:<22} {count}")

    if diagnostics["gate_failure_counts"]:
        print("\n  Gate failures (which gate blocked an override):")
        for gate, count in sorted(
            diagnostics["gate_failure_counts"].items(), key=lambda kv: -kv[1]
        ):
            print(f"    - {gate:<28} {count}")

    print("\n  Adjudication reasons:")
    for reason, count in sorted(
        diagnostics["adjudication_reason_counts"].items(), key=lambda kv: -kv[1]
    ):
        print(f"    - {count:>4}  {reason}")

    if diagnostics["overridden_claims"]:
        print("\n  Overridden claims (every override has a recorded reason):")
        for record in diagnostics["overridden_claims"][:20]:
            mark = "OK " if record["gold"] == "CONTRADICT" else "REG"
            print(f"    [{mark}] claim {record['claim_index']} gold={record['gold']}"
                  f" {record['baseline_prediction']} -> {record['prediction']}")
            print(f"           {record['adjudication_reason']}")
            print(f"           counter[{record['counter_section']}"
                  f" doc={record['counter_evidence_doc_id']}]:"
                  f" {str(record['counter_evidence'])[:92]}")
    print("==========================================================================================\n")


def get_eval_subset(unique_claims: list, n_claims=None, stratified: bool = False, seed: int = 42) -> list:
    """Select a subset of claims for evaluation, optionally with stratified sampling by gold label."""
    if n_claims is None or n_claims >= len(unique_claims):
        return list(unique_claims)

    if not stratified:
        return list(unique_claims[:n_claims])

    import random
    rng = random.Random(seed)
    by_label = {}
    for item in unique_claims:
        by_label.setdefault(item["evidence_label"], []).append(item)

    sampled = []
    total_avail = len(unique_claims)
    for label, items in by_label.items():
        label_n = max(1, round(n_claims * len(items) / total_avail))
        sampled_items = rng.sample(items, min(label_n, len(items)))
        sampled.extend(sampled_items)

    # Reconcile exact target count if rounding differed
    if len(sampled) > n_claims:
        sampled = sampled[:n_claims]
    elif len(sampled) < n_claims:
        remaining = [c for c in unique_claims if c not in sampled]
        sampled.extend(rng.sample(remaining, n_claims - len(sampled)))

    return sampled


def run_eval(
    n_claims=None,
    stratified: bool = False,
    seed: int = 42,
    output_dir: str = "eval",
    save_json: bool = True,
    evidence_mode: str = DEFAULT_EVIDENCE_MODE,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    section_weight: float = DEFAULT_SECTION_WEIGHT,
    prosecution_mode: str = DEFAULT_PROSECUTION_MODE,
):
    """Run evaluation over the SciFact dataset with accuracy and dual faithfulness reporting.

    The accuracy and faithfulness methodology is unchanged by the evidence
    mode: `evidence_mode` / `max_sentences` / `section_weight` only select how
    the evidence premise is extracted before the identical verifier and the
    identical faithfulness computation run on it. That is what makes ablation
    arms A/B/C/D comparable.

    `prosecution_mode="passive"` (arm E) changes no prediction at all: it adds
    a counter-evidence search whose findings are recorded alongside the
    unchanged metrics. Its accuracy/faithfulness numbers must match the same
    configuration run with prosecution_mode="off"; the prosecution diagnostics
    are reported in a separate block and replace nothing.
    """
    claims_dataset, corpus_dataset = load_scifact()
    valid_claims = get_valid_claims(claims_dataset)
    unique_claims = get_unique_claims(valid_claims)
    documents = build_document_corpus(corpus_dataset)

    pipeline = JurisVerifyPipeline(
        documents,
        evidence_mode=evidence_mode,
        max_sentences=max_sentences,
        section_weight=section_weight,
        prosecution_mode=prosecution_mode,
    )

    eval_claims = get_eval_subset(unique_claims, n_claims=n_claims, stratified=stratified, seed=seed)

    print(f"Starting evaluation on {len(eval_claims)} claims (total pool: {len(unique_claims)} unique claims)...")
    start_time = time.time()

    correct = 0
    wrong_cases = []
    prosecution_records = []
    strict_faithfulness_scores = []
    weighted_faithfulness_scores = []
    strict_by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    weighted_by_label = {"SUPPORT": [], "CONTRADICT": [], "NEI": []}
    all_eval_details = []

    for idx, item in enumerate(eval_claims, 1):
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

        claim_record = {
            "claim_index": idx,
            "claim": claim_text,
            "gold": gold,
            "prediction": prediction,
            "verdict": result["verdict"],
            "evidence": result["evidence"],
            "evidence_doc_id": result.get("evidence_doc_id"),
            "strict_faithfulness": strict_score,
            "weighted_faithfulness": weighted_score,
            "statements": weighted_details,
        }

        # Experimental modes only: record which scientific section each
        # selected sentence came from, so the ablation can be analysed by
        # section without re-running extraction. Baseline ("single") records
        # keep exactly the key set they had before.
        # Arm E only: carry the prosecution record onto the claim record.
        # Purely additive -- the prediction fields above are already final.
        if prosecution_mode != "off":
            for key in sorted(PROSECUTION_RESULT_KEYS | ADJUDICATION_RESULT_KEYS):
                if key in result:
                    claim_record[key] = result[key]
            # Arm F only: what arm D WOULD have predicted, so the tradeoff can
            # be read off a single arm-F run. The scored `prediction` above is
            # untouched and still comes from result["verdict"] via the
            # unchanged normalize_for_scifact_eval.
            if "baseline_verdict" in result:
                claim_record["baseline_prediction"] = normalize_for_scifact_eval(
                    result["baseline_verdict"]
                )
            prosecution_records.append(claim_record)

        spans = result.get("evidence_spans")
        if spans is not None:
            claim_record["evidence_spans"] = spans
            claim_record["evidence_sections"] = [span["section"] for span in spans]
            primary = next((sp for sp in spans if sp["rank"] == 0), None)
            claim_record["primary_evidence_section"] = (
                primary["section"] if primary else None
            )

        all_eval_details.append(claim_record)

        if prediction == gold:
            correct += 1
        else:
            wrong_cases.append(claim_record)

    elapsed = time.time() - start_time
    sec_per_claim = elapsed / len(eval_claims) if eval_claims else 0.0

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

    print(f"\n============================ EVALUATION SUMMARY ============================")
    print(f"Total Evaluated Claims: {len(eval_claims)} (of {len(unique_claims)} total available)")
    print(f"Correct Predictions:    {correct}")
    print(f"Accuracy:               {accuracy}%")
    print(f"Execution Wall Time:    {elapsed:.2f}s ({sec_per_claim:.2f}s per claim)")

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
        print(f"  - {label:<12} ({len(s_scores):>3} claims): Strict = {s_avg:.4f} | Weighted = {w_avg:.4f}")
    print(f"=================================================================================\n")

    prosecution_diagnostics = (
        compute_prosecution_diagnostics(prosecution_records)
        if prosecution_mode != "off"
        else None
    )
    if prosecution_diagnostics is not None:
        print_prosecution_diagnostics(prosecution_diagnostics)

    adjudication_diagnostics = (
        compute_adjudication_diagnostics(prosecution_records)
        if prosecution_mode == "active"
        else None
    )
    if adjudication_diagnostics is not None:
        print_adjudication_diagnostics(adjudication_diagnostics)

    results_payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_claims": len(eval_claims),
            "total_available_unique_claims": len(unique_claims),
            "stratified": stratified,
            "seed": seed,
            "evidence_mode": evidence_mode,
            "max_sentences": (
                max_sentences if evidence_mode in ("multi", "section") else None
            ),
            "section_weight": section_weight if evidence_mode == "section" else None,
            "prosecution_mode": prosecution_mode,
            "runtime_seconds": round(elapsed, 2),
            "seconds_per_claim": round(sec_per_claim, 4),
        },
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(eval_claims),
            "strict_faithfulness": avg_strict_faithfulness,
            "weighted_faithfulness": avg_weighted_faithfulness,
            "strict_faithfulness_by_label": {
                label: round(sum(scores) / len(scores), 4) if scores else 0.0
                for label, scores in strict_by_label.items()
            },
            "weighted_faithfulness_by_label": {
                label: round(sum(scores) / len(scores), 4) if scores else 0.0
                for label, scores in weighted_by_label.items()
            },
            "claim_counts_by_label": {
                label: len(scores) for label, scores in strict_by_label.items()
            },
        },
        "wrong_cases": wrong_cases,
        "eval_details": all_eval_details,
    }

    # Separate blocks: the existing "metrics" above are untouched by arms E/F.
    if prosecution_diagnostics is not None:
        results_payload["prosecution_diagnostics"] = prosecution_diagnostics
    if adjudication_diagnostics is not None:
        results_payload["adjudication_diagnostics"] = adjudication_diagnostics

    if save_json:
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Non-baseline runs get a mode suffix so experimental artifacts can
        # never be confused with the validated single-sentence results.
        if evidence_mode == "single":
            mode_suffix = ""
        elif evidence_mode == "section":
            # Weight goes in the filename too: two "section" arms that differ
            # only by section_weight are different experiments.
            mode_suffix = f"_section{max_sentences}_w{section_weight:g}"
        else:
            mode_suffix = f"_{evidence_mode}{max_sentences}"
        if prosecution_mode != "off":
            # Arm E artifacts must never be mistaken for their arm D twin,
            # even though the prediction metrics are identical by design.
            mode_suffix += f"_pros{prosecution_mode}"
        out_filename = os.path.join(
            output_dir, f"results_{timestamp_str}_n{len(eval_claims)}{mode_suffix}.json"
        )
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(results_payload, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation results to: {out_filename}")

        # If evaluating first 50 claims unstratified, update/archive results_n50.json.
        # Baseline mode only: an experimental run must not overwrite the
        # tracked reference artifact.
        if (
            len(eval_claims) == 50
            and not stratified
            and evidence_mode == "single"
            and prosecution_mode == "off"
        ):
            n50_file = os.path.join(output_dir, "results_n50.json")
            with open(n50_file, "w", encoding="utf-8") as f:
                json.dump(results_payload, f, indent=2, ensure_ascii=False)
            print(f"Archived n=50 reference results to: {n50_file}")

    return results_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Juris-Verify Scaled Evaluation Harness")
    parser.add_argument(
        "--n_claims",
        type=str,
        default="504",
        help="Number of claims to evaluate (e.g. '50', '504', 'all'). Default: '504' (full train set)",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Perform stratified sampling by gold label when n_claims < total",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval",
        help="Directory to save JSON results artifacts (default: 'eval')",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable saving JSON results file",
    )
    parser.add_argument(
        "--evidence_mode",
        type=str,
        default=DEFAULT_EVIDENCE_MODE,
        choices=list(EVIDENCE_MODES),
        help=(
            "Evidence extraction mode: 'single' (validated baseline, default), "
            "'multi' (experimental multi-sentence MMR bundle) or 'section' "
            "(experimental: that bundle with section-aware ranking)"
        ),
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=DEFAULT_MAX_SENTENCES,
        help=(
            f"Bundle size cap for --evidence_mode multi/section "
            f"(1..{MAX_BUNDLE_SENTENCES}, default: {DEFAULT_MAX_SENTENCES}); "
            "ignored in single mode"
        ),
    )
    parser.add_argument(
        "--section_weight",
        type=float,
        default=DEFAULT_SECTION_WEIGHT,
        help=(
            f"Section-prior magnitude for --evidence_mode section "
            f"(default: {DEFAULT_SECTION_WEIGHT}); 0.0 reduces it exactly to "
            "multi, negative inverts the prior (control arm). Ignored in "
            "single/multi modes"
        ),
    )
    parser.add_argument(
        "--prosecution_mode",
        type=str,
        default=DEFAULT_PROSECUTION_MODE,
        choices=list(PROSECUTION_MODES),
        help=(
            "Prosecution mode: 'off' (default, arm D), 'passive' (arm E -- "
            "records counter-evidence and changes no prediction, so its "
            "numbers must match the identical run with 'off') or 'active' "
            "(arm F -- the Adjudicator may flip a baseline SUPPORT to "
            "CONTRADICT when it validates the contradiction)"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_val = None if args.n_claims.lower() in ["all", "none"] else int(args.n_claims)
    run_eval(
        n_claims=n_val,
        stratified=args.stratified,
        seed=args.seed,
        output_dir=args.output_dir,
        save_json=not args.no_save,
        evidence_mode=args.evidence_mode,
        max_sentences=args.max_sentences,
        section_weight=args.section_weight,
        prosecution_mode=args.prosecution_mode,
    )

