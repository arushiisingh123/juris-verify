"""
Arm E acceptance check: prove prosecution_mode="passive" changes no prediction.

Arm E is only useful as a control if its predictions are IDENTICAL to arm D's.
This script proves that on the real 504 claims and exits non-zero on the first
discrepancy, so "if any prediction changes, STOP and debug" is enforced rather
than eyeballed.

WHY IT RUNS BOTH MODES IN ONE PROCESS
-------------------------------------
`HybridRetriever._build_semantic_index` calls
`chromadb.Client().create_collection(name="scifact_collection")` with a fixed
name, so constructing a second JurisVerifyPipeline in the same process raises
`InternalError: Collection [scifact_collection] already exists` (verified on
chromadb 1.5.9). It would also load a second DeBERTa-v3-large.

So one pipeline is built and `prosecution_mode` is flipped on the instance
between the two passes. That shares the corpus index and both models, makes the
comparison exact by construction, and costs one index build instead of two.

The baseline NLI call is unaffected by the flip in either direction: it is
always a single-item call made before prosecution runs, so its inputs are
bit-identical in both passes. Prosecution's own batched calls only ever
influence the prosecution record.

USAGE
-----
    # Full acceptance check (GPU, same environment as eval/evaluate.py)
    python verify_arm_e_identity.py --n_claims 504

    # Compare two artifacts already produced by eval/evaluate.py instead
    python verify_arm_e_identity.py --compare eval/<armD>.json eval/<armE>.json

    # Offline logic self-test -- no models, no dataset
    python verify_arm_e_identity.py --selftest

Exit code 0 means every prediction matched. Non-zero means it did not.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Arm D, exactly as eval/evaluate.py configures it.
ARM_D = {"evidence_mode": "section", "max_sentences": 2, "section_weight": 0.05}

# Fields that must be byte-identical between the two passes. These are every
# prediction-bearing field predict_claim produces.
PREDICTION_FIELDS = (
    "verdict",
    "matrix_label",
    "evidence",
    "evidence_doc_id",
    "evidence_doc_ids",
    "evidence_spans",
)


def compare_predictions(off_records, passive_records):
    """Diff two prediction sequences field by field.

    Returns (mismatches, compared) where each mismatch names the claim index,
    the field, and both values.
    """
    mismatches = []
    if len(off_records) != len(passive_records):
        mismatches.append(
            {
                "claim_index": None,
                "field": "record_count",
                "off": len(off_records),
                "passive": len(passive_records),
            }
        )
        return mismatches, 0

    for off, passive in zip(off_records, passive_records):
        if off["claim"] != passive["claim"]:
            mismatches.append(
                {
                    "claim_index": off["claim_index"],
                    "field": "claim",
                    "off": off["claim"],
                    "passive": passive["claim"],
                }
            )
            continue
        for field in PREDICTION_FIELDS:
            if off.get(field) != passive.get(field):
                mismatches.append(
                    {
                        "claim_index": off["claim_index"],
                        "field": field,
                        "off": off.get(field),
                        "passive": passive.get(field),
                    }
                )
        if off["prediction"] != passive["prediction"]:
            mismatches.append(
                {
                    "claim_index": off["claim_index"],
                    "field": "prediction",
                    "off": off["prediction"],
                    "passive": passive["prediction"],
                }
            )
    return mismatches, len(off_records)


def summarize(records):
    """Accuracy overall and per gold label, using the recorded predictions."""
    by_label = {}
    correct = 0
    for record in records:
        gold = record["gold"]
        hit = record["prediction"] == gold
        correct += hit
        bucket = by_label.setdefault(gold, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += hit

    rate = lambda c, n: round(c / n * 100, 2) if n else 0.0
    return {
        "total": len(records),
        "correct": correct,
        "overall_accuracy": rate(correct, len(records)),
        "by_label": {
            label: {
                "n": stats["n"],
                "correct": stats["correct"],
                "accuracy": rate(stats["correct"], stats["n"]),
            }
            for label, stats in sorted(by_label.items())
        },
    }


def print_summary(title, summary):
    print(f"  {title}")
    print(f"    overall      {summary['overall_accuracy']:>6}%  "
          f"({summary['correct']}/{summary['total']})")
    for label, stats in summary["by_label"].items():
        print(f"    {label:<12} {stats['accuracy']:>6}%  "
              f"({stats['correct']}/{stats['n']})")


def run_identity_check(n_claims, stratified, seed, batch_size):
    from src.data_loader import (
        build_document_corpus,
        get_unique_claims,
        get_valid_claims,
        load_scifact,
    )
    from src.pipeline import JurisVerifyPipeline
    from src.verification import normalize_for_scifact_eval
    from eval.evaluate import compute_prosecution_diagnostics, print_prosecution_diagnostics

    claims_dataset, corpus_dataset = load_scifact()
    unique_claims = get_unique_claims(get_valid_claims(claims_dataset))
    documents = build_document_corpus(corpus_dataset)

    from eval.evaluate import get_eval_subset

    eval_claims = get_eval_subset(
        unique_claims, n_claims=n_claims, stratified=stratified, seed=seed
    )

    # ONE pipeline; prosecution_mode is flipped on the instance between passes.
    pipeline = JurisVerifyPipeline(documents, **ARM_D, prosecution_mode="off")

    print(f"Arm E acceptance check over {len(eval_claims)} claims.")
    print(f"  arm D config: {ARM_D}")
    print("  pass 1: prosecution_mode='off'   (arm D)")
    print("  pass 2: prosecution_mode='passive' (arm E)")
    print("  sharing one retriever index and one DeBERTa instance.\n")

    def run_pass(mode):
        pipeline.prosecution_mode = mode
        records = []
        start = time.time()
        for index, item in enumerate(eval_claims, 1):
            result = pipeline.predict_claim(item["claim"])
            record = {
                "claim_index": index,
                "claim": item["claim"],
                "gold": item["evidence_label"],
                "prediction": normalize_for_scifact_eval(result["verdict"]),
            }
            for field in PREDICTION_FIELDS:
                if field in result:
                    record[field] = result[field]
            if mode != "off":
                for key, value in result.items():
                    if key.startswith(("prosecution_", "counter_", "candidate_", "n_contradictory", "max_p")):
                        record[key] = value
            records.append(record)
            if index % 50 == 0 or index == len(eval_claims):
                print(f"    [{mode}] {index}/{len(eval_claims)} "
                      f"({time.time() - start:.1f}s)")
        return records

    off_records = run_pass("off")
    passive_records = run_pass("passive")

    mismatches, compared = compare_predictions(off_records, passive_records)

    print("\n" + "=" * 78)
    print("ARM D vs ARM E -- PREDICTION IDENTITY")
    print("=" * 78)
    off_summary = summarize(off_records)
    passive_summary = summarize(passive_records)
    print_summary("Arm D (prosecution_mode=off):", off_summary)
    print()
    print_summary("Arm E (prosecution_mode=passive):", passive_summary)

    print(f"\n  claims compared:       {compared}")
    print(f"  prediction mismatches: {len(mismatches)}")
    if mismatches:
        print("\n  FIRST MISMATCHES:")
        for mismatch in mismatches[:10]:
            print(f"    claim {mismatch['claim_index']} field {mismatch['field']!r}")
            print(f"      off:     {str(mismatch['off'])[:160]}")
            print(f"      passive: {str(mismatch['passive'])[:160]}")
    print("=" * 78)

    if mismatches:
        print("\nFAILED: arm E altered the measured path. STOP and debug before\n"
              "treating any arm-E artifact as a valid control.\n")
        return 1, off_summary, passive_summary, passive_records

    print("\nPASSED: arm E reproduced every arm-D prediction exactly.\n")
    diagnostics = compute_prosecution_diagnostics(passive_records)
    print_prosecution_diagnostics(diagnostics)
    return 0, off_summary, passive_summary, passive_records


def run_compare(path_a, path_b):
    """Compare two eval/evaluate.py artifacts instead of re-running."""
    with open(path_a, encoding="utf-8") as handle:
        payload_a = json.load(handle)
    with open(path_b, encoding="utf-8") as handle:
        payload_b = json.load(handle)

    def to_records(payload):
        records = []
        for detail in payload["eval_details"]:
            record = {
                "claim_index": detail["claim_index"],
                "claim": detail["claim"],
                "gold": detail["gold"],
                "prediction": detail["prediction"],
            }
            for field in PREDICTION_FIELDS:
                if field in detail:
                    record[field] = detail[field]
            records.append(record)
        return records

    print(f"A: {path_a}")
    print(f"   {payload_a['metadata'].get('evidence_mode')} / "
          f"prosecution={payload_a['metadata'].get('prosecution_mode')}")
    print(f"B: {path_b}")
    print(f"   {payload_b['metadata'].get('evidence_mode')} / "
          f"prosecution={payload_b['metadata'].get('prosecution_mode')}\n")

    mismatches, compared = compare_predictions(to_records(payload_a), to_records(payload_b))
    print_summary("A:", summarize(to_records(payload_a)))
    print()
    print_summary("B:", summarize(to_records(payload_b)))
    print(f"\n  claims compared:       {compared}")
    print(f"  prediction mismatches: {len(mismatches)}")
    for mismatch in mismatches[:10]:
        print(f"    claim {mismatch['claim_index']} field {mismatch['field']!r}")
    if mismatches:
        print("\nFAILED: the two artifacts disagree on at least one prediction.\n")
        return 1
    print("\nPASSED: identical predictions across both artifacts.\n")
    return 0


def _selftest():
    failures = []

    def check(name, got, expected):
        if got != expected:
            failures.append(f"{name}: got {got!r}, expected {expected!r}")

    def rec(index, gold, prediction, verdict, evidence="e"):
        return {
            "claim_index": index,
            "claim": f"claim {index}",
            "gold": gold,
            "prediction": prediction,
            "verdict": verdict,
            "matrix_label": "AGREE",
            "evidence": evidence,
            "evidence_doc_id": "doc-1",
        }

    off = [
        rec(1, "CONTRADICT", "CONTRADICT", "CONTRADICTED"),
        rec(2, "SUPPORT", "SUPPORT", "PARTIALLY_SUPPORTED"),
        rec(3, "CONTRADICT", "SUPPORT", "PARTIALLY_SUPPORTED"),
    ]

    # Identical passes -> no mismatches.
    mismatches, compared = compare_predictions(off, [dict(r) for r in off])
    check("identical mismatches", mismatches, [])
    check("identical compared", compared, 3)

    # A changed verdict is caught, and so is the derived prediction.
    perturbed = [dict(r) for r in off]
    perturbed[2]["verdict"] = "CONTRADICTED"
    perturbed[2]["prediction"] = "CONTRADICT"
    mismatches, _ = compare_predictions(off, perturbed)
    check("verdict change detected", len(mismatches), 2)
    check("verdict field named", mismatches[0]["field"], "verdict")
    check("prediction field named", mismatches[1]["field"], "prediction")

    # A changed premise is caught even when the verdict survives.
    perturbed = [dict(r) for r in off]
    perturbed[1]["evidence"] = "something else"
    mismatches, _ = compare_predictions(off, perturbed)
    check("evidence change detected", len(mismatches), 1)
    check("evidence field named", mismatches[0]["field"], "evidence")

    # Length and ordering problems are caught.
    mismatches, compared = compare_predictions(off, off[:2])
    check("length mismatch detected", mismatches[0]["field"], "record_count")
    check("length mismatch compared", compared, 0)
    reordered = [off[1], off[0], off[2]]
    mismatches, _ = compare_predictions(off, reordered)
    check("reorder detected", mismatches[0]["field"], "claim")

    # Accuracy summary.
    summary = summarize(off)
    check("summary total", summary["total"], 3)
    check("summary correct", summary["correct"], 2)
    check("summary overall", summary["overall_accuracy"], round(2 / 3 * 100, 2))
    check("summary CONTRADICT", summary["by_label"]["CONTRADICT"]["accuracy"], 50.0)
    check("summary SUPPORT", summary["by_label"]["SUPPORT"]["accuracy"], 100.0)
    check("empty summary", summarize([])["overall_accuracy"], 0.0)

    print("\n" + "=" * 78)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} checks)")
        for failure in failures:
            print("  -", failure)
        print("=" * 78)
        return 1
    print("SELFTEST PASSED -- prediction diffing (verdict, premise, ordering,")
    print("length) and accuracy summarisation verified offline.")
    print("=" * 78)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prove arm E (passive prosecution) changes no arm-D prediction"
    )
    parser.add_argument("--n_claims", type=str, default="504")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compare", nargs=2, metavar=("ARM_D_JSON", "ARM_E_JSON"),
                        help="compare two eval artifacts instead of re-running")
    parser.add_argument("--selftest", action="store_true",
                        help="run the offline logic self-test and exit")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()
    if args.compare:
        return run_compare(*args.compare)

    n_claims = None if args.n_claims.lower() in ("all", "none") else int(args.n_claims)
    status, _, _, _ = run_identity_check(
        n_claims, args.stratified, args.seed, args.batch_size
    )
    return status


if __name__ == "__main__":
    sys.exit(main())
