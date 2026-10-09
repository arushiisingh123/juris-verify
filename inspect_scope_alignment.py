"""
Scope Alignment Diagnostic -- can the existing embedding model tell a direct
refutation from an NLI contradiction caused by outcome/scope mismatch?

READ-ONLY DIAGNOSTIC. Changes nothing: no arm F logic, no new gate, no
threshold, no prediction, no retrieval, no second model, no LLM. It loads the
project's existing SentenceTransformer (src.retrieval.EMBEDDING_MODEL_NAME) and
reports cosine similarities with the same `sentence_transformers.util.cos_sim`
that src/extraction.py already uses.

THE QUESTION
------------
Arm F overrode 3 claims on the stratified n=50 run: two correct (gold
CONTRADICT) and one wrong (gold SUPPORT). The hypothesis under test is that the
false override differs from the true ones in SCOPE: a genuine refutation
addresses the same population and the same outcome as the claim, in the
opposite direction, while a scope-mismatched sentence contradicts something the
claim never asserted (a different endpoint, a different subgroup, a harm rather
than the primary outcome).

If that is true, embedding cosine should already see it -- a refutation that
shares the claim's entities and outcome should sit CLOSER to the claim than a
sentence about a different endpoint does. This script measures whether it does.

WHAT IT REPORTS, PER CASE
-------------------------
    cos(claim, counter)      is the counter-evidence on the claim's topic?
    cos(claim, primary)      how close was the premise arm D actually chose?
    cos(primary, counter)    do the two evidence sentences discuss the same
                             thing, or different aspects of the paper?

plus two derived margins that need no threshold to be informative:

    counter_minus_primary = cos(claim,counter) - cos(claim,primary)
    scope_alignment       = min(cos(claim,counter), cos(primary,counter))
                            -- a counter-evidence sentence that is close to BOTH
                            the claim and the premise is discussing the paper's
                            main outcome; one that is close to neither is
                            discussing something else.

Arm D's selected spans are also scored, since the artifact records them. The
FULL prosecutor candidate pool is NOT recoverable from the artifact (it stores
the chosen counter-evidence, not every candidate), and rebuilding it would
require re-running retrieval -- explicitly out of scope here.

USAGE
-----
    # The real diagnostic, over the 3 overridden claims
    python inspect_scope_alignment.py \\
        --artifact eval/results_20261009_073610_n50_section2_w0.05_prosactive.json

    # Any claim indices
    python inspect_scope_alignment.py --artifact <path> --claims 8,9,46

    # Calibration probe: constructed refutation vs scope-mismatch pairs,
    # scored with the same model. Needs no artifact.
    python inspect_scope_alignment.py --probe
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

DEFAULT_ARTIFACT = "eval/results_20261009_073610_n50_section2_w0.05_prosactive.json"
DEFAULT_CLAIMS = "8,9,46"


def load_embedder():
    """The project's existing embedding model. No new model is introduced."""
    from sentence_transformers import SentenceTransformer

    from src.retrieval import EMBEDDING_MODEL_NAME

    print(f"Loading the project's embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME), EMBEDDING_MODEL_NAME


def cosine(embedder, left, right):
    """cos(left, right) using the same util.cos_sim src/extraction.py uses."""
    from sentence_transformers import util

    if not left or not right:
        return None
    vectors = embedder.encode([left, right], convert_to_tensor=True)
    return round(float(util.cos_sim(vectors[0], vectors[1])[0][0]), 4)


def extract_cases(artifact, claim_indices):
    """Pull the overridden claims out of an arm F artifact.

    Reads `eval_details` (claim text, primary evidence, prosecution and
    adjudication fields) and cross-checks against
    `adjudication_diagnostics.overridden_claims` when present.
    """
    with open(artifact, encoding="utf-8") as handle:
        payload = json.load(handle)

    metadata = payload.get("metadata", {})
    by_index = {d["claim_index"]: d for d in payload.get("eval_details", [])}

    overridden = {
        record["claim_index"]: record
        for record in (
            payload.get("adjudication_diagnostics", {}).get("overridden_claims", [])
        )
    }

    cases = []
    for index in claim_indices:
        detail = by_index.get(index)
        if detail is None:
            print(f"  WARNING: claim_index {index} not found in {artifact}")
            continue
        cases.append(
            {
                "claim_index": index,
                "claim": detail.get("claim"),
                "gold": detail.get("gold"),
                "baseline_prediction": detail.get("baseline_prediction"),
                "prediction": detail.get("prediction"),
                "primary_evidence": detail.get("evidence"),
                "primary_doc_id": detail.get("evidence_doc_id"),
                "counter_evidence": detail.get("counter_evidence"),
                "counter_doc_id": detail.get("counter_evidence_doc_id"),
                "counter_section": detail.get("counter_section"),
                "counter_p_contradiction": detail.get("counter_p_contradiction"),
                "reverse_p_contradiction": detail.get("reverse_p_contradiction"),
                "primary_counter_p_contradiction": detail.get(
                    "primary_counter_p_contradiction"
                ),
                "adjudication_status": detail.get("adjudication_status"),
                "adjudication_reason": detail.get("adjudication_reason"),
                "override_applied": detail.get("override_applied"),
                "counter_in_d_selection": detail.get("counter_in_d_selection"),
                "candidate_pool_size": detail.get("candidate_pool_size"),
                "evidence_spans": detail.get("evidence_spans") or [],
                "listed_as_override": index in overridden,
            }
        )
    return metadata, cases


def score_case(embedder, case):
    """Attach the three requested similarities plus the derived margins."""
    claim = case["claim"]
    primary = case["primary_evidence"]
    counter = case["counter_evidence"]

    case["cos_claim_counter"] = cosine(embedder, claim, counter)
    case["cos_claim_primary"] = cosine(embedder, claim, primary)
    case["cos_primary_counter"] = cosine(embedder, primary, counter)

    if case["cos_claim_counter"] is not None and case["cos_claim_primary"] is not None:
        case["counter_minus_primary"] = round(
            case["cos_claim_counter"] - case["cos_claim_primary"], 4
        )
    else:
        case["counter_minus_primary"] = None

    if case["cos_claim_counter"] is not None and case["cos_primary_counter"] is not None:
        case["scope_alignment"] = round(
            min(case["cos_claim_counter"], case["cos_primary_counter"]), 4
        )
    else:
        case["scope_alignment"] = None

    # Arm D's own selected spans, which the artifact does record.
    case["span_similarities"] = [
        {
            "rank": span.get("rank"),
            "section": span.get("section"),
            "doc_id": span.get("doc_id"),
            "selection_relevance": span.get("relevance"),
            "cos_claim_span": cosine(embedder, claim, span.get("sentence")),
            "cos_counter_span": cosine(embedder, counter, span.get("sentence")),
            "sentence": span.get("sentence"),
        }
        for span in case["evidence_spans"]
    ]
    return case


def print_cases(cases, model_name):
    print("\n" + "=" * 78)
    print("SCOPE ALIGNMENT DIAGNOSTIC -- arm F overridden claims")
    print(f"embedding model: {model_name}   (no prediction was modified)")
    print("=" * 78)

    for case in cases:
        verdict_tag = (
            "VALID OVERRIDE" if case["gold"] == "CONTRADICT" else "FALSE OVERRIDE"
        )
        print(f"\n--- Claim {case['claim_index']} | gold={case['gold']} | {verdict_tag} ---")
        print(f"  {case['baseline_prediction']} -> {case['prediction']}"
              f"   status={case['adjudication_status']}")
        print(f"  reason: {case['adjudication_reason']}")
        print(f"\n  CLAIM:\n    {case['claim']}")
        print(f"\n  PRIMARY EVIDENCE (doc {case['primary_doc_id']}):\n"
              f"    {case['primary_evidence']}")
        print(f"\n  COUNTER-EVIDENCE (doc {case['counter_doc_id']},"
              f" section={case['counter_section']},"
              f" in_bundle={case['counter_in_d_selection']}):\n"
              f"    {case['counter_evidence']}")
        print("\n  SEMANTIC SIMILARITIES (MiniLM cosine):")
        print(f"    cos(claim,   counter)   = {case['cos_claim_counter']}")
        print(f"    cos(claim,   primary)   = {case['cos_claim_primary']}")
        print(f"    cos(primary, counter)   = {case['cos_primary_counter']}")
        print(f"    counter_minus_primary   = {case['counter_minus_primary']}")
        print(f"    scope_alignment (min)   = {case['scope_alignment']}")
        print("\n  NLI probabilities already recorded by arm F (for contrast):")
        print(f"    P(contradiction) counter->claim   = {case['counter_p_contradiction']}")
        print(f"    P(contradiction) claim->counter   = {case['reverse_p_contradiction']}")
        print(f"    P(contradiction) primary->counter = "
              f"{case['primary_counter_p_contradiction']}")
        if case["span_similarities"]:
            print("\n  Arm D selected spans:")
            for span in case["span_similarities"]:
                print(f"    rank {span['rank']} [{span['section']}]"
                      f" cos(claim,span)={span['cos_claim_span']}"
                      f" cos(counter,span)={span['cos_counter_span']}")
                print(f"      {str(span['sentence'])[:92]}")

    # The comparison that answers the question.
    valid = [c for c in cases if c["gold"] == "CONTRADICT"]
    false = [c for c in cases if c["gold"] == "SUPPORT"]
    if valid and false:
        print("\n" + "=" * 78)
        print("SEPARATION CHECK -- valid overrides vs false override")
        print("=" * 78)
        header = f"  {'feature':<26}{'valid (gold C)':>26}{'false (gold S)':>18}"
        print(header)
        for feature in (
            "cos_claim_counter",
            "cos_claim_primary",
            "cos_primary_counter",
            "counter_minus_primary",
            "scope_alignment",
        ):
            valid_values = [c[feature] for c in valid if c[feature] is not None]
            false_values = [c[feature] for c in false if c[feature] is not None]
            valid_text = ", ".join(f"{v:.4f}" for v in valid_values)
            false_text = ", ".join(f"{v:.4f}" for v in false_values)
            print(f"  {feature:<26}{valid_text:>26}{false_text:>18}")

            if valid_values and false_values:
                gap_low = min(valid_values) - max(false_values)
                gap_high = min(false_values) - max(valid_values)
                if gap_low > 0:
                    note = f"separates: valid all higher by >= {gap_low:.4f}"
                elif gap_high > 0:
                    note = f"separates: valid all lower by >= {gap_high:.4f}"
                else:
                    note = "OVERLAPS -- no separation on this feature"
                print(f"  {'':<26}{note}")
        print("\n  n=3 (2 valid, 1 false). Separation here is a NECESSARY but not")
        print("  sufficient condition for a scope-alignment gate: a single false")
        print("  override cannot establish a decision boundary.")
    print("=" * 78 + "\n")


# ==========================================================================
# Calibration probe: does the model see scope at all?
# ==========================================================================
#
# Constructed pairs, written to isolate ONE difference. Group A sentences
# refute the claim on the SAME population and SAME outcome. Group B sentences
# are the kind NLI calls contradiction for the wrong reason: they conflict with
# something the claim never asserted -- a different endpoint, a different
# population, or a harm rather than the primary outcome.
#
# This does not use the real cases. It asks the narrower question: is cosine
# sensitive to scope at all, or is it dominated by topical overlap?

PROBE = [
    {
        "claim": "Vitamin D supplementation reduces the risk of fractures in older adults.",
        "primary": (
            "OBJECTIVE To determine whether vitamin D supplementation reduces "
            "fracture risk in community-dwelling older adults."
        ),
        "aligned_refutation": (
            "RESULTS Fracture incidence did not differ between the vitamin D and "
            "placebo groups among adults older than 70 years."
        ),
        "scope_mismatch": (
            "RESULTS Vitamin D supplementation increased the incidence of "
            "hypercalcemia in breastfed infants."
        ),
        "hard_mismatch": (
            "RESULTS Vitamin D supplementation did not reduce the risk of falls in older adults."
        ),
    },
    {
        "claim": "Aspirin reduces cardiovascular mortality in diabetic patients.",
        "primary": (
            "OBJECTIVE To assess whether low-dose aspirin lowers cardiovascular "
            "death among adults with type 2 diabetes."
        ),
        "aligned_refutation": (
            "RESULTS Cardiovascular death occurred more frequently in the aspirin "
            "arm than in the placebo arm among participants with diabetes."
        ),
        "scope_mismatch": (
            "RESULTS Aspirin significantly increased the rate of major "
            "gastrointestinal bleeding events."
        ),
        "hard_mismatch": (
            "RESULTS Aspirin did not reduce all-cause mortality in patients with diabetes."
        ),
    },
    {
        "claim": "Statin therapy lowers LDL cholesterol in adults with hyperlipidemia.",
        "primary": (
            "OBJECTIVE To evaluate the effect of statin therapy on lipid levels in "
            "adults with hyperlipidemia."
        ),
        "aligned_refutation": (
            "RESULTS LDL cholesterol levels did not differ significantly between "
            "statin-treated and untreated adults with hyperlipidemia."
        ),
        "scope_mismatch": (
            "RESULTS Statin therapy did not reduce all-cause mortality over five "
            "years of follow-up."
        ),
        "hard_mismatch": (
            "RESULTS Statin therapy did not lower triglyceride levels in adults with hyperlipidemia."
        ),
    },
    {
        "claim": "Antibiotic prophylaxis prevents surgical site infection after colorectal surgery.",
        "primary": (
            "OBJECTIVE To determine whether preoperative antibiotics prevent "
            "surgical site infection in colorectal procedures."
        ),
        "aligned_refutation": (
            "RESULTS Surgical site infection rates were higher in the antibiotic "
            "prophylaxis group than in the control group after colorectal surgery."
        ),
        "scope_mismatch": (
            "RESULTS Antibiotic prophylaxis was associated with a marked rise in "
            "Clostridium difficile colitis among hospitalized patients."
        ),
        "hard_mismatch": (
            "RESULTS Antibiotic prophylaxis did not prevent anastomotic leakage after colorectal surgery."
        ),
    },
    {
        "claim": "Intensive glucose control reduces microvascular complications in type 2 diabetes.",
        "primary": (
            "OBJECTIVE To compare intensive and standard glucose control for "
            "microvascular outcomes in type 2 diabetes."
        ),
        "aligned_refutation": (
            "RESULTS Microvascular complications were no less frequent under "
            "intensive glucose control than under standard control in type 2 diabetes."
        ),
        "scope_mismatch": (
            "RESULTS Intensive glucose control increased the frequency of severe "
            "hypoglycemic episodes requiring assistance."
        ),
        "hard_mismatch": (
            "RESULTS Intensive glucose control did not reduce macrovascular complications in type 2 diabetes."
        ),
    },
]


def run_probe(embedder, model_name):
    print("\n" + "=" * 78)
    print("CALIBRATION PROBE -- is MiniLM cosine sensitive to scope at all?")
    print(f"embedding model: {model_name}")
    print("=" * 78)
    print("\nConstructed pairs. Group A refutes the claim on the SAME population and")
    print("outcome. Group B contradicts something the claim never asserted (a")
    print("different endpoint, population, or a harm rather than the primary")
    print("outcome). These are NOT the real arm F cases.\n")

    rows = []
    for case in PROBE:
        aligned = {
            "claim": case["claim"],
            "primary_evidence": case["primary"],
            "counter_evidence": case["aligned_refutation"],
            "evidence_spans": [],
            "group": "A_aligned_refutation",
        }
        mismatch = {
            "claim": case["claim"],
            "primary_evidence": case["primary"],
            "counter_evidence": case["scope_mismatch"],
            "evidence_spans": [],
            "group": "B_scope_mismatch_distant_vocabulary",
        }
        # Group C is the HARD case and the honest one: same drug, same
        # population, near-identical phrasing, but a DIFFERENT endpoint
        # (falls vs fractures, all-cause vs cardiovascular mortality,
        # macro- vs microvascular). This is what a real scope mismatch inside
        # one abstract looks like, and it controls for the vocabulary
        # confound that makes group B easy.
        hard = {
            "claim": case["claim"],
            "primary_evidence": case["primary"],
            "counter_evidence": case["hard_mismatch"],
            "evidence_spans": [],
            "group": "C_scope_mismatch_shared_vocabulary",
        }
        rows.append(score_case(embedder, aligned))
        rows.append(score_case(embedder, mismatch))
        rows.append(score_case(embedder, hard))

    print(f"  {'grp':<4}{'c,counter':>11}{'c,primary':>11}{'p,counter':>11}"
          f"{'cnt-prim':>11}{'scope_al':>11}   claim")
    for row in rows:
        tag = row["group"][0]
        print(f"  {tag:<4}{row['cos_claim_counter']:>11.4f}"
              f"{row['cos_claim_primary']:>11.4f}{row['cos_primary_counter']:>11.4f}"
              f"{row['counter_minus_primary']:>11.4f}{row['scope_alignment']:>11.4f}"
              f"   {row['claim'][:38]}")

    print("\n  Group means. A vs C is the test that matters: group B shares little")
    print("  vocabulary with the claim and is therefore the easy case. Group C is")
    print("  what a real scope mismatch inside one abstract looks like.")
    print(f"  {'feature':<24}{'A aligned':>11}{'B distant':>11}{'C shared':>11}"
          f"{'A-C gap':>10}{'A vs B':>10}{'A vs C':>10}")
    verdicts = {}
    for feature in (
        "cos_claim_counter",
        "cos_claim_primary",
        "cos_primary_counter",
        "counter_minus_primary",
        "scope_alignment",
    ):
        values = {
            group: [r[feature] for r in rows if r["group"].startswith(group)]
            for group in ("A", "B", "C")
        }
        means = {g: sum(v) / len(v) for g, v in values.items()}
        sep_ab = (
            min(values["A"]) > max(values["B"]) or min(values["B"]) > max(values["A"])
        )
        sep_ac = (
            min(values["A"]) > max(values["C"]) or min(values["C"]) > max(values["A"])
        )
        verdicts[feature] = {"A_vs_B": sep_ab, "A_vs_C": sep_ac}
        print(f"  {feature:<24}{means['A']:>11.4f}{means['B']:>11.4f}"
              f"{means['C']:>11.4f}{means['A'] - means['C']:>+10.4f}"
              f"{('SEP' if sep_ab else 'overlap'):>10}"
              f"{('SEP' if sep_ac else 'OVERLAP'):>10}")
        print(f"  {'':<24}  A range [{min(values['A']):.4f}, {max(values['A']):.4f}]"
              f"   C range [{min(values['C']):.4f}, {max(values['C']):.4f}]")

    sep_ab = [name for name, v in verdicts.items() if v["A_vs_B"]]
    sep_ac = [name for name, v in verdicts.items() if v["A_vs_C"]]
    print(f"\n  Separate A from B (easy, distant vocabulary): {sep_ab or 'NONE'}")
    print(f"  Separate A from C (hard, shared vocabulary):   {sep_ac or 'NONE'}")
    if not sep_ac:
        print("\n  >>> VERDICT: A does NOT separate from C on any feature.")
        print("  >>> On hand-built, maximally clean examples this embedding cannot")
        print("  >>> tell a direct refutation from a same-vocabulary endpoint")
        print("  >>> mismatch. Since real SciFact pairs are no cleaner than these,")
        print("  >>> a scope-alignment gate on this representation is NOT justified.")
        print("  >>> Group B separating only shows cosine tracks topical overlap,")
        print("  >>> which the pipeline already exploits -- it is not scope.")
    else:
        print("\n  >>> A separates from C on: " + str(sep_ac))
        print("  >>> Necessary condition met. Still not sufficient: measure the real")
        print("  >>> arm F cases before building any gate.")
    print("=" * 78 + "\n")
    return rows, verdicts


def main():
    parser = argparse.ArgumentParser(
        description="Scope alignment diagnostic (read-only; no arm F changes)"
    )
    parser.add_argument("--artifact", type=str, default=DEFAULT_ARTIFACT,
                        help="arm F results JSON (prosecution_mode=active)")
    parser.add_argument("--claims", type=str, default=DEFAULT_CLAIMS,
                        help="comma-separated claim_index values (default: 8,9,46)")
    parser.add_argument("--probe", action="store_true",
                        help="run the constructed-pair calibration probe and exit")
    parser.add_argument("--out", type=str, default=None,
                        help="optional path to write the diagnostic JSON")
    args = parser.parse_args()

    embedder, model_name = load_embedder()

    if args.probe:
        rows, verdicts = run_probe(embedder, model_name)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(
                    {"probe": rows, "separating_features": verdicts},
                    handle, indent=2, ensure_ascii=False,
                )
            print(f"Saved probe results to: {args.out}")
        return 0

    if not os.path.exists(args.artifact):
        print(f"\nERROR: artifact not found: {args.artifact}")
        print("Arm F ran in Colab, so the results JSON is there, not in this repo.")
        print("Copy it into eval/ (or pass --artifact <path>) and re-run.")
        print("To check the embedding signal without it: --probe\n")
        return 2

    claim_indices = [int(token) for token in args.claims.split(",") if token.strip()]
    metadata, cases = extract_cases(args.artifact, claim_indices)
    if not cases:
        print("No matching claims found; nothing to report.")
        return 2

    print(f"artifact: {args.artifact}")
    print(f"  evidence_mode={metadata.get('evidence_mode')}"
          f" max_sentences={metadata.get('max_sentences')}"
          f" section_weight={metadata.get('section_weight')}"
          f" prosecution_mode={metadata.get('prosecution_mode')}")

    cases = [score_case(embedder, case) for case in cases]
    print_cases(cases, model_name)

    out = args.out or os.path.join("diagnostics", "scope_alignment.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "experiment": "scope_alignment_diagnostic",
                    "diagnostic_only": True,
                    "embedding_model": model_name,
                    "source_artifact": args.artifact,
                    "source_metadata": metadata,
                },
                "cases": cases,
            },
            handle, indent=2, ensure_ascii=False,
        )
    print(f"Saved diagnostic to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
