"""
Arm H diagnostic: ORACLE CEILING for contradiction detection.

READ-ONLY DIAGNOSTIC. This script imports the existing pipeline and scores
candidate sentences with the existing verifier. It modifies nothing:
  - no change to evidence selection, retrieval, or the verifier;
  - no change to normalize_for_scifact_eval or any faithfulness computation;
  - the arm-D prediction it reports comes from JurisVerifyPipeline.predict_claim
    itself, unaltered;
  - output goes to its own directory (default ./diagnostics), never eval/.

QUESTION IT ANSWERS
-------------------
Arm D (section-aware Multi-2) reaches 66.47% CONTRADICT accuracy. Of the
~33.5% it misses, how many are missed because the contradictory sentence was
NEVER RETRIEVED, versus missed because it WAS in the candidate pool and simply
was not selected (or was selected but diluted)? The first is a retrieval
problem; the second is a selection/calibration problem. Building a Prosecutor
is only worthwhile in the second case, and only up to the ceiling measured
here.

WHAT IT DOES, PER CLAIM
-----------------------
1. `pipeline.predict_claim(claim)` -- the authoritative arm-D prediction, on
   the exact measured path `eval/evaluate.py` uses. Not reimplemented here.
2. `pipeline.retriever.retrieve(claim)` -- the SAME retrieval configuration
   (semantic_k=3, bm25_k=3, top_n=2 defaults). Retrieval is deterministic
   (fixed embeddings, BM25, CrossEncoder, stable sort), so this reproduces the
   document set step 1 used internally. The reproduction is CHECKED, not
   assumed: arm D's rank-0 evidence span must appear in the reconstructed pool,
   and a mismatch is counted and reported rather than silently ignored.
3. Truncate each document to [:3000] -- identical to pipeline.predict_claim.
4. Build the candidate sentence pool with the extractor's own
   `_collect_candidate_sentences`. This deliberately reuses the private helper
   instead of re-deriving the pool: it is the exact deduplicated,
   section-labelled pool arm D ranks over, and reimplementing it would risk
   diverging from what D actually saw.
5. Score EVERY pool sentence with the existing NLI verifier, premise=sentence,
   hypothesis=claim -- the same orientation `NLIVerifier.verify` uses.
6. Record whether ANY candidate receives a raw NLI verdict of CONTRADICTED.

The oracle result is recorded only. It NEVER feeds back into the arm-D
prediction; the rules in section D/G of the report are computed afterwards from
the recorded data, as a what-if.

HOW THE 3-WAY DISTRIBUTION IS OBTAINED WITHOUT TOUCHING src/verification.py
---------------------------------------------------------------------------
`NLIVerifier.verify` returns the argmax label plus that label's score, so
P(contradiction) is not available through it. This script instead calls the
already-constructed HF pipeline object on the already-loaded model --
`pipeline.verifier.nli_pipeline` -- with `top_k=None`, which returns all three
class probabilities. Nothing is modified; a public attribute is read.

That path is only trustworthy if it agrees with production, so the agreement is
VERIFIED at runtime: for the first `--selfcheck N` claims the script also calls
`pipeline.verifier.verify(...)` on the same sentence and asserts the derived
verdict matches. The derivation takes the top-scoring entry of the `top_k=None`
output, which is the same entry `verify` reads as `result[0]`, and applies the
identical label mapping -- so agreement is expected by construction and the
check exists to prove it rather than to discover it.

RUN
---
Requires the same environment as eval/evaluate.py (GPU; NLIVerifier defaults to
device=0) plus the SciFact dataset. See the report for the exact command.

    python inspect_oracle_ceiling.py --n_claims 504

Offline logic self-test -- no models, no dataset, runs in under a second:

    python inspect_oracle_ceiling.py --selftest
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- arm D, exactly as eval/evaluate.py configures it -----------------------
ARM_D = {
    "evidence_mode": "section",
    "max_sentences": 2,
    "section_weight": 0.05,
}

# Label strings, matching src/verification.py's mapping.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"


# ==========================================================================
# NLI scoring (reads the existing verifier's pipeline; changes nothing)
# ==========================================================================

def nli_results(nli_pipeline, pairs, batch_size=16):
    """Score (premise, hypothesis) pairs, returning the full 3-way distribution.

    Args:
        nli_pipeline: the HF text-classification pipeline held by NLIVerifier
        pairs: list of (premise, hypothesis) tuples. Premise is the candidate
            sentence and hypothesis the claim -- the orientation
            NLIVerifier.verify uses (`{"text": evidence, "text_pair": claim}`).
        batch_size: forwarded to the pipeline

    Returns:
        list of (top_label, distribution) in input order, where top_label is
        the lower-cased argmax label exactly as NLIVerifier.verify would read
        it, and distribution maps every lower-cased label to its probability.

    Raises on an unexpected output shape rather than degrading quietly -- a
    silently truncated distribution would corrupt every threshold in section G.
    """
    if not pairs:
        return []

    inputs = [{"text": premise, "text_pair": hypothesis} for premise, hypothesis in pairs]
    raw = nli_pipeline(inputs, top_k=None, batch_size=batch_size)

    # top_k=None yields a 3-entry list per input, sorted by score descending:
    # list[dict] for a single input, list[list[dict]] for several.
    if len(inputs) == 1 and raw and isinstance(raw[0], dict):
        raw = [raw]

    if len(raw) != len(inputs):
        raise RuntimeError(
            "NLI pipeline returned {} results for {} inputs".format(len(raw), len(inputs))
        )

    results = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[0], dict):
            raise RuntimeError(
                "expected a multi-class score list per input (top_k=None), got: "
                + repr(entry)[:200]
            )
        distribution = {item["label"].lower(): float(item["score"]) for item in entry}
        # entry[0] is the highest-scoring class, i.e. what verify() reads.
        results.append((entry[0]["label"].lower(), distribution))
    return results


def verdict_from_top_label(top_label):
    """Map an NLI label to a verdict, replicating NLIVerifier.verify exactly.

    verify() lower-cases the label and branches entailment -> SUPPORTED,
    contradiction -> CONTRADICTED, anything else -> PARTIALLY_SUPPORTED. The
    final `else` is reproduced as an else so an unexpected label behaves the
    same here as it would in production.
    """
    if top_label == "entailment":
        return SUPPORTED
    elif top_label == "contradiction":
        return CONTRADICTED
    else:
        return PARTIALLY_SUPPORTED


def p_contradiction(distribution):
    """P(contradiction); 0.0 if the model exposes no such class."""
    return float(distribution.get("contradiction", 0.0))


# ==========================================================================
# Per-claim diagnostic
# ==========================================================================

def diagnose_claim(
    claim,
    gold,
    claim_index,
    pipeline,
    collect_pool,
    normalize_prediction,
    batch_size=16,
    selfcheck=False,
):
    """Record arm D's prediction alongside the oracle scan of its candidate pool.

    `collect_pool` and `normalize_prediction` are injected so the offline
    self-test can exercise this logic without models.
    """
    d_result = pipeline.predict_claim(claim)
    d_verdict = d_result["verdict"]
    d_prediction = normalize_prediction(d_verdict)

    d_spans = d_result.get("evidence_spans") or []
    d_primary = next((span for span in d_spans if span["rank"] == 0), None)
    d_span_texts = {span["sentence"] for span in d_spans}

    # Same retrieval config, same truncation, same pool construction as D.
    candidate_docs = pipeline.retriever.retrieve(claim)
    truncated = [
        {"doc_id": doc["doc_id"], "text": doc["text"][:3000]} for doc in candidate_docs
    ]
    pool = collect_pool(truncated)

    # Did the reconstruction reproduce what D actually ranked over?
    pool_texts = [item["sentence"] for item in pool]
    pool_reproduced = d_primary is None or d_primary["sentence"] in pool_texts

    scored = nli_results(
        pipeline.verifier.nli_pipeline,
        [(item["sentence"], claim) for item in pool],
        batch_size=batch_size,
    )

    candidates = []
    for item, (top_label, distribution) in zip(pool, scored):
        candidates.append(
            {
                "sentence": item["sentence"],
                "doc_id": item["doc_id"],
                "section": item["section"],
                "nli_verdict": verdict_from_top_label(top_label),
                "p_contradiction": p_contradiction(distribution),
                "distribution": distribution,
                "selected_by_d": item["sentence"] in d_span_texts,
            }
        )

    # Prove the top_k=None derivation agrees with the production verify().
    selfcheck_result = None
    if selfcheck and d_primary is not None:
        primary_scored = nli_results(
            pipeline.verifier.nli_pipeline,
            [(d_primary["sentence"], claim)],
            batch_size=1,
        )
        derived = verdict_from_top_label(primary_scored[0][0])
        production = pipeline.verifier.verify(claim, d_primary["sentence"])
        selfcheck_result = {
            "derived": derived,
            "production": production,
            "agree": derived == production,
        }

    contradictory = [c for c in candidates if c["nli_verdict"] == CONTRADICTED]
    # Deterministic oracle pick: strongest contradiction, ties to pool order.
    oracle = None
    if contradictory:
        best = max(
            range(len(contradictory)),
            key=lambda i: (contradictory[i]["p_contradiction"], -i),
        )
        oracle = contradictory[best]

    max_p = max((c["p_contradiction"] for c in candidates), default=0.0)

    return {
        "claim_index": claim_index,
        "claim": claim,
        "gold": gold,
        "d_verdict": d_verdict,
        "d_prediction": d_prediction,
        "d_evidence": d_result["evidence"],
        "d_evidence_doc_id": d_result.get("evidence_doc_id"),
        "d_spans": [
            {
                "sentence": span["sentence"],
                "doc_id": span["doc_id"],
                "section": span.get("section"),
                "rank": span["rank"],
                "relevance": span.get("relevance"),
                "ranking_score": span.get("ranking_score"),
            }
            for span in d_spans
        ],
        "candidate_count": len(candidates),
        "pool_doc_ids": sorted({str(item["doc_id"]) for item in pool}),
        "pool_reproduced": pool_reproduced,
        "oracle_has_contradiction": oracle is not None,
        "oracle_sentence": oracle["sentence"] if oracle else None,
        "oracle_doc_id": oracle["doc_id"] if oracle else None,
        "oracle_section": oracle["section"] if oracle else None,
        "oracle_nli_verdict": oracle["nli_verdict"] if oracle else None,
        "oracle_distribution": oracle["distribution"] if oracle else None,
        "oracle_p_contradiction": oracle["p_contradiction"] if oracle else 0.0,
        "oracle_in_d_selection": bool(oracle and oracle["selected_by_d"]),
        "n_contradictory_candidates": len(contradictory),
        "max_p_contradiction": max_p,
        "selfcheck": selfcheck_result,
        "candidates": candidates,
    }


# ==========================================================================
# Diagnostics A-G
# ==========================================================================

def _rate(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _confusion(records, predict):
    """2x2 confusion plus accuracies for a what-if rule.

    `predict(record)` returns "CONTRADICT" or "SUPPORT".
    """
    counts = {}
    for record in records:
        counts[(record["gold"], predict(record))] = (
            counts.get((record["gold"], predict(record)), 0) + 1
        )
    n_contradict = sum(1 for r in records if r["gold"] == "CONTRADICT")
    n_support = sum(1 for r in records if r["gold"] == "SUPPORT")
    contradict_correct = counts.get(("CONTRADICT", "CONTRADICT"), 0)
    support_correct = counts.get(("SUPPORT", "SUPPORT"), 0)
    return {
        "confusion": {
            "CONTRADICT_as_CONTRADICT": contradict_correct,
            "CONTRADICT_as_SUPPORT": counts.get(("CONTRADICT", "SUPPORT"), 0),
            "SUPPORT_as_CONTRADICT": counts.get(("SUPPORT", "CONTRADICT"), 0),
            "SUPPORT_as_SUPPORT": support_correct,
        },
        "overall_accuracy": _rate(contradict_correct + support_correct, len(records)),
        "contradict_accuracy": _rate(contradict_correct, n_contradict),
        "support_accuracy": _rate(support_correct, n_support),
        "correct": contradict_correct + support_correct,
        "total": len(records),
    }


def compute_diagnostics(records, thresholds=None):
    """Compute sections A-G from the recorded per-claim data."""
    if thresholds is None:
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]

    contradict = [r for r in records if r["gold"] == "CONTRADICT"]
    support = [r for r in records if r["gold"] == "SUPPORT"]
    other = [r for r in records if r["gold"] not in ("CONTRADICT", "SUPPORT")]

    hit = lambda r: r["oracle_has_contradiction"]
    d_says_contradict = lambda r: r["d_prediction"] == "CONTRADICT"

    # --- A: ceiling on CONTRADICT ---------------------------------------
    c_hit = [r for r in contradict if hit(r)]
    section_a = {
        "gold_contradict_total": len(contradict),
        "with_contradictory_candidate": len(c_hit),
        "max_possible_contradict_accuracy": _rate(len(c_hit), len(contradict)),
    }

    # --- B: false-positive exposure -------------------------------------
    s_hit = [r for r in support if hit(r)]
    section_b = {
        "gold_support_total": len(support),
        "with_contradictory_candidate": len(s_hit),
        "false_positive_exposure_rate": _rate(len(s_hit), len(support)),
    }

    # --- C: where the CONTRADICT misses actually come from ---------------
    d_correct = [r for r in contradict if d_says_contradict(r)]
    d_missed = [r for r in contradict if not d_says_contradict(r)]
    rescuable = [r for r in d_missed if hit(r)]
    absent = [r for r in d_missed if not hit(r)]
    # Of the rescuable, was the contradictory sentence already inside D's
    # 2-sentence bundle? If so the premise was diluted, not mis-selected.
    diluted = [r for r in rescuable if r["oracle_in_d_selection"]]
    not_selected = [r for r in rescuable if not r["oracle_in_d_selection"]]
    section_c = {
        "gold_contradict_total": len(contradict),
        "d_correct": len(d_correct),
        "d_missed": len(d_missed),
        "missed_but_contradiction_in_pool": len(rescuable),
        "missed_contradiction_in_d_selection_premise_diluted": len(diluted),
        "missed_contradiction_in_pool_not_selected": len(not_selected),
        "missed_no_contradiction_anywhere_in_pool": len(absent),
        "share_of_misses_recoverable_without_retrieval": _rate(
            len(rescuable), len(d_missed)
        ),
        "share_of_misses_requiring_retrieval_change": _rate(len(absent), len(d_missed)),
    }

    # --- the three-way split the experiment was asked to separate --------
    three_way = {
        "contradiction_in_pool_and_predicted_by_d": len(d_correct),
        "contradiction_in_pool_but_not_predicted_by_d": len(rescuable),
        "contradiction_absent_from_pool": len(absent),
    }

    # --- D: the naive oracle rule, two readings -------------------------
    scored_records = [r for r in records if r["gold"] in ("CONTRADICT", "SUPPORT")]
    section_d = {
        "baseline_arm_d": _confusion(scored_records, lambda r: r["d_prediction"]),
        "override_on_hit_else_keep_d": _confusion(
            scored_records,
            lambda r: "CONTRADICT" if hit(r) else r["d_prediction"],
        ),
        "pure_oracle_rule": _confusion(
            scored_records, lambda r: "CONTRADICT" if hit(r) else "SUPPORT"
        ),
    }

    # --- E / F: precision and recall of the oracle signal ---------------
    tp, fp = len(c_hit), len(s_hit)
    section_e = {
        "true_positives_gold_contradict_with_hit": tp,
        "false_positives_gold_support_with_hit": fp,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
    }
    section_f = {
        "true_positives": tp,
        "all_gold_contradict": len(contradict),
        "recall": round(tp / len(contradict), 4) if contradict else 0.0,
    }

    # --- G: threshold sweep on P(contradiction) -------------------------
    baseline = section_d["baseline_arm_d"]
    sweep = []
    for tau in thresholds:
        over = lambda r, t=tau: r["max_p_contradiction"] >= t
        row = {
            "threshold": tau,
            "override_else_keep_d": _confusion(
                scored_records,
                lambda r, t=tau: "CONTRADICT" if over(r, t) else r["d_prediction"],
            ),
            "pure_rule": _confusion(
                scored_records,
                lambda r, t=tau: "CONTRADICT" if over(r, t) else "SUPPORT",
            ),
            "n_triggered_gold_contradict": sum(1 for r in contradict if over(r)),
            "n_triggered_gold_support": sum(1 for r in support if over(r)),
        }
        triggered = row["n_triggered_gold_contradict"] + row["n_triggered_gold_support"]
        row["precision_at_threshold"] = (
            round(row["n_triggered_gold_contradict"] / triggered, 4) if triggered else 0.0
        )
        sweep.append(row)

    best_overall = max(
        sweep, key=lambda row: row["override_else_keep_d"]["overall_accuracy"]
    )
    # Best CONTRADICT that does not regress SUPPORT below arm D's level.
    no_regression = [
        row
        for row in sweep
        if row["override_else_keep_d"]["support_accuracy"]
        >= baseline["support_accuracy"] - 1e-9
    ]
    best_no_regression = (
        max(
            no_regression,
            key=lambda row: row["override_else_keep_d"]["contradict_accuracy"],
        )
        if no_regression
        else None
    )
    section_g = {
        "sweep": sweep,
        "best_overall_accuracy": best_overall,
        "best_contradict_without_support_regression": best_no_regression,
    }

    # --- integrity + context --------------------------------------------
    pool_sizes = [r["candidate_count"] for r in records]
    integrity = {
        "pool_reproduction_failures": sum(1 for r in records if not r["pool_reproduced"]),
        "selfchecks_run": sum(1 for r in records if r["selfcheck"]),
        "selfcheck_disagreements": sum(
            1 for r in records if r["selfcheck"] and not r["selfcheck"]["agree"]
        ),
        "gold_labels_outside_support_contradict": len(other),
        "mean_candidate_pool_size": round(sum(pool_sizes) / len(pool_sizes), 2)
        if pool_sizes
        else 0.0,
        "min_candidate_pool_size": min(pool_sizes) if pool_sizes else 0,
        "max_candidate_pool_size": max(pool_sizes) if pool_sizes else 0,
    }

    # Which sections do the oracle's contradictory sentences live in? Tells us
    # whether the existing section prior is already pointed at them.
    oracle_sections = {}
    for record in contradict:
        if hit(record):
            key = record["oracle_section"] or "unknown"
            oracle_sections[key] = oracle_sections.get(key, 0) + 1

    return {
        "A_contradict_ceiling": section_a,
        "B_support_false_positive_exposure": section_b,
        "C_miss_decomposition": section_c,
        "three_way_split": three_way,
        "D_naive_rule_impact": section_d,
        "E_precision": section_e,
        "F_recall": section_f,
        "G_threshold_sweep": section_g,
        "oracle_section_distribution_gold_contradict": oracle_sections,
        "integrity": integrity,
    }


# ==========================================================================
# Reporting
# ==========================================================================

def print_report(diag):
    a, b, c = (
        diag["A_contradict_ceiling"],
        diag["B_support_false_positive_exposure"],
        diag["C_miss_decomposition"],
    )
    d, e, f, g = (
        diag["D_naive_rule_impact"],
        diag["E_precision"],
        diag["F_recall"],
        diag["G_threshold_sweep"],
    )
    integ = diag["integrity"]

    print("\n" + "=" * 78)
    print("ARM H -- ORACLE CEILING DIAGNOSTIC (no prediction was modified)")
    print("=" * 78)

    print("\n[INTEGRITY]")
    print(f"  pool reproduction failures:      {integ['pool_reproduction_failures']}")
    print(
        f"  verify() selfcheck:              {integ['selfchecks_run']} run, "
        f"{integ['selfcheck_disagreements']} disagreements"
    )
    print(
        f"  candidate pool size:             mean {integ['mean_candidate_pool_size']}, "
        f"range {integ['min_candidate_pool_size']}-{integ['max_candidate_pool_size']}"
    )

    print("\n[A] CONTRADICT ceiling in the current retrieval pool")
    print(f"  gold CONTRADICT claims:                      {a['gold_contradict_total']}")
    print(f"  with >=1 raw-CONTRADICTED candidate:         {a['with_contradictory_candidate']}")
    print(f"  max possible CONTRADICT accuracy:            {a['max_possible_contradict_accuracy']}%")

    print("\n[B] False-positive exposure on gold SUPPORT")
    print(f"  gold SUPPORT claims:                         {b['gold_support_total']}")
    print(f"  with >=1 raw-CONTRADICTED candidate:         {b['with_contradictory_candidate']}")
    print(f"  exposure rate:                               {b['false_positive_exposure_rate']}%")

    print("\n[C] Where arm D's CONTRADICT misses come from")
    print(f"  arm D correct:                               {c['d_correct']}")
    print(f"  arm D missed:                                {c['d_missed']}")
    print(f"    -> contradiction WAS in the pool:          {c['missed_but_contradiction_in_pool']}")
    print(f"         of which inside D's bundle (diluted): {c['missed_contradiction_in_d_selection_premise_diluted']}")
    print(f"         of which never selected:              {c['missed_contradiction_in_pool_not_selected']}")
    print(f"    -> no contradiction anywhere in pool:      {c['missed_no_contradiction_anywhere_in_pool']}")
    print(f"  recoverable without retrieval change:        {c['share_of_misses_recoverable_without_retrieval']}% of misses")
    print(f"  requires retrieval change:                   {c['share_of_misses_requiring_retrieval_change']}% of misses")

    print("\n[THREE-WAY SPLIT]")
    for key, value in diag["three_way_split"].items():
        print(f"  {key:<48} {value}")

    print("\n[D] Naive rule: any candidate CONTRADICTED -> predict CONTRADICT")
    for name in ("baseline_arm_d", "override_on_hit_else_keep_d", "pure_oracle_rule"):
        row = d[name]
        cm = row["confusion"]
        print(f"  {name}:")
        print(
            f"    overall {row['overall_accuracy']}%  "
            f"CONTRADICT {row['contradict_accuracy']}%  "
            f"SUPPORT {row['support_accuracy']}%   ({row['correct']}/{row['total']})"
        )
        print(
            f"    confusion  C->C {cm['CONTRADICT_as_CONTRADICT']}  "
            f"C->S {cm['CONTRADICT_as_SUPPORT']}  "
            f"S->C {cm['SUPPORT_as_CONTRADICT']}  "
            f"S->S {cm['SUPPORT_as_SUPPORT']}"
        )

    print("\n[E] Precision of the oracle contradictory signal")
    print(f"  TP {e['true_positives_gold_contradict_with_hit']}  "
          f"FP {e['false_positives_gold_support_with_hit']}  -> precision {e['precision']}")
    print("\n[F] Recall of the oracle contradictory signal")
    print(f"  TP {f['true_positives']} / {f['all_gold_contradict']}  -> recall {f['recall']}")

    print("\n[G] Threshold sweep on max P(contradiction) over the pool")
    print(f"  {'tau':>5}  {'overall':>8}  {'CONTRA':>8}  {'SUPPORT':>8}  {'trig C':>7}  {'trig S':>7}  {'prec':>6}")
    for row in g["sweep"]:
        r = row["override_else_keep_d"]
        print(
            f"  {row['threshold']:>5}  {r['overall_accuracy']:>7}%  "
            f"{r['contradict_accuracy']:>7}%  {r['support_accuracy']:>7}%  "
            f"{row['n_triggered_gold_contradict']:>7}  {row['n_triggered_gold_support']:>7}  "
            f"{row['precision_at_threshold']:>6}"
        )
    best = g["best_overall_accuracy"]
    print(
        f"  best overall: tau={best['threshold']} -> "
        f"{best['override_else_keep_d']['overall_accuracy']}% overall, "
        f"{best['override_else_keep_d']['contradict_accuracy']}% CONTRADICT"
    )
    bnr = g["best_contradict_without_support_regression"]
    if bnr:
        print(
            f"  best CONTRADICT with no SUPPORT regression: tau={bnr['threshold']} -> "
            f"{bnr['override_else_keep_d']['contradict_accuracy']}% CONTRADICT, "
            f"{bnr['override_else_keep_d']['overall_accuracy']}% overall"
        )
    else:
        print("  no threshold avoids a SUPPORT regression")

    dist = diag["oracle_section_distribution_gold_contradict"]
    if dist:
        print("\n[CONTEXT] Section of the oracle sentence (gold CONTRADICT, hits only)")
        for key, value in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<14} {value}")
    print("=" * 78 + "\n")


# ==========================================================================
# Offline self-test: verifies all of the arithmetic with no models/dataset
# ==========================================================================

def _selftest():
    failures = []

    def check(name, got, expected):
        if got != expected:
            failures.append(f"{name}: got {got!r}, expected {expected!r}")

    # --- verdict mapping mirrors NLIVerifier.verify ----------------------
    check("verdict entailment", verdict_from_top_label("entailment"), SUPPORTED)
    check("verdict contradiction", verdict_from_top_label("contradiction"), CONTRADICTED)
    check("verdict neutral", verdict_from_top_label("neutral"), PARTIALLY_SUPPORTED)
    check("verdict unexpected", verdict_from_top_label("weird"), PARTIALLY_SUPPORTED)
    check("p_contradiction missing", p_contradiction({"neutral": 1.0}), 0.0)
    check("p_contradiction present", p_contradiction({"contradiction": 0.7}), 0.7)

    # --- the mapping is identical to the real verify() implementation ----
    import src.verification as V

    class _FakePipe:
        def __init__(self, label):
            self.label = label

        def __call__(self, payload, **kwargs):
            return [{"label": self.label, "score": 0.9}]

    for label in ("entailment", "contradiction", "neutral", "ENTAILMENT", "weird"):
        verifier = V.NLIVerifier.__new__(V.NLIVerifier)  # no model load
        verifier.nli_pipeline = _FakePipe(label)
        check(
            "verify parity for " + label,
            verdict_from_top_label(label.lower()),
            verifier.verify("claim", "evidence"),
        )

    # --- pipeline output-shape handling ---------------------------------
    class _BatchPipe:
        def __init__(self, mode):
            self.mode = mode

        def __call__(self, inputs, top_k=None, batch_size=None):
            assert top_k is None, "diagnostic must request the full distribution"
            three = lambda p: sorted(
                [
                    {"label": "contradiction", "score": p},
                    {"label": "neutral", "score": 1.0 - p - 0.01},
                    {"label": "entailment", "score": 0.01},
                ],
                key=lambda d: -d["score"],
            )
            if self.mode == "single":
                return three(0.8)
            return [three(0.8) for _ in inputs]

    got = nli_results(_BatchPipe("single"), [("premise", "claim")])
    check("single-input shape label", got[0][0], "contradiction")
    check("single-input shape classes", len(got[0][1]), 3)
    got = nli_results(_BatchPipe("batch"), [("a", "c"), ("b", "c")])
    check("batch shape length", len(got), 2)
    check("empty input", nli_results(_BatchPipe("batch"), []), [])

    class _TruncatingPipe:
        def __call__(self, inputs, top_k=None, batch_size=None):
            return [[{"label": "neutral", "score": 1.0}] for _ in inputs]

    try:
        nli_results(_TruncatingPipe(), [("a", "c"), ("b", "c")])
        failures.append("truncated distribution should have raised")
    except RuntimeError:
        pass

    # --- end-to-end per-claim record with injected fakes -----------------
    SENT_OBJ = "OBJECTIVE To determine whether the treatment reduces mortality."
    SENT_RES = "RESULTS Mortality was higher in the treatment arm than placebo."
    SENT_MET = "METHODS Participants were randomized across twelve study sites."
    scripted = {SENT_OBJ: 0.05, SENT_RES: 0.93, SENT_MET: 0.10}

    class _FakeNLI:
        def __call__(self, inputs, top_k=None, batch_size=None):
            out = []
            for item in inputs:
                p = scripted.get(item["text"], 0.02)
                out.append(
                    sorted(
                        [
                            {"label": "contradiction", "score": p},
                            {"label": "neutral", "score": 1.0 - p - 0.01},
                            {"label": "entailment", "score": 0.01},
                        ],
                        key=lambda d: -d["score"],
                    )
                )
            return out

    class _FakeVerifier:
        def __init__(self):
            self.nli_pipeline = _FakeNLI()

        def verify(self, claim, evidence, return_dict=False):
            p = scripted.get(evidence, 0.02)
            return CONTRADICTED if p > 0.5 else PARTIALLY_SUPPORTED

    class _FakeRetriever:
        def retrieve(self, claim):
            return [{"doc_id": "doc-A", "text": "T " + " ".join(scripted)}]

    class _FakePipeline:
        def __init__(self):
            self.verifier = _FakeVerifier()
            self.retriever = _FakeRetriever()

        def predict_claim(self, claim):
            return {
                "verdict": PARTIALLY_SUPPORTED,
                "evidence": SENT_OBJ,
                "evidence_doc_id": "doc-A",
                "evidence_spans": [
                    {
                        "sentence": SENT_OBJ,
                        "doc_id": "doc-A",
                        "section": "background",
                        "rank": 0,
                        "relevance": 0.6,
                        "ranking_score": 0.55,
                    }
                ],
            }

    pool = [
        {"sentence": SENT_OBJ, "doc_id": "doc-A", "section": "background"},
        {"sentence": SENT_RES, "doc_id": "doc-A", "section": "results"},
        {"sentence": SENT_MET, "doc_id": "doc-A", "section": "methods"},
    ]
    record = diagnose_claim(
        "The treatment reduces mortality.",
        "CONTRADICT",
        1,
        _FakePipeline(),
        collect_pool=lambda truncated: pool,
        normalize_prediction=lambda v: "SUPPORT"
        if v in (SUPPORTED, PARTIALLY_SUPPORTED)
        else "CONTRADICT",
        selfcheck=True,
    )
    check("record d_prediction", record["d_prediction"], "SUPPORT")
    check("record oracle hit", record["oracle_has_contradiction"], True)
    check("record oracle sentence", record["oracle_sentence"], SENT_RES)
    check("record oracle section", record["oracle_section"], "results")
    check("record oracle not in D selection", record["oracle_in_d_selection"], False)
    check("record candidate count", record["candidate_count"], 3)
    check("record max p", round(record["max_p_contradiction"], 2), 0.93)
    check("record pool reproduced", record["pool_reproduced"], True)
    check("record selfcheck agrees", record["selfcheck"]["agree"], True)

    # --- diagnostics arithmetic on a hand-built fixture ------------------
    def rec(gold, d_pred, hit, max_p, in_sel=False):
        return {
            "gold": gold,
            "d_prediction": d_pred,
            "oracle_has_contradiction": hit,
            "oracle_in_d_selection": in_sel,
            "oracle_section": "results" if hit else None,
            "max_p_contradiction": max_p,
            "candidate_count": 10,
            "pool_reproduced": True,
            "selfcheck": None,
        }

    fixture = [
        rec("CONTRADICT", "CONTRADICT", True, 0.95),              # D correct
        rec("CONTRADICT", "SUPPORT", True, 0.80),                 # rescuable, unselected
        rec("CONTRADICT", "SUPPORT", True, 0.70, in_sel=True),    # rescuable, diluted
        rec("CONTRADICT", "SUPPORT", False, 0.10),                # absent from pool
        rec("SUPPORT", "SUPPORT", False, 0.05),
        rec("SUPPORT", "SUPPORT", True, 0.60),                    # false positive
        rec("SUPPORT", "CONTRADICT", False, 0.20),
    ]
    diag = compute_diagnostics(fixture, thresholds=[0.5, 0.75, 0.9])

    check("A total", diag["A_contradict_ceiling"]["gold_contradict_total"], 4)
    check("A hits", diag["A_contradict_ceiling"]["with_contradictory_candidate"], 3)
    check("A ceiling", diag["A_contradict_ceiling"]["max_possible_contradict_accuracy"], 75.0)
    check("B hits", diag["B_support_false_positive_exposure"]["with_contradictory_candidate"], 1)
    c = diag["C_miss_decomposition"]
    check("C d_correct", c["d_correct"], 1)
    check("C d_missed", c["d_missed"], 3)
    check("C rescuable", c["missed_but_contradiction_in_pool"], 2)
    check("C diluted", c["missed_contradiction_in_d_selection_premise_diluted"], 1)
    check("C unselected", c["missed_contradiction_in_pool_not_selected"], 1)
    check("C absent", c["missed_no_contradiction_anywhere_in_pool"], 1)
    check("C recoverable share", c["share_of_misses_recoverable_without_retrieval"], 66.67)
    check(
        "three-way",
        diag["three_way_split"],
        {
            "contradiction_in_pool_and_predicted_by_d": 1,
            "contradiction_in_pool_but_not_predicted_by_d": 2,
            "contradiction_absent_from_pool": 1,
        },
    )
    check("E precision", diag["E_precision"]["precision"], round(3 / 4, 4))
    check("F recall", diag["F_recall"]["recall"], round(3 / 4, 4))

    base = diag["D_naive_rule_impact"]["baseline_arm_d"]
    check("D baseline correct", base["correct"], 3)  # 1 C->C + 2 S->S
    check("D baseline contradict acc", base["contradict_accuracy"], 25.0)
    over = diag["D_naive_rule_impact"]["override_on_hit_else_keep_d"]
    check("D override contradict acc", over["contradict_accuracy"], 75.0)
    check("D override support acc", over["support_accuracy"], round(1 / 3 * 100, 2))
    pure = diag["D_naive_rule_impact"]["pure_oracle_rule"]
    check("D pure contradict acc", pure["contradict_accuracy"], 75.0)
    check("D pure support acc", pure["support_accuracy"], round(2 / 3 * 100, 2))

    sweep = {row["threshold"]: row for row in diag["G_threshold_sweep"]["sweep"]}
    check("G tau=0.9 triggered C", sweep[0.9]["n_triggered_gold_contradict"], 1)
    check("G tau=0.9 triggered S", sweep[0.9]["n_triggered_gold_support"], 0)
    check("G tau=0.9 precision", sweep[0.9]["precision_at_threshold"], 1.0)
    check("G tau=0.5 triggered S", sweep[0.5]["n_triggered_gold_support"], 1)
    check("G tau=0.75 contradict acc", sweep[0.75]["override_else_keep_d"]["contradict_accuracy"], 50.0)
    # tau=0.75 fires on the 0.95 and 0.80 CONTRADICT claims while leaving the
    # 0.60 SUPPORT false positive alone, so SUPPORT holds at the arm-D level
    # (66.67%) and CONTRADICT doubles to 50% -- strictly better than tau=0.9,
    # which only fires on the claim arm D already got right.
    bnr = diag["G_threshold_sweep"]["best_contradict_without_support_regression"]
    check("G best-no-regression tau", bnr["threshold"], 0.75)
    check("G best-no-regression contradict acc",
          bnr["override_else_keep_d"]["contradict_accuracy"], 50.0)
    check("G best-no-regression support acc",
          bnr["override_else_keep_d"]["support_accuracy"], base["support_accuracy"])

    empty = compute_diagnostics([], thresholds=[0.5])
    check("empty ceiling", empty["A_contradict_ceiling"]["max_possible_contradict_accuracy"], 0.0)
    check("empty precision", empty["E_precision"]["precision"], 0.0)

    print("\n" + "=" * 78)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} checks)")
        for failure in failures:
            print("  -", failure)
        print("=" * 78)
        return 1
    print("SELFTEST PASSED -- mapping parity, output-shape handling, per-claim")
    print("record construction and all of sections A-G verified offline.")
    print("=" * 78)
    return 0


# ==========================================================================
# Entry point
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arm H oracle-ceiling diagnostic (read-only; modifies no predictions)"
    )
    parser.add_argument("--n_claims", type=str, default="504",
                        help="claims to diagnose ('50', '504', 'all'); default 504")
    parser.add_argument("--stratified", action="store_true",
                        help="stratified sampling by gold label when n_claims < total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="NLI batch size for scoring the candidate pool")
    parser.add_argument("--selfcheck", type=int, default=25,
                        help="claims on which to cross-check the derived verdict "
                             "against NLIVerifier.verify (default 25; 0 disables)")
    parser.add_argument("--output_dir", type=str, default="diagnostics",
                        help="where the JSON artifact goes; never eval/ (default: diagnostics)")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="run the offline logic self-test and exit (no models/dataset)")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    from src.data_loader import (
        build_document_corpus,
        get_unique_claims,
        get_valid_claims,
        load_scifact,
    )
    from src.extraction import _collect_candidate_sentences, _normalize_candidates
    from src.pipeline import JurisVerifyPipeline
    from src.verification import normalize_for_scifact_eval
    from eval.evaluate import get_eval_subset

    n_claims = None if args.n_claims.lower() in ("all", "none") else int(args.n_claims)

    claims_dataset, corpus_dataset = load_scifact()
    unique_claims = get_unique_claims(get_valid_claims(claims_dataset))
    documents = build_document_corpus(corpus_dataset)

    # Same claim population and ordering the eval harness uses.
    eval_claims = get_eval_subset(
        unique_claims, n_claims=n_claims, stratified=args.stratified, seed=args.seed
    )

    # Arm D, configured exactly as eval/evaluate.py would.
    pipeline = JurisVerifyPipeline(documents, **ARM_D)
    collect_pool = lambda truncated: _collect_candidate_sentences(
        _normalize_candidates(truncated)
    )

    print(f"Arm H diagnostic over {len(eval_claims)} claims "
          f"(arm D = {ARM_D}); scoring every candidate sentence with the "
          f"existing verifier. No prediction is modified.")
    start = time.time()

    records = []
    for index, item in enumerate(eval_claims, 1):
        records.append(
            diagnose_claim(
                item["claim"],
                item["evidence_label"],
                index,
                pipeline,
                collect_pool=collect_pool,
                normalize_prediction=normalize_for_scifact_eval,
                batch_size=args.batch_size,
                selfcheck=index <= args.selfcheck,
            )
        )
        if index % 25 == 0 or index == len(eval_claims):
            print(f"  {index}/{len(eval_claims)} claims "
                  f"({time.time() - start:.1f}s elapsed)")

    elapsed = time.time() - start
    diag = compute_diagnostics(records)
    print_report(diag)

    integ = diag["integrity"]
    if integ["pool_reproduction_failures"]:
        print(f"WARNING: candidate pool did not reproduce arm D's rank-0 span on "
              f"{integ['pool_reproduction_failures']} claims -- treat the pool-based "
              f"numbers as approximate on those.\n")
    if integ["selfcheck_disagreements"]:
        print(f"WARNING: derived verdict disagreed with NLIVerifier.verify on "
              f"{integ['selfcheck_disagreements']} of {integ['selfchecks_run']} "
              f"selfchecks -- investigate before trusting these results.\n")

    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            args.output_dir, f"oracle_ceiling_{stamp}_n{len(records)}.json"
        )
        payload = {
            "metadata": {
                "experiment": "arm_H_oracle_ceiling",
                "diagnostic_only": True,
                "note": "Oracle results were recorded only; no arm-D prediction "
                        "was modified and no evaluation methodology was changed.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "arm_d_config": ARM_D,
                "n_claims": len(records),
                "total_available_unique_claims": len(unique_claims),
                "stratified": args.stratified,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "runtime_seconds": round(elapsed, 2),
            },
            "diagnostics": diag,
            "records": records,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved diagnostic artifact to: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
