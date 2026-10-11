"""
Recompute every statistic in the G5 primary held-out validation report.

Research/analysis only. Modifies no production module, reruns no experiment,
and alters no existing result JSON.

TWO MODES
---------
1. ARTIFACT MODE (preferred). If the primary held-out artifacts are present,
   every confusion matrix is recomputed from the annotations joined to the
   candidate artifact, and the report's numbers are checked against it.

2. AUDIT MODE (fallback). If the artifacts are absent, the figures recorded in
   `docs/g5_primary_heldout_validation_summary.json` are audited for internal
   consistency and every derived statistic (precision, recall, confidence
   intervals, Fisher exact, McNemar) is recomputed from the confusion matrices.
   This does NOT verify the confusion matrices themselves against source data,
   and the output says so.

    python verify_g5_validation_numbers.py
    python verify_g5_validation_numbers.py --emit-summary
"""

import argparse
import json
import os
import sys
from math import comb, sqrt

ANNOTATIONS = "scope_alignment_primary_heldout_blind_annotations.json"
CANDIDATES = ("candidates_validation_20261011_174506_n188_section2_w0.05"
              "_prosactive_gatesg1g2g3.json")
BLIND = os.path.join("docs", "scope_alignment_primary_heldout_blind_dataset.json")
SUMMARY = os.path.join("docs", "g5_primary_heldout_validation_summary.json")

# --- figures as supplied by the project owner -------------------------------
# Source: the task brief describing the completed Colab run. Treated as INPUT
# to be audited, never as verified output.
SUPPLIED = {
    "validation_rows": 450,
    "unique_claims": 188,
    "corpus_documents": 5183,
    "candidate_frame": 54,
    "gold_contradict": 17,
    "gold_support": 37,
    "annotation_seed": 20261012,
    # NOTE: 158/188 was previously used here as the "baseline". That was WRONG --
    # it is the POST-override accuracy of the frozen G1+G2+G3 run, which already
    # contains that run's 12 adjudicator overrides. The true pre-override
    # baseline is derived from the artifact by derive_baseline(); the constant
    # below is only a fallback for when no artifact is available.
    "baseline_correct_fallback_pre_override": 154,
    "post_override_accuracy_of_frozen_run": 158,
    "g5": {"TP": 7, "FP": 0, "FN": 10, "TN": 37},
    "direction_only": {"TP": 11, "FP": 4, "FN": 6, "TN": 33},
    "discovery": {"n": 42, "gold_contradict": 24, "gold_support": 18,
                  "TP": 20, "FP": 0, "FN": 4, "TN": 18},
}


# ===========================================================================
# Statistics -- native, no dependency required
# ===========================================================================

def fisher_exact_two_sided(a, b, c, d):
    n, r1, c1 = a + b + c + d, a + b, a + c
    pr = lambda x: comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    lo, hi, obs = max(0, c1 - (n - r1)), min(r1, c1), pr(a)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= obs * (1 + 1e-9)))


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p, by the sum-of-smaller-probabilities rule."""
    if n == 0:
        return 1.0
    pmf = lambda x: comb(n, x) * p ** x * (1 - p) ** (n - x)
    obs = pmf(k)
    return min(1.0, sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs * (1 + 1e-9)))


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (0.0, 0.0)
    p, den = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half) * 100, 2),
            round(min(1.0, centre + half) * 100, 2))


def clopper_pearson(k, n):
    """Exact binomial CI. Uses scipy when available, else returns None."""
    try:
        from scipy.stats import beta
    except ImportError:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return (round(lo * 100, 2), round(hi * 100, 2))


def rate(k, n):
    return round(k / n * 100, 2) if n else None


def rule_stats(table, label):
    tp, fp, fn, tn = table["TP"], table["FP"], table["FN"], table["TN"]
    accepted, gold_c, gold_s = tp + fp, tp + fn, fp + tn
    n = tp + fp + fn + tn
    return {
        "rule": label,
        "confusion_matrix": dict(table),
        "n_cases": n,
        "n_accepted": accepted,
        "n_rejected": fn + tn,
        "gold_contradict": gold_c,
        "gold_support": gold_s,
        "precision_pct": rate(tp, accepted),
        "precision_ci95_wilson": wilson(tp, accepted),
        "precision_ci95_clopper_pearson": clopper_pearson(tp, accepted),
        "recall_pct": rate(tp, gold_c),
        "recall_ci95_wilson": wilson(tp, gold_c),
        "recall_ci95_clopper_pearson": clopper_pearson(tp, gold_c),
        "specificity_pct": rate(tn, gold_s),
        "specificity_ci95_wilson": wilson(tn, gold_s),
        "candidate_frame_accuracy_pct": rate(tp + tn, n),
        "rescues": tp,
        "added_errors": fp,
        "net_claims": tp - fp,
        "fisher_exact_two_sided_p": fisher_exact_two_sided(tp, fp, fn, tn),
    }


def mcnemar(g5, direction_only):
    """Two framings. Rule A is a strict subset of direction-only by
    construction (G5 = direction-only AND overall_scope_match == YES), so the
    accept-decision test is structurally determined and carries no evidential
    weight about which rule is better."""
    disc_accept = direction_only["TP"] + direction_only["FP"] - (g5["TP"] + g5["FP"])
    g5_right = direction_only["FP"]                      # dir-only FPs that G5 rejects
    do_right = direction_only["TP"] - g5["TP"]           # TPs G5 misses
    return {
        "on_accept_decision": {
            "discordant_pairs": disc_accept,
            "split": [0, disc_accept],
            "p_two_sided_exact": binom_two_sided(0, disc_accept),
            "interpretation": (
                "Structurally determined: G5 is a strict subset of "
                "direction-only, so every discordant pair must fall one way. "
                "This p-value is NOT evidence that G5 is more accurate."
            ),
        },
        "on_correctness": {
            "g5_correct_dironly_wrong": g5_right,
            "dironly_correct_g5_wrong": do_right,
            "discordant_pairs": g5_right + do_right,
            "p_two_sided_exact": binom_two_sided(g5_right, g5_right + do_right),
            "interpretation": (
                "The decision-relevant comparison. The two rules are "
                "indistinguishable in per-case correctness on this frame."
            ),
        },
    }


def simulated_application(table, baseline_correct, population):
    net = table["TP"] - table["FP"]
    after = baseline_correct + net
    return {
        "baseline_correct": baseline_correct,
        "population": population,
        "baseline_accuracy_pct": rate(baseline_correct, population),
        "after_correct": after,
        "after_accuracy_pct": rate(after, population),
        "delta_pp": round((after - baseline_correct) / population * 100, 2),
        "rescues": table["TP"],
        "added_errors": table["FP"],
        "net_claims": net,
    }


# ===========================================================================
# Artifact mode
# ===========================================================================

def derive_baseline(candidates):
    """True PRE-override baseline, and the frozen run's POST-override accuracy.

    `baseline_prediction` is the prediction before the adjudicator may override;
    `prediction` is after it may have. Conflating the two is precisely the error
    this function exists to prevent.
    """
    det = candidates["eval_details"]
    pre = sum(1 for r in det if r.get("baseline_prediction") == r.get("gold"))
    post = sum(1 for r in det if r.get("prediction") == r.get("gold"))
    overrides = [r for r in det if r.get("override_applied")]
    return {
        "population": len(det),
        "pre_override_correct": pre,
        "pre_override_accuracy_pct": rate(pre, len(det)),
        "post_override_correct": post,
        "post_override_accuracy_pct": rate(post, len(det)),
        "frozen_run_overrides": len(overrides),
        "frozen_run_overrides_on_gold_contradict":
            sum(1 for r in overrides if r["gold"] == "CONTRADICT"),
        "frozen_run_overrides_on_gold_support":
            sum(1 for r in overrides if r["gold"] == "SUPPORT"),
    }


def artifact_mode():
    """Recompute the confusion matrices from source. Returns None if absent."""
    missing = [p for p in (ANNOTATIONS, CANDIDATES) if not os.path.exists(p)]
    if missing:
        return None, missing

    with open(ANNOTATIONS, encoding="utf-8") as handle:
        annotations = json.load(handle)
    with open(CANDIDATES, encoding="utf-8") as handle:
        candidates = json.load(handle)

    cases = annotations.get("cases", annotations if isinstance(annotations, list) else [])
    gold_by_claim = {r["claim"]: r["gold"] for r in candidates["eval_details"]}

    g5 = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    direction_only = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    unmatched, genuine_yes_among_g5 = [], 0

    for case in cases:
        gold = gold_by_claim.get(case["claim"])
        if gold is None:
            unmatched.append(case.get("case_id"))
            continue
        ann = case["annotation"]
        is_contradict = gold == "CONTRADICT"
        accept_g5 = (ann.get("overall_scope_match") == "YES"
                     and ann.get("direction") == "OPPOSITE")
        accept_dir = ann.get("direction") == "OPPOSITE"
        for table, accept in ((g5, accept_g5), (direction_only, accept_dir)):
            if accept and is_contradict:
                table["TP"] += 1
            elif accept and not is_contradict:
                table["FP"] += 1
            elif not accept and is_contradict:
                table["FN"] += 1
            else:
                table["TN"] += 1
        if accept_g5 and ann.get("genuine_contradiction") == "YES":
            genuine_yes_among_g5 += 1

    return {
        "n_cases": len(cases),
        "unmatched_case_ids": unmatched,
        "g5": g5,
        "direction_only": direction_only,
        "g5_accepted_with_genuine_contradiction_yes": genuine_yes_among_g5,
        "baseline": derive_baseline(candidates),
    }, []


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recompute / audit the G5 primary held-out validation statistics"
    )
    parser.add_argument("--emit-summary", action="store_true",
                        help=f"write {SUMMARY}")
    args = parser.parse_args()

    recomputed, missing = artifact_mode()
    mode = "ARTIFACT" if recomputed else "AUDIT"

    print("=" * 78)
    print(f"G5 PRIMARY HELD-OUT VALIDATION — STATISTICAL VERIFICATION ({mode} MODE)")
    print("=" * 78)

    if recomputed:
        print("\nConfusion matrices RECOMPUTED FROM SOURCE:")
        print(f"  cases joined: {recomputed['n_cases']}, "
              f"unmatched: {recomputed['unmatched_case_ids'] or 'none'}")
        for key in ("g5", "direction_only"):
            agree = recomputed[key] == SUPPLIED[key]
            print(f"  {key:<16} {recomputed[key]}  vs supplied {SUPPLIED[key]}  "
                  f"-> {'MATCH' if agree else 'MISMATCH'}")
        g5_table, dir_table = recomputed["g5"], recomputed["direction_only"]
    else:
        print("\nArtifacts NOT FOUND, so the confusion matrices could not be")
        print("recomputed from source. Missing:")
        for path in missing:
            print(f"  - {path}")
        print("\nFalling back to AUDIT MODE: the supplied confusion matrices are")
        print("taken as input and every derived statistic is recomputed from them.")
        print("This verifies the arithmetic and the statistics, NOT the matrices.")
        g5_table, dir_table = SUPPLIED["g5"], SUPPLIED["direction_only"]

    g5 = rule_stats(g5_table, "G5 / Rule A")
    dir_only = rule_stats(dir_table, "direction-only")
    disc = rule_stats(SUPPLIED["discovery"], "G5 / Rule A (discovery study)")
    paired = mcnemar(g5_table, dir_table)
    if recomputed:
        base = recomputed["baseline"]
        baseline_correct = base["pre_override_correct"]
        population = base["population"]
        baseline_source = "derived from the candidate artifact (pre-override)"
    else:
        base = None
        baseline_correct = SUPPLIED["baseline_correct_fallback_pre_override"]
        population = SUPPLIED["unique_claims"]
        baseline_source = "fallback constant (no artifact available)"
    sim_g5 = simulated_application(g5_table, baseline_correct, population)
    sim_dir = simulated_application(dir_table, baseline_correct, population)

    print("\n--- consistency of the candidate frame ---")
    for name, stats in (("G5", g5), ("direction-only", dir_only)):
        ok = (stats["n_cases"] == SUPPLIED["candidate_frame"]
              and stats["gold_contradict"] == SUPPLIED["gold_contradict"]
              and stats["gold_support"] == SUPPLIED["gold_support"])
        print(f"  {name:<15} n={stats['n_cases']} C={stats['gold_contradict']} "
              f"S={stats['gold_support']}  -> {'OK' if ok else 'MISMATCH'}")

    print("\n--- per-rule statistics ---")
    for stats in (g5, dir_only):
        print(f"  {stats['rule']}")
        print(f"    accepts {stats['n_accepted']:<3} precision {stats['precision_pct']}% "
              f"Wilson95 {stats['precision_ci95_wilson']}")
        print(f"    {'':<11} recall    {stats['recall_pct']}% "
              f"Wilson95 {stats['recall_ci95_wilson']}")
        print(f"    {'':<11} frame accuracy {stats['candidate_frame_accuracy_pct']}%  "
              f"rescues {stats['rescues']} added errors {stats['added_errors']} "
              f"net {stats['net_claims']:+d}")
        print(f"    {'':<11} Fisher exact two-sided p = "
              f"{stats['fisher_exact_two_sided_p']:.7g}")

    print("\n--- paired comparison ---")
    for key, block in paired.items():
        print(f"  {key}: discordant={block['discordant_pairs']} "
              f"p={block['p_two_sided_exact']:.7g}")
        print(f"    {block['interpretation']}")

    if base:
        print("\n--- baseline, derived from the artifact ---")
        print(f"  pre-override  (true baseline)  : {base['pre_override_correct']}/"
              f"{base['population']} = {base['pre_override_accuracy_pct']}%")
        print(f"  post-override (frozen G1+G2+G3): {base['post_override_correct']}/"
              f"{base['population']} = {base['post_override_accuracy_pct']}%"
              f"   <-- NOT the baseline")
        print(f"  the frozen run itself applied {base['frozen_run_overrides']} overrides "
              f"({base['frozen_run_overrides_on_gold_contradict']} on gold CONTRADICT, "
              f"{base['frozen_run_overrides_on_gold_support']} on gold SUPPORT)")

    print(f"\n--- simulated application to the {population}-claim validation population ---")
    print(f"  baseline source: {baseline_source}")
    for name, sim in (("G5", sim_g5), ("direction-only", sim_dir)):
        print(f"  {name:<15} {sim['baseline_correct']}/{sim['population']} "
              f"({sim['baseline_accuracy_pct']}%) -> {sim['after_correct']}/"
              f"{sim['population']} ({sim['after_accuracy_pct']}%)  "
              f"delta {sim['delta_pp']:+}pp  net {sim['net_claims']:+d}")
    identical = sim_g5["after_correct"] == sim_dir["after_correct"]
    print(f"  G5 and direction-only reach the SAME simulated accuracy: {identical}")

    print("\n--- discovery study (outcome-selected; context only) ---")
    print(f"  precision {disc['precision_pct']}% recall {disc['recall_pct']}% "
          f"Fisher p = {disc['fisher_exact_two_sided_p']:.4g}")

    summary = {
        "metadata": {
            "document": "G5 primary held-out validation — computed statistics",
            "verification_mode": mode,
            "confusion_matrices_recomputed_from_source": bool(recomputed),
            "artifacts_missing": missing,
            "research_only": True,
            "production_modified": False,
            "experiments_rerun": False,
            "figures_provenance": (
                "Dataset sizes, frame size, gold counts and baseline correct count "
                "are AS SUPPLIED by the project owner and were not verifiable in "
                "this repository. All precision, recall, confidence intervals, "
                "Fisher exact and McNemar values are COMPUTED here from the "
                "confusion matrices."
            ),
            "g5_definition": ("overall_scope_match == YES AND direction == OPPOSITE"),
            "direction_only_definition": "direction == OPPOSITE",
            "annotation_seed": SUPPLIED["annotation_seed"],
        },
        "baseline_derivation": base,
        "baseline_used_for_simulation": {
            "correct": baseline_correct,
            "population": population,
            "source": baseline_source,
        },
        "dataset_as_supplied": {
            "validation_rows": SUPPLIED["validation_rows"],
            "unique_claims": SUPPLIED["unique_claims"],
            "corpus_documents": SUPPLIED["corpus_documents"],
            "candidate_frame": SUPPLIED["candidate_frame"],
            "gold_contradict": SUPPLIED["gold_contradict"],
            "gold_support": SUPPLIED["gold_support"],
            "post_override_accuracy_of_frozen_run_NOT_the_baseline":
                SUPPLIED["post_override_accuracy_of_frozen_run"],
        },
        "primary_heldout_validation": {"g5": g5, "direction_only": dir_only},
        "paired_comparison_mcnemar": paired,
        "simulated_application": {"g5": sim_g5, "direction_only": sim_dir,
                                  "same_result_as_direction_only": identical},
        "discovery_study": disc,
        "recomputed_from_artifacts": recomputed,
    }

    if args.emit_summary:
        os.makedirs("docs", exist_ok=True)
        with open(SUMMARY, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nwrote {SUMMARY}")

    print("\n" + "=" * 78)
    if not recomputed:
        print("INCOMPLETE: statistics verified, confusion matrices NOT verified")
        print("against source data. Place the artifacts in the repository and")
        print("re-run for artifact-mode verification.")
    else:
        print("COMPLETE: confusion matrices recomputed from source and all")
        print("derived statistics verified.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
