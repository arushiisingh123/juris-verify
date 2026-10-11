"""
Proposition-level scope-alignment diagnostic. RESEARCH ONLY.

Tests the hypothesis: "Proposition-level scope alignment can distinguish genuine
contradiction rescues from false SUPPORT->CONTRADICT overrides."

Modifies no production module. Implements no gate. Tunes nothing.

=============================================================================
THREE PHASES, SEPARATED IN CODE
=============================================================================
The scientific value of this diagnostic rests entirely on the rules being
specified before any outcome is consulted. That is enforced structurally, not
by convention:

  Phase 1  describe_annotations()   -- annotation distributions only.
  Phase 2  apply_rules()            -- accept/reject per case. The rule
                                       predicates receive ONLY an annotation
                                       dict; they are given no access to the
                                       case record, the artifact, or any label.
  Phase 3  score_against_outcomes() -- the ONLY function that reads gold
                                       labels, and the only one that may.

The rules are frozen constants below, and `rule_fingerprint()` hashes their
source so the report can prove they were not edited after results were seen.

=============================================================================
WHAT IS DELIBERATELY NOT USED AS A FEATURE
=============================================================================
gold label, rescue/regression status, override correctness, adjudication
status, adjudication reason, gate outcomes, document ids, section labels,
counter_section, and every NLI probability the adjudicator recorded. The only
inputs to the rules are the human/annotator judgements over claim, primary
evidence and counter-evidence -- information available at inference time.

Section labels and document ids are excluded for a second reason beyond the
brief: both correlate with the outcome in this dataset, so using them would
leak the answer into a feature.

=============================================================================
USAGE
=============================================================================
    python inspect_proposition_scope_alignment.py            # real run
    python inspect_proposition_scope_alignment.py --selftest  # verify the harness

The real run REFUSES to proceed while the blind dataset's annotation fields are
empty. It will not substitute, infer, or generate annotations.
"""

import argparse
import hashlib
import inspect as _inspect
import itertools
import json
import math
import os
import sys

BLIND = os.path.join("docs", "scope_alignment_blind_dataset.json")
ARTIFACT = "eval/results_20261011_160640_n504_section2_w0.05_prosactive_gatesg1g2g3.json"
OUT_JSON = os.path.join("docs", "proposition_scope_alignment_diagnostic.json")

# --- enum values, taken verbatim from the worksheet schema -------------------
ALIGNMENT = ("SAME", "DIFFERENT", "NOT_APPLICABLE", "UNCLEAR")
DIRECTION = ("OPPOSITE", "SAME", "NOT_APPLICABLE", "UNCLEAR")
JUDGEMENT = ("YES", "NO", "UNCLEAR")

SCHEMA = {
    "population_alignment": ALIGNMENT,
    "intervention_alignment": ALIGNMENT,
    "outcome_alignment": ALIGNMENT,
    "endpoint_alignment": ALIGNMENT,
    "context_alignment": ALIGNMENT,
    "causal_relation_alignment": ALIGNMENT,
    "direction": DIRECTION,
    "overall_scope_match": JUDGEMENT,
    "genuine_contradiction": JUDGEMENT,
}

SCOPE_DIMENSIONS = (
    "population_alignment",
    "intervention_alignment",
    "outcome_alignment",
    "endpoint_alignment",
    "context_alignment",
    "causal_relation_alignment",
)

# ===========================================================================
# PRE-SPECIFIED RULES -- frozen before any outcome is consulted
# ===========================================================================
# Each predicate takes ONLY the annotation dict. "Not explicitly different"
# means != "DIFFERENT", so SAME / NOT_APPLICABLE / UNCLEAR all pass -- the
# reading of the brief's wording, and the conservative one, since it does not
# let an UNCLEAR judgement veto an override.
#
# Scope alignment and directional contradiction are kept conceptually separate
# throughout: every rule is (scope condition) AND (direction condition), and
# the two halves are reported separately as well as jointly.


def rule_a(ann):
    """Rule A: overall proposition scope compatible AND direction opposite."""
    return ann["overall_scope_match"] == "YES" and ann["direction"] == "OPPOSITE"


def rule_b(ann):
    """Rule B: population, outcome and endpoint not explicitly DIFFERENT,
    AND direction opposite."""
    core = ("population_alignment", "outcome_alignment", "endpoint_alignment")
    return (
        all(ann[dim] != "DIFFERENT" for dim in core)
        and ann["direction"] == "OPPOSITE"
    )


def rule_c(ann):
    """Rule C: no explicit mismatch on any of the six scope dimensions,
    AND direction opposite."""
    return (
        all(ann[dim] != "DIFFERENT" for dim in SCOPE_DIMENSIONS)
        and ann["direction"] == "OPPOSITE"
    )


# Reported for contrast only, to keep the two concepts separable. Not proposed
# as decision rules.
def scope_only_a(ann):
    """Scope half of Rule A, with no direction condition."""
    return ann["overall_scope_match"] == "YES"


def scope_only_c(ann):
    """Scope half of Rule C, with no direction condition."""
    return all(ann[dim] != "DIFFERENT" for dim in SCOPE_DIMENSIONS)


def direction_only(ann):
    """Direction condition alone -- the current adjudicator's G1 in spirit."""
    return ann["direction"] == "OPPOSITE"


RULES = {
    "A": (rule_a, "overall_scope_match == YES AND direction == OPPOSITE"),
    "B": (rule_b, "population/outcome/endpoint != DIFFERENT AND direction == OPPOSITE"),
    "C": (rule_c, "no scope dimension == DIFFERENT AND direction == OPPOSITE"),
}

CONTRASTS = {
    "scope_only_A": (scope_only_a, "overall_scope_match == YES (no direction term)"),
    "scope_only_C": (scope_only_c, "no scope dimension == DIFFERENT (no direction term)"),
    "direction_only": (direction_only, "direction == OPPOSITE (no scope term)"),
}


def rule_fingerprint():
    """sha256 over the rule source, so the report can evidence that the rules
    were not altered after outcomes were seen."""
    source = "".join(
        _inspect.getsource(fn)
        for fn, _ in list(RULES.values()) + list(CONTRASTS.values())
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# ===========================================================================
# Statistics -- implemented natively so no dependency is added
# ===========================================================================

def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for the table [[a, b], [c, d]].

    Exact hypergeometric enumeration; summing every table with the same
    margins whose probability is <= the observed one (the standard two-sided
    convention). No scipy needed; `--selftest` cross-checks against scipy when
    it happens to be installed.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x):
        return (
            math.comb(row1, x)
            * math.comb(n - row1, col1 - x)
            / math.comb(n, col1)
        )

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    observed = prob(a)
    # 1e-9 relative tolerance so float noise does not drop the observed table.
    return min(
        1.0,
        sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)),
    )


def odds_ratio(a, b, c, d):
    """Odds ratio with the Haldane-Anscombe 0.5 correction when a cell is 0."""
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        corrected = True
    else:
        corrected = False
    return (a * d) / (b * c), corrected


def phi_coefficient(a, b, c, d):
    """Phi (Matthews) coefficient for a 2x2 table; 0.0 when a margin is empty."""
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return 0.0 if denom == 0 else (a * d - b * c) / denom


def rate(num, den):
    return round(num / den * 100, 2) if den else None


# ===========================================================================
# Loading and joining
# ===========================================================================

def load_blind(path=BLIND):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_artifact(path=ARTIFACT):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def annotation_completeness(cases):
    """Which cases have a usable, schema-valid annotation."""
    complete, incomplete, invalid = [], [], []
    for case in cases:
        ann = case.get("annotation") or {}
        missing = [f for f in SCHEMA if ann.get(f) in (None, "", [])]
        bad = [
            (f, ann[f]) for f in SCHEMA
            if ann.get(f) not in (None, "", []) and ann[f] not in SCHEMA[f]
        ]
        if bad:
            invalid.append({"case_id": case["case_id"], "invalid": bad})
        elif missing:
            incomplete.append({"case_id": case["case_id"], "missing": missing})
        else:
            complete.append(case["case_id"])
    return complete, incomplete, invalid


def join_to_artifact(cases, payload):
    """Join by exact claim text. Returns {case_id: claim_index} and diagnostics.

    The join carries the claim_index ONLY; no outcome field crosses here.
    Phase 3 looks outcomes up separately.
    """
    overrides = [r for r in payload["eval_details"] if r.get("override_applied")]
    by_claim = {}
    for record in overrides:
        by_claim.setdefault(record["claim"], []).append(record["claim_index"])

    mapping, unmatched, ambiguous = {}, [], []
    for case in cases:
        hits = by_claim.get(case["claim"])
        if not hits:
            unmatched.append(case["case_id"])
        elif len(hits) > 1:
            ambiguous.append({"case_id": case["case_id"], "claim_indices": hits})
        else:
            mapping[case["case_id"]] = hits[0]
    return mapping, {
        "n_overrides_in_artifact": len(overrides),
        "n_cases": len(cases),
        "n_joined": len(mapping),
        "unmatched_case_ids": unmatched,
        "ambiguous": ambiguous,
    }


# ===========================================================================
# PHASE 1 -- describe the annotations. No outcome is touched.
# ===========================================================================

def describe_annotations(cases):
    out = {}
    for field, allowed in SCHEMA.items():
        counts = {value: 0 for value in allowed}
        for case in cases:
            value = (case.get("annotation") or {}).get(field)
            if value in counts:
                counts[value] += 1
        n = sum(counts.values())
        out[field] = {
            "counts": counts,
            "n_annotated": n,
            "pct": {v: rate(c, n) for v, c in counts.items()},
        }
    # Co-occurrence of the two conceptually separate halves.
    pairs = {}
    for scope in JUDGEMENT:
        for direction in DIRECTION:
            pairs[f"{scope}|{direction}"] = sum(
                1 for c in cases
                if (c.get("annotation") or {}).get("overall_scope_match") == scope
                and (c.get("annotation") or {}).get("direction") == direction
            )
    out["_scope_x_direction"] = pairs
    return out


# ===========================================================================
# PHASE 2 -- apply the frozen rules. No outcome is touched.
# ===========================================================================

def apply_rules(cases):
    """Returns {rule_name: {case_id: bool}} using only annotation dicts."""
    decisions = {}
    for name, (predicate, _desc) in list(RULES.items()) + list(CONTRASTS.items()):
        verdicts = {}
        for case in cases:
            ann = case["annotation"]
            # The predicate is handed the annotation dict alone -- by
            # construction it cannot reach the case record or any label.
            verdicts[case["case_id"]] = bool(predicate(ann))
        decisions[name] = verdicts
    return decisions


# ===========================================================================
# PHASE 3 -- the ONLY place gold labels are read
# ===========================================================================

def outcome_lookup(mapping, payload):
    """case_id -> "rescue" | "regression", from the artifact's gold labels."""
    by_index = {r["claim_index"]: r for r in payload["eval_details"]}
    out = {}
    for case_id, claim_index in mapping.items():
        record = by_index[claim_index]
        out[case_id] = "rescue" if record["gold"] == "CONTRADICT" else "regression"
    return out


def score_rule(verdicts, outcomes):
    """Confusion matrix and statistics for one rule.

    accept = the rule would allow the override to stand.
    A rescue accepted is a true positive; a regression accepted is a false
    positive.
    """
    ids = [cid for cid in verdicts if cid in outcomes]
    tp = sum(1 for c in ids if verdicts[c] and outcomes[c] == "rescue")
    fp = sum(1 for c in ids if verdicts[c] and outcomes[c] == "regression")
    fn = sum(1 for c in ids if not verdicts[c] and outcomes[c] == "rescue")
    tn = sum(1 for c in ids if not verdicts[c] and outcomes[c] == "regression")

    accepted, rejected = tp + fp, fn + tn
    n_rescue, n_regr = tp + fn, fp + tn
    p = fisher_exact_two_sided(tp, fp, fn, tn)
    orr, corrected = odds_ratio(tp, fp, fn, tn)

    # Net claims relative to leaving every override in place: each rejected
    # regression is +1 claim recovered, each rejected rescue is -1 lost.
    return {
        "n_scored": len(ids),
        "confusion_matrix": {
            "accepted_rescue_TP": tp,
            "accepted_regression_FP": fp,
            "rejected_rescue_FN": fn,
            "rejected_regression_TN": tn,
        },
        "n_accepted": accepted,
        "n_rejected": rejected,
        "precision_pct": rate(tp, accepted),
        "recall_pct": rate(tp, n_rescue),
        "specificity_pct": rate(tn, n_regr),
        "true_rescues_retained": tp,
        "true_rescues_lost": fn,
        "regressions_rejected": tn,
        "regressions_retained": fp,
        "net_claims_vs_accepting_all": tn - fn,
        "fisher_exact_two_sided_p": round(p, 6),
        "odds_ratio": round(orr, 4),
        "odds_ratio_haldane_corrected": corrected,
        "phi_coefficient": round(phi_coefficient(tp, fp, fn, tn), 4),
    }


def dimension_discriminativeness(cases, outcomes):
    """Per-dimension 2x2: (DIFFERENT vs not) x (regression vs rescue).

    Phase 3 -- descriptive, not a rule, and deliberately computed after the
    rules were frozen.
    """
    out = {}
    for field in SCOPE_DIMENSIONS + ("direction", "overall_scope_match"):
        if field == "direction":
            flag = lambda v: v != "OPPOSITE"
            label = "direction != OPPOSITE"
        elif field == "overall_scope_match":
            flag = lambda v: v != "YES"
            label = "overall_scope_match != YES"
        else:
            flag = lambda v: v == "DIFFERENT"
            label = f"{field} == DIFFERENT"

        ids = [c["case_id"] for c in cases if c["case_id"] in outcomes]
        anns = {c["case_id"]: c["annotation"][field] for c in cases}
        a = sum(1 for c in ids if flag(anns[c]) and outcomes[c] == "regression")
        b = sum(1 for c in ids if flag(anns[c]) and outcomes[c] == "rescue")
        cc = sum(1 for c in ids if not flag(anns[c]) and outcomes[c] == "regression")
        d = sum(1 for c in ids if not flag(anns[c]) and outcomes[c] == "rescue")
        p = fisher_exact_two_sided(a, b, cc, d)
        orr, corrected = odds_ratio(a, b, cc, d)
        out[field] = {
            "flag": label,
            "flagged_regressions": a,
            "flagged_rescues": b,
            "unflagged_regressions": cc,
            "unflagged_rescues": d,
            "flag_precision_for_regression_pct": rate(a, a + b),
            "fisher_exact_two_sided_p": round(p, 6),
            "odds_ratio": round(orr, 4),
            "odds_ratio_haldane_corrected": corrected,
            "phi_coefficient": round(phi_coefficient(a, b, cc, d), 4),
        }
    return out


# ===========================================================================
# Self-test: verifies the harness on synthetic annotations only
# ===========================================================================

def _selftest():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # --- rule logic -------------------------------------------------
    def ann(**kw):
        base = {f: "SAME" for f in SCOPE_DIMENSIONS}
        base["direction"] = "OPPOSITE"
        base["overall_scope_match"] = "YES"
        base["genuine_contradiction"] = "YES"
        base.update(kw)
        return base

    check("A all-aligned", rule_a(ann()), True)
    check("A scope NO", rule_a(ann(overall_scope_match="NO")), False)
    check("A dir SAME", rule_a(ann(direction="SAME")), False)
    check("A scope UNCLEAR", rule_a(ann(overall_scope_match="UNCLEAR")), False)

    check("B all-aligned", rule_b(ann()), True)
    check("B population DIFFERENT", rule_b(ann(population_alignment="DIFFERENT")), False)
    check("B outcome DIFFERENT", rule_b(ann(outcome_alignment="DIFFERENT")), False)
    check("B endpoint DIFFERENT", rule_b(ann(endpoint_alignment="DIFFERENT")), False)
    # B ignores intervention/context/causal by design
    check("B intervention DIFFERENT passes",
          rule_b(ann(intervention_alignment="DIFFERENT")), True)
    check("B context DIFFERENT passes", rule_b(ann(context_alignment="DIFFERENT")), True)
    check("B UNCLEAR is not DIFFERENT",
          rule_b(ann(population_alignment="UNCLEAR")), True)
    check("B N/A is not DIFFERENT",
          rule_b(ann(population_alignment="NOT_APPLICABLE")), True)
    check("B dir not OPPOSITE", rule_b(ann(direction="UNCLEAR")), False)

    check("C all-aligned", rule_c(ann()), True)
    for dim in SCOPE_DIMENSIONS:
        check(f"C {dim} DIFFERENT", rule_c(ann(**{dim: "DIFFERENT"})), False)
    check("C dir SAME", rule_c(ann(direction="SAME")), False)

    check("C is at least as strict as B",
          all(
              not rule_c(a) or rule_b(a)
              for a in (ann(**{d: v}) for d in SCOPE_DIMENSIONS for v in ALIGNMENT)
          ), True)

    check("scope_only_A ignores direction", scope_only_a(ann(direction="SAME")), True)
    check("direction_only ignores scope",
          direction_only(ann(overall_scope_match="NO")), True)

    # --- rule predicates cannot see labels --------------------------
    for name, (predicate, _d) in list(RULES.items()) + list(CONTRASTS.items()):
        params = list(_inspect.signature(predicate).parameters)
        check(f"{name} takes exactly one arg", params, ["ann"])

    # --- Fisher exact, against known values --------------------------
    # Classic tea-tasting table: p = 0.4857
    check("fisher tea-tasting", round(fisher_exact_two_sided(3, 1, 1, 3), 4), 0.4857)
    check("fisher no association", round(fisher_exact_two_sided(5, 5, 5, 5), 4), 1.0)
    check("fisher perfect 10/10", fisher_exact_two_sided(10, 0, 0, 10) < 1e-4, True)
    try:
        from scipy.stats import fisher_exact as _sp
        for table in [(3, 1, 1, 3), (5, 5, 5, 5), (10, 0, 0, 10), (24, 4, 0, 14),
                      (12, 7, 12, 11), (1, 0, 23, 18)]:
            mine = fisher_exact_two_sided(*table)
            theirs = _sp([[table[0], table[1]], [table[2], table[3]]])[1]
            if abs(mine - theirs) > 1e-9:
                failures.append(f"fisher mismatch {table}: {mine} vs scipy {theirs}")
        print("  (Fisher cross-checked against scipy on 6 tables)")
    except ImportError:
        print("  (scipy absent; native Fisher checked against known values only)")

    # --- effect sizes ------------------------------------------------
    orr, corrected = odds_ratio(10, 2, 3, 15)
    check("odds ratio", round(orr, 4), round((10 * 15) / (2 * 3), 4))
    check("odds ratio uncorrected", corrected, False)
    _, corrected0 = odds_ratio(10, 0, 3, 15)
    check("odds ratio corrected on zero cell", corrected0, True)
    check("phi perfect", round(phi_coefficient(10, 0, 0, 10), 4), 1.0)
    check("phi none", round(phi_coefficient(5, 5, 5, 5), 4), 0.0)

    # --- end-to-end scoring on synthetic data -------------------------
    synth = []
    for i in range(10):   # 10 rescues the rule should accept
        synth.append({"case_id": f"S{i:02d}", "claim": f"c{i}", "annotation": ann()})
    for i in range(10, 16):  # 6 regressions the rule should reject
        synth.append({"case_id": f"S{i:02d}", "claim": f"c{i}",
                      "annotation": ann(population_alignment="DIFFERENT")})
    for i in range(16, 18):  # 2 rescues the rule wrongly rejects
        synth.append({"case_id": f"S{i:02d}", "claim": f"c{i}",
                      "annotation": ann(outcome_alignment="DIFFERENT")})
    outcomes = {f"S{i:02d}": ("rescue" if i < 10 or i >= 16 else "regression")
                for i in range(18)}
    decisions = apply_rules(synth)
    scored = score_rule(decisions["C"], outcomes)
    cm = scored["confusion_matrix"]
    check("synth TP", cm["accepted_rescue_TP"], 10)
    check("synth FP", cm["accepted_regression_FP"], 0)
    check("synth FN", cm["rejected_rescue_FN"], 2)
    check("synth TN", cm["rejected_regression_TN"], 6)
    check("synth precision", scored["precision_pct"], 100.0)
    check("synth recall", scored["recall_pct"], round(10 / 12 * 100, 2))
    check("synth specificity", scored["specificity_pct"], 100.0)
    check("synth net", scored["net_claims_vs_accepting_all"], 4)
    check("synth n_scored", scored["n_scored"], 18)

    desc = describe_annotations(synth)
    check("describe counts population DIFFERENT",
          desc["population_alignment"]["counts"]["DIFFERENT"], 6)
    check("describe n_annotated", desc["direction"]["n_annotated"], 18)

    dims = dimension_discriminativeness(synth, outcomes)
    check("dim population flagged regressions",
          dims["population_alignment"]["flagged_regressions"], 6)
    check("dim population flagged rescues",
          dims["population_alignment"]["flagged_rescues"], 0)

    # --- completeness + join detection --------------------------------
    blank = [{"case_id": "X1", "claim": "q", "annotation": {f: None for f in SCHEMA}}]
    complete, incomplete, invalid = annotation_completeness(blank)
    check("blank -> incomplete", (complete, len(incomplete), invalid), ([], 1, []))
    badval = [{"case_id": "X2", "claim": "q",
               "annotation": {**ann(), "direction": "SIDEWAYS"}}]
    _, _, invalid = annotation_completeness(badval)
    check("invalid enum detected", len(invalid), 1)

    print("\n" + "=" * 74)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} checks)")
        for failure in failures:
            print("  -", failure)
        print("=" * 74)
        return 1
    print("SELFTEST PASSED -- rule logic (incl. C-implies-B), label isolation,")
    print("Fisher exact, odds ratio, phi, scoring, per-dimension tables, and")
    print("annotation-completeness detection all verified on synthetic data.")
    print("=" * 74)
    return 0


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Proposition scope-alignment diagnostic (research only)"
    )
    parser.add_argument("--blind", default=BLIND)
    parser.add_argument("--artifact", default=ARTIFACT)
    parser.add_argument("--out", default=OUT_JSON)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    for path in (args.blind, args.artifact):
        if not os.path.exists(path):
            print(f"ERROR: required file not found: {path}")
            return 2

    blind = load_blind(args.blind)
    payload = load_artifact(args.artifact)
    cases = blind["cases"]

    mapping, join_info = join_to_artifact(cases, payload)
    complete, incomplete, invalid = annotation_completeness(cases)
    fingerprint = rule_fingerprint()

    print("=" * 74)
    print("PROPOSITION SCOPE-ALIGNMENT DIAGNOSTIC")
    print("=" * 74)
    print(f"blind dataset : {args.blind}  ({len(cases)} cases)")
    print(f"artifact      : {args.artifact}")
    print(f"  gate rule   : {payload['metadata'].get('adjudication_gates')}")
    print(f"join by exact claim text: {join_info['n_joined']}/{join_info['n_cases']} "
          f"matched against {join_info['n_overrides_in_artifact']} overrides")
    if join_info["unmatched_case_ids"]:
        print(f"  UNMATCHED: {join_info['unmatched_case_ids']}")
    if join_info["ambiguous"]:
        print(f"  AMBIGUOUS: {join_info['ambiguous']}")
    print(f"frozen rule fingerprint (sha256): {fingerprint[:32]}...")
    print(f"\nannotations: {len(complete)} complete, {len(incomplete)} incomplete, "
          f"{len(invalid)} with invalid enum values")

    result = {
        "metadata": {
            "diagnostic": "proposition_scope_alignment",
            "research_only": True,
            "production_modified": False,
            "blind_dataset": args.blind,
            "artifact": args.artifact,
            "artifact_gate_rule": payload["metadata"].get("adjudication_gates"),
            "join_method": "exact claim text",
            "join": join_info,
            "rule_fingerprint_sha256": fingerprint,
            "rules_prespecified": {
                name: desc for name, (_fn, desc) in RULES.items()
            },
            "contrast_conditions": {
                name: desc for name, (_fn, desc) in CONTRASTS.items()
            },
            "features_excluded_by_design": [
                "gold label", "rescue/regression label", "override correctness",
                "adjudication status", "adjudication reason", "gate outcomes",
                "document ids", "section labels", "counter_section",
                "all adjudicator NLI probabilities",
            ],
            "annotation_schema": {f: list(v) for f, v in SCHEMA.items()},
            "annotation_status": {
                "n_complete": len(complete),
                "n_incomplete": len(incomplete),
                "n_invalid": len(invalid),
                "complete_case_ids": complete,
                "incomplete": incomplete,
                "invalid": invalid,
            },
        }
    }

    if len(complete) != len(cases):
        print("\n" + "!" * 74)
        print("BLOCKED: the blind dataset carries no usable annotations.")
        print("!" * 74)
        print("Phase 1 (describe), phase 2 (apply rules) and phase 3 (score)")
        print("all require annotation values. This script will NOT substitute,")
        print("infer, or generate them: the rules are frozen and the dataset is")
        print("the only admissible input.")
        print("\nTo proceed: have an annotator who has NOT read")
        print("  docs/scope_alignment_failure_analysis.{md,json}")
        print("  docs/scope_alignment_reconciliation.md")
        print("fill the `annotation` object for each case in")
        print(f"  {args.blind}")
        print("following docs/scope_alignment_blind_annotation.md, then re-run.")
        result["status"] = "BLOCKED_NO_ANNOTATIONS"
        result["phase_1_descriptive"] = None
        result["phase_2_rule_decisions"] = None
        result["phase_3_outcome_scoring"] = None
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"\nWrote status to: {args.out}")
        return 3

    # ---- Phase 1 ----------------------------------------------------
    print("\n--- PHASE 1: annotation distributions (no outcomes read) ---")
    descriptive = describe_annotations(cases)
    for field in SCHEMA:
        counts = descriptive[field]["counts"]
        print(f"  {field:<28} " + "  ".join(f"{v}={n}" for v, n in counts.items() if n))

    # ---- Phase 2 ----------------------------------------------------
    print("\n--- PHASE 2: frozen rules applied (no outcomes read) ---")
    decisions = apply_rules(cases)
    for name in list(RULES) + list(CONTRASTS):
        accepted = sum(1 for v in decisions[name].values() if v)
        print(f"  {name:<16} accepts {accepted:>3} / {len(cases)}")

    # ---- Phase 3 ----------------------------------------------------
    print("\n--- PHASE 3: outcome-based scoring (gold labels read HERE only) ---")
    outcomes = outcome_lookup(mapping, payload)
    scoring = {}
    for name in list(RULES) + list(CONTRASTS):
        scoring[name] = score_rule(decisions[name], outcomes)
        s = scoring[name]
        print(f"  {name:<16} acc={s['n_accepted']:>3} prec={s['precision_pct']}% "
              f"rec={s['recall_pct']}% spec={s['specificity_pct']}% "
              f"net={s['net_claims_vs_accepting_all']:+d} "
              f"p={s['fisher_exact_two_sided_p']} phi={s['phi_coefficient']}")

    dims = dimension_discriminativeness(cases, outcomes)
    print("\n  per-dimension discriminativeness:")
    for field, stats in sorted(dims.items(), key=lambda kv: kv[1]["fisher_exact_two_sided_p"]):
        print(f"    {field:<28} p={stats['fisher_exact_two_sided_p']:<10} "
              f"phi={stats['phi_coefficient']:<8} "
              f"flagged {stats['flagged_regressions']}R/{stats['flagged_rescues']}C")

    result["status"] = "SCORED"
    result["phase_1_descriptive"] = descriptive
    result["phase_2_rule_decisions"] = decisions
    result["phase_3_outcome_scoring"] = {
        "outcome_counts": {
            "rescues": sum(1 for v in outcomes.values() if v == "rescue"),
            "regressions": sum(1 for v in outcomes.values() if v == "regression"),
        },
        "rules": scoring,
        "dimension_discriminativeness": dims,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
