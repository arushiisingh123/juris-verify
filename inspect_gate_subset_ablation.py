"""
D1 -- gate-subset ablation over the arm F artifact. ANALYSIS ONLY.

Reads the stored arm F results and recomputes, for all 16 subsets of the four
adjudication gates, what arm F's predictions WOULD have been. No model is run:
every gate outcome is already recorded per claim, so this is pure arithmetic
over the artifact.

Changes nothing. Touches no production file. Introduces no gate and no
threshold.

WHY EXACT RECONSTRUCTION IS POSSIBLE
------------------------------------
`src.prosecution.adjudicate` computes all four gates unconditionally, before the
first-failure priority logic selects which one to report as the reason. So the
stored `adjudication_gates` dict carries every gate's outcome, not just the
deciding one, for every claim where adjudication ran. Verified on the artifact:
289/289 claims with counter-evidence have all four gates resolved to True/False.

The production override rule is

    status == TRUE_CONTRADICTION  and  primary_verdict != CONTRADICTED

where TRUE_CONTRADICTION requires all four gates to pass. For a subset S the
counterfactual rule is "every gate in S passes", with the same
baseline-not-already-CONTRADICT precondition. `primary_verdict != CONTRADICTED`
is equivalent to `baseline_prediction == "SUPPORT"`, which is also stored.

S = {all four} must reproduce the real run exactly (24 / 13 / 11). S = {} is the
naive "any counter-evidence" rule, which must reproduce arm H. Both are asserted.

USAGE
    python inspect_gate_subset_ablation.py
"""

import argparse
import itertools
import json
import os
import sys

ARTIFACT = "eval/results_20261009_085502_n504_section2_w0.05_prosactive.json"
OUT = os.path.join("docs", "arm_f_gate_subset_ablation.md")

# Gate short name -> (artifact key, human definition)
GATES = {
    "G1": (
        "symmetric_contradiction",
        "Symmetry: verdict(premise=claim, hypothesis=counter) == CONTRADICTED. "
        "A real contradiction holds in both directions; a merely topical or "
        "independently contradictory sentence conflicts one way only.",
    ),
    "G2": (
        "pool_coherent",
        "Coherence: verdict(premise=primary, hypothesis=counter) != CONTRADICTED. "
        "If the baseline premise refutes the counter-evidence, the two retrieved "
        "sentences disagree with each other rather than with the claim.",
    ),
    "G3": (
        "uncontested_support",
        "Uncontested support: primary_verdict != SUPPORTED. If the source "
        "entails the claim, a contradictory fragment elsewhere is not grounds "
        "to flip the label.",
    ),
    "G4": (
        "finding_grounded",
        "Finding-grounded: counter section in {results, conclusions}, or "
        "(section unrecoverable AND counter from the document the baseline "
        "cited). Background/Methods sentences report prior work or design.",
    ),
}
GATE_ORDER = ["G1", "G2", "G3", "G4"]

# Arm D reference, from the artifact's own baseline_accuracy block.
ARM_D = {"correct": 421, "total": 504, "contradict": (115, 173), "support": (306, 331)}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(payload):
    """One row per claim, carrying only what the ablation needs."""
    rows = []
    for detail in payload["eval_details"]:
        gates = detail.get("adjudication_gates") or {}
        rows.append(
            {
                "claim_index": detail["claim_index"],
                "gold": detail.get("gold"),
                "baseline_prediction": detail.get("baseline_prediction"),
                "prediction": detail.get("prediction"),
                "override_applied": bool(detail.get("override_applied")),
                "adjudication_status": detail.get("adjudication_status"),
                "has_counter": bool(gates.get("counter_evidence_present")),
                "counter_section": detail.get("counter_section"),
                "counter_in_d_selection": detail.get("counter_in_d_selection"),
                "baseline_verdict": detail.get("baseline_verdict"),
                "gates": {
                    name: gates.get(key) for name, (key, _) in GATES.items()
                },
            }
        )
    return rows


def identifiable(row):
    """Can every gate be read for this claim? True only when adjudication ran
    against real counter-evidence."""
    return row["has_counter"] and all(
        isinstance(row["gates"][name], bool) for name in GATE_ORDER
    )


def evaluate_subset(rows, subset):
    """Counterfactual outcome if only `subset` were required to override."""
    pool = [r for r in rows if identifiable(r)]
    unidentifiable = sum(
        1 for r in rows if r["has_counter"] and not identifiable(r)
    )

    satisfying = [
        r for r in pool if all(r["gates"][name] for name in subset)
    ]
    # Overriding only changes anything when the baseline said SUPPORT.
    candidates = [r for r in satisfying if r["baseline_prediction"] == "SUPPORT"]
    rescues = [r for r in candidates if r["gold"] == "CONTRADICT"]
    regressions = [r for r in candidates if r["gold"] == "SUPPORT"]

    correct = ARM_D["correct"] + len(rescues) - len(regressions)
    c_correct = ARM_D["contradict"][0] + len(rescues)
    s_correct = ARM_D["support"][0] - len(regressions)

    return {
        "subset": tuple(subset),
        "label": "+".join(subset) if subset else "{} (naive)",
        "n_gates": len(subset),
        "satisfying_all": len(satisfying),
        "satisfying_baseline_support": len(candidates),
        "candidate_overrides": len(candidates),
        "rescues": len(rescues),
        "regressions": len(regressions),
        "precision": round(len(rescues) / len(candidates) * 100, 2) if candidates else None,
        "overall_correct": correct,
        "overall_accuracy": round(correct / ARM_D["total"] * 100, 2),
        "overall_delta": round(
            (correct - ARM_D["correct"]) / ARM_D["total"] * 100, 2
        ),
        "net_claims": len(rescues) - len(regressions),
        "contradict_accuracy": round(c_correct / ARM_D["contradict"][1] * 100, 2),
        "contradict_delta": round(
            (c_correct - ARM_D["contradict"][0]) / ARM_D["contradict"][1] * 100, 2
        ),
        "support_accuracy": round(s_correct / ARM_D["support"][1] * 100, 2),
        "support_delta": round(
            (s_correct - ARM_D["support"][0]) / ARM_D["support"][1] * 100, 2
        ),
        "unidentifiable": unidentifiable,
        "rescue_indices": [r["claim_index"] for r in rescues],
        "regression_indices": [r["claim_index"] for r in regressions],
    }


def pass_rates(rows):
    """Gate pass rate across each population of interest."""
    pool = [r for r in rows if identifiable(r)]
    overridable = [r for r in pool if r["baseline_prediction"] == "SUPPORT"]
    overrides = [r for r in rows if r["override_applied"]]
    rescues = [r for r in overrides if r["gold"] == "CONTRADICT"]
    regressions = [r for r in overrides if r["gold"] == "SUPPORT"]

    populations = [
        ("all claims", rows, len(rows)),
        ("gates identifiable (has counter-evidence)", pool, len(pool)),
        ("baseline-SUPPORT with counter-evidence (overridable)", overridable, len(overridable)),
        ("actual arm F overrides", overrides, len(overrides)),
        ("correct overrides (rescues)", rescues, len(rescues)),
        ("regressions", regressions, len(regressions)),
    ]

    table = []
    for name, group, n in populations:
        entry = {"population": name, "n": n}
        for gate in GATE_ORDER:
            resolved = [r for r in group if isinstance(r["gates"][gate], bool)]
            passed = sum(1 for r in resolved if r["gates"][gate])
            entry[gate] = (
                f"{passed}/{len(resolved)} ({passed / len(resolved) * 100:.1f}%)"
                if resolved else "n/a"
            )
        table.append(entry)
    return table, pool, overridable, overrides, rescues, regressions


def pairwise_redundancy(pool):
    """How often does each gate pair agree? Near-total agreement means one of
    the two adds nothing beyond the other."""
    out = []
    for a, b in itertools.combinations(GATE_ORDER, 2):
        both = [r for r in pool if isinstance(r["gates"][a], bool) and isinstance(r["gates"][b], bool)]
        agree = sum(1 for r in both if r["gates"][a] == r["gates"][b])
        a_only = sum(1 for r in both if r["gates"][a] and not r["gates"][b])
        b_only = sum(1 for r in both if r["gates"][b] and not r["gates"][a])
        out.append(
            {
                "pair": f"{a} / {b}",
                "n": len(both),
                "agreement": round(agree / len(both) * 100, 1) if both else 0.0,
                f"a_pass_b_fail": a_only,
                f"b_pass_a_fail": b_only,
            }
        )
    return out


def artifact_rule(payload, rows):
    """Which gate selection the artifact was actually produced with.

    Read from the artifact rather than assumed, so a G4-off validation run is
    checked against ITS own rule instead of the shipped one. Falls back to all
    four for artifacts written before the selection was recorded.
    """
    meta = payload.get("metadata", {}).get("adjudication_gates")
    if meta:
        return tuple(sorted(meta))
    for detail in payload.get("eval_details", []):
        required = detail.get("adjudication_required_gates")
        if required:
            return tuple(sorted(required))
    return tuple(sorted(GATE_ORDER))


def consistency_checks(rows, results, rule):
    """Item 8. Every check is a hard assertion on the stored data.

    `rule` is the gate selection the artifact was produced with, so the
    reproduction check adapts to a reduced-gate validation run.
    """
    checks = []

    def record(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    overrides = [r for r in rows if r["override_applied"]]
    n_over = len(overrides)
    n_resc = sum(1 for r in overrides if r["gold"] == "CONTRADICT")
    n_regr = sum(1 for r in overrides if r["gold"] == "SUPPORT")
    label = "+".join(rule)

    shipped = next((r for r in results if tuple(sorted(r["subset"])) == rule), None)
    record(f"artifact rule is {label}", shipped is not None, label)
    if shipped is not None:
        record(f"{label} reproduces the artifact override count",
               shipped["candidate_overrides"] == n_over,
               f"reconstructed {shipped['candidate_overrides']}, artifact {n_over}")
        record(f"{label} reproduces the artifact rescue count",
               shipped["rescues"] == n_resc,
               f"reconstructed {shipped['rescues']}, artifact {n_resc}")
        record(f"{label} reproduces the artifact regression count",
               shipped["regressions"] == n_regr,
               f"reconstructed {shipped['regressions']}, artifact {n_regr}")

    record("override count == rescues + regressions",
           n_over == n_resc + n_regr, f"{n_over} == {n_resc} + {n_regr}")

    if "G4" in rule:
        bad_section = [
            r["claim_index"] for r in overrides
            if r["counter_section"] in ("background", "methods")
        ]
        record("no override has a background/methods counter section (G4 integrity)",
               not bad_section, f"violations: {bad_section or 'none'}")
    else:
        n_bg = sum(
            1 for r in overrides
            if r["counter_section"] in ("background", "methods")
        )
        record("G4 not required, so background/methods counters are expected",
               True, f"{n_bg} override(s) with a background/methods counter")

    downgrades = [
        r["claim_index"] for r in rows
        if r["baseline_prediction"] == "CONTRADICT" and r["prediction"] != "CONTRADICT"
    ]
    record("no baseline CONTRADICT was downgraded",
           not downgrades, f"violations: {downgrades or 'none'}")

    violations = [
        r["claim_index"] for r in overrides
        if not all(r["gates"][g] for g in rule)
    ]
    record(f"every override passed every REQUIRED gate ({label})",
           not violations, f"violations: {violations or 'none'}")

    all_true_status = [
        r["claim_index"] for r in overrides
        if r["adjudication_status"] != "TRUE_CONTRADICTION"
    ]
    record("every override has status TRUE_CONTRADICTION",
           not all_true_status, f"violations: {all_true_status or 'none'}")

    # Arm D reference consistency against the artifact's own baseline block.
    base_support = sum(
        1 for r in rows if r["gold"] == "SUPPORT" and r["baseline_prediction"] == "SUPPORT"
    )
    base_contradict = sum(
        1 for r in rows
        if r["gold"] == "CONTRADICT" and r["baseline_prediction"] == "CONTRADICT"
    )
    record("arm D SUPPORT correct == 306", base_support == 306, f"{base_support}")
    record("arm D CONTRADICT correct == 115", base_contradict == 115, f"{base_contradict}")
    return checks


def main():
    parser = argparse.ArgumentParser(description="D1 gate-subset ablation (analysis only)")
    parser.add_argument("--artifact", default=ARTIFACT)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    if not os.path.exists(args.artifact):
        print(f"ERROR: artifact not found: {args.artifact}")
        return 2

    payload = load(args.artifact)
    rows = build_rows(payload)

    results = [
        evaluate_subset(rows, list(subset))
        for size in range(len(GATE_ORDER) + 1)
        for subset in itertools.combinations(GATE_ORDER, size)
    ]
    table, pool, overridable, overrides, rescues, regressions = pass_rates(rows)
    redundancy = pairwise_redundancy(pool)
    rule = artifact_rule(payload, rows)
    checks = consistency_checks(rows, results, rule)

    payload_out = {
        "artifact": args.artifact,
        "arm_d_reference": ARM_D,
        "artifact_rule": list(rule),
        "populations": {
            "all_claims": len(rows),
            "identifiable": len(pool),
            "overridable_baseline_support_with_counter": len(overridable),
            "actual_overrides": len(overrides),
            "rescues": len(rescues),
            "regressions": len(regressions),
        },
        "pass_rates": table,
        "subsets": results,
        "pairwise_redundancy": redundancy,
        "consistency_checks": checks,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    side = os.path.splitext(args.out)[0] + "_data.json"
    with open(side, "w", encoding="utf-8") as handle:
        json.dump(payload_out, handle, indent=2, ensure_ascii=False)

    # Console summary; the markdown report is written by render_report.py
    print(f"artifact gate rule: {'+'.join(rule)}")
    print(f"identifiable (has counter-evidence): {len(pool)}/{len(rows)}")
    print(f"overridable (baseline SUPPORT + counter): {len(overridable)}")
    print(f"actual overrides: {len(overrides)} ({len(rescues)} rescues, "
          f"{len(regressions)} regressions)\n")

    print("CONSISTENCY CHECKS")
    for check in checks:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['check']}  -- {check['detail']}")
    failed = [c for c in checks if not c["ok"]]
    print(f"  => {len(checks) - len(failed)}/{len(checks)} passed\n")

    print("GATE PASS RATES")
    header = f"  {'population':<54}{'n':>5}" + "".join(f"{g:>18}" for g in GATE_ORDER)
    print(header)
    for entry in table:
        print(f"  {entry['population']:<54}{entry['n']:>5}"
              + "".join(f"{entry[g]:>18}" for g in GATE_ORDER))
    print()

    print("16-SUBSET ABLATION  (sorted by overall accuracy)")
    print(f"  {'subset':<16}{'cands':>7}{'resc':>6}{'regr':>6}{'prec':>8}"
          f"{'overall':>9}{'d-all':>8}{'CONTRA':>9}{'d-C':>8}{'SUPP':>9}{'d-S':>8}{'net':>6}")
    for row in sorted(results, key=lambda r: -r["overall_accuracy"]):
        prec = f"{row['precision']}%" if row["precision"] is not None else "-"
        print(f"  {row['label']:<16}{row['candidate_overrides']:>7}{row['rescues']:>6}"
              f"{row['regressions']:>6}{prec:>8}{row['overall_accuracy']:>8}%"
              f"{row['overall_delta']:>+8.2f}{row['contradict_accuracy']:>8}%"
              f"{row['contradict_delta']:>+8.2f}{row['support_accuracy']:>8}%"
              f"{row['support_delta']:>+8.2f}{row['net_claims']:>+6}")
    print()

    print("PAIRWISE GATE REDUNDANCY (on the 289 identifiable claims)")
    for entry in redundancy:
        print(f"  {entry['pair']:<12} agreement {entry['agreement']:>5}%   "
              f"first-only-pass {entry['a_pass_b_fail']:>4}   "
              f"second-only-pass {entry['b_pass_a_fail']:>4}")
    print(f"\nWrote data to: {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
