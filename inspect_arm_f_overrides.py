"""
Arm F override error analysis -- read-only diagnostic.

Reads an arm F (`--prosecution_mode active`) results artifact, extracts every
overridden claim, and writes docs/arm_f_override_error_analysis.md with the
compact table and all of the concentration analyses.

Changes nothing: no production code, no retrieval, no extraction, no
verification, no prosecution, no adjudication, no evaluation logic. Reads JSON
and writes one markdown file.

MECHANICAL VS JUDGEMENT
-----------------------
Everything this script computes is mechanical and reproducible: the table, the
correct-vs-regression comparison, and the concentration breakdowns by section,
document, adjudication reason, confidence, margin and in-bundle status.

The failure CATEGORY per regression (semantic scope mismatch / unrelated
evidence / NLI error / provenance failure / other) and the "what made this
rescue genuinely relevant" note require reading the sentences. The script emits
the text for each case and leaves those fields blank. Pass `--categories
<json>` to fold in a `{"claim_index": {"category": "A", "note": "..."}}`
mapping once that pass is done, and the report regenerates with them included.

No threshold is proposed and no scope criterion is invented: distributions are
reported descriptively, with explicit overlap statements.

USAGE
    python inspect_arm_f_overrides.py \\
        --artifact eval/results_20261009_085502_n504_section2_w0.05_prosactive.json

    python inspect_arm_f_overrides.py --artifact <path> --categories cats.json
"""

import argparse
import json
import os
import statistics
import sys

DEFAULT_ARTIFACT = "eval/results_20261009_085502_n504_section2_w0.05_prosactive.json"
DEFAULT_OUT = os.path.join("docs", "arm_f_override_error_analysis.md")

# Arm H's measured figures, used to derive the like-for-like override baseline.
ARM_H_RESCUABLE = 44
ARM_H_NAIVE_SUPPORT_ACCURACY = 0.6073

FAILURE_CATEGORIES = {
    "A": "semantic scope mismatch",
    "B": "unrelated evidence",
    "C": "NLI error",
    "D": "provenance / grounding failure",
    "E": "other",
}


def load(artifact):
    with open(artifact, encoding="utf-8") as handle:
        return json.load(handle)


def extract_overrides(payload):
    """Every claim whose prediction arm F changed, with all recorded fields."""
    details = {d["claim_index"]: d for d in payload.get("eval_details", [])}
    diagnostics = payload.get("adjudication_diagnostics", {}) or {}

    indices = [
        record["claim_index"]
        for record in diagnostics.get("overridden_claims", [])
    ]
    # Fall back to scanning eval_details if the diagnostics block is absent or
    # truncated (print_adjudication_diagnostics caps its console listing).
    scanned = [
        index for index, detail in details.items() if detail.get("override_applied")
    ]
    if len(scanned) > len(indices):
        indices = sorted(scanned)

    overrides = []
    for index in indices:
        detail = details.get(index)
        if detail is None:
            continue
        gold = detail.get("gold")
        correct = detail.get("prediction") == gold
        overrides.append(
            {
                "claim_index": index,
                "gold": gold,
                "baseline_prediction": detail.get("baseline_prediction"),
                "prediction": detail.get("prediction"),
                "override_correct": correct,
                "outcome": "RESCUE" if correct else "REGRESSION",
                "claim": detail.get("claim"),
                "primary_evidence": detail.get("evidence"),
                "primary_doc_id": detail.get("evidence_doc_id"),
                "primary_section": detail.get("primary_evidence_section"),
                "counter_evidence": detail.get("counter_evidence"),
                "counter_doc_id": detail.get("counter_evidence_doc_id"),
                "counter_section": detail.get("counter_section"),
                "counter_confidence": detail.get("counter_confidence"),
                "counter_margin": detail.get("counter_score"),
                "counter_p_contradiction": detail.get("counter_p_contradiction"),
                "counter_distribution": detail.get("counter_distribution"),
                "counter_rank": detail.get("counter_rank"),
                "counter_pool_index": detail.get("counter_pool_index"),
                "counter_in_d_selection": detail.get("counter_in_d_selection"),
                "counter_selection_rank": detail.get("counter_selection_rank"),
                "candidate_pool_size": detail.get("candidate_pool_size"),
                "n_contradictory_candidates": detail.get("n_contradictory_candidates"),
                "max_p_contradiction": detail.get("max_p_contradiction"),
                "baseline_verdict": detail.get("baseline_verdict"),
                "adjudicated_verdict": detail.get("adjudicated_verdict"),
                "adjudication_status": detail.get("adjudication_status"),
                "adjudication_reason": detail.get("adjudication_reason"),
                "adjudication_gates": detail.get("adjudication_gates"),
                "primary_verdict_distribution": detail.get("primary_verdict_distribution"),
                "primary_p_contradiction": detail.get("primary_p_contradiction"),
                "primary_confidence": detail.get("primary_confidence"),
                "reverse_verdict": detail.get("reverse_verdict"),
                "reverse_p_contradiction": detail.get("reverse_p_contradiction"),
                "primary_counter_verdict": detail.get("primary_counter_verdict"),
                "primary_counter_p_contradiction": detail.get(
                    "primary_counter_p_contradiction"
                ),
                "cross_document": (
                    detail.get("counter_evidence_doc_id") is not None
                    and str(detail.get("counter_evidence_doc_id"))
                    != str(detail.get("evidence_doc_id"))
                ),
            }
        )
    return overrides


def counts_by(records, key, missing="(none)"):
    out = {}
    for record in records:
        value = record.get(key)
        value = missing if value is None else value
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def numeric_summary(records, key):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    if not values:
        return None
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def concentration_table(rescues, regressions, key, missing="(none)"):
    """Cross-tab one categorical field against outcome, with a rate column."""
    keys = set()
    for group in (rescues, regressions):
        for record in group:
            value = record.get(key)
            keys.add(missing if value is None else value)
    rows = []
    for value in sorted(keys, key=str):
        n_rescue = sum(
            1 for r in rescues if (r.get(key) if r.get(key) is not None else missing) == value
        )
        n_reg = sum(
            1 for r in regressions if (r.get(key) if r.get(key) is not None else missing) == value
        )
        total = n_rescue + n_reg
        rows.append(
            {
                "value": value,
                "rescues": n_rescue,
                "regressions": n_reg,
                "total": total,
                "precision": round(n_rescue / total * 100, 1) if total else 0.0,
            }
        )
    return sorted(rows, key=lambda row: -row["total"])


def overlap_statement(rescues, regressions, key):
    a = [r[key] for r in rescues if isinstance(r.get(key), (int, float))]
    b = [r[key] for r in regressions if isinstance(r.get(key), (int, float))]
    if not a or not b:
        return "insufficient data"
    if min(a) > max(b):
        return f"SEPARATES: rescues all above regressions by >= {min(a) - max(b):.4f}"
    if min(b) > max(a):
        return f"SEPARATES: regressions all above rescues by >= {min(b) - max(a):.4f}"
    lo = max(min(a), min(b))
    hi = min(max(a), max(b))
    return f"OVERLAPS on [{lo:.4f}, {hi:.4f}] -- no clean split"


def truncate(text, width=70):
    if text is None:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def derive_baseline():
    """The like-for-like override baseline, from arm H's reported SUPPORT drop."""
    naive_support_correct = round(ARM_H_NAIVE_SUPPORT_ACCURACY * 331)
    losable = 306 - naive_support_correct
    overridable = ARM_H_RESCUABLE + losable
    return {
        "losable": losable,
        "rescuable": ARM_H_RESCUABLE,
        "overridable": overridable,
        "base_rate": ARM_H_RESCUABLE / overridable,
    }


def build_report(payload, overrides, categories, artifact):
    metadata = payload.get("metadata", {})
    metrics = payload.get("metrics", {})
    adjudication = payload.get("adjudication_diagnostics", {}) or {}

    rescues = [r for r in overrides if r["override_correct"]]
    regressions = [r for r in overrides if not r["override_correct"]]
    baseline = derive_baseline()

    lines = []
    add = lines.append

    add("# Arm F override error analysis")
    add("")
    add("Read-only analysis. No production code was changed to produce it.")
    add("")
    add(f"- **Artifact:** `{artifact}`")
    add(f"- **Config:** evidence_mode=`{metadata.get('evidence_mode')}`, "
        f"max_sentences=`{metadata.get('max_sentences')}`, "
        f"section_weight=`{metadata.get('section_weight')}`, "
        f"prosecution_mode=`{metadata.get('prosecution_mode')}`")
    add(f"- **Claims:** {metadata.get('n_claims')}")
    add(f"- **Overrides found:** {len(overrides)} "
        f"({len(rescues)} rescues, {len(regressions)} regressions)")
    add("")

    # --- headline ----------------------------------------------------
    add("## 1. Effect on accuracy")
    add("")
    add("| metric | arm D | arm F | delta |")
    add("|---|---|---|---|")
    base_acc = adjudication.get("baseline_accuracy") or {}
    adj_acc = adjudication.get("adjudicated_accuracy") or {}
    if base_acc and adj_acc:
        add(f"| overall | {base_acc.get('overall')}% | {adj_acc.get('overall')}% | "
            f"{adj_acc.get('overall', 0) - base_acc.get('overall', 0):+.2f} |")
        for label in sorted(set(base_acc.get("by_label", {})) | set(adj_acc.get("by_label", {}))):
            b = base_acc.get("by_label", {}).get(label, {})
            a = adj_acc.get("by_label", {}).get(label, {})
            add(f"| {label} | {b.get('accuracy')}% ({b.get('correct')}/{b.get('n')}) | "
                f"{a.get('accuracy')}% ({a.get('correct')}/{a.get('n')}) | "
                f"{a.get('accuracy', 0) - b.get('accuracy', 0):+.2f} |")
    add("")
    add(f"Net: **{len(rescues) - len(regressions):+d} claims**. "
        f"Override precision **{len(rescues)}/{len(overrides)} = "
        f"{len(rescues) / len(overrides) * 100:.2f}%**" if overrides else "")
    add("")
    add("### Did the gates discriminate, or only reduce volume?")
    add("")
    add("The like-for-like baseline is NOT arm H's reported 54.86% precision: that")
    add("figure's denominator is all claims carrying counter-evidence, including")
    add("ones where an override changes nothing because arm D already said")
    add("CONTRADICT. The comparable denominator is the *overridable* population --")
    add("claims arm D predicted SUPPORT that carry counter-evidence:")
    add("")
    add(f"- rescuable (gold CONTRADICT, baseline SUPPORT, has counter): "
        f"**{baseline['rescuable']}** (arm H)")
    add(f"- losable (gold SUPPORT, baseline SUPPORT, has counter): "
        f"**{baseline['losable']}** (derived from arm H's naive-rule SUPPORT of "
        f"{ARM_H_NAIVE_SUPPORT_ACCURACY:.2%})")
    add(f"- overridable population: **{baseline['overridable']}**")
    add(f"- base rate of correctness: **{baseline['base_rate'] * 100:.2f}%**")
    add("")
    if overrides:
        precision = len(rescues) / len(overrides)
        expected = len(overrides) * baseline["base_rate"]
        add(f"Arm F's {precision * 100:.2f}% is **{precision / baseline['base_rate']:.2f}x** "
            f"that base rate. A random {len(overrides)} drawn from the "
            f"{baseline['overridable']} would be expected to yield "
            f"{expected:.1f} rescues and a net of "
            f"{expected - (len(overrides) - expected):+.1f} claims.")
    add("")

    # --- the compact table -------------------------------------------
    add("## 2. All overrides")
    add("")
    add("| # | gold | base | pred | outcome | cat | counter sec | x-doc | in bundle "
        "| conf | margin | rev p(C) | prim-cnt p(C) | rank | pool |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for record in overrides:
        category = (categories.get(str(record["claim_index"])) or {}).get("category", "")
        add(
            f"| {record['claim_index']} | {record['gold'][:1]} | "
            f"{str(record['baseline_prediction'])[:1]} | {str(record['prediction'])[:1]} | "
            f"{'rescue' if record['override_correct'] else '**REGR**'} | {category} | "
            f"{record['counter_section']} | {'yes' if record['cross_document'] else 'no'} | "
            f"{'yes' if record['counter_in_d_selection'] else 'no'} | "
            f"{record['counter_confidence']} | {record['counter_margin']} | "
            f"{record['reverse_p_contradiction']} | "
            f"{record['primary_counter_p_contradiction']} | "
            f"{record['counter_rank']} | {record['candidate_pool_size']} |"
        )
    add("")
    add("`conf` = counter_confidence (argmax class probability). `margin` = "
        "counter_score (P(contradiction) minus best competitor). `rev p(C)` = "
        "P(contradiction) for claim->counter (the symmetry gate). `prim-cnt p(C)` "
        "= P(contradiction) for primary->counter (the coherence gate).")
    add("")

    # --- concentration analyses --------------------------------------
    add("## 3. Where are the regressions concentrated?")
    add("")
    for key, title in [
        ("counter_section", "Counter-evidence section"),
        ("cross_document", "Cross-document counter-evidence"),
        ("counter_in_d_selection", "Counter already in arm D's bundle"),
        ("adjudication_reason", "Adjudication reason"),
        ("reverse_verdict", "Reverse-direction verdict"),
        ("primary_counter_verdict", "Primary-vs-counter verdict"),
        ("baseline_verdict", "Baseline verdict"),
    ]:
        rows = concentration_table(rescues, regressions, key)
        add(f"### {title}")
        add("")
        add("| value | rescues | regressions | total | precision |")
        add("|---|---|---|---|---|")
        for row in rows:
            add(f"| {row['value']} | {row['rescues']} | {row['regressions']} | "
                f"{row['total']} | {row['precision']}% |")
        add("")

    add("### Numeric fields: rescues vs regressions")
    add("")
    add("| field | rescues (min/med/max) | regressions (min/med/max) | separation |")
    add("|---|---|---|---|")
    for key in [
        "counter_confidence",
        "counter_margin",
        "counter_p_contradiction",
        "reverse_p_contradiction",
        "primary_counter_p_contradiction",
        "primary_p_contradiction",
        "max_p_contradiction",
        "counter_rank",
        "candidate_pool_size",
        "n_contradictory_candidates",
    ]:
        a = numeric_summary(rescues, key)
        b = numeric_summary(regressions, key)
        fmt = lambda s: f"{s['min']} / {s['median']} / {s['max']}" if s else "-"
        add(f"| {key} | {fmt(a)} | {fmt(b)} | "
            f"{overlap_statement(rescues, regressions, key)} |")
    add("")
    add("Descriptive only. No threshold is proposed; overlap statements say")
    add("whether a clean split exists at all, which is the prerequisite for")
    add("asking the question later.")
    add("")

    # --- per-case detail ---------------------------------------------
    for title, group in [
        ("4. SUPPORT regressions (wrong overrides)", regressions),
        ("5. Rescued CONTRADICT claims (correct overrides)", rescues),
    ]:
        add(f"## {title}")
        add("")
        if not group:
            add("_None._")
            add("")
            continue
        for record in group:
            entry = categories.get(str(record["claim_index"])) or {}
            add(f"### Claim {record['claim_index']} "
                f"(gold {record['gold']}, {record['baseline_prediction']} -> "
                f"{record['prediction']})")
            add("")
            add(f"- **Claim:** {truncate(record['claim'], 400)}")
            add(f"- **Primary evidence** (doc `{record['primary_doc_id']}`, "
                f"section `{record['primary_section']}`): "
                f"{truncate(record['primary_evidence'], 400)}")
            add(f"- **Counter-evidence** (doc `{record['counter_doc_id']}`, "
                f"section `{record['counter_section']}`, "
                f"cross-doc `{record['cross_document']}`, "
                f"in-bundle `{record['counter_in_d_selection']}`): "
                f"{truncate(record['counter_evidence'], 400)}")
            add(f"- **Adjudication:** {record['adjudication_status']} -- "
                f"`{record['adjudication_reason']}`")
            add(f"- **Gates:** `{record['adjudication_gates']}`")
            add(f"- **NLI:** counter->claim P(C)="
                f"{record['counter_p_contradiction']} (conf "
                f"{record['counter_confidence']}, margin {record['counter_margin']}); "
                f"claim->counter P(C)={record['reverse_p_contradiction']} "
                f"({record['reverse_verdict']}); "
                f"primary->counter P(C)="
                f"{record['primary_counter_p_contradiction']} "
                f"({record['primary_counter_verdict']}); "
                f"primary->claim P(C)={record['primary_p_contradiction']} "
                f"(conf {record['primary_confidence']})")
            add(f"- **Distributions:** counter `{record['counter_distribution']}`; "
                f"primary `{record['primary_verdict_distribution']}`")
            add(f"- **Pool:** size {record['candidate_pool_size']}, "
                f"{record['n_contradictory_candidates']} contradictory, "
                f"counter at contradiction-rank {record['counter_rank']} / "
                f"pool index {record['counter_pool_index']}, "
                f"max P(C) in pool {record['max_p_contradiction']}")
            if record["override_correct"]:
                add(f"- **Why genuinely relevant:** "
                    f"{entry.get('note', '_(pending qualitative pass)_')}")
            else:
                category = entry.get("category")
                label = FAILURE_CATEGORIES.get(category, "_(pending qualitative pass)_")
                add(f"- **Failure category:** "
                    f"{(category + ' - ' + label) if category else label}")
                add(f"- **Notes:** {entry.get('note', '_(pending)_')}")
            add("")

    add("## 6. Category legend")
    add("")
    for key, value in FAILURE_CATEGORIES.items():
        add(f"- **{key}** -- {value}")
    add("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Arm F override error analysis (read-only)"
    )
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--categories", default=None,
                        help='JSON: {"<claim_index>": {"category": "A", "note": "..."}}')
    args = parser.parse_args()

    if not os.path.exists(args.artifact):
        print(f"ERROR: artifact not found: {args.artifact}")
        print("Arm F ran in Colab, so the results JSON is there, not in this repo.")
        print("Copy it into eval/ (or pass --artifact <path>) and re-run.")
        return 2

    payload = load(args.artifact)
    if payload.get("metadata", {}).get("prosecution_mode") != "active":
        print("WARNING: artifact metadata does not say prosecution_mode=active; "
              "there may be no overrides to analyse.")

    overrides = extract_overrides(payload)
    if not overrides:
        print("No overrides found in the artifact.")
        return 2

    categories = {}
    if args.categories:
        with open(args.categories, encoding="utf-8") as handle:
            categories = json.load(handle)

    report = build_report(payload, overrides, categories, args.artifact)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)

    rescues = sum(1 for r in overrides if r["override_correct"])
    print(f"Extracted {len(overrides)} overrides "
          f"({rescues} rescues, {len(overrides) - rescues} regressions)")
    print(f"Wrote report to: {args.out}")

    side = os.path.splitext(args.out)[0] + "_data.json"
    with open(side, encoding="utf-8", mode="w") as handle:
        json.dump(
            {"artifact": args.artifact, "overrides": overrides}, handle,
            indent=2, ensure_ascii=False,
        )
    print(f"Wrote per-case data to: {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
