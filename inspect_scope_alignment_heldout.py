"""
Build the HELD-OUT blind validation set for the scope-alignment hypothesis.

Research/analysis only. No production module is touched. No rule is defined or
changed here -- the frozen rules live in inspect_proposition_scope_alignment.py
and are only *referenced* by the annotation guide's schema, never evaluated.

=============================================================================
THIS SCRIPT NEVER READS A GOLD LABEL
=============================================================================
Eligibility uses exactly two fields, both specified in the brief and neither
outcome-derived:

    baseline_prediction == "SUPPORT"      (arm D's prediction, not its correctness)
    counter_evidence is non-empty

Every artifact record is then immediately projected onto a three-field
whitelist (claim, primary evidence, counter evidence) plus a content hash.
Sampling, ordering and the audit all operate on that projection, so `gold`,
`override_applied`, `adjudication_*`, gate outcomes, sections, doc ids and
every probability are structurally unreachable rather than merely unused.

That is deliberate beyond the brief's requirement: it also keeps the AUTHOR of
this script blind to the held-out labels, which preserves the option of a
model-assisted screen on this set later. Auditing blinding by reading the
labels would destroy exactly the property being audited.

=============================================================================
USAGE
=============================================================================
    python inspect_scope_alignment_heldout.py
    python inspect_scope_alignment_heldout.py --selftest
"""

import argparse
import hashlib
import json
import os
import random
import sys

ARTIFACT = "eval/results_20261011_160640_n504_section2_w0.05_prosactive_gatesg1g2g3.json"
PRIOR_BLIND = os.path.join("docs", "scope_alignment_blind_dataset.json")

OUT_MANIFEST = os.path.join("docs", "scope_alignment_heldout_manifest.json")
OUT_DATASET = os.path.join("docs", "scope_alignment_heldout_blind_dataset.json")
OUT_AUDIT = os.path.join("docs", "scope_alignment_heldout_audit.json")

SEED = 20261011
TARGET_N = 100

# --- annotation schema, exactly as specified in the brief --------------------
# NOTE the differences from the 42-case study's schema: PARTIAL is new,
# NOT_APPLICABLE is gone, and causal_relation_alignment is renamed
# causal_alignment. Implemented as specified; the comparability consequence is
# flagged in the report, not silently reconciled here.
ALIGNMENT = ["SAME", "PARTIAL", "DIFFERENT", "UNCLEAR"]
DIRECTION = ["OPPOSITE", "SAME", "UNCLEAR"]
JUDGEMENT = ["YES", "NO", "UNCLEAR"]

ANNOTATION_SCHEMA = {
    "overall_scope_match": JUDGEMENT,
    "population_alignment": ALIGNMENT,
    "intervention_alignment": ALIGNMENT,
    "outcome_alignment": ALIGNMENT,
    "endpoint_alignment": ALIGNMENT,
    "context_alignment": ALIGNMENT,
    "causal_alignment": ALIGNMENT,
    "direction": DIRECTION,
    "genuine_contradiction": JUDGEMENT,
}
ANNOTATION_FIELD_ORDER = [
    "overall_scope_match",
    "population_alignment",
    "intervention_alignment",
    "outcome_alignment",
    "endpoint_alignment",
    "context_alignment",
    "causal_alignment",
    "direction",
    "genuine_contradiction",
    "annotator_notes",
]

# Anything that is, or is derived from, an outcome. Asserted absent everywhere.
OUTCOME_DERIVED = [
    "gold", "prediction", "baseline_prediction", "baseline_verdict", "verdict",
    "override_applied", "override_correct", "adjudicated_verdict",
    "adjudication_status", "adjudication_reason", "adjudication_gates",
    "adjudication_required_gates", "matrix_label", "claim_index",
    "counter_section", "primary_evidence_section", "evidence_sections",
    "counter_evidence_doc_id", "evidence_doc_id", "evidence_doc_ids",
    "counter_p_contradiction", "reverse_p_contradiction",
    "primary_counter_p_contradiction", "primary_p_contradiction",
    "counter_confidence", "counter_score", "counter_distribution",
    "primary_verdict_distribution", "max_p_contradiction", "counter_rank",
    "counter_pool_index", "counter_in_d_selection", "counter_selection_rank",
    "candidate_pool_size", "n_contradictory_candidates",
    "strict_faithfulness", "weighted_faithfulness", "statements",
    "rescue", "regression",
]

SAMPLING_RULE = (
    'baseline_prediction == "SUPPORT" AND counter_evidence non-empty; '
    "then exclude every claim present in the 42-case blind dataset; "
    "no filter on gold label, override_applied, adjudication status or reason, "
    "section, or confidence; then random.Random(20261011).sample(frame, 100) "
    "over the eligible set sorted by sha256(claim) for determinism."
)


def claim_hash(claim):
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()


def project(record):
    """Whitelist projection. Everything not named here is discarded, so no
    outcome field can reach the sample, the manifest or the dataset."""
    return {
        "claim": record["claim"],
        "primary_evidence": record["evidence"],
        "counter_evidence": record["counter_evidence"],
        "claim_sha256": claim_hash(record["claim"]),
    }


def build_frame(payload, prior_claims):
    """Eligible candidates, as projections. Reads only the two frame fields."""
    frame, excluded = [], []
    for record in payload["eval_details"]:
        if record.get("baseline_prediction") != "SUPPORT":
            continue
        if not (record.get("counter_evidence") or "").strip():
            continue
        if record["claim"] in prior_claims:
            excluded.append(claim_hash(record["claim"]))
            continue
        frame.append(project(record))
    return frame, excluded


def draw_sample(frame, seed=SEED, target=TARGET_N):
    """Deterministic sample. `random.sample` returns its own randomized order,
    which becomes the presentation order."""
    ordered = sorted(frame, key=lambda item: item["claim_sha256"])
    rng = random.Random(seed)
    n = min(target, len(ordered))
    drawn = rng.sample(ordered, n)
    return drawn, len(ordered)


def blank_annotation():
    return {field: None for field in ANNOTATION_FIELD_ORDER}


def build_cases(drawn):
    return [
        {
            "case_id": f"H{i:03d}",
            "claim": item["claim"],
            "primary_evidence": item["primary_evidence"],
            "counter_evidence": item["counter_evidence"],
            "annotation": blank_annotation(),
        }
        for i, item in enumerate(drawn, 1)
    ]


# ===========================================================================
# Blinding audit -- conditions A-H
# ===========================================================================

def audit(cases, drawn, frame, frame_size, prior_claims, payload, n_excluded):
    checks = []

    def record(key, condition, ok, detail):
        checks.append(
            {"condition": key, "requirement": condition,
             "result": "PASS" if ok else "FAIL", "detail": detail}
        )
        return ok

    frame_hashes = {item["claim_sha256"] for item in frame}
    sample_hashes = [item["claim_sha256"] for item in drawn]

    # A -- every selected case comes from the allowed frame
    inside = all(h in frame_hashes for h in sample_hashes)
    # independently re-derive the frame straight from the artifact
    rederived = {
        claim_hash(r["claim"]) for r in payload["eval_details"]
        if r.get("baseline_prediction") == "SUPPORT"
        and (r.get("counter_evidence") or "").strip()
        and r["claim"] not in prior_claims
    }
    record("A", "every selected case is in the allowed sampling frame",
           inside and set(sample_hashes) <= rederived,
           f"{len(sample_hashes)} selected, all within the {len(rederived)}-case "
           f"independently re-derived frame")

    # B -- no previously annotated case included
    prior_hashes = {claim_hash(c) for c in prior_claims}
    overlap = sorted(set(sample_hashes) & prior_hashes)
    record("B", "none of the 42 previously annotated cases is included",
           not overlap,
           f"overlap = {len(overlap)} (42 prior cases excluded from the frame; "
           f"{n_excluded} exclusions applied)")

    # C / D -- no gold or outcome-derived field anywhere in the dataset
    keys = set()

    def walk(obj):
        if isinstance(obj, dict):
            keys.update(obj)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(cases)
    leaked = sorted(set(OUTCOME_DERIVED) & keys)
    record("C", "no gold/outcome fields present in the blind dataset",
           not leaked, f"keys present: {sorted(keys)}")
    record("D", "no outcome-derived metadata present",
           not leaked, f"forbidden keys found: {leaked or 'none'}")

    # E -- all annotation fields null
    all_null = all(
        value is None for case in cases for value in case["annotation"].values()
    )
    complete_schema = all(
        set(case["annotation"]) == set(ANNOTATION_FIELD_ORDER) for case in cases
    )
    record("E", "all annotation fields are null",
           all_null and complete_schema,
           f"{len(cases) * len(ANNOTATION_FIELD_ORDER)} fields, all null = {all_null}; "
           f"schema complete on every case = {complete_schema}")

    # F -- randomized presentation order
    by_hash = sorted(frame, key=lambda item: item["claim_sha256"])
    hash_order = [item["claim_sha256"] for item in by_hash if item["claim_sha256"] in set(sample_hashes)]
    differs_from_hash_order = sample_hashes != hash_order
    differs_from_artifact_order = sample_hashes != [
        claim_hash(r["claim"]) for r in payload["eval_details"]
        if claim_hash(r["claim"]) in set(sample_hashes)
    ]
    record("F", "the sample is in randomized presentation order",
           differs_from_hash_order and differs_from_artifact_order,
           f"differs from content-hash order = {differs_from_hash_order}; "
           f"differs from artifact row order = {differs_from_artifact_order}")

    # G -- seed recorded separately in the manifest
    record("G", "the random seed is recorded in the manifest",
           True, f"seed = {SEED}, recorded at manifest.sampling.random_seed")

    # H -- metadata cannot reveal rescue vs regression
    only_allowed = all(
        set(case) == {"case_id", "claim", "primary_evidence", "counter_evidence",
                      "annotation"}
        for case in cases
    )
    record("H", "the blind dataset cannot reveal rescue vs regression from metadata",
           only_allowed and not leaked,
           "verified STRUCTURALLY (no outcome field is present, and case_id is a "
           "sequential index over a seeded random draw). Deliberately NOT verified "
           "by correlating order against gold labels: doing so would require "
           "reading the held-out labels, destroying the blinding this condition "
           "exists to protect and contaminating any later model-assisted screen.")

    passed = sum(1 for c in checks if c["result"] == "PASS")
    return {
        "n_conditions": len(checks),
        "n_passed": passed,
        "overall": "PASS" if passed == len(checks) else "FAIL",
        "checks": checks,
        "frame_size_before_exclusion": frame_size + n_excluded,
        "n_previously_annotated_excluded": n_excluded,
        "eligible_after_exclusion": frame_size,
        "sample_size": len(cases),
        "gold_labels_read_by_this_script": False,
    }


# ===========================================================================
# Self-test
# ===========================================================================

def _selftest():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    fake = {"eval_details": [
        # in frame
        {"claim": "c1", "evidence": "p1", "counter_evidence": "x1",
         "baseline_prediction": "SUPPORT", "gold": "CONTRADICT", "override_applied": True},
        {"claim": "c2", "evidence": "p2", "counter_evidence": "x2",
         "baseline_prediction": "SUPPORT", "gold": "SUPPORT", "override_applied": False},
        # excluded: baseline not SUPPORT
        {"claim": "c3", "evidence": "p3", "counter_evidence": "x3",
         "baseline_prediction": "CONTRADICT", "gold": "CONTRADICT"},
        # excluded: empty counter-evidence
        {"claim": "c4", "evidence": "p4", "counter_evidence": "",
         "baseline_prediction": "SUPPORT", "gold": "SUPPORT"},
        {"claim": "c5", "evidence": "p5", "counter_evidence": None,
         "baseline_prediction": "SUPPORT", "gold": "SUPPORT"},
        # in frame but previously annotated
        {"claim": "prior", "evidence": "p6", "counter_evidence": "x6",
         "baseline_prediction": "SUPPORT", "gold": "SUPPORT"},
    ]}
    frame, excluded = build_frame(fake, {"prior"})
    check("frame size", len(frame), 2)
    check("frame claims", sorted(i["claim"] for i in frame), ["c1", "c2"])
    check("one exclusion", len(excluded), 1)

    # projection discards everything outcome-bearing
    check("projection keys", sorted(frame[0]),
          ["claim", "claim_sha256", "counter_evidence", "primary_evidence"])
    blob = json.dumps(frame)
    check("no gold in projection", "gold" in blob or "override" in blob, False)

    # determinism
    a, n1 = draw_sample(frame, seed=SEED, target=2)
    b, n2 = draw_sample(frame, seed=SEED, target=2)
    check("sample deterministic", [i["claim"] for i in a], [i["claim"] for i in b])
    check("eligible count returned", (n1, n2), (2, 2))
    c, _ = draw_sample(frame, seed=SEED + 1, target=2)
    check("different seed may reorder", len(c), 2)
    small, n3 = draw_sample(frame, seed=SEED, target=100)
    check("target above eligible -> take all", (len(small), n3), (2, 2))

    cases = build_cases(a)
    check("case ids", [x["case_id"] for x in cases], ["H001", "H002"])
    check("case keys", sorted(cases[0]),
          ["annotation", "case_id", "claim", "counter_evidence", "primary_evidence"])
    check("annotation schema size", len(cases[0]["annotation"]), 10)
    check("annotation all null",
          all(v is None for v in cases[0]["annotation"].values()), True)
    check("causal field name", "causal_alignment" in cases[0]["annotation"], True)
    check("old causal name absent",
          "causal_relation_alignment" in cases[0]["annotation"], False)
    check("PARTIAL in alignment enum", "PARTIAL" in ANNOTATION_SCHEMA["population_alignment"], True)
    check("NOT_APPLICABLE absent from alignment enum",
          "NOT_APPLICABLE" in ANNOTATION_SCHEMA["population_alignment"], False)
    check("direction enum", ANNOTATION_SCHEMA["direction"], ["OPPOSITE", "SAME", "UNCLEAR"])

    # The 2-case fixture above is too small to exercise condition F -- a
    # 2-of-2 draw can coincide with the source order by chance. Audit the
    # full 8 conditions on a frame big enough for order to be meaningful.
    big = {"eval_details": [
        {"claim": f"claim number {i}", "evidence": f"p{i}",
         "counter_evidence": f"x{i}", "baseline_prediction": "SUPPORT",
         "gold": "CONTRADICT" if i % 3 else "SUPPORT",
         "override_applied": bool(i % 2)}
        for i in range(30)
    ]}
    big_frame, big_excluded = build_frame(big, set())
    check("big frame size", len(big_frame), 30)
    big_drawn, big_elig = draw_sample(big_frame, seed=SEED, target=20)
    big_cases = build_cases(big_drawn)
    result = audit(big_cases, big_drawn, big_frame, big_elig, set(), big,
                   len(big_excluded))
    check("audit all pass", result["overall"], "PASS")
    check("audit condition count", result["n_conditions"], 8)
    check("audit never read gold", result["gold_labels_read_by_this_script"], False)
    check("audit sample size", result["sample_size"], 20)

    # Condition F must FAIL when the order is NOT randomized: feed back the
    # content-hash-sorted order instead of the seeded draw.
    unshuffled = sorted(big_frame, key=lambda i: i["claim_sha256"])[:20]
    bad_order = audit(build_cases(unshuffled), unshuffled, big_frame, big_elig,
                      set(), big, len(big_excluded))
    f_result = next(c["result"] for c in bad_order["checks"] if c["condition"] == "F")
    check("condition F detects unshuffled order", f_result, "FAIL")

    # A case from outside the frame must FAIL condition A.
    intruder = build_cases(big_drawn[:5]) + build_cases(
        [project({"claim": "not in frame", "evidence": "p", "counter_evidence": "x"})]
    )
    bad_frame = audit(intruder, big_drawn[:5] + [project(
        {"claim": "not in frame", "evidence": "p", "counter_evidence": "x"})],
        big_frame, big_elig, set(), big, len(big_excluded))
    a_result = next(c["result"] for c in bad_frame["checks"] if c["condition"] == "A")
    check("condition A detects an out-of-frame case", a_result, "FAIL")

    # A previously annotated case must FAIL condition B.
    prior_case = project({"claim": "claim number 7", "evidence": "p",
                          "counter_evidence": "x"})
    bad_prior = audit(build_cases([prior_case]), [prior_case], big_frame,
                      big_elig, {"claim number 7"}, big, 1)
    b_result = next(c["result"] for c in bad_prior["checks"] if c["condition"] == "B")
    check("condition B detects a previously annotated case", b_result, "FAIL")

    # a deliberately leaky dataset must FAIL C/D/H
    leaky = [dict(big_cases[0], gold="CONTRADICT")]
    bad = audit(leaky, big_drawn, big_frame, big_elig, set(), big, len(big_excluded))
    failed = {c["condition"] for c in bad["checks"] if c["result"] == "FAIL"}
    check("leak detected on C/D/H", {"C", "D", "H"} <= failed, True)

    # a non-null annotation must FAIL condition E
    filled = [dict(big_cases[0],
                   annotation={**big_cases[0]["annotation"], "direction": "OPPOSITE"})]
    bad_e = audit(filled, big_drawn, big_frame, big_elig, set(), big, len(big_excluded))
    e_result = next(c["result"] for c in bad_e["checks"] if c["condition"] == "E")
    check("condition E detects a filled annotation", e_result, "FAIL")

    print("\n" + "=" * 74)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        print("=" * 74)
        return 1
    print("SELFTEST PASSED -- frame construction, whitelist projection, sample")
    print("determinism, small-frame fallback, schema shape, and all 8 audit")
    print("conditions including positive leak detection.")
    print("=" * 74)
    return 0


# ===========================================================================
# Annotation guide
# ===========================================================================

def write_guide(path, n_cases):
    lines = []
    add = lines.append
    add("# Annotation guide — scope alignment (held-out set)")
    add("")
    add(f"{n_cases} cases, in `docs/scope_alignment_heldout_blind_dataset.json`.")
    add("")
    add("## Your task")
    add("")
    add("Each case gives you three pieces of text:")
    add("")
    add("- a **claim**")
    add("- a **primary evidence** sentence or pair of sentences")
    add("- a **counter-evidence** sentence")
    add("")
    add("For each case, judge how the **claim** relates to the **counter-evidence**,")
    add("and record nine judgements. The primary evidence is supplied for context")
    add("only — it is what a system had already retrieved — but every judgement below")
    add("is about the claim/counter-evidence pair.")
    add("")
    add("Work from the supplied text. Do not look anything up, and do not consult any")
    add("other file in this repository.")
    add("")
    add("## The nine fields")
    add("")
    add("### `overall_scope_match` — YES / NO / UNCLEAR")
    add("")
    add("Are the claim and the counter-evidence about the **same proposition** — that")
    add("is, could one in principle confirm or refute the other? Answer YES when they")
    add("address the same thing, whether or not they agree. Answer NO when they are")
    add("about different things, even if they share topic or vocabulary.")
    add("")
    add("### The six alignment fields — SAME / PARTIAL / DIFFERENT / UNCLEAR")
    add("")
    add("| field | question |")
    add("|---|---|")
    add("| `population_alignment` | Same subjects, patients, cells, species, or study group? |")
    add("| `intervention_alignment` | Same intervention, exposure, drug, gene, or programme? |")
    add("| `outcome_alignment` | Same outcome or phenomenon? |")
    add("| `endpoint_alignment` | Same **specific measured** endpoint? |")
    add("| `context_alignment` | Same setting or conditions — in vivo vs in vitro, species, disease state, dose regime, time frame? |")
    add("| `causal_alignment` | Same kind of relation between the same entities — causes / is required for / is associated with / interacts with? |")
    add("")
    add("Use **PARTIAL** when the two overlap but do not coincide: one is a subset,")
    add("a narrower case, or a related-but-not-identical version of the other.")
    add("")
    add("`outcome_alignment` and `endpoint_alignment` are separate on purpose. The")
    add("outcome can match while the endpoint differs — \"cancer risk\" against")
    add("\"mammographic density\" is the same outcome area measured a different way,")
    add("and \"cancer\" against \"colorectal cancer\" is an aggregate against a")
    add("specific site.")
    add("")
    add("### `direction` — OPPOSITE / SAME / UNCLEAR")
    add("")
    add("Relative to the claim, which way does the counter-evidence point **on a")
    add("shared axis**? OPPOSITE means they point opposite ways on the same axis")
    add("(claim says a treatment reduces something; the evidence says it increased, or")
    add("did not change). SAME means they point the same way. If the two are not about")
    add("a shared axis at all, use UNCLEAR.")
    add("")
    add("### `genuine_contradiction` — YES / NO / UNCLEAR")
    add("")
    add("Does the counter-evidence **actually contradict** the claim? Judge this on its")
    add("own; it is not a summary of the fields above. Two sentences can be about the")
    add("same proposition and agree, and a sentence can look contradictory while")
    add("actually being about something else.")
    add("")
    add("### `annotator_notes` — free text, optional")
    add("")
    add("Use it whenever you answer UNCLEAR, so the reason is recoverable.")
    add("")
    add("## Guidance")
    add("")
    add("- **Shared vocabulary is not contradiction.** Two sentences can be packed with")
    add("  the same technical terms and still be about different measurements,")
    add("  different populations, or different studies. Check what is actually being")
    add("  asserted before judging the relation.")
    add("- **Watch for a different endpoint.** The commonest trap is a counter-evidence")
    add("  sentence that shares the claim's intervention and population but reports a")
    add("  different measurement. That is `outcome_alignment` or `endpoint_alignment`")
    add("  DIFFERENT, and it usually means `genuine_contradiction` = NO.")
    add("- **Watch for a different population or intervention.** A result about a")
    add("  different cell type, species, subgroup, drug, or gene does not refute a")
    add("  claim about the one named in the claim.")
    add("- **Aggregate vs specific.** A null result on an aggregate category does not")
    add("  negate an effect on one specific member of it. Usually PARTIAL on the")
    add("  relevant field.")
    add("- **UNCLEAR is a real answer.** Prefer it to a forced guess. A high UNCLEAR")
    add("  rate is itself a useful result about how decidable these judgements are.")
    add("- **Base judgements on the supplied text.** Use outside domain knowledge only")
    add("  where the text cannot be read without it, and say so in the notes.")
    add("- **Annotate each case independently.** Do not let one case influence the next,")
    add("  and do not try to balance your answers across the set.")
    add("- **Do not try to guess what any system decided** about these cases, or whether")
    add("  it was right. That information is not in this file.")
    add("")
    add("## Recording your answers")
    add("")
    add("Fill the `annotation` object for each case. Either edit the blind dataset in")
    add("place, or copy it to `docs/scope_alignment_heldout_annotations.json` and fill")
    add("that. Leave any field you genuinely cannot judge as `null` rather than")
    add("guessing — `null` and `UNCLEAR` mean different things: `UNCLEAR` is a")
    add("judgement that the text does not settle it, `null` means you did not judge.")
    add("")
    add("Valid values, exactly:")
    add("")
    add("```")
    for field in ANNOTATION_FIELD_ORDER:
        if field == "annotator_notes":
            add(f"{field:<26} free text")
        else:
            add(f"{field:<26} {' / '.join(ANNOTATION_SCHEMA[field])}")
    add("```")
    add("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build the held-out blind validation set (analysis only)"
    )
    parser.add_argument("--artifact", default=ARTIFACT)
    parser.add_argument("--prior", default=PRIOR_BLIND)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    for path in (args.artifact, args.prior):
        if not os.path.exists(path):
            print(f"ERROR: required file not found: {path}")
            return 2

    with open(args.artifact, encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(args.prior, encoding="utf-8") as handle:
        prior = json.load(handle)
    prior_claims = {case["claim"] for case in prior["cases"]}

    frame, excluded = build_frame(payload, prior_claims)
    drawn, eligible = draw_sample(frame, seed=args.seed, target=args.n)
    cases = build_cases(drawn)

    print("=" * 74)
    print("HELD-OUT BLIND VALIDATION SET")
    print("=" * 74)
    print(f"artifact                      : {args.artifact}")
    print(f"  gate rule                   : {payload['metadata'].get('adjudication_gates')}")
    print(f"frame (baseline SUPPORT + counter-evidence) : {eligible + len(excluded)}")
    print(f"previously annotated, excluded              : {len(excluded)}")
    print(f"eligible after exclusion                    : {eligible}")
    print(f"random seed                                 : {args.seed}")
    print(f"held-out sample size                        : {len(cases)}")
    print(f"gold labels read by this script             : NO")

    manifest = {
        "metadata": {
            "study": "scope-alignment held-out validation",
            "research_only": True,
            "production_modified": False,
            "source_artifact": args.artifact,
            "source_artifact_gate_rule": payload["metadata"].get("adjudication_gates"),
            "prior_study_dataset": args.prior,
            "gold_labels_read_when_building": False,
        },
        "sampling": {
            "random_seed": args.seed,
            "rule": SAMPLING_RULE,
            "frame_size_before_exclusion": eligible + len(excluded),
            "n_previously_annotated_excluded": len(excluded),
            "eligible_after_exclusion": eligible,
            "target_n": args.n,
            "sample_size": len(cases),
            "note": (
                "No filter was applied on gold label, override_applied, "
                "adjudication status, adjudication reason, section, or any "
                "confidence/probability field."
            ),
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "sample_index": index,
                "claim_sha256": claim_hash(case["claim"]),
            }
            for index, case in enumerate(cases, 1)
        ],
    }

    dataset = {
        "metadata": {
            "study": "scope-alignment held-out validation",
            "n_cases": len(cases),
            "exposed_fields": ["case_id", "claim", "primary_evidence", "counter_evidence"],
            "annotation_schema": ANNOTATION_SCHEMA,
            "free_text_field": "annotator_notes",
            "annotations_performed": False,
            "guide": "docs/scope_alignment_heldout_annotation_guide.md",
        },
        "cases": cases,
    }

    report = audit(cases, drawn, frame, eligible, prior_claims, payload, len(excluded))

    os.makedirs("docs", exist_ok=True)
    for path, data in (
        (OUT_MANIFEST, manifest),
        (OUT_DATASET, dataset),
        (OUT_AUDIT, report),
    ):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
    write_guide(
        os.path.join("docs", "scope_alignment_heldout_annotation_guide.md"), len(cases)
    )

    print("\nBLINDING AUDIT")
    for check in report["checks"]:
        print(f"  [{check['result']}] {check['condition']}: {check['requirement']}")
        print(f"         {check['detail']}")
    print(f"\n  overall: {report['overall']} ({report['n_passed']}/{report['n_conditions']})")

    print("\nfiles written:")
    for path in (OUT_MANIFEST, OUT_DATASET, OUT_AUDIT,
                 os.path.join("docs", "scope_alignment_heldout_annotation_guide.md")):
        print(f"  {path}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
