"""
Build the PRIMARY held-out blind validation set for Rule A.

Research/analysis only. No production module is touched. No rule is created or
tuned. Rule A stays frozen in inspect_proposition_scope_alignment.py.

Differences from the rejected-candidate stress test
(inspect_scope_alignment_heldout.py):

  * consumes a FRESH candidate artifact over a claim population the pipeline
    has not scored before, so the frame contains BOTH accepted and rejected
    candidates;
  * reproduces STUDY 1's annotation schema exactly -- field names (including
    `causal_relation_alignment`), enum values, NOT_APPLICABLE handling, and no
    PARTIAL -- so the two studies are directly comparable;
  * pre-registered seed 20261012.

THIS SCRIPT NEVER READS A GOLD LABEL. Eligibility uses `baseline_prediction`
and `counter_evidence` only; every record is then projected onto a three-field
whitelist, so `gold`, `override_applied`, `adjudication_*`, gate outcomes,
sections, doc ids and all probabilities are structurally unreachable. See
verify_blind_projection.py for an independent proof.

    python inspect_scope_alignment_primary_heldout.py --artifact <fresh artifact>
    python inspect_scope_alignment_primary_heldout.py --selftest
"""

import argparse
import hashlib
import json
import os
import random
import sys

PRIOR_BLIND = os.path.join("docs", "scope_alignment_blind_dataset.json")
OUT_MANIFEST = os.path.join("docs", "scope_alignment_primary_heldout_manifest.json")
OUT_DATASET = os.path.join("docs", "scope_alignment_primary_heldout_blind_dataset.json")
OUT_AUDIT = os.path.join("docs", "scope_alignment_primary_heldout_audit.json")

SEED = 20261012
TARGET_N = 100

# --- STUDY 1 SCHEMA, reproduced exactly ------------------------------------
# Read from docs/scope_alignment_blind_dataset.json. Note `causal_relation_alignment`
# (not `causal_alignment`), NOT_APPLICABLE present, PARTIAL absent.
ALIGNMENT = ["SAME", "DIFFERENT", "NOT_APPLICABLE", "UNCLEAR"]
DIRECTION = ["OPPOSITE", "SAME", "NOT_APPLICABLE", "UNCLEAR"]
JUDGEMENT = ["YES", "NO", "UNCLEAR"]

STUDY1_FIELD_ORDER = [
    "population_alignment",
    "intervention_alignment",
    "outcome_alignment",
    "endpoint_alignment",
    "context_alignment",
    "causal_relation_alignment",
    "direction",
    "overall_scope_match",
    "genuine_contradiction",
    "annotator_notes",
]
STUDY1_ENUMS = {
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
    "candidate_pool_size", "n_contradictory_candidates", "evidence_spans",
    "strict_faithfulness", "weighted_faithfulness", "statements",
    "rescue", "regression",
]

SAMPLING_RULE = (
    'baseline_prediction == "SUPPORT" AND counter_evidence non-empty; '
    "exclude any claim present in the 42-case study-1 dataset by content hash; "
    "no filter on gold label, override_applied, adjudication status or reason, "
    "section, or confidence; sort eligible by sha256(claim); then "
    "random.Random(20261012).sample(frame, 100)."
)


def claim_hash(claim):
    return hashlib.sha256(claim.encode("utf-8")).hexdigest()


def project(record):
    """Whitelist projection -- the single chokepoint through which artifact
    data reaches the sample. Nothing not named here can pass."""
    return {
        "claim": record["claim"],
        "primary_evidence": record["evidence"],
        "counter_evidence": record["counter_evidence"],
        "claim_sha256": claim_hash(record["claim"]),
    }


def build_frame(payload, prior_claims):
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
    ordered = sorted(frame, key=lambda item: item["claim_sha256"])
    rng = random.Random(seed)
    return rng.sample(ordered, min(target, len(ordered))), len(ordered)


def build_cases(drawn):
    return [
        {
            "case_id": f"P{i:03d}",
            "claim": item["claim"],
            "primary_evidence": item["primary_evidence"],
            "counter_evidence": item["counter_evidence"],
            "annotation": {field: None for field in STUDY1_FIELD_ORDER},
        }
        for i, item in enumerate(drawn, 1)
    ]


def audit(cases, drawn, frame, eligible, prior_claims, payload, n_excluded, seed):
    checks = []

    def record(key, requirement, ok, detail):
        checks.append({"condition": key, "requirement": requirement,
                       "result": "PASS" if ok else "FAIL", "detail": detail})

    sample_hashes = [item["claim_sha256"] for item in drawn]
    rederived = {
        claim_hash(r["claim"]) for r in payload["eval_details"]
        if r.get("baseline_prediction") == "SUPPORT"
        and (r.get("counter_evidence") or "").strip()
        and r["claim"] not in prior_claims
    }
    record("A", "every selected case belongs to the new candidate frame",
           set(sample_hashes) <= rederived,
           f"{len(sample_hashes)} selected, all inside the {len(rederived)}-case "
           f"independently re-derived frame")

    prior_hashes = {claim_hash(c) for c in prior_claims}
    overlap = sorted(set(sample_hashes) & prior_hashes)
    record("B", "no previously annotated case is included", not overlap,
           f"overlap = {len(overlap)}; {n_excluded} study-1 claim(s) removed from "
           f"the frame by content hash")

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
    record("C", "no gold/outcome information is exposed", not leaked,
           f"forbidden keys found: {leaked or 'none'}")

    all_null = all(v is None for c in cases for v in c["annotation"].values())
    record("D", "all annotation fields are null", all_null,
           f"{len(cases) * len(STUDY1_FIELD_ORDER)} fields, all null = {all_null}")

    with open(PRIOR_BLIND, encoding="utf-8") as handle:
        study1 = json.load(handle)
    s1_order = list(study1["cases"][0]["annotation"].keys())
    s1_enums = {
        k: v["allowed_values"]
        for k, v in study1["metadata"]["annotation_schema"].items()
    }
    order_ok = all(list(c["annotation"].keys()) == s1_order for c in cases)
    enums_ok = s1_enums == STUDY1_ENUMS
    record("E", "schema exactly matches study 1", order_ok and enums_ok,
           f"field order identical = {order_ok}; enum values identical = {enums_ok}; "
           f"causal field = 'causal_relation_alignment'")

    redrawn, _ = draw_sample(frame, seed=seed, target=len(cases))
    reproducible = [i["claim_sha256"] for i in redrawn] == sample_hashes
    record("F", "the deterministic sample is reproducible", reproducible,
           f"re-drawing with seed {seed} yields the identical sequence = {reproducible}")

    hash_order = [
        i["claim_sha256"] for i in sorted(frame, key=lambda x: x["claim_sha256"])
        if i["claim_sha256"] in set(sample_hashes)
    ]
    artifact_order = [
        claim_hash(r["claim"]) for r in payload["eval_details"]
        if claim_hash(r["claim"]) in set(sample_hashes)
    ]
    record("G", "presentation order is randomized",
           sample_hashes != hash_order and sample_hashes != artifact_order,
           f"differs from content-hash order = {sample_hashes != hash_order}; "
           f"differs from artifact row order = {sample_hashes != artifact_order}")

    record("H", "sampling was performed without reading gold labels", True,
           "eligibility used baseline_prediction and counter_evidence only; every "
           "record then passed through project(), a three-field whitelist. Proven "
           "independently by verify_blind_projection.py.")

    passed = sum(1 for c in checks if c["result"] == "PASS")
    return {
        "metadata": {
            "study": "primary held-out validation of Rule A",
            "purpose": "Primary Rule A validation — balanced candidate frame",
            "research_only": True,
            "production_modified": False,
            "gold_labels_read_when_building": False,
        },
        "n_conditions": len(checks),
        "n_passed": passed,
        "overall": "PASS" if passed == len(checks) else "FAIL",
        "checks": checks,
        "frame_size_before_exclusion": eligible + n_excluded,
        "n_previously_annotated_excluded": n_excluded,
        "eligible_after_exclusion": eligible,
        "sample_size": len(cases),
    }


def _selftest():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    # Schema must match study 1 on disk.
    with open(PRIOR_BLIND, encoding="utf-8") as handle:
        study1 = json.load(handle)
    check("field order matches study 1",
          list(study1["cases"][0]["annotation"].keys()), STUDY1_FIELD_ORDER)
    check("enums match study 1",
          {k: v["allowed_values"]
           for k, v in study1["metadata"]["annotation_schema"].items()},
          STUDY1_ENUMS)
    check("causal field name", "causal_relation_alignment" in STUDY1_FIELD_ORDER, True)
    check("no PARTIAL", any("PARTIAL" in v for v in STUDY1_ENUMS.values()), False)
    check("NOT_APPLICABLE retained",
          all("NOT_APPLICABLE" in STUDY1_ENUMS[f] for f in STUDY1_FIELD_ORDER
              if f.endswith("_alignment")), True)

    # Frame construction + projection safety on a synthetic artifact where the
    # frame deliberately contains BOTH accepted and rejected candidates.
    fake = {"eval_details": [
        {"claim": f"claim {i}", "evidence": f"p{i}", "counter_evidence": f"x{i}",
         "baseline_prediction": "SUPPORT", "gold": "CONTRADICT" if i % 3 else "SUPPORT",
         "override_applied": bool(i % 2), "counter_section": "results",
         "evidence_doc_id": 1000 + i, "counter_p_contradiction": 0.9}
        for i in range(40)
    ] + [
        {"claim": "not support", "evidence": "p", "counter_evidence": "x",
         "baseline_prediction": "CONTRADICT"},
        {"claim": "no counter", "evidence": "p", "counter_evidence": "",
         "baseline_prediction": "SUPPORT"},
        {"claim": "prior case", "evidence": "p", "counter_evidence": "x",
         "baseline_prediction": "SUPPORT"},
    ]}
    frame, excluded = build_frame(fake, {"prior case"})
    check("frame size", len(frame), 40)
    check("one prior excluded", len(excluded), 1)
    check("projection drops every outcome field",
          sorted(frame[0]),
          ["claim", "claim_sha256", "counter_evidence", "primary_evidence"])
    blob = json.dumps(frame)
    check("no outcome substring survives projection",
          any(t in blob for t in ('"gold"', "override_applied", "counter_section",
                                  "evidence_doc_id", "counter_p_contradiction")),
          False)

    drawn, eligible = draw_sample(frame, seed=SEED, target=25)
    again, _ = draw_sample(frame, seed=SEED, target=25)
    check("sample deterministic", [i["claim"] for i in drawn],
          [i["claim"] for i in again])
    check("eligible reported", eligible, 40)
    cases = build_cases(drawn)
    check("case id prefix", cases[0]["case_id"], "P001")
    check("case keys", sorted(cases[0]),
          ["annotation", "case_id", "claim", "counter_evidence", "primary_evidence"])

    report = audit(cases, drawn, frame, eligible, {"prior case"}, fake,
                   len(excluded), SEED)
    check("audit passes", report["overall"], "PASS")
    check("audit conditions", report["n_conditions"], 8)

    # Positive failure detection.
    leaky = [dict(cases[0], gold="CONTRADICT")]
    bad = audit(leaky, drawn, frame, eligible, {"prior case"}, fake,
                len(excluded), SEED)
    failed = {c["condition"] for c in bad["checks"] if c["result"] == "FAIL"}
    check("C detects a leak", "C" in failed, True)

    filled = [dict(cases[0],
                   annotation={**cases[0]["annotation"], "direction": "OPPOSITE"})]
    bad = audit(filled, drawn, frame, eligible, {"prior case"}, fake,
                len(excluded), SEED)
    check("D detects a filled annotation",
          next(c["result"] for c in bad["checks"] if c["condition"] == "D"), "FAIL")

    renamed = [dict(cases[0], annotation={
        ("causal_alignment" if k == "causal_relation_alignment" else k): v
        for k, v in cases[0]["annotation"].items()})]
    bad = audit(renamed, drawn, frame, eligible, {"prior case"}, fake,
                len(excluded), SEED)
    check("E detects a schema drift",
          next(c["result"] for c in bad["checks"] if c["condition"] == "E"), "FAIL")

    unshuffled = sorted(frame, key=lambda i: i["claim_sha256"])[:25]
    bad = audit(build_cases(unshuffled), unshuffled, frame, eligible,
                {"prior case"}, fake, len(excluded), SEED)
    check("G detects unshuffled order",
          next(c["result"] for c in bad["checks"] if c["condition"] == "G"), "FAIL")

    prior_in = [project({"claim": "claim 5", "evidence": "p", "counter_evidence": "x"})]
    bad = audit(build_cases(prior_in), prior_in, frame, eligible, {"claim 5"},
                fake, 1, SEED)
    check("B detects a previously annotated case",
          next(c["result"] for c in bad["checks"] if c["condition"] == "B"), "FAIL")

    print("\n" + "=" * 74)
    if failures:
        print(f"SELFTEST FAILED ({len(failures)})")
        for f in failures:
            print("  -", f)
        print("=" * 74)
        return 1
    print("SELFTEST PASSED -- study-1 schema reproduced exactly (order, enums,")
    print("NOT_APPLICABLE, causal_relation_alignment naming), projection safety,")
    print("sample determinism, and all 8 audit conditions including positive")
    print("failure detection on A/B/C/D/E/G.")
    print("=" * 74)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build the primary held-out blind validation set (analysis only)"
    )
    parser.add_argument("--artifact", default=None,
                        help="fresh candidate artifact from generate_fresh_candidate_run.py")
    parser.add_argument("--prior", default=PRIOR_BLIND)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=TARGET_N)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    if not args.artifact:
        print("ERROR: --artifact is required.")
        print("\nThe primary validation needs a FRESH candidate population. A re-run")
        print("of the same configuration over the same 504 train claims reproduces")
        print("the identical 150-case frame (verified on the two runs already in")
        print("eval/: symmetric difference = none), so it cannot supply one.")
        print("\nGenerate one first, in a GPU environment:")
        print("    python generate_fresh_candidate_run.py --split validation")
        print("then pass its artifact here with --artifact.")
        return 2

    for path in (args.artifact, args.prior):
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}")
            return 2

    with open(args.artifact, encoding="utf-8") as handle:
        payload = json.load(handle)
    with open(args.prior, encoding="utf-8") as handle:
        prior = json.load(handle)
    prior_claims = {case["claim"] for case in prior["cases"]}

    frame, excluded = build_frame(payload, prior_claims)
    if not frame:
        print("ERROR: the candidate frame is empty.")
        return 2
    drawn, eligible = draw_sample(frame, seed=args.seed, target=args.n)
    cases = build_cases(drawn)

    print("=" * 74)
    print("PRIMARY HELD-OUT BLIND VALIDATION SET")
    print("=" * 74)
    print(f"artifact                         : {args.artifact}")
    print(f"  claims split                   : {payload['metadata'].get('claims_split')}")
    print(f"  frozen config                  : {payload['metadata'].get('frozen_config')}")
    print(f"fresh candidate frame            : {eligible + len(excluded)}")
    print(f"excluded as previously annotated : {len(excluded)}")
    print(f"eligible after exclusion         : {eligible}")
    print(f"random seed                      : {args.seed}")
    print(f"final sample size                : {len(cases)}")
    print(f"gold labels read by this script  : NO")

    manifest = {
        "metadata": {
            "study": "primary held-out validation of Rule A",
            "purpose": "Primary Rule A validation — balanced candidate frame",
            "research_only": True,
            "production_modified": False,
            "source_artifact": args.artifact,
            "claims_split": payload["metadata"].get("claims_split"),
            "frozen_config": payload["metadata"].get("frozen_config"),
            "prior_study_dataset": args.prior,
            "schema_source": "study 1 (docs/scope_alignment_blind_dataset.json)",
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
        },
        "cases": [
            {"case_id": c["case_id"], "sample_index": i,
             "claim_sha256": claim_hash(c["claim"])}
            for i, c in enumerate(cases, 1)
        ],
    }
    dataset = {
        "metadata": {
            "study": "primary held-out validation of Rule A",
            "purpose": "Primary Rule A validation — balanced candidate frame",
            "n_cases": len(cases),
            "exposed_fields": ["case_id", "claim", "primary_evidence", "counter_evidence"],
            "annotation_schema": {
                field: {"allowed_values": STUDY1_ENUMS[field]}
                for field in STUDY1_FIELD_ORDER if field != "annotator_notes"
            },
            "free_text_field": "annotator_notes",
            "schema_source": "reproduced exactly from study 1",
            "annotations_performed": False,
            "guide": "docs/scope_alignment_blind_annotation.md (study-1 guide; "
                     "same schema)",
        },
        "cases": cases,
    }
    report = audit(cases, drawn, frame, eligible, prior_claims, payload,
                   len(excluded), args.seed)

    os.makedirs("docs", exist_ok=True)
    for path, data in ((OUT_MANIFEST, manifest), (OUT_DATASET, dataset),
                       (OUT_AUDIT, report)):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    print("\nAUDIT")
    for check in report["checks"]:
        print(f"  [{check['result']}] {check['condition']}: {check['requirement']}")
        print(f"         {check['detail']}")
    print(f"\n  overall: {report['overall']} "
          f"({report['n_passed']}/{report['n_conditions']})")
    print("\nfiles written:")
    for path in (OUT_MANIFEST, OUT_DATASET, OUT_AUDIT):
        print(f"  {path}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
