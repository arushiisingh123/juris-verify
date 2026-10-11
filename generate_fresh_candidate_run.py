"""
Generate a FRESH candidate run for the primary Rule A validation.

Research script. Imports production code and calls it with the frozen
configuration; modifies nothing. No file under src/ or eval/ is touched.

=============================================================================
WHY A RE-RUN ON THE SAME CLAIMS CANNOT WORK
=============================================================================
`src/data_loader.get_valid_claims` reads `claims_dataset["train"]` only, so the
504 unique labelled claims are the ENTIRE population the harness has ever
scored. Within that population the candidate frame

    baseline_prediction == "SUPPORT" AND counter_evidence non-empty

is 150 claims, of which 42 are accepted (overridden) and 108 rejected. The 42
accepted are exactly the 42 already annotated in study 1.

And the frame is stable across runs. Measured on the two independent runs
already in `eval/`:

    2026-10-09 default-mode (g1-g4) : frame = 150
    2026-10-11 G4-off      (g1-g3)  : frame = 150
    symmetric difference            : none

So a third run of the same configuration over the same claims reproduces the
same 150 candidates, hence the same 42/108 split, hence -- after excluding the
previously annotated 42 -- the same all-rejected residue. **A fresh run of the
same split cannot deliver a fresh candidate population.** The constraint is
structural, not run-to-run noise.

=============================================================================
WHAT THIS SCRIPT DOES INSTEAD
=============================================================================
It scores a claim population the pipeline has never seen: the SciFact
**validation** split, against the same document corpus, with the identical
frozen configuration

    evidence_mode=section  max_sentences=2  section_weight=0.05
    prosecution_mode=active  adjudication_gates=g1,g2,g3

Only the claim population differs, so the candidate frame it yields contains
BOTH accepted and rejected candidates and -- by construction, since the splits
are disjoint -- no previously annotated case.

The split is selected here, in this research script. `src/data_loader.py` is
NOT modified; `load_scifact()` and `build_document_corpus()` are called exactly
as they are.

=============================================================================
REQUIREMENTS
=============================================================================
Same environment as `eval/evaluate.py`: a GPU (NLIVerifier defaults to
device=0), the DeBERTa verifier, the CrossEncoder reranker and the SciFact
dataset. This script cannot run on a CPU-only machine.

    python generate_fresh_candidate_run.py --split validation

Writes eval/candidates_<split>_<timestamp>_section2_w0.05_prosactive_gatesg1g2g3.json
in the same shape `eval/evaluate.py` produces, so the sampler can consume it
unchanged. Gold labels ARE recorded in the artifact (they are needed for
scoring later) but the sampler never reads them -- see
inspect_scope_alignment_primary_heldout.py and verify_blind_projection.py.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# The frozen configuration. Do not vary these: the validation must test the
# same system study 1 characterised.
FROZEN_CONFIG = {
    "evidence_mode": "section",
    "max_sentences": 2,
    "section_weight": 0.05,
    "prosecution_mode": "active",
    "adjudication_gates": "g1,g2,g3",
}

# Fields the sampler needs. Everything else the pipeline returns is recorded
# too, for later scoring, but the sampler reads only the first three.
SAMPLER_FIELDS = ("claim", "evidence", "counter_evidence", "baseline_prediction")


def main():
    parser = argparse.ArgumentParser(
        description="Score a fresh SciFact split with the frozen configuration"
    )
    parser.add_argument(
        "--split", default="validation",
        help="SciFact claims split to score (default: validation). 'train' would "
             "reproduce the existing 504-claim population and is not useful here.",
    )
    parser.add_argument("--n_claims", type=str, default="all")
    parser.add_argument("--output_dir", default="eval")
    parser.add_argument("--dry_run", action="store_true",
                        help="report the claim population and exit without "
                             "loading any model")
    args = parser.parse_args()

    from src.data_loader import build_document_corpus, get_unique_claims, load_scifact

    print("=" * 74)
    print("FRESH CANDIDATE RUN")
    print("=" * 74)
    print(f"frozen config : {FROZEN_CONFIG}")
    print(f"claims split  : {args.split}")

    claims_dataset, corpus_dataset = load_scifact()
    available = list(claims_dataset.keys())
    print(f"splits available in allenai/scifact claims: {available}")
    if args.split not in claims_dataset:
        print(f"ERROR: split {args.split!r} not present; available: {available}")
        return 2

    raw = list(claims_dataset[args.split])
    labelled = [item for item in raw if item.get("evidence_label")]
    unique = get_unique_claims(labelled)
    print(f"  {args.split}: {len(raw)} rows, {len(labelled)} with an evidence_label, "
          f"{len(unique)} unique claims")

    if not unique:
        print(f"ERROR: split {args.split!r} has no labelled claims. The SciFact test "
              f"split withholds labels; use 'validation'.")
        return 2

    if args.dry_run:
        print("\n--dry_run: no model loaded, nothing scored.")
        print("Re-run without --dry_run in a GPU environment to produce the artifact.")
        return 0

    documents = build_document_corpus(corpus_dataset)
    print(f"  corpus: {len(documents)} documents")

    from src.pipeline import JurisVerifyPipeline
    from src.verification import normalize_for_scifact_eval

    pipeline = JurisVerifyPipeline(
        documents,
        evidence_mode=FROZEN_CONFIG["evidence_mode"],
        max_sentences=FROZEN_CONFIG["max_sentences"],
        section_weight=FROZEN_CONFIG["section_weight"],
        prosecution_mode=FROZEN_CONFIG["prosecution_mode"],
        adjudication_gates=FROZEN_CONFIG["adjudication_gates"],
    )

    subset = unique if args.n_claims.lower() in ("all", "none") else unique[: int(args.n_claims)]
    print(f"\nscoring {len(subset)} claims...")
    start = time.time()

    details = []
    for index, item in enumerate(subset, 1):
        result = pipeline.predict_claim(item["claim"])
        record = {
            "claim_index": index,
            "claim": item["claim"],
            "gold": item["evidence_label"],
            "prediction": normalize_for_scifact_eval(result["verdict"]),
        }
        record.update(result)
        if "baseline_verdict" in result:
            record["baseline_prediction"] = normalize_for_scifact_eval(
                result["baseline_verdict"]
            )
        else:
            record["baseline_prediction"] = record["prediction"]
        details.append(record)
        if index % 25 == 0 or index == len(subset):
            print(f"  {index}/{len(subset)}  ({time.time() - start:.1f}s)")

    frame = [
        r for r in details
        if r.get("baseline_prediction") == "SUPPORT"
        and (r.get("counter_evidence") or "").strip()
    ]
    print(f"\ncandidate frame (baseline SUPPORT + counter-evidence): {len(frame)}")
    print("  gold composition deliberately NOT reported.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(
        args.output_dir,
        f"candidates_{args.split}_{stamp}_n{len(subset)}_section2_w0.05"
        f"_prosactive_gatesg1g2g3.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "purpose": "fresh candidate population for primary Rule A validation",
                    "claims_split": args.split,
                    "n_claims": len(subset),
                    "frozen_config": FROZEN_CONFIG,
                    "candidate_frame_size": len(frame),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "runtime_seconds": round(time.time() - start, 2),
                    "production_modified": False,
                    "note": (
                        "Split selection happens in this research script; "
                        "src/data_loader.py is unmodified and still reads only the "
                        "train split for the standard harness."
                    ),
                },
                "eval_details": details,
            },
            handle, indent=2, ensure_ascii=False,
        )
    print(f"\nwrote {out}")
    print("\nNext: python inspect_scope_alignment_primary_heldout.py "
          f"--artifact {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
