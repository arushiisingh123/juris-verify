# Primary Rule A validation — preparation and the blocking constraint

Research/analysis only. No production module was modified. Rule A is unchanged
and remains frozen in `inspect_proposition_scope_alignment.py`
(`sha256 = 0449d3d0…`). Nothing was annotated, and no gold composition is
reported anywhere in this document.

---

## 1. Status

| step | state |
|---|---|
| stress-test artifacts relabelled and preserved | **done** |
| smallest reproducible way to get a fresh candidate run | **determined** |
| fresh candidate run executed | **BLOCKED — needs GPU + models + dataset** |
| primary blind dataset / manifest / audit | **blocked on the run above** |
| projection-safety verification script | **done, and passing** |

---

## 2. A re-run of the same configuration cannot supply a fresh frame

This is the finding that changes the plan, and it is measured rather than
argued.

`src/data_loader.get_valid_claims` reads `claims_dataset["train"]` only, so the
504 unique labelled claims are the **entire population** the harness has ever
scored. Inside that population:

| | count |
|---|---|
| candidate frame (`baseline_prediction == "SUPPORT"` AND counter-evidence non-empty) | **150** |
| accepted (overridden by G1+G2+G3) | **42** — exactly the study-1 annotated set |
| rejected | **108** |

And the frame is **stable across independent runs**. Measured on the two runs
already in `eval/`:

```
2026-10-09 default-mode (g1–g4) : frame = 150
2026-10-11 G4-off      (g1–g3)  : frame = 150
symmetric difference            : none
```

A third run of the same configuration over the same claims therefore reproduces
the same 150 candidates, the same 42/108 split, and — after excluding the
previously annotated 42 — the same all-rejected residue. **The limitation is
structural, not run-to-run noise**, so "generate a new independent run" does not
by itself fix it. (The ~0.4% per-claim retrieval instability documented in
`scope_alignment_reconciliation.md` perturbs individual case fields but left the
frame membership identical.)

## 3. The smallest change that does supply one

Score a **claim population the pipeline has never seen**, holding everything
else fixed: the SciFact **validation** split, against the same document corpus,
with the identical frozen configuration.

```
evidence_mode=section  max_sentences=2  section_weight=0.05
prosecution_mode=active  adjudication_gates=g1,g2,g3
```

Only the claim population differs. Consequences:

- the frame contains **both accepted and rejected** candidates, so it is a
  balanced validation frame rather than a rejection-set stress test;
- **no overlap with study 1 is possible** — the splits are disjoint — so
  exclusion becomes a belt-and-braces check rather than the thing that
  unbalances the sample;
- the system under test is unchanged, so study-1 findings remain comparable.

`generate_fresh_candidate_run.py` does this. It **imports** production code and
calls it; it modifies nothing. Split selection happens inside the research
script, so `src/data_loader.py` keeps reading only `train` for the standard
harness.

### Why this is blocked here

| requirement | status in this environment |
|---|---|
| GPU (`NLIVerifier` defaults to `device=0`) | missing — `torch 2.14.1+cpu` |
| DeBERTa verifier | not cached |
| CrossEncoder reranker | not cached |
| SciFact dataset | not cached (and the dataset download fails on an SSL cert error) |

The `--dry_run` path reaches `load_scifact()` and fails only on the dataset
download, confirming the blocker is environmental rather than a code fault.

### Command

```bash
# 1. in a GPU environment (same as eval/evaluate.py)
python generate_fresh_candidate_run.py --split validation

# 2. then, anywhere
python inspect_scope_alignment_primary_heldout.py \
    --artifact eval/candidates_validation_<ts>_n<N>_section2_w0.05_prosactive_gatesg1g2g3.json
```

Step 2 writes the manifest, blind dataset and audit, with seed **20261012**.

### One unknown to check when it runs

The yield of the validation split is not verifiable without the dataset. The
script reports `rows / with an evidence_label / unique claims` before scoring and
refuses if the split has no labelled claims (the SciFact **test** split withholds
labels, so `validation` is the right one). If the resulting frame is smaller than
100, `inspect_scope_alignment_primary_heldout.py` takes every eligible case and
reports the exact count — it does not pad, and it does not fall back to the
train split.

## 4. Fallback, if the validation frame proves too small

Stated now so the choice is pre-registered rather than made after seeing a
disappointing count:

> Sample 100 from the **full 150-case train frame without excluding the 42**,
> and record per case whether it was in study 1. That yields a balanced
> accepted/rejected mix at the cost of ~28 previously annotated cases, which
> must then be reported separately and excluded from the headline figure.

This is strictly worse than a fresh split and should only be used if
`validation` cannot supply a usable frame.

## 5. The stress-test artifacts are preserved and relabelled

Not deleted, not annotated, case payloads byte-identical (hash-verified before
and after). A `purpose` label was added to the metadata of each:

> **Rejected-Candidate Stress Test — not primary Rule A validation**

with a note recording why: after excluding the 42 previously annotated cases
(which are exactly the 42 accepted candidates), every remaining candidate is one
the adjudicator rejected.

Files: `scope_alignment_heldout_manifest.json`,
`scope_alignment_heldout_blind_dataset.json`,
`scope_alignment_heldout_audit.json` (100 cases, all 1,000 annotation values
still `null`).

It remains a legitimate secondary experiment — "does Rule A correctly decline
the candidates the gates already declined?" — just not the primary validation.

## 6. Annotation schema: study 1, reproduced exactly

Read from `docs/scope_alignment_blind_dataset.json` and asserted field-by-field
in the sampler's self-test:

| field | allowed values |
|---|---|
| `population_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `intervention_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `outcome_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `endpoint_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `context_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| **`causal_relation_alignment`** | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `direction` | OPPOSITE / SAME / NOT_APPLICABLE / UNCLEAR |
| `overall_scope_match` | YES / NO / UNCLEAR |
| `genuine_contradiction` | YES / NO / UNCLEAR |
| `annotator_notes` | free text |

Field **order** is preserved as well, and audit condition E fails if either the
order or the enums drift — verified by a positive test that renaming
`causal_relation_alignment` to `causal_alignment` makes E fail.

Note this reverts the schema used for the stress test, which had introduced
`PARTIAL`, dropped `NOT_APPLICABLE`, and renamed the causal field. Using study
1's schema is what keeps the two studies comparable, and is why Rule B's meaning
is stable across them.

Annotators use the existing study-1 guide,
`docs/scope_alignment_blind_annotation.md` — same schema, same instructions.

## 7. Blinding guarantees

`verify_blind_projection.py` proves, independently of the sampler's own claims,
that the blind projection cannot reach an outcome field. Three methods, all
passing on both samplers:

1. **Static (AST).** `project()` reads only `claim`, `evidence`,
   `counter_evidence`; no forbidden key is subscripted anywhere in
   `build_frame` / `draw_sample` / `build_cases` / `project`; `build_frame`
   routes every record through `project()`; `build_cases` reads only projected
   keys.
2. **Dynamic.** A record stuffed with every outcome field, each carrying a unique
   sentinel, produces output containing no sentinel — through `project()` alone
   and through the full path to a blind case.
3. **Instrumented.** A `dict` subclass that raises on any read of an outcome key
   survives the entire path untouched; the keys actually read are
   `{baseline_prediction, claim, counter_evidence, evidence}`. A positive control
   confirms the tripwire fires on a deliberate read.

`baseline_prediction` is read for eligibility only, by `build_frame`. It is a
model prediction, not a correctness signal, and never reaches the output.

## 8. What the annotator receives

Only the primary blind dataset and
`docs/scope_alignment_blind_annotation.md`. **Not** the evaluation artifact, the
42-case results, the stress-test results, the failure analysis, the
reconciliation, or any gold label. The study-1 guide already carries the
unblinding warning naming the files that must not be read first.
