# G5 validation — reproducibility

How to re-derive every figure in `g5_primary_heldout_validation_report.md` and
`g5_primary_heldout_validation_tables.md`.

---

## 1. Blocking prerequisite

**Five artifacts are referenced by the report but are not in this repository.**
Until they are present, the confusion matrices cannot be verified against source
data and the report's `AS SUPPLIED` figures remain unverified.

| artifact | expected location |
|---|---|
| completed annotations | `scope_alignment_primary_heldout_blind_annotations.json` (repo root) |
| candidate run | `candidates_validation_20261011_174506_n188_section2_w0.05_prosactive_gatesg1g2g3.json` (repo root) |
| blind dataset | `docs/scope_alignment_primary_heldout_blind_dataset.json` |
| manifest | `docs/scope_alignment_primary_heldout_manifest.json` |
| sampling audit | `docs/scope_alignment_primary_heldout_audit.json` |

Place them, then run the command in §3. The verifier switches from AUDIT mode to
ARTIFACT mode and prints MATCH / MISMATCH per confusion matrix.

## 2. Environment

| component | value |
|---|---|
| embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| verifier | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| dataset | `allenai/scifact` — `claims` (validation split), `corpus` |
| GPU | required (`NLIVerifier` defaults to `device=0`) |

`scipy` is used only for Clopper–Pearson intervals and as a cross-check on the
native Fisher implementation. Every statistic the report needs is computable
without it; `requirements.txt` is unchanged.

## 3. Commands

```bash
# A. Verify / recompute every statistic in the report.
#    AUDIT mode if the artifacts are absent; ARTIFACT mode if present.
python verify_g5_validation_numbers.py
python verify_g5_validation_numbers.py --emit-summary   # rewrites the summary JSON

# B. Re-prove the blinding of the candidate-frame sampler.
python verify_blind_projection.py --module inspect_scope_alignment_primary_heldout
python verify_blind_projection.py --module inspect_scope_alignment_heldout

# C. Confirm the G5 / Rule A definition is still the frozen one.
python inspect_proposition_scope_alignment.py --selftest

# D. Re-derive the candidate frame from a fresh run (GPU required).
python generate_fresh_candidate_run.py --split validation
python inspect_scope_alignment_primary_heldout.py --artifact eval/candidates_validation_<ts>_...json

# E. Sampler self-test (schema fidelity + audit conditions, no data needed).
python inspect_scope_alignment_primary_heldout.py --selftest
```

## 4. Frozen definitions

```
G5 / Rule A     ==  overall_scope_match == YES  AND  direction == OPPOSITE
direction-only  ==  direction == OPPOSITE
```

Both live in `inspect_proposition_scope_alignment.py`, frozen before any
annotation existed and fingerprinted:

```
sha256(rule source) = 0449d3d00fc7cf0a65cdc65ad0b1b080bbcb87b9f7986b7df845b308caa50d77
```

Check it has not drifted:

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from inspect_proposition_scope_alignment import rule_fingerprint; print(rule_fingerprint())"
```

Neither rule contains a threshold or tunable parameter; every condition is an
enum equality, asserted by a structural test that fails if any numeric constant
appears in a comparison.

## 5. Seeds and sampling

| parameter | value |
|---|---|
| annotation / sampling seed (primary) | **20261012** |
| sampling seed (rejected-candidate stress test) | 20261011 |
| ordering | `sorted(frame, key=sha256(claim))`, then `random.Random(seed).sample(frame, n)` |

Content-hash ordering rather than a seeded shuffle of the source order, so the
`case_id → claim` mapping is recomputable from the artifact alone and no key file
needs to exist.

## 6. Pipeline configuration (frozen across all arms)

```
evidence_mode      = section
max_sentences      = 2
section_weight     = 0.05
prosecution_mode   = active
adjudication_gates = g1,g2,g3
```

G5 is **not** among `adjudication_gates`. It is evaluated offline against
annotations and is not part of the pipeline.

## 7. Statistical methods

| quantity | method | implementation |
|---|---|---|
| precision / recall / specificity | direct from the confusion matrix | `verify_g5_validation_numbers.py` |
| 95% CI (primary) | Wilson score interval | native |
| 95% CI (exact) | Clopper–Pearson | `scipy.stats.beta` |
| association test | Fisher exact, two-sided | native via `math.comb`, cross-checked against `scipy.stats.fisher_exact` |
| paired comparison | McNemar, exact two-sided binomial | native |

Fisher's two-sided p sums every table with the observed margins whose
probability is ≤ the observed table's (the standard convention), with a 1e-9
relative tolerance so float noise cannot drop the observed table.

**Two framings of McNemar are reported, deliberately.** G5 is a strict subset of
direction-only, so the accept-decision test is structurally determined
(p = 0.0078125) and carries no evidence about which rule is better. The
decision-relevant test is on per-case correctness (p = 1.0). Report both or
report only the latter; reporting only the former would be misleading.

## 8. Simulated application — exactly how it is computed

```
after_correct = baseline_correct + (TP - FP)
```

with `population = 188`, TP/FP from the rule's confusion matrix on the 54-case
candidate frame, and

```
baseline_correct = count(baseline_prediction == gold) = 154      # pre-override
```

**derived from the candidate artifact, not hardcoded.** `verify_g5_validation_numbers.py`
computes it in `derive_baseline()` precisely so the earlier error cannot recur.

> ⚠ An earlier revision used `baseline_correct = 158`. That was wrong: 158/188
> is the frozen G1+G2+G3 run's **post**-override accuracy (`prediction == gold`),
> which already contains that run's 12 overrides, so using it as the baseline
> double-counted them. Correct: **154/188 = 81.91% → 161/188 = 85.64%**, delta
> unchanged at +3.72 pp. See the report §0.0.

Valid because the frame is **exhaustive** over the population either rule can
affect: a rule can only change a claim that is baseline-SUPPORT **and** carries
counter-evidence, and all 54 such claims in the validation population were
annotated. No claim outside the frame can change, so no extrapolation occurs.

It is nonetheless a **simulated application, not an independently rerun
end-to-end production evaluation**, and must be described that way.

## 9. Files

### Research documentation (this package)

```
docs/g5_primary_heldout_validation_report.md
docs/g5_primary_heldout_validation_tables.md
docs/g5_primary_heldout_validation_summary.json
docs/g5_reproducibility.md
```

### Analysis scripts

```
verify_g5_validation_numbers.py              statistics: recompute / audit
verify_blind_projection.py                   blinding proof (3 methods)
inspect_proposition_scope_alignment.py       frozen rules + fingerprint
inspect_scope_alignment_primary_heldout.py   primary candidate-frame sampler
inspect_scope_alignment_heldout.py           stress-test sampler
generate_fresh_candidate_run.py              fresh candidate run (GPU)
inspect_gate_subset_ablation.py              D1 gate ablation
inspect_arm_f_overrides.py                   arm F override extraction
inspect_oracle_ceiling.py                    arm H oracle ceiling
```

### Prior analyses the report cites

```
docs/scope_alignment_failure_analysis.md     hypothesis generation (outcome-aware)
docs/scope_alignment_reconciliation.md       retrieval-nondeterminism measurement
docs/arm_f_gate_subset_ablation.md           D1: G3 necessary, G4 net-harmful
docs/scope_alignment_primary_heldout_plan.md why the validation split was needed
```

### Production source — unmodified

```
src/prosecution.py  src/pipeline.py  src/extraction.py  src/retrieval.py
src/verification.py  src/sections.py  src/orchestration.py  src/data_loader.py
src/matrix.py  eval/evaluate.py  requirements.txt  and every test_*.py
```

Confirm:

```bash
git diff --stat HEAD -- src/ eval/ test_*.py requirements.txt   # expect empty
```

## 10. Known reproducibility limits

1. **Per-claim retrieval instability ~0.4%.** Two independent runs over the same
   504 train claims differed on 2 claims' retrieved documents, which changed 1 of
   42 override decisions. Aggregate figures were unaffected (identical baselines),
   but a candidate frame is not guaranteed identical across runs. Measured in
   `scope_alignment_reconciliation.md`. Seeding the Chroma HNSW index, or exact
   search over the 5,183 documents, would remove it — a production change, not
   made.
2. **Single annotator.** No inter-annotator agreement; the
   `NOT_APPLICABLE` / `DIFFERENT` boundary is unmeasured.
3. **The five artifacts are absent** (§1), so the report's confusion matrices are
   unverified against source data.
