# G5 primary held-out validation — paper-ready tables

Transfer-ready. Every statistic here is **computed** by
`verify_g5_validation_numbers.py`, which now runs in **ARTIFACT MODE**: both
confusion matrices and the baseline are recomputed from the recovered source
artifacts. Only the 450 validation rows and 5,183 corpus documents remain **as
supplied**. See `g5_primary_heldout_validation_report.md` §0.

> ⚠ **Superseded figures.** An earlier revision gave the baseline as 158/188
> (84.04%) and the G5 result as 165/188 (87.77%). Both were wrong — 158/188 is
> the frozen G1+G2+G3 run's *post*-override accuracy, so using it as a baseline
> double-counted that run's 12 overrides. Correct: **81.91% (154/188) → 85.64%
> (161/188)**; the delta is unchanged at +3.72 pp. See report §0.0.

Notation: TP = accepted and gold CONTRADICT · FP = accepted and gold SUPPORT ·
FN = rejected and gold CONTRADICT · TN = rejected and gold SUPPORT. CIs are
Wilson 95%; Clopper–Pearson 95% given where the two differ materially.

---

### Table 1. Study design

| | Discovery study | **Primary held-out validation** |
|---|---|---|
| SciFact split | train | **validation** |
| unique labelled claims | 504 | 188 |
| candidate frame (n) | 42 | **54** |
| gold CONTRADICT / SUPPORT | 24 / 18 | **17 / 37** |
| candidate selection | outcome-selected (the adjudicator's overrides) | **outcome-blind** (`baseline_prediction == SUPPORT` ∧ counter-evidence non-empty) |
| annotator knew outcomes | yes | **no** |
| rule frozen before annotation | no | **yes (fingerprinted)** |
| overlap between studies | — | **none** (disjoint splits) |
| evidential status | discovery / hypothesis-generating | **primary independent validation** |

---

### Table 2. G5 confusion matrix — primary held-out (n = 54)

| | G5 accepts | G5 rejects | total |
|---|---|---|---|
| gold CONTRADICT | **7** (TP) | 10 (FN) | 17 |
| gold SUPPORT | **0** (FP) | 37 (TN) | 37 |
| total | 7 | 47 | 54 |

---

### Table 3. direction-only confusion matrix — primary held-out (n = 54)

| | accepts | rejects | total |
|---|---|---|---|
| gold CONTRADICT | 11 (TP) | 6 (FN) | 17 |
| gold SUPPORT | **4** (FP) | 33 (TN) | 37 |
| total | 15 | 39 | 54 |

---

### Table 4. Primary held-out performance

| metric | **G5** | direction-only |
|---|---|---|
| rule | `overall_scope_match == YES ∧ direction == OPPOSITE` | `direction == OPPOSITE` |
| accepted | 7 | 15 |
| Precision | **100.00%** (7/7) | 73.33% (11/15) |
| Precision 95% CI (Wilson) | [64.57, 100.00] | [48.05, 89.10] |
| Precision 95% CI (Clopper–Pearson) | [59.04, 100.00] | [44.90, 92.21] |
| Recall | 41.18% (7/17) | **64.71%** (11/17) |
| Recall 95% CI (Wilson) | [21.61, 63.99] | [41.30, 82.69] |
| Specificity | **100.00%** (37/37) | 89.19% (33/37) |
| Candidate-frame accuracy | 81.48% (44/54) | 81.48% (44/54) |
| Rescues | 7 | 11 |
| **Added errors** | **0** | **4** |
| Net claims | +7 | +7 |
| Fisher exact (two-sided) | **1.098 × 10⁻⁴** | 1.002 × 10⁻⁴ |

---

### Table 5. Paired comparison — G5 vs direction-only

| framing | discordant pairs | split | two-sided exact p | interpretation |
|---|---|---|---|---|
| accept decision | 8 | 0 / 8 | 0.0078125 | **structurally determined** — G5 is a strict subset of direction-only; carries no evidence of superiority |
| **per-case correctness** | 8 | 4 / 4 | **1.0** | **no detectable accuracy difference** |

The demonstrated difference is the error profile, not accuracy: G5 adds 0 false
contradictions where direction-only adds 4, at the cost of 4 true detections.

---

### Table 6. Simulated application to the 188-claim validation population

| rule | baseline correct | after | baseline acc. | after acc. | Δ (pp) | rescues | added errors | net |
|---|---|---|---|---|---|---|---|---|
| **G5** | 154/188 | **161/188** | 81.91% | **85.64%** | **+3.72** | 7 | **0** | +7 |
| direction-only | 154/188 | 161/188 | 81.91% | 85.64% | +3.72 | 11 | 4 | +7 |
| *frozen G1+G2+G3 run (reference)* | *154/188* | *158/188* | *81.91%* | *84.04%* | *+2.13* | *8* | *4* | *+4* |

Baseline is the **pre-override** accuracy (`baseline_prediction` in the candidate
artifact), derived from source. The third row is the frozen run's own
post-override result, for comparison only — it is not the baseline.

**Simulated, not an independently rerun end-to-end evaluation.** Computed by
applying each rule's decisions on the exhaustively annotated 54-case candidate
frame to the 188-claim baseline. Valid because the frame is exhaustive over the
population either rule can affect — only a baseline-SUPPORT claim with
counter-evidence can change, and all 54 such claims were annotated — so no
extrapolation is involved. Both rules reach the same figure.

---

### Table 7. Discovery study (n = 42) — context only

| | accepts | rejects | total |
|---|---|---|---|
| gold CONTRADICT | 20 (TP) | 4 (FN) | 24 |
| gold SUPPORT | 0 (FP) | 18 (TN) | 18 |
| total | 20 | 22 | 42 |

| metric | value |
|---|---|
| Precision | 100.00% (20/20), Wilson 95% CI [83.89, 100.00] |
| Recall | 83.33% (20/24), Wilson 95% CI [64.15, 93.32] |
| Fisher exact (two-sided) | 2.12 × 10⁻⁸ |

**Outcome-selected with outcome-aware labelling — not valid as validation.**

---

### Table 8. Annotation schema (study 1, reused unchanged)

| field | allowed values | consumed by G5 |
|---|---|---|
| `population_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| `intervention_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| `outcome_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| `endpoint_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| `context_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| `causal_relation_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR | no |
| **`direction`** | OPPOSITE / SAME / NOT_APPLICABLE / UNCLEAR | **yes** |
| **`overall_scope_match`** | YES / NO / UNCLEAR | **yes** |
| `genuine_contradiction` | YES / NO / UNCLEAR | no — independent check |

All 7 G5-accepted cases were annotated `genuine_contradiction = YES`. Since G5
does not consume that field, this is convergent rather than circular.

---

### Table 9. Leakage and overlap controls

| control | mechanism | verification |
|---|---|---|
| no gold at selection | whitelist projection to `claim` / `primary_evidence` / `counter_evidence` | static AST + poisoned-record + tripwire, all passing |
| no outcome metadata exposed | doc ids, sections, all probabilities withheld (section and provenance correlate with outcome) | audit condition C |
| no discovery-study overlap | disjoint splits; content-hash exclusion as a redundant check | audit condition B |
| blind at annotation time | all annotation fields initially `null` | audit condition D |
| schema stability | study-1 field names, enums and order asserted | audit condition E |
| sample reproducible | `random.Random(20261012).sample(...)` over sha256(claim)-sorted frame | audit condition F |
| order randomized | differs from content-hash and artifact row order | audit condition G |

---

### Table 10. Error profile — G5's 10 false negatives

| subset | n | `direction` | reachable by a scope refinement? |
|---|---|---|---|
| rejected by G5, accepted by direction-only | 4 | OPPOSITE | yes — fail only on `overall_scope_match` |
| rejected by both | 6 | not OPPOSITE | **no** — outside G5's reach regardless of the scope term |

Derived from the two confusion matrices. Per-case attribution requires the
annotation file, which was unavailable.
