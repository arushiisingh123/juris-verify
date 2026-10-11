# G5 scope-alignment gate — primary held-out validation

Research documentation. No production module was modified, no experiment was
rerun, and no existing result JSON was altered. G5 is **not** integrated into
`src/prosecution.py` or `src/pipeline.py`.

Companions: `g5_primary_heldout_validation_tables.md` (paper-ready tables),
`g5_primary_heldout_validation_summary.json` (computed statistics),
`g5_reproducibility.md`.

---

## 0. Correction notice and provenance

### 0.0 CORRECTION — the baseline figure was wrong

An earlier revision of this report used **158/188 = 84.04%** as the validation
baseline and reported G5 as reaching **165/188 = 87.77%**. **Both figures were
wrong and have been removed.**

- **158/188 is the POST-override accuracy of the frozen G1+G2+G3 run**, not a
  baseline. That run applied **12 overrides of its own** (8 on gold CONTRADICT,
  4 on gold SUPPORT, net +4).
- Using it as the base therefore **double-counted the adjudicator's existing
  overrides**: 5 of G5's 7 accepted cases had already been overridden by that
  same run.
- The **true pre-override baseline is 154/188 = 81.91%**, derived from the
  `baseline_prediction` field of the recovered candidate artifact.
- The corrected simulated result is **161/188 = 85.64%**.
- The **delta is numerically unchanged at +3.72 pp**, because G5's net effect
  (+7 claims) does not depend on the base; only the absolute figures were wrong.

**No experiment was rerun for this correction.** It is arithmetic over the
recovered source artifacts, and `verify_g5_validation_numbers.py` now *derives*
the baseline from the artifact instead of carrying a hardcoded constant, so the
error cannot recur silently.

### 0.1 Provenance — now verified against source

The primary held-out artifacts were recovered from the original run and are in
the repository:

| artifact | sha256 (first 16) |
|---|---|
| `candidates_validation_20261011_174506_n188_section2_w0.05_prosactive_gatesg1g2g3.json` | `96380ccb3646058e…` |
| `scope_alignment_primary_heldout_blind_annotations.json` | `bb6e8a4f8ecfdbb7…` |
| `docs/scope_alignment_primary_heldout_blind_dataset.json` | `6d26f0790e04157d…` |

`verify_g5_validation_numbers.py` runs in **ARTIFACT MODE** and recomputes both
confusion matrices from the annotations joined to the candidate artifact:

```
cases joined: 54, unmatched: none
g5              {TP 7, FP 0, FN 10, TN 37}   -> MATCH
direction_only  {TP 11, FP 4, FN 6, TN 33}   -> MATCH
```

Independently re-derived from the artifact: 188 claims, `claims_split:
validation`, frozen config `section/2/0.05/active/g1,g2,g3`, candidate frame
**54**, pre-override baseline **154/188**, and all 7 G5 accepts gold CONTRADICT
with `genuine_contradiction = YES`.

**Two artifacts remain absent** and were never written — the sampler that
produces them was not the route used for this run:

```
docs/scope_alignment_primary_heldout_manifest.json   NOT FOUND
docs/scope_alignment_primary_heldout_audit.json      NOT FOUND
```

Their absence does not affect any figure in this report: the candidate frame was
re-derived directly from the candidate artifact and matched at 54 cases. It does
mean the sampling audit trail (seed, order randomisation, exclusion checks) is
not independently recorded for this run.

Figures still taken **as supplied** and not independently verifiable here: 450
validation rows and 5,183 corpus documents. Everything else in this report is
either recomputed from the artifacts or computed from the confusion matrices.

## 0.2 Two further corrections to the framing in the brief

Both emerged from recomputing the statistics and both materially change what
may be claimed.

**(a) The McNemar p-value does not mean what it appears to mean.** The reported
`p = 0.0078125` is a test of the **accept-decision** disagreement between G5 and
direction-only. Because G5 is `direction-only AND overall_scope_match == YES`,
G5's accept set is a *strict subset* of direction-only's, so all 8 discordant
pairs must fall one way and the p-value is structurally determined. It is not
evidence that G5 is more accurate.

On **per-case correctness** — the decision-relevant framing — the discordance is
4 vs 4 and **McNemar two-sided exact p = 1.0**. The two rules are statistically
indistinguishable in accuracy on this frame.

**(b) The +3.72 pp gain is not specific to G5.** Direction-only produces 11
rescues and 4 added errors, net **+7** — identical to G5's 7 rescues and 0 added
errors. Both therefore simulate to the same 161/188 = 85.64%, and both score
44/54 = 81.48% on the candidate frame.

G5's real advantage is **zero added errors** — a precision/safety property — not
an accuracy advantage over the comparator.

---

## 1. Experimental objective

Determine whether a proposition-level scope-alignment test can identify, with
high precision, which of the adjudicator's contradiction decisions are genuine
refutations of the claim rather than artefacts of scope mismatch — and do so on
a candidate population selected before any outcome was consulted.

## 2. Hypothesis

Error analysis of the arm F override set indicated that false
SUPPORT → CONTRADICT decisions arise predominantly from **proposition-level
scope mismatch**: the counter-evidence diverges from the claim on population,
intervention, or outcome/endpoint, and the NLI model reads that divergence as
contradiction. The hypothesis under test:

> Requiring that the claim and the counter-evidence be about the same
> proposition (`overall_scope_match == YES`) **and** point opposite ways on a
> shared axis (`direction == OPPOSITE`) will accept genuine contradictions and
> reject scope-mismatch artefacts.

Scope alignment and directional contradiction are kept conceptually separate
throughout, and `direction == OPPOSITE` alone serves as the comparator.

## 3. G5 definition (frozen)

```
G5  ==  overall_scope_match == YES  AND  direction == OPPOSITE
```

Comparator:

```
direction-only  ==  direction == OPPOSITE
```

Both were frozen in `inspect_proposition_scope_alignment.py` before any
annotation existed, and fingerprinted
(`sha256 = 0449d3d00fc7cf0a65cdc65ad0b1b080bbcb87b9f7986b7df845b308caa50d77`) so
that the pre-specification is checkable. G5 contains no threshold and no tunable
parameter; every condition is an enum equality.

**G5 is a validated experimental safety gate and is not integrated into
production.**

## 4. Dataset construction

| property | value | class |
|---|---|---|
| source | SciFact **validation** split | as supplied |
| validation rows | 450 | as supplied |
| unique labelled claims | 188 | as supplied |
| corpus documents | 5,183 | as supplied |
| pipeline configuration | `evidence_mode=section`, `max_sentences=2`, `section_weight=0.05`, `prosecution_mode=active`, `adjudication_gates=g1,g2,g3` | frozen |

The validation split was chosen because a re-run over the training split cannot
supply a fresh candidate population: the 504 train claims are the entire
population the harness scores, and two independent runs already in `eval/`
produced an *identical* 150-case candidate frame (symmetric difference: none).
See `scope_alignment_primary_heldout_plan.md` §2.

## 5. Candidate-frame construction

The candidate frame was defined using **only** fields available before any
outcome is known:

```
baseline_prediction == "SUPPORT"   AND   counter_evidence is non-empty
```

**54 eligible candidate cases** resulted. No filter was applied on gold label,
`override_applied`, adjudication status, adjudication reason, section, or any
confidence or similarity value.

The frame therefore contains **both accepted and rejected** candidates — unlike
the earlier rejected-candidate stress test, where excluding the previously
annotated cases left only candidates the gates had declined.

## 6. Leakage and overlap controls

1. **No gold exposure during selection.** The sampler projects every artifact
   record onto a three-field whitelist (`claim`, `primary_evidence`,
   `counter_evidence`) before sampling, so outcome fields are structurally
   unreachable rather than merely unused.
2. **Independently proven.** `verify_blind_projection.py` establishes this three
   ways — static AST analysis, a poisoned record whose outcome-field sentinels
   must not survive, and a `dict` subclass that raises on any forbidden read —
   with a positive control confirming the tripwire fires.
3. **No overlap with the discovery study.** The validation and train splits are
   disjoint, so overlap is impossible by construction; content-hash exclusion
   against the 42-case dataset was applied as a redundant check.
4. **Blind at annotation time.** All annotation fields were initially `null`;
   document ids, section labels and every probability were withheld, because
   section and provenance both correlate with the outcome in this problem.
5. **Randomized presentation order**, audited to differ from both the
   content-hash order and the artifact row order.

## 7. Annotation protocol

The **study-1 annotation schema** was reused exactly — field names (including
`causal_relation_alignment`), enum values, `NOT_APPLICABLE` handling, field
order — so the discovery and validation studies remain directly comparable.

| field | allowed values |
|---|---|
| `population_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `intervention_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `outcome_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `endpoint_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `context_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `causal_relation_alignment` | SAME / DIFFERENT / NOT_APPLICABLE / UNCLEAR |
| `direction` | OPPOSITE / SAME / NOT_APPLICABLE / UNCLEAR |
| `overall_scope_match` | YES / NO / UNCLEAR |
| `genuine_contradiction` | YES / NO / UNCLEAR |

Annotation seed: **20261012**. Annotators judge the claim against the
counter-evidence; the primary evidence is supplied for context only.
`genuine_contradiction` is annotated independently of the alignment fields and
is **not** used by G5 — it serves as an independent check on G5's accepts.

**All 7 G5-accepted cases were independently annotated
`genuine_contradiction = YES`** (as supplied). Since `genuine_contradiction` is
not an input to G5, this is a genuine convergence rather than a tautology.

---

## 8. Results — primary held-out validation (n = 54)

### 8.1 G5

| | accepted | rejected |
|---|---|---|
| **gold CONTRADICT** (17) | **TP = 7** | FN = 10 |
| **gold SUPPORT** (37) | **FP = 0** | TN = 37 |

- Precision **100.00%** (7/7), Wilson 95% CI **[64.57, 100.00]**
- Recall **41.18%** (7/17), Wilson 95% CI **[21.61, 63.99]**
- Specificity **100.00%** (37/37), Wilson 95% CI [90.59, 100.00]
- Candidate-frame accuracy 81.48% (44/54)
- Fisher exact two-sided **p = 1.098 × 10⁻⁴**

### 8.2 direction-only comparator

| | accepted | rejected |
|---|---|---|
| **gold CONTRADICT** (17) | TP = 11 | FN = 6 |
| **gold SUPPORT** (37) | **FP = 4** | TN = 33 |

- Precision **73.33%** (11/15), Wilson 95% CI [48.05, 89.10]
- Recall **64.71%** (11/17), Wilson 95% CI [41.30, 82.69]
- Candidate-frame accuracy 81.48% (44/54)
- Fisher exact two-sided **p = 1.002 × 10⁻⁴**

### 8.3 Paired comparison

| framing | discordant | split | two-sided exact p | meaning |
|---|---|---|---|---|
| accept decision | 8 | 0 / 8 | **0.0078125** | structurally determined — G5 ⊂ direction-only. **Not** evidence of superiority. |
| per-case correctness | 8 | 4 / 4 | **1.0** | the decision-relevant test: **no detectable accuracy difference** |

The informative contrast is the precision/recall trade: G5 adds **0** errors
where direction-only adds **4**, at the cost of **4** rescues.

### 8.4 Simulated application to the 188-claim validation population

| rule | baseline (pre-override) | after | Δ | rescues | added errors | net |
|---|---|---|---|---|---|---|
| **G5** | 154/188 = 81.91% | **161/188 = 85.64%** | **+3.72 pp** | 7 | **0** | +7 |
| direction-only | 154/188 = 81.91% | 161/188 = 85.64% | +3.72 pp | 11 | 4 | +7 |
| *frozen G1+G2+G3 run, for reference* | *154/188 = 81.91%* | *158/188 = 84.04%* | *+2.13 pp* | *8* | *4* | *+4* |

The baseline is the **pre-override** accuracy, from `baseline_prediction` in the
candidate artifact. The third row is the frozen run's own post-override result,
shown only for comparison — it is **not** the baseline, and using it as one was
the error corrected in §0.0.

**This is a simulated application, not an independently rerun end-to-end
evaluation.** It is computed by applying each rule's decisions on the
exhaustively annotated 54-case candidate frame to the 188-claim validation
baseline. It is valid only because the frame is exhaustive over the population
where either rule can act — a rule can only change a claim that is
baseline-SUPPORT with counter-evidence, and all 54 such claims were annotated.
No claim outside the frame can be affected, so no extrapolation is involved.
It is nonetheless **not** a production evaluation run, and must not be described
as one.

Note that both rules reach the same simulated figure.

## 9. Discovery study (n = 42) — context only

| | accepted | rejected |
|---|---|---|
| **gold CONTRADICT** (24) | TP = 20 | FN = 4 |
| **gold SUPPORT** (18) | FP = 0 | TN = 18 |

Precision 100.00% (20/20), recall 83.33% (20/24), Fisher exact
p = 2.12 × 10⁻⁸.

**This is discovery evidence only.** The 42 cases were **outcome-selected** —
they are exactly the overrides the G1+G2+G3 adjudicator made — and the labels
were assigned by an annotator who knew each case's outcome while looking for a
scope pattern. Both facts make it unusable as a test of the hypothesis it
generated.

### 9.1 The primary held-out validation is the stronger evidence

| | discovery (n=42) | **primary held-out (n=54)** |
|---|---|---|
| candidate selection | outcome-selected | **outcome-blind** |
| annotator knew outcomes | yes | **no** |
| rule specified before annotation | no | **yes, frozen + fingerprinted** |
| overlap with the other study | — | **none (disjoint splits)** |
| recall | 83.33% | 41.18% |

The recall difference is the expected direction: the discovery study was drawn
from cases the gates had already accepted, which are enriched for the easy,
cleanly-aligned refutations G5 detects.

## 10. Error analysis — the 10 false negatives

G5 misses 10 of 17 gold CONTRADICT cases in the frame (recall 41.18%). Per-case
analysis requires the annotation file, which is absent, so the mechanism cannot
be attributed here. What the structure of the rule and the comparator imply:

- **4 of the 10** are accepted by direction-only, so their `direction` is
  `OPPOSITE` and they fail on `overall_scope_match != YES`. These are cases
  where the annotator judged the proposition scope not to match, or was unsure.
- **6 of the 10** are rejected by direction-only too, so their `direction` is
  not `OPPOSITE`. G5 cannot reach them regardless of the scope term, and they
  bound what any scope refinement could recover.

Earlier error analysis on the discovery set identified a floor that no scope
test can reach — NLI errors on paraphrased causal vocabulary, and
unrelated-evidence retrieval failures. Whether the 6 direction-rejected cases
here are of that kind is **not established** and is the first question for the
per-case pass once the annotation file is available.

## 11. Limitations

1. **The source artifacts could not be verified** (§0). The confusion matrices
   are as supplied.
2. **n = 54, with only 17 positives.** Every interval is wide: G5's 100%
   precision on 7 accepts has a Wilson lower bound of **64.57%**, so the data
   are compatible with true precision near 65%.
3. **Recall is low — 41.18%**, CI [21.61, 63.99]. G5 is a high-precision filter,
   not a contradiction detector.
4. **No accuracy advantage over the comparator** was demonstrated (McNemar on
   correctness p = 1.0; identical simulated accuracy). The demonstrated
   advantage is zero added errors.
5. **Single annotator, no inter-annotator agreement.** The
   `NOT_APPLICABLE` / `DIFFERENT` boundary is the likeliest source of
   disagreement and is unmeasured.
6. **The simulated application is not a production evaluation run** (§8.4).
7. **Per-claim retrieval instability** of ~0.4% was previously measured
   (2/504 claims retrieved different documents across runs), so the candidate
   frame is not perfectly reproducible.
8. **`genuine_contradiction = YES` on all 7 accepts** is supportive but rests on
   the same single annotator.
9. **Validated on SciFact only**, in one domain, with one retrieval
   configuration.

## 12. Paper-ready claims

Each is directly supported by the experiment as described, subject to §0.

1. On a 54-case outcome-blind candidate frame from the SciFact validation split,
   a frozen two-condition scope-alignment gate — `overall_scope_match == YES AND
   direction == OPPOSITE` — accepted 7 contradiction decisions, **all 7 of which
   were gold CONTRADICT (precision 100%, Wilson 95% CI [64.6, 100])**, and added
   **zero** false contradictions (specificity 100%, 37/37).
2. The association between the gate's decision and the gold label is
   significant: **Fisher exact two-sided p = 1.10 × 10⁻⁴**.
3. The gate's **recall is 41.18%** (CI [21.6, 64.0]): it is a high-precision
   safety filter, not a complete contradiction detector.
4. Against a `direction == OPPOSITE` comparator, the gate **eliminates all 4
   false contradictions** the comparator admits, at the cost of 4 true
   detections. Precision rises 73.33% → 100%; recall falls 64.71% → 41.18%.
5. Applying the gate's decisions to the 188-claim validation **pre-override**
   baseline **simulates** 81.91% (154/188) → **85.64% (161/188)**, **+3.72 pp**,
   from 7 rescues and 0 added errors. The comparator reaches the same simulated
   accuracy by a different route (11 rescues, 4 added errors). For reference, the
   frozen G1+G2+G3 run itself reached 158/188 = 84.04% via 8 rescues and 4 added
   errors.
6. All 7 gate-accepted cases were independently annotated
   `genuine_contradiction = YES`, a field the gate does not consume.
7. The rule was **frozen and fingerprinted before any annotation existed**, and
   the candidate frame was selected from `baseline_prediction` and
   `counter_evidence` only, with blinding verified by three independent methods.
8. The earlier 42-case study is **discovery evidence only** (outcome-selected,
   outcome-aware labelling); the 54-case study is the primary independent
   validation.

## 13. Claims we should NOT make

1. ❌ "G5 achieves 100% contradiction detection accuracy." It detects **41.18%**
   of contradictions. Precision ≠ detection rate.
2. ❌ "G5 has 100% precision" **without the interval**. n = 7; the Wilson 95% CI
   lower bound is 64.57%.
3. ❌ "G5 significantly outperforms direction-only." McNemar on per-case
   correctness gives **p = 1.0**, and simulated accuracy is identical. Only the
   error-free property is demonstrated.
4. ❌ Citing **McNemar p = 0.0078125** as evidence of superiority. It tests
   accept-decision disagreement, which is structurally guaranteed because G5 is
   a strict subset of direction-only.
5. ❌ "G5 improves validation accuracy by 3.72 pp" as a G5-specific result. The
   comparator achieves the same +3.72 pp.
6. ❌ "85.64% end-to-end validation accuracy." It is a **simulated** application
   to an exhaustively annotated candidate frame, not a rerun evaluation.
7. ❌ **"84.04% → 87.77%", or any use of 158/188 as the baseline.** Superseded
   and wrong — see §0.0. 158/188 is the frozen G1+G2+G3 run's *post*-override
   accuracy, so treating it as a baseline double-counted that run's 12 overrides.
   The correct figures are **81.91% (154/188) → 85.64% (161/188)**.
8. ❌ "G5 is in the production pipeline." It is **not** integrated; `src/` is
   unmodified.
9. ❌ Citing the discovery study's 83.33% recall as validation. It is
   outcome-selected with outcome-aware labels.
10. ❌ "Scope alignment is a novel contribution." Claim/evidence scope comparison
   is long-standing in claim verification. What is supported is narrower: that
   *this* pipeline's false contradictions are predominantly scope-mismatch
   artefacts, and that a two-condition test removes them at measured cost.
11. ❌ "Validated for high-stakes deployment." One dataset, one configuration,
    n = 54, one annotator, no inter-annotator agreement.
12. ❌ Any per-case claim about **why** the 10 false negatives fail. The
    annotation file was unavailable; §10 gives only what the rule structure
    implies.
