# Proposition-level scope alignment — diagnostic

Research only. No production module was modified. No G5 gate was implemented.
Nothing was tuned. Machine-readable companion:
`docs/proposition_scope_alignment_diagnostic.json`.

---

## Status: rules frozen, evaluation BLOCKED

**The annotations this diagnostic needs do not exist.** `docs/scope_alignment_blind_dataset.json`
carries all 420 annotation values (42 cases × 10 fields) as `null`, with
`annotations_performed: false`, and no annotation file exists anywhere in the
repository.

The brief states "we have exploratory model-assisted annotations for the 42
cases". Nothing in this workspace contains them. They were not produced in a
prior turn — the previous task's instruction was explicitly *"Do not perform the
annotations yourself"*, and the dataset was handed over deliberately empty.

**I did not supply them, and should not.** Two turns ago I enumerated every one
of these 42 cases by claim index with its outcome (`docs/scope_alignment_reconciliation.md`,
§2). Any annotation I produce now is outcome-aware by construction — not
"exploratory", but a reconstruction of the very labels the brief forbids using
to define the rule. Producing them would manufacture an apparent confirmation of
the hypothesis, which is the worst available outcome here.

So this document delivers what *can* be delivered without them: the inspection,
the harness, and the **pre-specified rules frozen to disk while the annotation
column is still empty** — the strongest form of "the rule must be specified
before outcome-based scoring" available.

---

## 1. Inspection findings

### 1.1 Allowed annotation values (read verbatim; nothing invented)

From `docs/scope_alignment_blind_annotation.md` and the schema block in
`docs/scope_alignment_blind_dataset.json` — both agree:

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

Plus free-text `annotator_notes`. Exposed case fields: `case_id`, `claim`,
`primary_evidence`, `counter_evidence` — nothing else.

### 1.2 The join is sound

The 504 artifact (`...gatesg1g2g3.json`, `adjudication_gates: ['g1','g2','g3']`)
holds **42 overrides, 24 gold CONTRADICT / 18 gold SUPPORT**. Claim texts are
unique across those 42, so an exact-claim-text join is unambiguous. Verified by
the harness: **42/42 matched, 0 unmatched, 0 ambiguous.**

### 1.3 Other documents inspected

| file | relevance |
|---|---|
| `scope_alignment_blind_annotation.md` | enum definitions, annotator instructions, unblinding warnings |
| `scope_alignment_blind_dataset.json` | the 42 cases; annotation column empty |
| `scope_alignment_reconciliation.md` | **outcome of every case by claim index** — an unblinding hazard, and the reason I cannot annotate |
| `arm_f_gate_subset_ablation.md` | D1: G3 necessary (8/8 subsets without it net-negative); G4 net-harmful |
| `scope_alignment_failure_analysis.{md,json}` | the hypothesis-generating analysis — **outcome-aware labels, not usable to define the rule** |

---

## 2. The three activities, kept separate

The brief asks for these to be explicitly distinguished. They are, and two of
the three are complete.

### Activity 1 — hypothesis generation (COMPLETE, and contaminated)

`docs/scope_alignment_failure_analysis.md` reported that regressions diverge on
population/intervention/outcome while rescues invert direction on an
otherwise-aligned proposition. **Those labels were assigned by one annotator
(me) who knew each case's outcome and was looking for that pattern.** That
analysis is legitimate as a source of a hypothesis and worthless as a test of
it. Per the brief it was **not** used to define the rules below — the three
rules come from the brief's own specification.

### Activity 2 — rule definition (COMPLETE, frozen)

Three rules, exactly as specified, implemented in
`inspect_proposition_scope_alignment.py`:

| rule | condition |
|---|---|
| **A** | `overall_scope_match == YES` **AND** `direction == OPPOSITE` |
| **B** | `population`, `outcome`, `endpoint` all `!= DIFFERENT` **AND** `direction == OPPOSITE` |
| **C** | none of the six scope dimensions `== DIFFERENT` **AND** `direction == OPPOSITE` |

Three contrast conditions are also computed, to keep scope alignment and
directional contradiction conceptually separate rather than fused:

| contrast | condition |
|---|---|
| `scope_only_A` | `overall_scope_match == YES` (no direction term) |
| `scope_only_C` | no scope dimension `== DIFFERENT` (no direction term) |
| `direction_only` | `direction == OPPOSITE` (no scope term) |

**Interpretation of "not explicitly different":** `!= DIFFERENT`, so SAME,
NOT_APPLICABLE and UNCLEAR all pass. This is the conservative reading — an
UNCLEAR judgement does not get to veto an override — and it matters, because the
alternative reading would make the rules strictly stricter and the comparison a
different experiment. It is fixed now, before any data.

**Frozen-rule fingerprint:** `sha256(rule source) = 0449d3d00fc7cf0a65cdc65ad0b1b080…`
(full value in the JSON). This was computed and written to disk with the
annotation column empty, so it can be checked later that the rules were not
adjusted after results were seen. No threshold exists to tune — every condition
is an enum equality.

### Activity 3 — outcome-based evaluation (BLOCKED)

Cannot run. Requires annotations.

---

## 3. How the harness enforces phase separation

Not by convention — structurally:

| phase | function | sees outcomes? |
|---|---|---|
| 1 | `describe_annotations(cases)` | no |
| 2 | `apply_rules(cases)` | no |
| 3 | `score_against_outcomes` / `outcome_lookup` | **yes — the only place** |

Each rule predicate takes a single argument: the annotation dict. It is handed
no case record, no artifact row and no label, so it *cannot* read an outcome
even accidentally. The self-test asserts this by signature inspection.

The join deliberately carries `claim_index` only; outcomes are looked up in a
separate phase-3 function.

### Features excluded by design

gold label · rescue/regression status · override correctness · adjudication
status · adjudication reason · gate outcomes · document ids · section labels ·
`counter_section` · every adjudicator NLI probability.

Section labels and document ids are excluded for a second reason beyond the
brief: both correlate with the outcome in this dataset (sections run `unknown`
15/24 vs 14/18; G4-blocked 45.8% vs 42.1%), so using them would leak the answer
into a feature.

### Harness verification

`--selftest` passes: rule logic across all four enum values per dimension,
the C⟹B implication, label isolation by signature, Fisher exact (**cross-checked
against `scipy.stats.fisher_exact` on 6 tables**, agreement < 1e-9), odds ratio
with Haldane–Anscombe correction, phi, end-to-end scoring on synthetic
annotations with a known confusion matrix, per-dimension tables, and detection
of both missing and out-of-enum annotation values.

Fisher exact is implemented natively via `math.comb`, so **no dependency is
added**; scipy is used only in the self-test cross-check, when present.

---

## 4. What the diagnostic will report once annotations exist

Per rule and per contrast: accepted/rejected counts, the 2×2 confusion matrix
(accepted-rescue / accepted-regression / rejected-rescue / rejected-regression),
precision, recall, specificity, true rescues retained and lost, regressions
rejected and retained, net claims versus accepting every override, two-sided
Fisher exact p, odds ratio, and phi. Plus a per-dimension discriminativeness
table (DIFFERENT vs not, against outcome) — computed in phase 3, after the rules
were frozen, and descriptive only.

---

## 5. Answers to the six questions

**1. Does proposition-level scope alignment appear promising?**
**Undetermined from this repository's current state.** The only supporting
evidence is Activity 1, whose labels are outcome-aware. That makes the
hypothesis worth testing, which is not the same as promising.

**2. Which dimensions appear most discriminative?**
**Cannot be answered.** Activity 1 nominated population, intervention and
outcome (38.9% each among regressions), but those counts come from the
contaminated labelling and cannot be cited as a finding.

**3. Does a simple interpretable rule separate the two groups?**
**Unknown.** Rules A, B and C are specified and implemented; none has been
evaluated. Note in advance that they are nested — C ⟹ B is asserted in the
self-test — so if C separates and B does not, the extra dimensions carry the
signal; if both separate equally, the three core dimensions suffice.

**4. What failure cases remain?**
Activity 1 identified a floor that no scope test can reach: ~3 of 18 regressions
were plain NLI errors (claim 285: "initiation patterns" vs "origin firing" —
which agree) or unrelated-evidence retrieval failures (claim 452). Treat that as
a hypothesis about the ceiling, not a measurement.

**5. Is a fresh held-out validation justified?**
**Not yet — it is premature by one step.** Held-out validation tests whether a
rule *generalises*. No rule has yet been shown to work on the 42 cases at all,
on uncontaminated labels. Commissioning a held-out set now would spend a second
sample on a question the first has not answered.

**6. Should we proceed to a production G5 experiment?**
**No**, and not implemented here regardless. D1 additionally showed G3 is
necessary and G4 net-harmful, so any G5 would have to be evaluated against the
G1+G2+G3 configuration rather than the shipped four-gate one.

---

## 6. Recommendation

**Neither STOP nor PROCEED TO HELD-OUT VALIDATION can be justified from the
current evidence.** Forcing the binary now would mean choosing between
abandoning a hypothesis that has never been fairly tested, and spending a new
sample on one that has never passed a first test. Both are avoidable by doing
the cheap missing step.

**The blocking action is one step, and it is cheap:**

> Have someone who has **not** read `scope_alignment_failure_analysis.{md,json}`
> or `scope_alignment_reconciliation.md` fill the 42 × 9 annotation fields in
> `docs/scope_alignment_blind_dataset.json`, following
> `docs/scope_alignment_blind_annotation.md`. Then re-run the diagnostic.

The rules are already frozen and fingerprinted, so that annotation pass
constitutes a genuine pre-registered test. **Then** the STOP / PROCEED decision
becomes answerable:

| result on the 42 cases | next step |
|---|---|
| a rule retains most of the 24 rescues while rejecting most of the 18 regressions, with Fisher p < 0.05 | **PROCEED TO HELD-OUT VALIDATION** |
| no rule separates, or separation is driven only by `direction_only` (so scope adds nothing over the existing G1) | **STOP / REJECT** |
| separation exists but `UNCLEAR` rates are high | annotation guidance is the problem, not the hypothesis — refine and re-annotate before deciding |

**On model-assisted annotation.** If a model is used instead of a human, it must
be a fresh context with no access to this repository's analysis documents or the
artifacts, and the result must be labelled **model-assisted exploratory
annotation, not independent human validation** — in the report, not just in a
footnote. A model annotating from the raw sentence pairs is a reasonable
*screen*. It is not validation of a hypothesis about human-interpretable
proposition structure, and a positive result from it would still leave item 5
open.

---

## 7. Limitations (applying to any result this diagnostic later produces)

1. **n = 42** (24 / 18), from a single 504-claim run. Small for a 2×2 with
   several candidate rules.
2. **Three rules plus three contrasts = six comparisons** on one sample. No
   multiplicity correction is applied; a nominal p < 0.05 on one of six should
   be read accordingly.
3. **The cases are not an independent sample of the problem.** They are the
   overrides the G1+G2+G3 adjudicator happened to make, so they are already
   filtered by G1, G2 and G3. A rule that separates here may not separate on
   claims the current adjudicator rejects.
4. **Per-claim retrieval instability.** `scope_alignment_reconciliation.md`
   established that 2/504 claims (0.40%) retrieve different documents between
   runs, and that this changed 1 of 42 overrides. The case set is therefore not
   perfectly reproducible across runs.
5. **Annotation reliability is unmeasured.** With one annotator there is no
   inter-annotator agreement; the `NOT_APPLICABLE` / `DIFFERENT` boundary is the
   likeliest source of disagreement and the harness reports its rate for that
   reason.
6. **No claim of validation can rest on these 42 cases**, whatever the result —
   they are the hypothesis-generating sample. At best a positive result licenses
   a held-out test.

---

## Appendix — commands

```bash
# verify the harness (no real data touched)
python inspect_proposition_scope_alignment.py --selftest

# run the diagnostic (currently exits 3: BLOCKED, no annotations)
python inspect_proposition_scope_alignment.py
```
