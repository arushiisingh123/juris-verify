# Arm F override error analysis

Read-only analysis. No production code was changed to produce it.

> **Status.** Part 1 below is complete and derived only from the reported summary
> figures plus the arm H/E measurements. Parts 2–5 require the per-claim data in
> `eval/results_20261009_085502_n504_section2_w0.05_prosactive.json`, which is
> **not in this repository** — the run happened in Colab. Copy the artifact into
> `eval/` and run:
>
> ```
> python inspect_arm_f_overrides.py \
>     --artifact eval/results_20261009_085502_n504_section2_w0.05_prosactive.json
> ```
>
> That regenerates this file with the 24-row table, every concentration
> breakdown, and a per-case block for each override. It reproduces Part 1's
> base-rate analysis, so nothing here is lost on regeneration; the qualitative
> categorisation in Part 4/5 is then filled by a reading pass and folded back in
> via `--categories`.

---

## Part 1 — What the summary figures already establish

### 1.1 The reported numbers are internally consistent

| metric | arm D | arm F | delta |
|---|---|---|---|
| overall | 83.53% (421/504) | 83.93% (423/504) | **+0.40** |
| CONTRADICT | 66.47% (115/173) | 73.99% (128/173) | **+7.52** |
| SUPPORT | 92.45% (306/331) | 89.12% (295/331) | **−3.33** |

Every figure reconciles: `115 + 13 = 128`, `306 − 11 = 295`, `421 + 13 − 11 = 423`,
and `13/24 = 54.17%`. Net effect **+2 claims out of 504**.

### 1.2 The gates discriminate — they are not just a volume filter

This is the central question, and it is easy to get wrong. Arm H reported
"precision 54.86%" for the naive rule, and arm F reports 54.17%. Read side by
side those look identical, suggesting the four gates added nothing. **They are
not comparable.** Arm H's denominator is all 289 claims carrying
counter-evidence, including the ~131 where an override changes nothing because
arm D already predicted CONTRADICT.

The like-for-like denominator is the **overridable population**: claims arm D
predicted SUPPORT that carry counter-evidence.

| quantity | value | source |
|---|---|---|
| rescuable (gold CONTRADICT, baseline SUPPORT, has counter) | 44 | arm H |
| losable (gold SUPPORT, baseline SUPPORT, has counter) | 105 | derived: `306 − round(0.6073 × 331)` |
| **overridable population** | **149** | sum |
| **base rate of correctness** | **29.53%** | 44/149 |

Cross-check on the 105: arm E found 131 gold-SUPPORT claims with
counter-evidence; `131 − 105 = 26`, which matches arm D's 25 SUPPORT errors
(claims already predicted CONTRADICT, where an override is a no-op). Consistent.

Against that baseline:

- Arm F's **54.17% is 1.83× the 29.53% base rate**.
- A random 24 drawn from the 149 would be expected to yield **7.1 rescues** and
  a net of **−9.8 claims**. Arm F produced 13 rescues and **+2 claims**.
- `P(X ≥ 13 | n=24, p=0.2953) = 0.0100`.

So the gates are distinguishable from random selection at the 5% level, and they
converted an expected ~10-claim loss into a 2-claim gain. The prosecution
mechanism is doing real work. The caveat is equally real: with n=24 the 95%
interval on 54.17% is roughly ±20pp, so the *size* of the effect is poorly
determined even though its existence is supported.

### 1.3 What the gate design already tells us about the 11 regressions

Every override requires `adjudication_status == TRUE_CONTRADICTION`, which means
all four gates passed. So **all 11 regressions passed all four gates**, and by
construction they cannot be:

- **asymmetric** — gate 1 required `verdict(claim → counter) == CONTRADICTED`;
- **internally inconsistent** — gate 2 required `verdict(primary → counter) != CONTRADICTED`;
- **contested** — gate 3 required `primary_verdict != SUPPORTED`, so no regression
  had a premise that entailed the claim;
- **background or methods counter-evidence** — gate 4 rejects those outright.

Gate 4's only remaining admissions are `results`, `conclusions`, or `unknown`
with the counter in the same document the baseline cited. **This yields a
falsifiable integrity check:** no override may have
`counter_section ∈ {background, methods}`. If the artifact shows one, gate 4 has
a bug. `inspect_arm_f_overrides.py` surfaces this in its section breakdown.

The practical consequence is that the failure mode lives entirely in the space
the four gates do not represent — see Part 6.3.

### 1.4 Prior evidence bearing on the scope hypothesis

A separate calibration probe (`inspect_scope_alignment.py`, untracked and not
added to the repo) scored constructed pairs with the project's own MiniLM. On
five triples holding the primary fixed:

| feature | aligned refutation | mismatch, distant vocabulary | mismatch, **shared** vocabulary |
|---|---|---|---|
| cos(claim, counter) | 0.8749 | 0.5817 | 0.8346 |
| cos(primary, counter) | 0.8710 | 0.5246 | 0.7926 |

Aligned refutations separate cleanly from distant-vocabulary mismatches and
**overlap almost entirely** with shared-vocabulary ones (ranges [0.7583, 0.9233]
vs [0.7391, 0.9188]). In two of five cases the mismatch scored *higher* than the
genuine refutation — the macrovascular/microvascular and all-cause/cardiovascular
pairs. Cosine tracks topical overlap, and the distinctions that define scope
mismatch are precisely the lexically adjacent ones.

That is constructed data, not the real cases. It argues against a cosine-based
scope feature; it says nothing yet about whether the real regressions are scope
failures at all.

---

## Part 2 — All 24 overrides

_Awaiting the artifact. `inspect_arm_f_overrides.py` emits a row per override
with: claim index, gold, baseline and adjudicated prediction, outcome, counter
section, cross-document flag, in-bundle flag, counter confidence, counter
margin, reverse P(contradiction), primary→counter P(contradiction),
contradiction rank and pool size._

## Part 3 — Concentration analysis

_Awaiting the artifact. Cross-tabulated rescue/regression counts and precision
per level of: counter section, cross-document, in-bundle, adjudication reason,
reverse verdict, primary→counter verdict, baseline verdict; plus
min/median/max and an explicit overlap statement for every numeric field
(confidence, margin, all four NLI directions, rank, pool size,
contradictory-candidate count)._

## Part 4 — SUPPORT regressions, categorised

_Awaiting the artifact. Categories: **A** semantic scope mismatch, **B**
unrelated evidence, **C** NLI error, **D** provenance/grounding failure, **E**
other. Assignment requires reading the sentences and is deliberately not
automated._

## Part 5 — Rescued CONTRADICT claims

_Awaiting the artifact. What made each counter-evidence genuinely relevant._

---

## Part 6 — Conclusions

### 6.1 Strongest observed failure pattern

**Not yet determinable** — it needs the per-case data. What *is* determined is
the shape of the space it must live in (§1.3): all 11 regressions are symmetric,
pool-coherent, uncontested, finding-grounded contradictions. They are not
sloppy matches that slipped a filter; they are cases where the NLI model and the
section heuristic both say "this is a grounded contradiction of the claim" and
are wrong together. That makes a fifth gate of the same *kind* — another binary
sentence-pair relation — unlikely to help, since the existing four already agree.

### 6.2 Strongest evidence that prosecution is useful

The base-rate comparison in §1.2, and it is the most defensible number in this
analysis:

> Arm F's override precision is **1.83× the 29.53% base rate** of the overridable
> population (p = 0.0100), turning an expected **−9.8 claims** under random
> selection into **+2**.

Supporting: CONTRADICT accuracy **+7.52pp (115 → 128)**, which is 13 of arm H's
44 identified recoverable misses — **29.5% of the available headroom** — with no
change to retrieval, extraction, or the verifier.

The honest counterweight: overall accuracy moved **+0.40pp**, within noise for
n=504, and SUPPORT fell 3.33pp. Arm F is a demonstration that the mechanism has
signal, not yet a system improvement worth shipping.

### 6.3 What information the adjudicator appears to be missing

All four gates are **binary relations between two text spans**, computed by a
model trained on generic NLI. None of them represents, explicitly:

1. **Population / subgroup** the claim asserts about, vs the counter's;
2. **Outcome / endpoint** measured (all-cause vs cardiovascular mortality,
   macro- vs microvascular, falls vs fractures);
3. **Direction and magnitude** of the reported effect;
4. **Comparison arm** the counter's effect is relative to.

A claim and a counter can match on topic and diverge on any of (1)–(4), and
generic NLI will frequently call that a contradiction. §1.4 adds that the
project's existing sentence embedding does not recover (1) or (2) in a usable
form, so the missing information is not retrievable by swapping a similarity
measure — it would need extracting as structure.

### 6.4 Three experimentally testable directions

**D1 — Offline gate-subset ablation.** Arm F records all four gate outcomes for
every adjudicated claim, not just the overrides. So the precision and volume of
**all 16 gate subsets** can be recomputed from the existing artifact with *zero*
model runs and zero code change. Answers: which gates are load-bearing, which
are inert, and whether any is actively harmful. (A prior concern: with mean
counter confidence 0.9254, gate 1's symmetry test may fire for nearly everything
and contribute nothing — this settles it.)

**D2 — Random-k control arm.** Override a random 24 from the overridable
population. §1.2 predicts 7.1 rescues and −9.8 claims analytically; a run gives
the empirical distribution and converts the 1.83× point estimate into a proper
interval. Directly falsifies "the gain is just volume reduction."

**D3 — Scope-feature measurement study (measurement, not a gate).** Over all
~149 overridable claims, measure whether any *extractable structured* feature
separates rescues from regressions: shared numeric quantities between claim and
counter, overlap of population noun phrases, overlap of endpoint terms, presence
of an explicit comparison. Report separation and overlap only — no criterion, no
threshold. This tests §6.3's hypothesis without committing to a design.

### 6.5 Recommended first: D1

Three reasons:

1. **Zero marginal cost.** Pure offline recomputation over an artifact that
   already exists. No GPU, no re-run, no production change — the cheapest
   experiment available by a wide margin.
2. **It audits what is already built before adding anything.** §1.3 shows all
   four gates fired on all 11 regressions, so at least one may be contributing
   nothing. Adding a fifth signal while an existing one is inert would confound
   the next result.
3. **It sizes the remaining headroom per gate.** D3 is the expensive direction
   and the one that needs a design; knowing which gate is weakest tells you what
   D3 should actually measure. Running D3 first risks measuring features that
   duplicate a gate already working.

D2 is a close second and could run concurrently — it needs no new logic either,
only a seeded random selection over the overridable set, and it hardens the
headline claim in §6.2 against the "volume filter" objection.
