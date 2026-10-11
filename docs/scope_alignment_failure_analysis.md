# Proposition-level scope alignment — failure analysis

Analysis only. No production code was changed. Machine-readable companion:
`docs/scope_alignment_failure_analysis.json`.

---

## 0. Data provenance and two caveats that bound everything below

**The G4-off artifact is not in this repository.**
`eval/results_20261011_160640_n504_section2_w0.05_prosactive_gatesg1g2g3.json`
is absent (`eval/` holds only the four earlier files; a tree-wide search for
`*gatesg1g2g3*` and `*20261011*` returns nothing). The run happened in Colab.

The analysis is still possible because the **default-mode** arm F artifact
(`results_20261009_085502_n504_section2_w0.05_prosactive.json`) stores *every*
gate outcome, all four NLI directions, and the full text for all 504 claims. The
G1+G2+G3 case list is therefore reconstructable exactly — the same property that
made the D1 ablation possible.

**Caveat 1 — a one-case discrepancy.** The reconstruction yields:

| | reconstructed | reported in the G4-off run |
|---|---|---|
| rescues | **24** | 24 ✓ |
| regressions | **19** | **18** ✗ |

Rescues match exactly; regressions are off by one. This should have been an
exact match, so **one of the 19 cases analysed here is probably not in the real
regression set.** Ranked by reverse-direction `P(contradiction)` — the quantity
closest to G1's decision boundary and so likeliest to flip between processes —
the candidates are **claim 391 (0.641)**, **338 (0.713)**, **452 (0.743)**,
**285 (0.772)**. Notably 452 and 285 are the two cases this analysis labels
`ambiguous/other`, so if the missing case is one of those, the pattern below
*strengthens*. Reconciling this is the first item in §7.

**Caveat 2 — label provenance.** Case selection, provenance and all NLI figures
are computed from the artifact. **The mismatch labels and preserved-dimension
labels are my judgements**, assigned after reading all 43 claim/counter-evidence
pairs. They are recorded per case in the JSON so each can be checked or
overruled. Multi-label assignment was used where a case mismatches on more than
one dimension.

---

## 1. Regressions: mismatch taxonomy (n = 19)

Multi-label, so percentages sum above 100%.

| mismatch type | n | % of regressions |
|---|---|---|
| **intervention mismatch** | **8** | **42.1%** |
| **outcome mismatch** | **8** | **42.1%** |
| **population mismatch** | **7** | **36.8%** |
| evidence granularity mismatch | 5 | 26.3% |
| endpoint mismatch | 2 | 10.5% |
| direction mismatch | 2 | 10.5% |
| context/condition mismatch | 2 | 10.5% |
| causal/relation mismatch | 2 | 10.5% |
| ambiguous/other | 2 | 10.5% |
| temporal mismatch | 0 | 0.0% |

### The three recurring shapes

**(a) Different agent, same topic — 8 cases.** The counter-evidence studies a
different molecule, drug or programme than the claim.

> **150** — claim: *G-CSF increases expansion and infiltration of MDSCs into
> tumors.* counter: *CSF1 produced by tumor cells caused HDAC2-mediated
> downregulation of granulocyte-specific chemokine expression in CAF…*
> Different cytokine, different cell populations.
>
> **338** — claim about **OCT3/4**; counter about **FoxA**. A different pioneer
> factor says nothing about OCT3/4's interactions.
>
> **418 / 419 / 420** — the three DESMOND claims, each refuted by evidence about
> a *different* programme (dietary intervention in pregnancy; TTM-based
> feedback). In 418 the primary evidence is itself off-topic (pre-eclampsia, ALK
> inhibitors), so retrieval failed before adjudication ever ran.

**(b) Different measured outcome, same intervention — 8 cases.** The sharpest
illustration shares *everything* except the endpoint:

> **356** — claim: *Pyridostatin reduces **proliferation** of homologous
> recombination-defective cells.* counter: *Treatment with pyridostatin
> increases **telomere fragility** in BRCA2-deficient cells.*
> Same compound, same cell population, different measurement. The counter was
> already inside arm D's own evidence bundle.

**(c) Aggregate vs specific — 5 cases.** An aggregate null is read as negating a
specific effect:

> **55 / 56** — claims name *colorectal cancer* and *postmenopausal breast
> cancer*; the counter reports *"No association was detected between bariatric
> surgery and fractures, **cancer**, or stroke."*
>
> **95** — *three ORFs* (claim) vs *a single genome-length transcript*
> (counter). One transcript can carry three ORFs; the counts do not conflict.

### Two cases where the NLI model is simply wrong

> **285** — claim: *termination events dictated by **initiation patterns***;
> counter: *replication profile predominantly determined by the kinetics of
> **origin firing***. Origin firing *is* initiation. The two agree.
>
> **69** — both claim and counter say CD28 increases exhaustion. The
> contradiction comes from the contrastive framing *"CD28 augments, whereas
> 4-1BB reduces"*, plus a population shift (conventional T cells vs CAR
> signalling).

And one that no proposition test would be needed for:

> **452** — claim about *Wdr5 recruitment depending on Kat8*; counter about
> *PAX5 deletion via 9p breakpoints detected by FISH*. Entirely unrelated — a
> retrieval failure.

---

## 2. Rescues: preserved proposition dimensions (n = 24)

| dimension | preserved | mismatched | not applicable | preserved, of applicable |
|---|---|---|---|---|
| **context/condition** | 24 | 0 | 0 | **100%** |
| **causal/relation** | 24 | 0 | 0 | **100%** |
| **population** | 22 | 0 | 2 | **100%** |
| **intervention** | 18 | 0 | 6 | **100%** |
| **outcome** | 23 | 1 | 0 | **95.8%** |
| endpoint | 22 | 2 | 0 | 91.7% |
| temporal | 1 | 1 | 22 | — |

**Direction or polarity is inverted in 24/24 rescues (100%).**

"Not applicable" marks dimensions the claim does not assert — six rescues are
observational claims with no intervention (e.g. 204 heart-failure incidence in
women, 324 deprivation and emergency admission), and two have no identifiable
population. Treating those as "not preserved" would understate the pattern, so
they are recorded separately.

### The canonical rescue shape

Every rescue shares the claim's subject, intervention and outcome, and flips the
direction:

> **412** — claim: *Taking 400mg α-tocopheryl acetate with vitamin C **reduces**
> the risk of prostate cancer.* counter: *neither vitamin E nor C supplementation
> **reduced** the risk of prostate or total cancer.*
>
> **294** — claim: natriuretic peptides **increase** diabetes susceptibility.
> counter: a **21% decreased** risk of incident T2D per SD higher log-NP.
>
> **85** — CDCA **decreases** BAT activity vs CDCA treatment **increased** BAT
> activity.
>
> **324** — deprived-area patients **less likely** to have an emergency first
> admission vs **more likely** to be admitted as emergencies.

### The two exceptions, both informative

> **494** — the only rescue whose **outcome axis does not match**. Same
> population and exposure (birth weight), direction inverted, but the counter's
> endpoint is **mammographic density** — a surrogate, not breast cancer. It was
> correct here, but by substitution rather than by refutation.
>
> **83** — endpoint narrows from general effectiveness to **acute kidney
> injury**, while direction still inverts on a shared axis.

### Three rescues that contradict G4's rationale outright

**85**, **294** and **449** were all blocked by G4 in the default run and are all
correct. **449** is the most direct refutation of G4's design: its counter sits
in a **`CONTEXT`/background** section —

> claim: *The peak incidence of fractures occurs in **toddlers**.* counter:
> *CONTEXT The incidence of distal forearm fractures in children peaks around
> the time of the **pubertal growth spurt**…*

— a genuine, correctly-adjudicated refutation whose discriminating axis is
*temporal*, in exactly the section G4 was built to exclude on the grounds that
background sentences cannot ground a refutation.

---

## 3. Most discriminative dimensions

This is the headline result.

| criterion | regressions (n=19) | rescues (n=24) |
|---|---|---|
| **≥1 non-direction dimension mismatch** | **19 (100%)** | **2 (8.3%)** |
| **direction/polarity inversion on a shared outcome** | **2 (10.5%)** | **24 (100%)** |

Read together: **a rescue is a direction inversion with every other dimension
held fixed; a regression is a dimension divergence with no direction inversion.**
The two populations are almost perfectly separated by that single structural
distinction, on a feature the current adjudicator does not represent at all.

The two regressions with direction inversion (**88**, **368**) are consistent
rather than exceptional — in both, the inversion sits on a *different*
intervention or population:

- **88** — cholesterol **loading** in VSMCs vs HDL-mediated **depletion** in
  macrophages. Opposite exposure, different cell type.
- **368** — MLL1 inhibition driving EpiSC→naive vs 2i/LIF withdrawal driving
  naive→primed. Different intervention, opposite transition.

So direction inversion alone is not sufficient; it has to be *direction
inversion on an otherwise-aligned proposition*.

### Provenance features, for contrast

| feature | rescues | regressions |
|---|---|---|
| cross-document counter | 12/24 (50.0%) | 8/19 (42.1%) |
| G4 would have blocked | 11/24 (45.8%) | 8/19 (42.1%) |
| counter inside arm D's bundle | 6/24 (25.0%) | 2/19 (10.5%) |
| counter section = `unknown` | 15/24 (62.5%) | 14/19 (73.7%) |

Cross-document rate (50.0% vs 42.1%) and G4-blocked rate (45.8% vs 42.1%) differ
by under 8 points on n=24 and n=19 — far from separating, and nowhere near the
100%-vs-8.3% split above. The in-bundle rate differs more (25.0% vs 10.5%) but on
6 and 2 cases respectively, so it is not interpretable. **The provenance and
section signals that G1–G4 do represent carry little discriminative information
here**, which is what the D1 ablation indicated for G4 and this now shows case
by case.

---

## 4. Observed empirical findings

Stated separately from hypotheses, per the brief. These are descriptions of the
43 cases, nothing more.

1. **F1.** 19/19 regressions mismatch the claim on at least one proposition
   dimension other than direction. 2/24 rescues do.
2. **F2.** 24/24 rescues invert direction or polarity on a shared outcome axis.
   2/19 regressions do, and in both the inversion sits on a different
   intervention or population.
3. **F3.** The most frequent mismatch axes are **intervention** (42.1%),
   **outcome** (42.1%) and **population** (36.8%).
4. **F4.** 5/19 regressions are aggregate-vs-specific granularity errors: an
   aggregate null read as negating a site- or subgroup-specific effect.
5. **F5.** 2/19 regressions are plain NLI errors on paraphrased causal
   vocabulary (285) or contrastive framing (69), with no scope divergence to
   detect. 1/19 (452) is an unrelated-evidence retrieval failure.
6. **F6.** Cross-document rate (50.0% rescues vs 42.1% regressions) and
   G4-blocked rate (45.8% vs 42.1%) differ by under 8 points at these sample
   sizes — not a separating signal.
7. **F7.** Three correct rescues (85, 294, 449) were blocked by G4 in the
   default run, one of them (449) from a background section — independent
   case-level confirmation of D1's finding that G4 is net harmful.
8. **F8.** The rescue population is dominated by polarity negation of a
   quantitative finding; 1/24 (494) succeeds through a surrogate-endpoint
   substitution rather than true refutation.

## 5. Hypotheses for the next method

Explicitly hypotheses, not findings, and **no novelty is claimed**: "compare the
scope of a claim and its evidence" is a long-standing idea in claim
verification. What this analysis supports is narrower — that *this* pipeline's
specific failure mode is scope divergence, and that the dimensions above are
where it occurs.

- **H1.** A component that checks alignment of **population, intervention and
  outcome** between claim and counter-evidence, and requires **direction
  inversion on an aligned outcome**, would reject most regressions while
  retaining most rescues. Supported by F1/F2; untested.
- **H2.** Direction inversion alone is insufficient (F2's two exceptions);
  alignment must be checked *first*.
- **H3.** The required alignment is **asymmetric in granularity**: the
  counter-evidence's outcome must be at least as specific as the claim's.
  Aggregate-vs-specific is 26.3% of regressions (F4), and rescue 494 shows the
  converse failure (surrogate substitution) passing by luck.
- **H4.** A floor of ~3/19 regressions (F5) is **not** reachable by any scope
  test — they are NLI errors and retrieval failures. Any scope component's
  ceiling on this population is therefore about 16/19.
- **H5.** The signal is **not** recoverable from sentence-level embedding
  similarity. The earlier scope-alignment probe found aligned refutations and
  shared-vocabulary endpoint mismatches overlapping almost entirely under MiniLM
  cosine (ranges [0.7583, 0.9233] vs [0.7391, 0.9188]), with the mismatch
  scoring *higher* in 2 of 5 constructed cases. H1 would need explicit
  extraction of the three dimensions, not a distance.

## 6. Is a scope-alignment component justified?

**On this evidence, yes as a hypothesis worth testing — not yet as a design.**

Supporting: the 100%-vs-8.3% and 100%-vs-10.5% separations (F1, F2) are the
cleanest structural distinction found anywhere in this investigation, they have
a mechanistic explanation, and they sit on dimensions the current adjudicator
does not represent.

Against, and these are load-bearing:

- **n = 43**, from one run, hand-labelled by one annotator (me), with the labels
  assigned knowing each case's outcome. That is a real bias risk: I was reading
  *for* a scope pattern. The labels are published per-case in the JSON precisely
  so they can be independently re-assigned.
- One of the 19 regressions is probably not in the real set (§0).
- F5 caps the achievable gain at ~16/19 regressions.
- H5 says the obvious cheap implementation does not work, and no implementation
  that does work has been demonstrated.

## 7. Recommended next experiment

**Blind re-annotation before any implementation.**

The weakest link is not the pattern; it is that one annotator who expected the
pattern produced the labels. So:

1. **Reconcile the 19-vs-18 discrepancy** (cheap, and a prerequisite): copy the
   G4-off artifact into `eval/` and diff its override set against the
   reconstruction. If a borderline case flipped between processes, that also
   implies the pipeline is not bit-deterministic across runs — which would
   matter for every arm comparison made so far and should be checked by
   re-running default mode and diffing.
2. **Blind re-annotation.** Present all 43 claim/counter pairs **with gold
   labels and outcomes stripped and the order shuffled**, and label each only on
   whether population / intervention / outcome / direction are aligned. Then
   recover the outcomes and recompute the separation. If F1/F2 survive at
   anything near 100%-vs-10%, the pattern is real; if they collapse, they were
   annotation artefact and H1 should be dropped.
3. Only then, **measure extractability**: for the three dimensions, can they be
   recovered from the sentence pairs by any deterministic means available in
   this repo? H5 has already ruled out embedding cosine, so this is a genuine
   open question and the real cost driver of H1.

Step 2 before step 3, and step 3 before any component. Jumping to an
implementation now would be building on 43 outcome-aware labels.

---

## 8. Case-level data

Full per-case records — claim, primary evidence, counter-evidence, all four NLI
directions and distributions, provenance, gate outcomes, assigned labels and the
reasoning for each — are in `docs/scope_alignment_failure_analysis.json` under
`regressions` (19) and `rescues` (24).
