# Juris-Verify

Adversarial Agentic RAG for contradiction-aware summarization — Legal & Scientific meta-analysis.

Migrated from an exploratory Colab notebook (`Juris_Verify_MVP.ipynb`, kept in
`/legacy` for reference only — do not build on top of it directly, it's full of
dead-end experiments). The code in `src/` and `eval/` is the cleaned, validated
version of what actually worked.

## Current state (Phase 1 — Scientific track, SciFact)

Working, validated pipeline:

```
Claim -> HybridRetriever (semantic + BM25 -> CrossEncoder rerank)
      -> extract_best_evidence (embedding sentence-ranking extractor)
      -> NLIVerifier (DeBERTa-v3-large-mnli)
      -> label (SUPPORT / CONTRADICT / NEI)
```

**Validated result:** 78.97% accuracy (398/504) on the full SciFact unique
labeled population (`eval/evaluate.py`), with Strict Faithfulness of 0.1766
and Weighted Faithfulness of 0.4762.

*Note on Earlier Figures:* The earlier 86% figure (43/50) and 88% prototype
figure were measured on an unstratified first-50-claims slice (`unique_claims[:50]`)
in dataset order, which contained an unrepresentatively easy distribution. The
78.97% figure across the entire 504-claim population (331 SUPPORT / 173 CONTRADICT)
is the first unbiased, validated reference benchmark for this pipeline.

### Design decisions worth knowing before touching this code

- **Evidence extraction is NOT generative.** A Qwen2.5-7B-Instruct extractor
  was tried first and rejected — it paraphrased from its own knowledge
  instead of quoting the document verbatim, which silently breaks
  citation-grounding. The current extractor (`src/extraction.py`) selects an
  existing sentence via embedding similarity, so its output is always a
  verbatim substring of the source doc.
- **Per-document title stripping:** SciFact corpus texts are formatted as
  `title + abstract`. Sentence extraction cleans titles per-document. (In the
  pre-refactor single-string version, Doc 2's title leaked through into candidate
  pools — e.g. Claim #36, which previously extracted a title string that happened
  to score a correct verdict).
- **The verifier is a dedicated NLI model, not a generative LLM.** FLAN-T5,
  Mistral-7B-Instruct, and Qwen2.5-7B were all tried as verifiers and
  rejected (too weak / over-predicted "partially supported" on clear-cut
  cases). `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` is what's
  scoring 78.97% overall (93.05% on SUPPORT, 52.02% on CONTRADICT) — don't swap
  the verifier without re-running `eval/evaluate.py` and comparing.

## Experimental evidence selection (opt-in, not yet validated)

The validated pipeline above picks **one** sentence: the highest cosine
similarity to the claim. `eval/evaluate.py` exposes two opt-in experimental
alternatives to that choice. Both are **measurement tools, not results** —
nothing below has been validated, and `evidence_mode` defaults to `single`
everywhere, so the validated baseline is what you get unless you ask otherwise.

### Why touch evidence selection at all

The $N=504$ audit (item #6 below) attributes **84% of missed contradictions to
evidence extraction, not to the verifier**: the retrieved document *did* contain
a sentence the existing `NLIVerifier` scores as `CONTRADICTED`, but
`extract_best_evidence` returned a different one. That makes selection, not
verification, the place to spend effort on CONTRADICT accuracy.

### Task B — Multi-sentence MMR bundle (`evidence_mode="multi"`)

Greedy Maximal Marginal Relevance over the pooled candidate sentences:
relevance to the claim minus redundancy against what is already selected,
`mmr_lambda=0.7`, bundle size 1–4. Rank 0 is always the sentence `single` would
have returned, so a bundle is a superset of the baseline evidence and the two
stay directly comparable. See `src/extraction.py`.

### Task C — Section-aware ranking (`evidence_mode="section"`)

**Hypothesis.** The 84% extraction failures are not random — they are
*structurally* biased. A sentence like

> `OBJECTIVE To determine whether high doses of folic acid and B vitamins`
> `administered daily reduce mortality in patients with chronic kidney disease.`

restates the research question in almost the claim's own words, so cosine
similarity loves it, while carrying no finding at all and therefore no way to
resolve the claim either way. The sentence that *does* conflict with the claim
lives under `RESULTS` or `CONCLUSIONS` and is phrased in outcome language, which
scores lower on surface similarity. If that is right, then knowing **which
section a candidate came from** should be enough to break the tie in favour of
the sentence that can actually settle the claim — with no new model, no new
dependency, and no change to the verifier.

**Does SciFact expose section headings?** Not as a metadata field — the HF
`allenai/scifact` corpus schema has no per-sentence section label, and
`build_document_corpus` flattens each record to `title + abstract`. But its
structured abstracts carry headings *inline*, as an ALL-CAPS prefix on the
sentence that opens each section. Verbatim from
`eval/results_20260907_095418_n504.json`:

```
OBJECTIVE To determine whether high doses of folic acid and B vitamins ...
RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP ...
DESIGN, SETTING, AND PATIENTS Retrospective analysis of German and US cohorts ...
CONCLUSIONS A dosage of folic acid of 0.8 mg/d appears necessary ...
```

So `src/sections.py` reads the headings the documents really do expose. It is a
deterministic, documented heuristic — no model, no learned component, no
per-claim special-casing:

1. Take the sentence's leading run of **upper-case** tokens (trailing `:`/`,`
   stripped, max 6 tokens). Upper case is required, so `Results showed ...` is
   not a heading and `RESULTS Of the ...` is.
2. Longest-match that run against a **closed vocabulary** of ~90 structured-
   abstract headings, shortening from the right until a phrase is known.
   `METHODS AND RESULTS ...` binds as the 3-token compound (→ results), and
   `CONCLUSIONS A dosage ...` falls back to the 1-gram (→ conclusions).
3. No match → the sentence does not open a section and **inherits** the most
   recent heading seen earlier in the same document (sticky carry-forward).
   Only the first sentence of a section carries its heading, so carry-forward is
   what labels the rest of it. Labels never cross a document boundary.
4. Sentences before any heading — and every sentence of an unstructured
   abstract — are `unknown`.

The vocabulary is deliberately **closed** rather than "any ALL-CAPS prefix":
auditing the $N=504$ artifact found `PTEN`, `IFI16`, `IL-10`, `IL-18`, `AMPK`,
`CD28`, `EBI2`, `FGF21`, `DUSP4`, `PDPN`, `HAND2`, `DNA` and `FISH` all opening
sentences and all matching a permissive rule. The closed vocabulary rejects
every one, at the cost of returning `unknown` for an unlisted heading — the safe
direction, since `unknown` carries a zero prior and therefore cannot change the
ranking.

*Known limitation:* `split_sentences` drops fragments of ≤20 characters, so a
corpus that put headings on their own short line (rather than inline, as SciFact
does) would lose them and fall back to carry-forward / `unknown`.

**How section information modifies ranking.** One additive term on the cosine
scale, and nothing else:

```
ranking_score(i) = cosine(claim, sentence_i) + section_weight * section_prior(i)
```

`section_prior` is a fixed ordinal table (`src/sections.py`) carrying the
*direction* of the preference; `section_weight` (default **0.05**) is the single
scalar carrying its *magnitude*:

| section | prior | adjustment at `section_weight=0.05` | reasoning |
|---|---|---|---|
| `results`, `conclusions` | **+1.0** | +0.05 | reports an outcome; a contradiction of a factual claim is almost always stated here |
| `unknown` | **0.0** | 0.00 | no heading recovered — stay neutral, never penalize a document for being unstructured |
| `methods` | **−0.5** | −0.025 | high topical overlap, low chance of resolving the claim; penalized less than background because it at least states concrete study facts |
| `background` | **−1.0** | −0.05 | Background / Objective / Introduction — the documented worst case: restates the question in the claim's own words |

*Why 0.05:* MiniLM claim-to-sentence cosines on SciFact abstracts sit in a
narrow band (~0.3–0.65), so the gap between a Background sentence that parrots
the claim and the Results sentence that answers it is typically well under 0.1.
A weight of 0.05 gives a ±0.1 swing between the extreme sections — enough to
reorder those near-ties, far too small to promote a genuinely irrelevant
sentence. It is a documented starting point to be swept, **not a tuned value**.

Three deliberate scoping choices:

- The prior applies at **rank 0**, not only to the MMR extensions — otherwise
  section-aware *single*-sentence selection (arm C) would be impossible.
- The `min_relevance` floor is tested against the **raw cosine**, never the
  adjusted score: a section bonus must not smuggle an off-topic sentence past
  the floor, and a penalty must not evict an on-topic one.
- `section_weight=0.0` is an **exact** no-op (the prior term is skipped, not
  computed as `+0.0`), so `section` reduces bit-for-bit to `multi`. That is what
  makes it a valid control arm. `evidence_mode="multi"` is pinned to `0.0`
  internally and cannot be perturbed by the knob; `evidence_mode="single"`
  ignores it entirely.

Bundles report the scoring per span (`section`, `section_prior`,
`ranking_score`, plus the unchanged raw `relevance`), and `eval/evaluate.py`
records `primary_evidence_section` per claim for experimental modes only — so
any ranking decision can be read off the artifact without re-running extraction.
Baseline (`single`) records keep exactly the key set they had before.

### Intended ablation (to be run in Colab, not locally)

`section` is a superset knob rather than a fourth code path: with
`--max_sentences 1` it is section-aware *single*-sentence selection, and with
`--max_sentences 2` it is Multi-2 plus section-aware ranking. All four arms are
therefore two orthogonal knobs over one implementation:

| arm | description | command |
|---|---|---|
| **A** | Single baseline (validated reference) | `--evidence_mode single` |
| **B** | Multi-2 (MMR, no section prior) | `--evidence_mode multi --max_sentences 2` |
| **C** | Section-aware single | `--evidence_mode section --max_sentences 1 --section_weight 0.05` |
| **D** | Multi-2 + section-aware ranking | `--evidence_mode section --max_sentences 2 --section_weight 0.05` |

Primary metric: **CONTRADICT accuracy**. Regression checks: SUPPORT accuracy and
overall accuracy — a CONTRADICT gain that costs more SUPPORT accuracy than it
wins is not an improvement. Strict / Weighted Faithfulness are reported
unchanged; the evidence mode only selects how the premise is extracted before
the identical verifier and the identical faithfulness computation run on it,
which is what makes the arms comparable.

Secondary sweeps worth running once A–D are in: `section_weight` ∈
{0.0, 0.025, 0.05, 0.10} — where 0.0 must reproduce A/B exactly, a correctness
check on the harness rather than a result — and `section_weight=-0.05` as an
**anti-prior control**: if inverting the prior does *not* hurt CONTRADICT
accuracy, the section signal is not doing what this hypothesis claims.

Artifacts are named by arm (`results_<ts>_n504_section2_w0.05.json`), and the
tracked `eval/results_n50.json` reference is only ever overwritten by
`--evidence_mode single`, so experimental runs cannot be mistaken for the
validated baseline.

### What is NOT yet known

**No accuracy claim and no novelty claim attaches to any of this.** The only
evidence in hand is *observational*, and correlational by construction —
re-running `src/sections.py` over the baseline's own selected evidence in
`eval/results_20260907_095418_n504.json`, grouped by the section of whatever
sentence the baseline happened to pick (91 of 504 selected sentences carry a
detectable heading):

| section of selected evidence | n | overall acc | CONTRADICT n | CONTRADICT acc |
|---|---|---|---|---|
| `results` / `conclusions` | 39 | 87.2% | 21 | **81.0%** |
| `methods` | 14 | 64.3% | 6 | 16.7% |
| `background` | 38 | 57.9% | 22 | **27.3%** |

That is consistent with the hypothesis and is why the feature was built, but it
**cannot** be read as the feature working: the selection was not randomized, the
groups differ in claim difficulty and document structure, and a claim whose
answer happens to sit in a Results sentence may simply be an easier claim.
Whether *steering* selection toward those sections actually moves CONTRADICT
accuracy is exactly the open question arms C and D exist to answer. Until that
experiment runs, `single` remains the only validated mode.

## What's NOT built yet — the actual next-phase work

1. **LangGraph orchestration.** `langgraph` is a dependency but nothing uses
   it yet. Need an actual state graph wiring Synthesizer → Prosecutor →
   Adjudicator as a loop, not the current single-pass classifier.
2. **Adjudicator + rewrite loop.** When the verifier returns CONTRADICT, feed
   the evidence span back to a Synthesizer LLM and force a rewrite with an
   explicit "Conflict Footnote" citing the contradicting source. Nothing
   currently loops back — it just returns a label.
3. **Persisted Contradiction Matrix.** Structure verifier output into
   `{claim_id, source_doc_id, label, confidence, evidence_span, timestamp}`
   records instead of returning one-off dicts. Should be queryable.
4. **Citation URIs.** (Implemented) Extracted evidence preserves and tags
   source `doc_id` throughout retrieval, sentence extraction, LangGraph
   orchestration, and persisted Contradiction Matrix records.
5. **Phase 3 — legal track.** Not started. Plan is CaseHOLD first (most
   tractable), then COLIEE, then EUR-Lex.
6. **Faithfulness scoring (SummaC/FactCC NLI method).** (Implemented in
   `eval/evaluate.py`) Dual factual consistency metrics reported alongside
   accuracy, broken out by gold label (SUPPORT / CONTRADICT / NEI) at $N=504$:
   - **Strict Faithfulness:** Binary entailment-only (SUPPORTED = 1.0, else 0.0).
     Overall: **0.1766** (SUPPORT: 0.2598, CONTRADICT: 0.0173).
   - **Weighted Faithfulness:** 3-way partial credit reflecting the project's
     AGREE / SCOPE-MISMATCH / CONTRADICT taxonomy (SUPPORTED = 1.0,
     PARTIALLY_SUPPORTED = 0.5, CONTRADICTED = 0.0).
     Overall: **0.4762** (SUPPORT: 0.5952, CONTRADICT: 0.2486).
   Reuses local `NLIVerifier` over atomic sub-statements without external LLM API
   dependencies.

   *General Limitation:* Atomic statements are evaluated against a single top
   evidence sentence (`extract_best_evidence`); multi-sentence support may
   understate faithfulness for multi-assertion claims.

   *Evidenced Single-Sentence Extraction Bottleneck vs. NLI Reasoning Limits:* An
   empirical audit of a randomized sample of 25 claims (`random.sample(seed=42)`)
   drawn from the full population of 80 `PARTIALLY_SUPPORTED` `CONTRADICT` cases at
   scale ($N=504$) revealed two distinct failure modes:
   - **Sentence-Extraction Bottleneck (84.0%, 21 of 25 cases):** The retrieved
     source document contained an explicit contradiction sentence that evaluated
     directly to `CONTRADICTED` under the existing `NLIVerifier`, but
     `extract_best_evidence` selected a high-overlap introductory/objective
     sentence due to cosine similarity bias.
   - **NLI Reasoning Limits (16.0%, 4 of 25 cases):** Even when checking all
     sentences in the source/gold document, every sentence evaluated to neutral
     (`PARTIALLY_SUPPORTED`) due to complex comparative framing, subtle statistical
     inferences, or domain-specific terminology.

   *Note on Earlier Pilot Audit:* An initial audit on the $N=50$ pilot found a
   50/50 split (3 of 6 cases); that estimate was drawn from an unstratified,
   non-random first-50 slice with an unrepresentatively small sample ($n=6$). The
   84/16 split drawn from the full, unbiased 504-claim population is the more
   statistically trustworthy measurement and supersedes the 50/50 figure as the
   validated reference distribution going forward.
7. **Scale the eval set.** (Implemented in `eval/evaluate.py`) Configurable
   evaluation harness scaled to the full SciFact unique training set ($N=504$,
   331 SUPPORT / 173 CONTRADICT) with support for stratified sampling
   (`--stratified`), deterministic random seed (`--seed 42`), execution wall-time
   benchmarking, and timestamped JSON artifact export (`eval/results_20260907_095418_n504.json`).
   - **Full-scale Accuracy:** 78.97% (398/504 total; SUPPORT: 93.05% [308/331],
     CONTRADICT: 52.02% [90/173]). The accuracy drop from the pilot is
     concentrated in contradiction detection, driven by the dual failure modes
     quantified above (84% extraction selection failure, 16% NLI reasoning limits).
   - **Runtime:** 124.62s (~2.08 min, 0.25s/claim on GPU).
   - *Scope Clarification:* `eval/evaluate.py` evaluates the base single-pass
     pipeline (`JurisVerifyPipeline.predict_claim`), measuring standalone
     retrieval, extraction, and verification accuracy and faithfulness across the
     corpus, NOT the LangGraph adjudicator retry loop from item #1 (which handles
     adversarial correction cycles).

## Suggested build order

Start with #1 (LangGraph loop) — everything else (#2, #3) hangs off having
an actual orchestration layer instead of a flat function call chain.

## Repo layout

```
src/
  data_loader.py   - SciFact loading (claims + corpus)
  retrieval.py     - HybridRetriever: semantic + BM25 + rerank
  extraction.py    - sentence extractor: single (validated) / multi / section
  sections.py      - scientific section detection + ranking prior (Task C)
  verification.py  - NLIVerifier + label mapping
  pipeline.py      - orchestrates retrieve -> extract -> verify
  orchestration.py - LangGraph Synthesizer -> Prosecutor -> Adjudicator loop
  matrix.py        - persisted Contradiction Matrix records
eval/
  evaluate.py      - N=504 accuracy + faithfulness harness, error analysis

test_extraction_modes.py        - Task B: single baseline + multi-sentence MMR
test_section_aware_evidence.py  - Task C: section detection + section-aware ranking
```

Both test files are CPU-only and offline (a deterministic fake embedder maps
known sentences onto hand-written unit vectors, so the real `cos_sim` still does
the similarity math):

```
pytest test_extraction_modes.py test_section_aware_evidence.py
```
