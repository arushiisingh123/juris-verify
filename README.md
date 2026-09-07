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

   *Evidenced Single-Sentence Extraction Bottleneck:* An audit of the gold
   `CONTRADICT` claims that received `PARTIALLY_SUPPORTED` revealed that exactly
   **50% (3 of 6 in pilot audit)** failed solely because cosine-similarity sentence
   extraction selected a high-overlap introductory/objective sentence instead of a
   conclusive finding elsewhere in the document (e.g. Claim #138 in Doc 26016929,
   Claim #122 in Doc 3272084, and Claim #112 in Doc 6157837). In each case,
   evaluating the document's conclusion or result sentences with the existing
   verifier yielded a decisive `CONTRADICTED` verdict.
7. **Scale the eval set.** (Implemented in `eval/evaluate.py`) Configurable
   evaluation harness scaled to the full SciFact unique training set ($N=504$,
   331 SUPPORT / 173 CONTRADICT) with support for stratified sampling
   (`--stratified`), deterministic random seed (`--seed 42`), execution wall-time
   benchmarking, and timestamped JSON artifact export (`eval/results_20260907_095418_n504.json`).
   - **Full-scale Accuracy:** 78.97% (398/504 total; SUPPORT: 93.05% [308/331],
     CONTRADICT: 52.02% [90/173]). The performance delta from the pilot is
     concentrated in contradiction detection, precisely matching the extraction
     bottleneck identified above.
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
  extraction.py     - embedding-based sentence extractor
  verification.py  - NLIVerifier + label mapping
  pipeline.py       - orchestrates retrieve -> extract -> verify
eval/
  evaluate.py       - reproduces the 88% accuracy result + error analysis
```
