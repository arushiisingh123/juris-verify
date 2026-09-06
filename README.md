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

**Validated result:** 86% accuracy on a 50-claim held-out SciFact eval
(`eval/evaluate.py`). Re-measured post-Citation URI refactor via
`compare_old_new_extraction.py` (0 regressions vs baseline; old 88% figure was
an approximation from the exploratory notebook). Scale `N_EVAL_CLAIMS` up
before citing this number anywhere formal.

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
  scoring 86% — don't swap the verifier without re-running `eval/evaluate.py`
  and comparing.

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
   `eval/evaluate.py`) Factual consistency metric reported alongside accuracy,
   broken out by gold label (SUPPORT / CONTRADICT / NEI). Reuses local
   `NLIVerifier` over atomic sub-statements without external LLM API
   dependencies. *Limitation:* Atomic statements are checked against the single
   top evidence sentence; multi-sentence support may understate faithfulness for
   complex multi-assertion claims.
7. **Scale the eval set.** 50 claims is fine for iteration, not for a paper
   headline number.

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
