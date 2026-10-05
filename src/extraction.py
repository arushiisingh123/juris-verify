"""
Evidence extraction for Juris-Verify.

IMPORTANT: A generative extractor (Qwen2.5-7B-Instruct, prompted to
"copy the sentence verbatim") was tried first and rejected -- it
paraphrased/generated from its own knowledge instead of extracting,
which silently breaks citation-grounding (the extracted "evidence"
would not actually appear in the source document).

This embedding-based sentence-ranking extractor replaced it and is
the validated approach: split the doc into sentences, embed each,
cosine-sim against the claim, take the top-scoring sentence. Because
it's a selection over existing text (not generation), the output is
always a substring of the document -- which is exactly what
citation-grounding requires.

TWO EVIDENCE MODES
------------------
evidence_mode="single" (DEFAULT, VALIDATED BASELINE)
    One best sentence across the candidate documents. This is the code
    path used for the N=504 SciFact run (78.97% accuracy) and is left
    unchanged. Nothing selects "multi" implicitly.

evidence_mode="multi" (EXPERIMENTAL, OPT-IN)
    A small complementary evidence bundle (1..MAX_BUNDLE_SENTENCES
    sentences) selected greedily with Maximal Marginal Relevance (MMR):
    relevance to the claim minus redundancy against already-selected
    sentences. Motivation: the full N=504 error analysis attributes ~84%
    of missed contradictions to evidence extraction, i.e. the single
    top-similarity sentence often is not the sentence carrying the
    conflicting quantity/direction. This mode is a measurement tool for
    that hypothesis -- it is not itself a validated result.

    Selection rank 0 is always the sentence baseline mode would have
    returned, so a "multi" bundle is a superset of the baseline evidence
    and the two modes stay directly comparable.
"""

import re
from sentence_transformers import util


# --- configuration defaults -------------------------------------------------
# Baseline stays the default everywhere; "multi" must be requested explicitly.
DEFAULT_EVIDENCE_MODE = "single"
EVIDENCE_MODES = ("single", "multi")

# Research scope for Task B: small bundles only (2-4 sentences), so the NLI
# premise stays short enough for the verifier to stay precise.
MAX_BUNDLE_SENTENCES = 4
DEFAULT_MAX_SENTENCES = 3

# MMR trade-off: 1.0 = pure claim relevance (degenerates toward top-k),
# 0.0 = pure diversity. 0.7 keeps relevance dominant while still rewarding
# sentences that add something the already-selected ones do not.
DEFAULT_MMR_LAMBDA = 0.7

# Sentences below this cosine similarity to the claim are not eligible as
# *additional* evidence (the rank-0 baseline sentence is always kept).
DEFAULT_MIN_RELEVANCE = 0.0

# Deterministic textual representation handed to the verifier.
EVIDENCE_JOIN = " "


def split_sentences(document):
    """Split a document into sentences, dropping short fragments and
    the leading title (SciFact docs are 'title + abstract').

    NOTE: Title stripping is applied per-document. In the legacy
    single-string implementation, only Doc 1's title was dropped,
    allowing Doc 2's title to leak into sentence candidate pools (e.g.
    Claim #36 where Doc 2's title was previously extracted).
    """
    sentences = re.split(r"(?<=[.!?])\s+", document)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 1:
        sentences = sentences[1:]  # drop title

    return sentences


def _normalize_candidates(candidates):
    """Accept a single candidate dict or a list of them; return a list."""
    if isinstance(candidates, dict):
        return [candidates]
    if not candidates:
        return []
    return list(candidates)


def _collect_candidate_sentences(candidates):
    """Flatten candidate documents into a provenance-carrying sentence pool.

    Returns:
        list: [{"sentence": str, "doc_id": Any, "doc_rank": int,
                "sentence_index": int}, ...] in source order. Exact duplicate
        sentence texts are dropped (first occurrence wins) so a sentence
        repeated across retrieved documents cannot fill the bundle twice.
    """
    pool = []
    seen_texts = set()

    for doc_rank, candidate in enumerate(candidates):
        text = candidate.get("text", "")
        doc_id = candidate.get("doc_id")

        for sentence_index, sentence in enumerate(split_sentences(text)):
            if sentence in seen_texts:
                continue
            seen_texts.add(sentence)
            pool.append(
                {
                    "sentence": sentence,
                    "doc_id": doc_id,
                    "doc_rank": doc_rank,
                    "sentence_index": sentence_index,
                }
            )

    return pool


def _argmax(values, eligible=None):
    """Deterministic argmax: strictly-greater comparison scanning in index
    order, so ties always resolve to the lowest index. Returns None when no
    index is eligible.
    """
    best_idx = None
    best_val = None
    for idx, val in enumerate(values):
        if eligible is not None and idx not in eligible:
            continue
        if best_val is None or val > best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _empty_bundle():
    return {
        "sentence": "",
        "doc_id": None,
        "evidence_mode": "multi",
        "spans": [],
        "doc_ids": [],
    }


def extract_evidence_bundle(
    claim,
    candidates,
    embedder,
    max_sentences=DEFAULT_MAX_SENTENCES,
    mmr_lambda=DEFAULT_MMR_LAMBDA,
    min_relevance=DEFAULT_MIN_RELEVANCE,
):
    """Select a small, complementary evidence bundle for `claim`.

    Greedy MMR over the pooled candidate sentences:
      rank 0       = argmax cosine(claim, sentence)   [== baseline evidence]
      rank k (k>0) = argmax  mmr_lambda * relevance(i)
                            - (1 - mmr_lambda) * max_over_selected_j cosine(i, j)

    Fully deterministic: no sampling, and ties resolve to the lowest pool
    index (i.e. earlier document, earlier sentence).

    Args:
        claim: the claim string to find evidence for
        candidates: list of {"doc_id": ..., "text": ...} dicts (or a single dict)
        embedder: a SentenceTransformer instance (reuse the retriever's
            embedding_model to avoid loading the model twice)
        max_sentences: bundle size cap, 1..MAX_BUNDLE_SENTENCES
        mmr_lambda: relevance/diversity trade-off in [0.0, 1.0]
        min_relevance: minimum claim similarity for *additional* sentences

    Returns:
        dict:
          "sentence": the bundle as one deterministic string (sentences in
              source order, joined by a single space) -- a drop-in premise for
              the existing verifier, which expects a string.
          "doc_id": doc_id of the rank-0 (most claim-relevant) sentence, so
              downstream single-citation consumers keep working.
          "evidence_mode": "multi"
          "spans": per-sentence provenance, in source order:
              [{"sentence", "doc_id", "relevance", "rank"}, ...]
          "doc_ids": unique doc_ids contributing to the bundle, in source order.

        Every span's "sentence" remains a verbatim substring of the document
        carrying that span's doc_id.
    """
    if isinstance(max_sentences, bool) or not isinstance(max_sentences, int):
        raise ValueError("max_sentences must be an int")
    if not 1 <= max_sentences <= MAX_BUNDLE_SENTENCES:
        raise ValueError(
            "max_sentences must be between 1 and "
            + str(MAX_BUNDLE_SENTENCES)
            + " (got " + str(max_sentences) + ")"
        )
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("mmr_lambda must be in [0.0, 1.0] (got " + str(mmr_lambda) + ")")

    pool = _collect_candidate_sentences(_normalize_candidates(candidates))
    if not pool:
        return _empty_bundle()

    sentences = [item["sentence"] for item in pool]

    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    relevance = util.cos_sim(claim_embedding, sentence_embeddings)[0].tolist()
    pairwise = util.cos_sim(sentence_embeddings, sentence_embeddings).tolist()

    k = min(max_sentences, len(pool))

    # Rank 0: identical choice to the single-sentence baseline.
    selected = [_argmax(relevance)]

    while len(selected) < k:
        chosen = set(selected)
        mmr_scores = []
        eligible = set()
        for idx in range(len(pool)):
            if idx in chosen or relevance[idx] < min_relevance:
                mmr_scores.append(None)
                continue
            redundancy = max(pairwise[idx][j] for j in selected)
            mmr_scores.append(
                mmr_lambda * relevance[idx] - (1.0 - mmr_lambda) * redundancy
            )
            eligible.add(idx)

        next_idx = _argmax(mmr_scores, eligible=eligible)
        if next_idx is None:
            break  # nothing left that clears min_relevance
        selected.append(next_idx)

    primary_idx = selected[0]
    rank_by_idx = {idx: rank for rank, idx in enumerate(selected)}

    # Present in source order so the joined premise reads like the source,
    # while "rank" preserves selection order (0 == baseline sentence).
    ordered = sorted(
        selected, key=lambda idx: (pool[idx]["doc_rank"], pool[idx]["sentence_index"])
    )

    spans = [
        {
            "sentence": pool[idx]["sentence"],
            "doc_id": pool[idx]["doc_id"],
            "relevance": float(relevance[idx]),
            "rank": rank_by_idx[idx],
        }
        for idx in ordered
    ]

    doc_ids = []
    for span in spans:
        if span["doc_id"] not in doc_ids:
            doc_ids.append(span["doc_id"])

    return {
        "sentence": EVIDENCE_JOIN.join(span["sentence"] for span in spans),
        "doc_id": pool[primary_idx]["doc_id"],
        "evidence_mode": "multi",
        "spans": spans,
        "doc_ids": doc_ids,
    }


def extract_best_evidence(
    claim,
    candidates,
    embedder,
    evidence_mode=DEFAULT_EVIDENCE_MODE,
    max_sentences=DEFAULT_MAX_SENTENCES,
    mmr_lambda=DEFAULT_MMR_LAMBDA,
    min_relevance=DEFAULT_MIN_RELEVANCE,
):
    """Select evidence for `claim` from candidate documents.

    Args:
        claim: the claim string to find evidence for
        candidates: list of {"doc_id": ..., "text": ...} dicts (or a single candidate dict)
        embedder: a SentenceTransformer instance (reuse the retriever's
            embedding_model to avoid loading the model twice)
        evidence_mode: "single" (default, validated baseline) or "multi"
            (experimental complementary bundle, see extract_evidence_bundle)
        max_sentences, mmr_lambda, min_relevance: "multi" mode only; ignored
            when evidence_mode="single"

    Returns:
        dict: with evidence_mode="single", exactly {"sentence": str, "doc_id": Any}
        -- unchanged from the validated baseline -- where sentence is guaranteed
        to be a verbatim substring of the source document with the given doc_id.
        With evidence_mode="multi", the richer bundle dict documented in
        extract_evidence_bundle (a superset of those two keys).
    """
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError(
            "evidence_mode must be one of " + str(EVIDENCE_MODES)
            + " (got " + repr(evidence_mode) + ")"
        )

    if evidence_mode == "multi":
        return extract_evidence_bundle(
            claim,
            candidates,
            embedder,
            max_sentences=max_sentences,
            mmr_lambda=mmr_lambda,
            min_relevance=min_relevance,
        )

    # --- validated single-sentence baseline: unchanged below this line ---
    if isinstance(candidates, dict):
        candidates = [candidates]
    elif not candidates:
        return {"sentence": "", "doc_id": None}

    all_sentences = []
    all_doc_ids = []

    for candidate in candidates:
        text = candidate.get("text", "")
        doc_id = candidate.get("doc_id")

        sentences = split_sentences(text)
        for s in sentences:
            all_sentences.append(s)
            all_doc_ids.append(doc_id)

    if not all_sentences:
        return {"sentence": "", "doc_id": None}

    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(all_sentences, convert_to_tensor=True)

    scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax().item()

    return {
        "sentence": all_sentences[best_idx],
        "doc_id": all_doc_ids[best_idx],
    }
