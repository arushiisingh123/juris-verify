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

THREE EVIDENCE MODES
--------------------
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

evidence_mode="section" (EXPERIMENTAL, OPT-IN -- Task C)
    Identical machinery to "multi", plus a structure-aware prior on
    candidate ranking: a candidate's claim relevance is adjusted by which
    scientific section it came from (see src/sections.py), favouring
    Results/Findings/Conclusions sentences over Background/Objective/
    Introduction ones.

    Rationale: the same error analysis shows the extractor repeatedly picks
    a Background/Objective sentence that restates the research question in
    the claim's own wording, which cosine similarity rewards but which
    cannot resolve the claim either way. Section provenance is a cheap,
    interpretable signal for "this sentence reports a finding".

    "section" is a superset knob, not a fourth code path: with
    max_sentences=1 it is section-aware *single*-sentence selection, and
    with max_sentences=2 it is Multi-2 plus section-aware ranking. The
    planned ablation (A single / B multi-2 / C section-aware single /
    D multi-2 + section-aware) is therefore two orthogonal knobs over one
    implementation. Setting section_weight=0.0 reduces it exactly to
    "multi".

    EXPERIMENTAL STATUS: this is a measurement tool, not a validated
    result. No accuracy claim attaches to it until the Colab ablation runs.
"""

import re
from sentence_transformers import util

from src.sections import assign_sections, section_prior


# --- configuration defaults -------------------------------------------------
# Baseline stays the default everywhere; "multi"/"section" must be requested
# explicitly.
DEFAULT_EVIDENCE_MODE = "single"
EVIDENCE_MODES = ("single", "multi", "section")

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

# Magnitude of the section-aware prior for evidence_mode="section", applied on
# the cosine scale as:
#
#     ranking_score(i) = cosine(claim, sentence_i)
#                        + section_weight * section_prior(section_i)
#
# where section_prior is the fixed ordinal table in src/sections.py
# (+1.0 results/conclusions, 0.0 unknown, -0.5 methods, -1.0 background).
#
# Why 0.05: MiniLM claim-to-sentence cosines on SciFact abstracts sit in a
# fairly narrow band (roughly 0.3-0.65), so the gap between a Background
# sentence that parrots the claim and the Results sentence that answers it is
# typically well under 0.1. A weight of 0.05 moves a Results/Conclusions
# candidate by +0.05 and a Background candidate by -0.05, i.e. it can flip a
# gap of up to 0.1 -- enough to reorder those near-ties, but far too small to
# promote a genuinely irrelevant sentence over a relevant one. This default is
# a starting point to be swept in the ablation, not a tuned value.
DEFAULT_SECTION_WEIGHT = 0.05

# Allowed range for section_weight. 0.0 disables the prior exactly (reducing
# "section" to "multi"); negative values invert it, which is a useful
# anti-prior control arm for the ablation.
MIN_SECTION_WEIGHT = -1.0
MAX_SECTION_WEIGHT = 1.0

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

    Each entry also carries its scientific section (src/sections.py). Section
    labels are computed over the document's FULL sentence list before
    deduplication, so a heading-bearing sentence that gets dropped as a
    cross-document duplicate still advances the sticky carry-forward for the
    sentences that follow it. Labels never cross a document boundary.

    Returns:
        list: [{"sentence": str, "doc_id": Any, "doc_rank": int,
                "sentence_index": int, "section": str}, ...] in source order.
        Exact duplicate sentence texts are dropped (first occurrence wins) so
        a sentence repeated across retrieved documents cannot fill the bundle
        twice.
    """
    pool = []
    seen_texts = set()

    for doc_rank, candidate in enumerate(candidates):
        text = candidate.get("text", "")
        doc_id = candidate.get("doc_id")

        doc_sentences = split_sentences(text)
        doc_sections = assign_sections(doc_sentences)

        for sentence_index, sentence in enumerate(doc_sentences):
            if sentence in seen_texts:
                continue
            seen_texts.add(sentence)
            pool.append(
                {
                    "sentence": sentence,
                    "doc_id": doc_id,
                    "doc_rank": doc_rank,
                    "sentence_index": sentence_index,
                    "section": doc_sections[sentence_index],
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


def _resolved_bundle_mode(section_weight):
    """Which ranking was actually applied, for the bundle's "evidence_mode".

    Reports what ran rather than what was requested, so an artifact produced
    with section_weight=0.0 is labelled "multi" -- which is exactly what it is,
    since a zero weight is a no-op. The mode the caller *asked* for is recorded
    separately in the eval metadata, so nothing is lost.
    """
    return "multi" if section_weight == 0.0 else "section"


def _empty_bundle(section_weight=0.0):
    return {
        "sentence": "",
        "doc_id": None,
        "evidence_mode": _resolved_bundle_mode(section_weight),
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
    section_weight=0.0,
):
    """Select a small, complementary evidence bundle for `claim`.

    Greedy MMR over the pooled candidate sentences:
      rank 0       = argmax ranking_score(i)
      rank k (k>0) = argmax  mmr_lambda * ranking_score(i)
                            - (1 - mmr_lambda) * max_over_selected_j cosine(i, j)

    where the ranking score folds in the section-aware prior:

      ranking_score(i) = relevance(i) + section_weight * section_prior(i)
      relevance(i)     = cosine(claim, sentence_i)
      section_prior(i) = fixed ordinal table in src/sections.py, i.e.
                         +1.0 results / conclusions
                          0.0 unknown  (no heading recovered)
                         -0.5 methods
                         -1.0 background / objective / introduction

    With the default section_weight=0.0 the prior term is skipped entirely and
    ranking_score IS relevance, making this function bit-for-bit the
    pre-Task-C MMR implementation. Only evidence_mode="section" passes a
    non-zero weight.

    Two deliberate scoping choices in how the prior is applied:
      * It applies at rank 0 as well, not only to the MMR extensions --
        otherwise section-aware *single*-sentence selection (ablation arm C)
        would be impossible.
      * The `min_relevance` floor is tested against the RAW cosine relevance,
        never the section-adjusted score. The floor is a statement about
        topical relevance; a section bonus must not smuggle an off-topic
        sentence past it, and a section penalty must not evict an on-topic one.

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
        section_weight: magnitude of the section prior, in
            [MIN_SECTION_WEIGHT, MAX_SECTION_WEIGHT]. 0.0 (default) disables
            it exactly; negative inverts it (anti-prior control arm).

    Returns:
        dict:
          "sentence": the bundle as one deterministic string (sentences in
              source order, joined by a single space) -- a drop-in premise for
              the existing verifier, which expects a string.
          "doc_id": doc_id of the top-ranked (rank-0) sentence, so downstream
              single-citation consumers keep working.
          "evidence_mode": "multi" when section_weight is 0.0, else "section"
              -- i.e. the ranking that actually ran.
          "spans": per-sentence provenance, in source order:
              [{"sentence", "doc_id", "relevance", "section",
                "section_prior", "ranking_score", "rank"}, ...]
              where "relevance" stays the raw cosine (unchanged meaning) and
              "ranking_score" is the section-adjusted value actually ranked on.
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
    if isinstance(section_weight, bool) or not isinstance(section_weight, (int, float)):
        raise ValueError("section_weight must be a number")
    if not MIN_SECTION_WEIGHT <= section_weight <= MAX_SECTION_WEIGHT:
        raise ValueError(
            "section_weight must be in ["
            + str(MIN_SECTION_WEIGHT) + ", " + str(MAX_SECTION_WEIGHT)
            + "] (got " + str(section_weight) + ")"
        )
    section_weight = float(section_weight)

    pool = _collect_candidate_sentences(_normalize_candidates(candidates))
    if not pool:
        return _empty_bundle(section_weight)

    sentences = [item["sentence"] for item in pool]

    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    relevance = util.cos_sim(claim_embedding, sentence_embeddings)[0].tolist()
    pairwise = util.cos_sim(sentence_embeddings, sentence_embeddings).tolist()

    # Section-aware ranking score. At section_weight == 0.0 we reuse the
    # `relevance` list object itself rather than recomputing r + 0.0, so the
    # "multi" path does strictly the same arithmetic it did before Task C.
    priors = [section_prior(item["section"]) for item in pool]
    if section_weight == 0.0:
        ranking_score = relevance
    else:
        ranking_score = [
            relevance[idx] + section_weight * priors[idx] for idx in range(len(pool))
        ]

    k = min(max_sentences, len(pool))

    # Rank 0: with section_weight == 0.0 this is the identical choice to the
    # single-sentence baseline. With a non-zero weight it is the section-aware
    # top sentence (ablation arm C at max_sentences=1).
    selected = [_argmax(ranking_score)]

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
                mmr_lambda * ranking_score[idx] - (1.0 - mmr_lambda) * redundancy
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
            "section": pool[idx]["section"],
            "section_prior": priors[idx],
            "ranking_score": float(ranking_score[idx]),
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
        "evidence_mode": _resolved_bundle_mode(section_weight),
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
    section_weight=DEFAULT_SECTION_WEIGHT,
):
    """Select evidence for `claim` from candidate documents.

    Args:
        claim: the claim string to find evidence for
        candidates: list of {"doc_id": ..., "text": ...} dicts (or a single candidate dict)
        embedder: a SentenceTransformer instance (reuse the retriever's
            embedding_model to avoid loading the model twice)
        evidence_mode: "single" (default, validated baseline), "multi"
            (experimental complementary bundle) or "section" (experimental
            bundle with section-aware ranking) -- see extract_evidence_bundle
        max_sentences, mmr_lambda, min_relevance: "multi"/"section" modes only;
            ignored when evidence_mode="single"
        section_weight: "section" mode only. Ignored by both "single" and
            "multi", so neither of those paths can be perturbed by it.

    Returns:
        dict: with evidence_mode="single", exactly {"sentence": str, "doc_id": Any}
        -- unchanged from the validated baseline -- where sentence is guaranteed
        to be a verbatim substring of the source document with the given doc_id.
        With evidence_mode="multi" or "section", the richer bundle dict
        documented in extract_evidence_bundle (a superset of those two keys).
    """
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError(
            "evidence_mode must be one of " + str(EVIDENCE_MODES)
            + " (got " + repr(evidence_mode) + ")"
        )

    if evidence_mode in ("multi", "section"):
        return extract_evidence_bundle(
            claim,
            candidates,
            embedder,
            max_sentences=max_sentences,
            mmr_lambda=mmr_lambda,
            min_relevance=min_relevance,
            # "multi" is pinned to 0.0 so the Task B implementation it names
            # stays exactly what it was; only "section" reads the knob.
            section_weight=0.0 if evidence_mode == "multi" else section_weight,
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
