"""Tests for Task B: contradiction-sensitive multi-sentence evidence extraction.

CPU-only and offline: no SentenceTransformer / NLI model is loaded. A
deterministic FakeEmbedder maps known sentences onto hand-written unit
vectors, so the real `sentence_transformers.util.cos_sim` still does the
similarity math (same code path as production) while the test controls the
geometry exactly.

Scope of these tests:
  - evidence_mode="single" (default) keeps the validated baseline behavior,
    return type, and exact key set;
  - evidence_mode="multi" produces a complementary bundle with provenance;
  - bundle size stays within 2-4 (MAX_BUNDLE_SENTENCES) and degrades safely
    for short documents and empty candidates;
  - selection is deterministic across repeated calls;
  - the pipeline/orchestration wrappers default to baseline.

Run: pytest test_extraction_modes.py
"""

import math

import pytest
import torch

from src.extraction import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_MAX_SENTENCES,
    EVIDENCE_JOIN,
    MAX_BUNDLE_SENTENCES,
    extract_best_evidence,
    extract_evidence_bundle,
    split_sentences,
)


# --------------------------------------------------------------------------
# Deterministic offline embedder
# --------------------------------------------------------------------------

DIM = 4


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class FakeEmbedder:
    """Maps text -> fixed unit vector. Unknown text gets a stable hash-derived
    vector so the embedder never fails, but tests that care about geometry
    register their sentences explicitly.
    """

    def __init__(self, vectors=None):
        self.vectors = dict(vectors or {})
        self.encode_calls = 0

    def register(self, text, vector):
        self.vectors[text] = _unit(vector)

    def _vector(self, text):
        if text in self.vectors:
            return self.vectors[text]
        # Stable fallback: deterministic for a given string, never zero.
        h = abs(hash(text))
        return _unit([1.0 + (h >> (8 * i)) % 7 for i in range(DIM)])

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        self.encode_calls += 1
        if isinstance(texts, str):
            return torch.tensor(self._vector(texts), dtype=torch.float32)
        return torch.tensor(
            [self._vector(t) for t in texts], dtype=torch.float32
        )


# --------------------------------------------------------------------------
# Fixtures: a claim plus a document whose sentences have controlled similarity
# --------------------------------------------------------------------------

CLAIM = "The treatment reduces mortality in elderly patients."

# Sentences are >20 chars (split_sentences drops shorter fragments).
TITLE = "Title line about the treatment and its effects on mortality."
S_SUPPORT = "The treatment reduced mortality among elderly patients overall."
S_SUPPORT_NEAR = "Mortality among elderly patients fell under the treatment regimen."
S_CONFLICT = "However, mortality increased in the elderly subgroup receiving it."
S_METHOD = "Participants were randomized across twelve independent study sites."
S_OFFTOPIC = "Funding statements and author disclosures appear at the end here."

DOC_TEXT = " ".join(
    [TITLE, S_SUPPORT, S_SUPPORT_NEAR, S_CONFLICT, S_METHOD, S_OFFTOPIC]
)


def build_embedder():
    """Unit vectors in 4-D, with axis 0 = the claim direction, chosen so the
    similarity spread resembles what MiniLM actually produces on SciFact
    abstracts (claim relevance ~0.3-0.65, near-duplicates ~0.99 to each other)
    rather than an artificially wide spread:

      sentence         cos(claim, s)   cos(s, S_SUPPORT)
      S_SUPPORT             0.62            1.00   <- baseline pick, rank 0
      S_SUPPORT_NEAR        0.60            1.00   <- 2nd most relevant, but
                                                      redundant: MMR skips it
      S_CONFLICT            0.55            0.35   <- complementary: MMR picks
      S_METHOD              0.30            0.19
      S_OFFTOPIC            0.00            0.78   <- orthogonal to the claim

    With mmr_lambda=0.7 that gives rank-1 scores of 0.126 (S_SUPPORT_NEAR) vs
    0.280 (S_CONFLICT), so the complementary sentence wins on redundancy
    rather than on raw relevance.
    """
    emb = FakeEmbedder()
    emb.register(CLAIM, [1.0, 0.0, 0.0, 0.0])
    emb.register(S_SUPPORT, [0.62, 0.7846, 0.0, 0.0])
    emb.register(S_SUPPORT_NEAR, [0.60, 0.80, 0.0, 0.0])
    emb.register(S_CONFLICT, [0.55, 0.0115, 0.8351, 0.0])
    emb.register(S_METHOD, [0.30, 0.0, 0.0, 0.9539])
    emb.register(S_OFFTOPIC, [0.0, 1.0, 0.0, 0.0])
    emb.register(TITLE, [0.58, 0.40, 0.40, 0.40])
    return emb


@pytest.fixture
def embedder():
    return build_embedder()


@pytest.fixture
def candidates():
    return [{"doc_id": "doc-A", "text": DOC_TEXT}]


# --------------------------------------------------------------------------
# 1. Baseline single-sentence behavior
# --------------------------------------------------------------------------

def test_default_mode_is_single():
    assert DEFAULT_EVIDENCE_MODE == "single"


def test_single_mode_returns_exactly_two_keys(embedder, candidates):
    """Baseline output shape must be unchanged: {"sentence", "doc_id"} only."""
    result = extract_best_evidence(CLAIM, candidates, embedder)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"sentence", "doc_id"}
    assert isinstance(result["sentence"], str)


def test_single_mode_picks_top_similarity_sentence(embedder, candidates):
    result = extract_best_evidence(CLAIM, candidates, embedder)

    assert result["sentence"] == S_SUPPORT
    assert result["doc_id"] == "doc-A"
    # Title is stripped by split_sentences, so it can never be the evidence.
    assert TITLE not in result["sentence"]


def test_single_mode_evidence_is_verbatim_substring(embedder, candidates):
    result = extract_best_evidence(CLAIM, candidates, embedder)
    assert result["sentence"] in DOC_TEXT


def test_single_mode_explicit_flag_matches_default(embedder, candidates):
    implicit = extract_best_evidence(CLAIM, candidates, embedder)
    explicit = extract_best_evidence(
        CLAIM, candidates, embedder, evidence_mode="single"
    )
    assert implicit == explicit


def test_single_mode_ignores_multi_only_knobs(embedder, candidates):
    """max_sentences / mmr_lambda must not affect baseline output."""
    baseline = extract_best_evidence(CLAIM, candidates, embedder)
    with_knobs = extract_best_evidence(
        CLAIM,
        candidates,
        embedder,
        evidence_mode="single",
        max_sentences=4,
        mmr_lambda=0.1,
        min_relevance=0.9,
    )
    assert with_knobs == baseline


def test_single_mode_empty_candidates(embedder):
    for empty in ([], None):
        assert extract_best_evidence(CLAIM, empty, embedder) == {
            "sentence": "",
            "doc_id": None,
        }


def test_single_mode_accepts_single_dict(embedder):
    result = extract_best_evidence(
        CLAIM, {"doc_id": "doc-A", "text": DOC_TEXT}, embedder
    )
    assert result["sentence"] == S_SUPPORT
    assert result["doc_id"] == "doc-A"


def test_invalid_evidence_mode_raises(embedder, candidates):
    with pytest.raises(ValueError):
        extract_best_evidence(CLAIM, candidates, embedder, evidence_mode="top_k")


# --------------------------------------------------------------------------
# 2. Multi-sentence bundle output
# --------------------------------------------------------------------------

def test_multi_mode_returns_bundle_superset(embedder, candidates):
    result = extract_best_evidence(
        CLAIM, candidates, embedder, evidence_mode="multi", max_sentences=3
    )

    assert {"sentence", "doc_id"} <= set(result.keys())
    assert result["evidence_mode"] == "multi"
    assert isinstance(result["sentence"], str)
    assert isinstance(result["spans"], list)
    assert len(result["spans"]) == 3


def test_multi_mode_rank0_equals_baseline_sentence(embedder, candidates):
    """The bundle must contain the baseline evidence, so the modes stay
    directly comparable and multi mode cannot lose baseline evidence."""
    baseline = extract_best_evidence(CLAIM, candidates, embedder)
    bundle = extract_best_evidence(
        CLAIM, candidates, embedder, evidence_mode="multi", max_sentences=3
    )

    rank0 = [s for s in bundle["spans"] if s["rank"] == 0]
    assert len(rank0) == 1
    assert rank0[0]["sentence"] == baseline["sentence"]
    assert bundle["doc_id"] == baseline["doc_id"]
    assert baseline["sentence"] in bundle["sentence"]


def test_multi_mode_prefers_complementary_over_redundant(embedder, candidates):
    """S_SUPPORT_NEAR is the 2nd most claim-similar sentence but is nearly a
    duplicate of the rank-0 pick; MMR should take the complementary
    S_CONFLICT instead. This is the behavior that distinguishes the bundle
    from a naive top-k."""
    bundle = extract_evidence_bundle(
        CLAIM, candidates, embedder, max_sentences=2
    )
    selected = [s["sentence"] for s in bundle["spans"]]

    assert S_SUPPORT in selected
    assert S_CONFLICT in selected
    assert S_SUPPORT_NEAR not in selected


def test_multi_mode_is_not_naive_top_k(embedder, candidates):
    """Explicit guard against "just take the first/top-k sentences"."""
    sentences = split_sentences(DOC_TEXT)
    bundle = extract_evidence_bundle(
        CLAIM, candidates, embedder, max_sentences=3
    )
    selected = [s["sentence"] for s in bundle["spans"]]

    first_k_in_document_order = sentences[:3]
    assert selected != first_k_in_document_order

    # Nor the top-3 by raw claim relevance (which would be SUPPORT,
    # SUPPORT_NEAR, CONFLICT).
    top_k_by_relevance = {S_SUPPORT, S_SUPPORT_NEAR, S_CONFLICT}
    assert set(selected) != top_k_by_relevance


def test_multi_mode_joined_text_is_source_ordered(embedder, candidates):
    """The premise string reads in document order, and every span text
    appears in it verbatim."""
    bundle = extract_evidence_bundle(
        CLAIM, candidates, embedder, max_sentences=3
    )
    spans = bundle["spans"]

    keys = [(s["doc_id"], DOC_TEXT.index(s["sentence"])) for s in spans]
    assert keys == sorted(keys)
    assert bundle["sentence"] == EVIDENCE_JOIN.join(
        s["sentence"] for s in spans
    )
    for span in spans:
        assert span["sentence"] in bundle["sentence"]


def test_multi_mode_no_duplicate_sentences(embedder, candidates):
    bundle = extract_evidence_bundle(
        CLAIM, candidates, embedder, max_sentences=4
    )
    selected = [s["sentence"] for s in bundle["spans"]]
    assert len(selected) == len(set(selected))


def test_multi_mode_dedupes_identical_sentences_across_docs(embedder):
    """A sentence repeated in two retrieved documents must not fill the
    bundle twice."""
    dupe_docs = [
        {"doc_id": "doc-A", "text": " ".join([TITLE, S_SUPPORT, S_CONFLICT])},
        {"doc_id": "doc-B", "text": " ".join([TITLE, S_SUPPORT, S_METHOD])},
    ]
    bundle = extract_evidence_bundle(
        CLAIM, dupe_docs, embedder, max_sentences=4
    )
    selected = [s["sentence"] for s in bundle["spans"]]

    assert selected.count(S_SUPPORT) == 1
    assert len(selected) == len(set(selected))


# --------------------------------------------------------------------------
# 3. Bundle size limits (2-4)
# --------------------------------------------------------------------------

def test_default_max_sentences_within_research_range():
    assert MAX_BUNDLE_SENTENCES == 4
    assert 2 <= DEFAULT_MAX_SENTENCES <= MAX_BUNDLE_SENTENCES


@pytest.mark.parametrize("k", [2, 3, 4])
def test_multi_mode_respects_requested_size(embedder, candidates, k):
    bundle = extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=k)
    assert len(bundle["spans"]) == k
    assert len(bundle["spans"]) <= MAX_BUNDLE_SENTENCES


def test_multi_mode_rejects_oversized_bundle(embedder, candidates):
    with pytest.raises(ValueError):
        extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=5)


def test_multi_mode_rejects_nonpositive_bundle(embedder, candidates):
    for bad in (0, -1):
        with pytest.raises(ValueError):
            extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=bad)


def test_multi_mode_rejects_bad_mmr_lambda(embedder, candidates):
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            extract_evidence_bundle(CLAIM, candidates, embedder, mmr_lambda=bad)


def test_multi_mode_size_one_matches_baseline_sentence(embedder, candidates):
    baseline = extract_best_evidence(CLAIM, candidates, embedder)
    bundle = extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=1)

    assert bundle["sentence"] == baseline["sentence"]
    assert bundle["doc_id"] == baseline["doc_id"]
    assert len(bundle["spans"]) == 1


# --------------------------------------------------------------------------
# 4. Provenance / document IDs
# --------------------------------------------------------------------------

def test_spans_carry_doc_id_and_verbatim_text(embedder):
    docs = [
        {"doc_id": "doc-A", "text": " ".join([TITLE, S_SUPPORT, S_SUPPORT_NEAR])},
        {"doc_id": "doc-B", "text": " ".join([TITLE, S_CONFLICT, S_METHOD])},
    ]
    by_id = {d["doc_id"]: d["text"] for d in docs}

    bundle = extract_evidence_bundle(CLAIM, docs, embedder, max_sentences=3)

    assert bundle["spans"]
    for span in bundle["spans"]:
        assert span["doc_id"] in by_id
        # Citation-grounding: verbatim substring of its OWN source document.
        assert span["sentence"] in by_id[span["doc_id"]]
        assert isinstance(span["relevance"], float)
        assert isinstance(span["rank"], int)


def test_bundle_spans_cross_documents(embedder):
    """Evidence from more than one document is kept with per-span doc_ids,
    not collapsed onto a single citation."""
    docs = [
        {"doc_id": "doc-A", "text": " ".join([TITLE, S_SUPPORT, S_SUPPORT_NEAR])},
        {"doc_id": "doc-B", "text": " ".join([TITLE, S_CONFLICT, S_METHOD])},
    ]
    bundle = extract_evidence_bundle(CLAIM, docs, embedder, max_sentences=2)

    span_docs = {s["doc_id"] for s in bundle["spans"]}
    assert span_docs == {"doc-A", "doc-B"}
    assert bundle["doc_ids"] == ["doc-A", "doc-B"]
    # Primary citation stays the rank-0 document.
    assert bundle["doc_id"] == "doc-A"


def test_doc_ids_are_unique_and_source_ordered(embedder):
    docs = [
        {"doc_id": "doc-A", "text": " ".join([TITLE, S_SUPPORT, S_SUPPORT_NEAR])},
        {"doc_id": "doc-B", "text": " ".join([TITLE, S_CONFLICT, S_METHOD])},
    ]
    bundle = extract_evidence_bundle(CLAIM, docs, embedder, max_sentences=4)

    assert len(bundle["doc_ids"]) == len(set(bundle["doc_ids"]))
    assert bundle["doc_ids"] == ["doc-A", "doc-B"]


def test_ranks_are_contiguous_from_zero(embedder, candidates):
    bundle = extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=4)
    ranks = sorted(s["rank"] for s in bundle["spans"])
    assert ranks == list(range(len(bundle["spans"])))


# --------------------------------------------------------------------------
# 5. Fewer sentences than requested
# --------------------------------------------------------------------------

def test_fewer_sentences_than_k_returns_what_exists(embedder):
    """Doc yields only 2 usable sentences after title stripping; asking for 4
    must return 2, not pad or fail."""
    short_doc = [{"doc_id": "doc-S", "text": " ".join([TITLE, S_SUPPORT, S_CONFLICT])}]
    assert len(split_sentences(short_doc[0]["text"])) == 2

    bundle = extract_evidence_bundle(CLAIM, short_doc, embedder, max_sentences=4)

    assert len(bundle["spans"]) == 2
    assert bundle["doc_id"] == "doc-S"


def test_single_sentence_document(embedder):
    """One sentence after title stripping: the bundle degrades to the
    baseline sentence."""
    one_doc = [{"doc_id": "doc-1", "text": " ".join([TITLE, S_SUPPORT])}]
    bundle = extract_evidence_bundle(CLAIM, one_doc, embedder, max_sentences=3)

    assert len(bundle["spans"]) == 1
    assert bundle["spans"][0]["sentence"] == S_SUPPORT
    assert bundle["sentence"] == S_SUPPORT
    assert bundle["spans"][0]["rank"] == 0


def test_min_relevance_can_shrink_bundle(embedder, candidates):
    """A high min_relevance filters out weak additional sentences while the
    rank-0 baseline sentence is always kept."""
    bundle = extract_evidence_bundle(
        CLAIM, candidates, embedder, max_sentences=4, min_relevance=0.99
    )
    assert len(bundle["spans"]) == 1
    assert bundle["spans"][0]["sentence"] == S_SUPPORT


def test_offtopic_sentence_excluded_by_default_threshold(embedder, candidates):
    """S_OFFTOPIC is orthogonal to the claim (relevance 0.0) and must not be
    pulled in ahead of relevant sentences."""
    bundle = extract_evidence_bundle(CLAIM, candidates, embedder, max_sentences=3)
    selected = [s["sentence"] for s in bundle["spans"]]
    assert S_OFFTOPIC not in selected


# --------------------------------------------------------------------------
# 6. Empty / degenerate candidates
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "empty_candidates",
    [
        [],
        None,
        [{"doc_id": "doc-E", "text": ""}],
        [{"doc_id": "doc-E", "text": "Too short."}],  # <20 chars, filtered out
        [{"doc_id": "doc-E"}],  # missing text key
    ],
)
def test_multi_mode_empty_candidates_safe(embedder, empty_candidates):
    bundle = extract_best_evidence(
        CLAIM, empty_candidates, embedder, evidence_mode="multi"
    )

    assert bundle["sentence"] == ""
    assert bundle["doc_id"] is None
    assert bundle["spans"] == []
    assert bundle["doc_ids"] == []
    assert bundle["evidence_mode"] == "multi"


def test_empty_candidates_skip_embedding(embedder):
    """No embedder call at all when there is nothing to rank."""
    extract_best_evidence(CLAIM, [], embedder, evidence_mode="multi")
    assert embedder.encode_calls == 0


def test_multi_mode_accepts_single_dict(embedder):
    bundle = extract_best_evidence(
        CLAIM,
        {"doc_id": "doc-A", "text": DOC_TEXT},
        embedder,
        evidence_mode="multi",
        max_sentences=2,
    )
    assert len(bundle["spans"]) == 2
    assert all(s["doc_id"] == "doc-A" for s in bundle["spans"])


# --------------------------------------------------------------------------
# 7. Determinism
# --------------------------------------------------------------------------

def test_multi_mode_repeated_calls_identical(candidates):
    results = [
        extract_evidence_bundle(CLAIM, candidates, build_embedder(), max_sentences=3)
        for _ in range(5)
    ]
    first = results[0]
    for other in results[1:]:
        assert other == first


def test_multi_mode_deterministic_under_ties(embedder):
    """Two sentences with identical embeddings: selection must resolve to the
    earlier pool position, every time."""
    tie_a = "Identical scoring sentence number one appears earlier here."
    tie_b = "Identical scoring sentence number two appears later on here."
    emb = FakeEmbedder()
    emb.register(CLAIM, [1.0, 0.0, 0.0, 0.0])
    emb.register(TITLE, [0.1, 1.0, 0.0, 0.0])
    emb.register(tie_a, [0.9, 0.1, 0.0, 0.0])
    emb.register(tie_b, [0.9, 0.1, 0.0, 0.0])

    docs = [{"doc_id": "doc-T", "text": " ".join([TITLE, tie_a, tie_b])}]

    for _ in range(5):
        bundle = extract_evidence_bundle(CLAIM, docs, emb, max_sentences=1)
        assert bundle["spans"][0]["sentence"] == tie_a


def test_single_and_multi_agree_on_primary_across_repeats(candidates):
    for _ in range(3):
        baseline = extract_best_evidence(CLAIM, candidates, build_embedder())
        bundle = extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="multi",
            max_sentences=4,
        )
        assert bundle["doc_id"] == baseline["doc_id"]
        rank0 = next(s for s in bundle["spans"] if s["rank"] == 0)
        assert rank0["sentence"] == baseline["sentence"]


# --------------------------------------------------------------------------
# 8. Wrapper defaults (pipeline / orchestration / eval stay baseline)
# --------------------------------------------------------------------------

def test_pipeline_and_orchestration_default_to_single_mode():
    """Import-light signature check: the wrappers must default to the
    validated baseline so existing callers are unaffected."""
    import inspect

    import src.orchestration  # noqa: F401  (imports langgraph; present in requirements)
    import src.pipeline

    pipeline_sig = inspect.signature(src.pipeline.JurisVerifyPipeline.__init__)
    assert pipeline_sig.parameters["evidence_mode"].default == "single"
    assert (
        pipeline_sig.parameters["max_sentences"].default == DEFAULT_MAX_SENTENCES
    )

    run_sig = inspect.signature(src.orchestration.run_pipeline)
    assert run_sig.parameters["evidence_mode"].default == "single"
    assert run_sig.parameters["max_retries"].default == 3
