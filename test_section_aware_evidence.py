"""Tests for Task C: structure-aware (section-aware) evidence ranking.

CPU-only and offline: no SentenceTransformer / NLI model is loaded. A
deterministic FakeEmbedder maps known sentences onto hand-written unit vectors,
so the real `sentence_transformers.util.cos_sim` still does the similarity math
(the same code path production uses) while the test controls the geometry
exactly.

Scope of these tests:
  - src/sections.py heading detection: the real inline ALL-CAPS headings SciFact
    exposes, sticky carry-forward, longest-match on compound headings, and
    rejection of biomedical acronyms that merely look like headings;
  - Results/Conclusions candidates outranking Background/Objective ones when
    claim relevance is comparable (the point of the feature);
  - deterministic ranking and tie-breaking;
  - missing / unrecoverable section metadata degrading to a neutral prior;
  - the evidence_mode="single" baseline and the evidence_mode="multi" MMR
    implementation being bit-for-bit unaffected.

Run: pytest test_section_aware_evidence.py
"""

import math

import pytest
import torch

from src.extraction import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_SECTION_WEIGHT,
    EVIDENCE_MODES,
    MAX_SECTION_WEIGHT,
    MIN_SECTION_WEIGHT,
    extract_best_evidence,
    extract_evidence_bundle,
    split_sentences,
)
from src.sections import (
    SECTION_BACKGROUND,
    SECTION_CONCLUSIONS,
    SECTION_METHODS,
    SECTION_RESULTS,
    SECTION_UNKNOWN,
    assign_sections,
    detect_section_heading,
    section_prior,
)


# --------------------------------------------------------------------------
# Deterministic offline embedder
# --------------------------------------------------------------------------

DIM = 6


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def axis_vector(cos_to_claim, axis):
    """Unit vector with exactly `cos_to_claim` along axis 0 (the claim
    direction) and all remaining mass on `axis`.

    Two such vectors built on *different* axes have pairwise cosine
    c1 * c2, which keeps the MMR redundancy term predictable by hand.
    """
    vector = [0.0] * DIM
    vector[0] = cos_to_claim
    vector[axis] = math.sqrt(max(0.0, 1.0 - cos_to_claim * cos_to_claim))
    return vector


class FakeEmbedder:
    """Maps text -> fixed unit vector. Unknown text gets a stable hash-derived
    vector so the embedder never fails, but every sentence whose geometry a
    test depends on is registered explicitly.
    """

    def __init__(self, vectors=None):
        self.vectors = dict(vectors or {})

    def register(self, text, vector):
        self.vectors[text] = _unit(vector)

    def _vector(self, text):
        if text in self.vectors:
            return self.vectors[text]
        h = abs(hash(text))
        return _unit([1.0 + (h >> (8 * i)) % 7 for i in range(DIM)])

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        if isinstance(texts, str):
            return torch.tensor(self._vector(texts), dtype=torch.float32)
        return torch.tensor([self._vector(t) for t in texts], dtype=torch.float32)


# --------------------------------------------------------------------------
# Fixture: one structured SciFact-shaped abstract
# --------------------------------------------------------------------------
#
# Headings are inline ALL-CAPS prefixes, exactly as the real corpus formats
# them (see src/sections.py for verbatim examples from the N=504 artifact).
# S_BG_INHERIT carries no heading of its own, so it must inherit "background"
# from the OBJECTIVE sentence above it -- that is the carry-forward rule under
# test.
#
#   sentence        cos(claim, s)   section       prior   adjusted @ w=0.05
#   S_OBJECTIVE          0.60       background    -1.0          0.55
#   S_BG_INHERIT         0.58       background    -1.0          0.53
#   S_METHODS            0.50       methods       -0.5          0.475
#   S_RESULTS            0.56       results       +1.0          0.61
#   S_CONCLUSIONS        0.45       conclusions   +1.0          0.50
#
# So raw cosine ranks S_OBJECTIVE first and the section-aware score ranks
# S_RESULTS first. The gap it has to overcome (0.60 - 0.56 = 0.04) is smaller
# than the 0.10 swing the default weight provides between a +1.0 and a -1.0
# section, which is the regime this feature is designed for.

CLAIM = "The treatment reduces mortality in elderly patients."

TITLE = "Folic acid supplementation and mortality in chronic kidney disease."
S_OBJECTIVE = (
    "OBJECTIVE To determine whether the treatment reduces mortality in "
    "elderly patients with chronic disease."
)
S_BG_INHERIT = (
    "Earlier observational work had suggested a mortality benefit in older "
    "patients receiving the treatment."
)
S_METHODS = (
    "METHODS Participants were randomized across twelve independent study "
    "sites and followed for five years."
)
S_RESULTS = (
    "RESULTS Mortality was significantly higher among elderly patients in "
    "the treatment arm than in the placebo arm."
)
S_CONCLUSIONS = (
    "CONCLUSIONS The treatment did not reduce mortality and may increase it "
    "in the elderly subgroup."
)

DOC_TEXT = " ".join(
    [TITLE, S_OBJECTIVE, S_BG_INHERIT, S_METHODS, S_RESULTS, S_CONCLUSIONS]
)

SECTION_OF = {
    S_OBJECTIVE: SECTION_BACKGROUND,
    S_BG_INHERIT: SECTION_BACKGROUND,
    S_METHODS: SECTION_METHODS,
    S_RESULTS: SECTION_RESULTS,
    S_CONCLUSIONS: SECTION_CONCLUSIONS,
}


def build_embedder():
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(S_OBJECTIVE, axis_vector(0.60, 1))
    emb.register(S_BG_INHERIT, axis_vector(0.58, 2))
    emb.register(S_METHODS, axis_vector(0.50, 3))
    emb.register(S_RESULTS, axis_vector(0.56, 4))
    emb.register(S_CONCLUSIONS, axis_vector(0.45, 5))
    return emb


@pytest.fixture
def embedder():
    return build_embedder()


@pytest.fixture
def candidates():
    return [{"doc_id": "doc-A", "text": DOC_TEXT}]


def rank0(bundle):
    """The rank-0 (top-ranked) span of a bundle."""
    return next(span for span in bundle["spans"] if span["rank"] == 0)


def texts(bundle):
    return [span["sentence"] for span in bundle["spans"]]


# ==========================================================================
# 1. Section detection (src/sections.py)
# ==========================================================================

@pytest.mark.parametrize(
    "sentence, expected_section, expected_heading",
    [
        # Verbatim shapes observed in eval/results_20260907_095418_n504.json.
        (
            "OBJECTIVE To determine whether high doses of folic acid reduce mortality.",
            SECTION_BACKGROUND,
            "OBJECTIVE",
        ),
        (
            "RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP.",
            SECTION_RESULTS,
            "RESULTS",
        ),
        (
            "CONCLUSIONS A dosage of folic acid of 0.8 mg/d appears necessary.",
            SECTION_CONCLUSIONS,
            "CONCLUSIONS",
        ),
        (
            "DESIGN, SETTING, AND PATIENTS Retrospective analysis of German cohorts.",
            SECTION_METHODS,
            "DESIGN SETTING AND PATIENTS",
        ),
        (
            "MAIN OUTCOME MEASURES Overall survival at five years.",
            SECTION_METHODS,
            "MAIN OUTCOME MEASURES",
        ),
        (
            "CONCLUSIONS AND RELEVANCE The association was not causal.",
            SECTION_CONCLUSIONS,
            "CONCLUSIONS AND RELEVANCE",
        ),
        ("BACKGROUND Prior trials were underpowered.", SECTION_BACKGROUND, "BACKGROUND"),
        ("INTERVENTIONS Oral folic acid versus placebo.", SECTION_METHODS, "INTERVENTIONS"),
        ("FINDINGS Event rates diverged after month six.", SECTION_RESULTS, "FINDINGS"),
        ("IMPORTANCE Mortality remains high in this group.", SECTION_BACKGROUND, "IMPORTANCE"),
    ],
)
def test_detect_real_scifact_headings(sentence, expected_section, expected_heading):
    section, heading = detect_section_heading(sentence)
    assert section == expected_section
    assert heading == expected_heading


def test_compound_heading_resolves_by_longest_match():
    """"METHODS AND RESULTS" must bind as the 3-token compound (-> results),
    not greedily as the 1-token "METHODS" (-> methods)."""
    section, heading = detect_section_heading(
        "METHODS AND RESULTS We enrolled 400 patients and event rates diverged."
    )
    assert heading == "METHODS AND RESULTS"
    assert section == SECTION_RESULTS


def test_longest_match_shortens_until_known():
    """"CONCLUSIONS A ..." offers a 2-token run; the 2-gram is unknown, so the
    match must fall back to the 1-gram rather than failing outright."""
    section, heading = detect_section_heading(
        "CONCLUSIONS A single dose was sufficient for most participants."
    )
    assert (section, heading) == (SECTION_CONCLUSIONS, "CONCLUSIONS")


@pytest.mark.parametrize(
    "sentence",
    [
        # Biomedical acronyms that a permissive "any ALL-CAPS prefix" rule
        # would misread as headings. All were found doing exactly that in the
        # N=504 artifact audit.
        "PTEN loss was observed in a third of the tumours examined.",
        "IFI16 restricts replication of the virus in primary macrophages.",
        "IL-10 and IL-18 levels were measured at baseline and week twelve.",
        "AMPK activation suppressed lipogenesis in the treated animals.",
        "CD28 costimulation was required for the proliferative response.",
        "EBI2 mediates positioning of the responding cells in the follicle.",
        "FGF21 analogues improved glycaemic control in the cohort studied.",
        "DNA methylation patterns differed between the two tissue types.",
        "FISH analysis confirmed the rearrangement in all nine samples.",
        # Lower case is not a heading, even for a vocabulary word.
        "Results showed no difference between the two treatment arms.",
        "Conclusions drawn by the authors were later disputed in print.",
        # Ordinary prose.
        "The treatment reduced mortality among elderly patients overall.",
    ],
)
def test_non_headings_are_rejected(sentence):
    assert detect_section_heading(sentence) == (None, None)


def test_detect_section_heading_on_empty_input():
    assert detect_section_heading("") == (None, None)
    assert detect_section_heading(None) == (None, None)


def test_assign_sections_carries_forward_within_document():
    """Only the first sentence of a section carries its heading; the rest must
    inherit it."""
    sentences = split_sentences(DOC_TEXT)
    sections = assign_sections(sentences)

    assert len(sections) == len(sentences)
    assert dict(zip(sentences, sections)) == {
        S_OBJECTIVE: SECTION_BACKGROUND,
        S_BG_INHERIT: SECTION_BACKGROUND,  # inherited, no heading of its own
        S_METHODS: SECTION_METHODS,
        S_RESULTS: SECTION_RESULTS,
        S_CONCLUSIONS: SECTION_CONCLUSIONS,
    }


def test_assign_sections_before_first_heading_is_unknown():
    lead = "This cohort study followed participants for a decade or longer."
    sentences = [lead, S_RESULTS, S_BG_INHERIT]
    assert assign_sections(sentences) == [
        SECTION_UNKNOWN,  # nothing seen yet
        SECTION_RESULTS,
        SECTION_RESULTS,  # inherited from the RESULTS heading
    ]


def test_assign_sections_empty_list():
    assert assign_sections([]) == []


# ==========================================================================
# 2. Missing / unrecoverable section metadata
# ==========================================================================

UNSTRUCTURED_A = "The treatment reduced mortality among elderly patients overall."
UNSTRUCTURED_B = "Mortality increased in the elderly subgroup receiving the drug."
UNSTRUCTURED_C = "Participants were randomized across twelve independent sites."
UNSTRUCTURED_TITLE = "An unstructured abstract about treatment and mortality."
UNSTRUCTURED_DOC = " ".join(
    [UNSTRUCTURED_TITLE, UNSTRUCTURED_A, UNSTRUCTURED_B, UNSTRUCTURED_C]
)


def unstructured_embedder():
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(UNSTRUCTURED_TITLE, axis_vector(0.10, 1))
    emb.register(UNSTRUCTURED_A, axis_vector(0.60, 1))
    emb.register(UNSTRUCTURED_B, axis_vector(0.56, 2))
    emb.register(UNSTRUCTURED_C, axis_vector(0.30, 3))
    return emb


def test_unstructured_document_yields_unknown_sections():
    sentences = split_sentences(UNSTRUCTURED_DOC)
    assert assign_sections(sentences) == [SECTION_UNKNOWN] * len(sentences)


def test_missing_section_metadata_is_neutral_not_penalized():
    """A document whose structure cannot be read must rank exactly as it would
    without the feature -- "unknown" carries a zero prior, so the section-aware
    bundle has to equal the plain MMR bundle."""
    docs = [{"doc_id": "doc-U", "text": UNSTRUCTURED_DOC}]

    plain = extract_evidence_bundle(
        CLAIM, docs, unstructured_embedder(), max_sentences=3, section_weight=0.0
    )
    section_aware = extract_evidence_bundle(
        CLAIM, docs, unstructured_embedder(), max_sentences=3, section_weight=0.5
    )

    assert texts(section_aware) == texts(plain)
    assert rank0(section_aware)["sentence"] == rank0(plain)["sentence"]
    for span in section_aware["spans"]:
        assert span["section"] == SECTION_UNKNOWN
        assert span["section_prior"] == 0.0
        # No heading recovered => the score ranked on is the raw cosine.
        assert span["ranking_score"] == pytest.approx(span["relevance"])


def test_section_prior_defaults_to_neutral_for_unknown_values():
    assert section_prior(SECTION_UNKNOWN) == 0.0
    assert section_prior(None) == 0.0
    assert section_prior("not-a-section") == 0.0


def test_partially_structured_document_mixes_known_and_unknown(embedder):
    """A document that only starts carrying headings partway through must get
    real sections for the tail and "unknown" (neutral) for the head."""
    lead = "This multicentre cohort followed elderly participants for a decade."
    text = " ".join([TITLE, lead, S_RESULTS, S_CONCLUSIONS])
    emb = build_embedder()
    emb.register(lead, axis_vector(0.59, 2))

    bundle = extract_evidence_bundle(
        CLAIM,
        [{"doc_id": "doc-P", "text": text}],
        emb,
        max_sentences=3,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    by_text = {span["sentence"]: span["section"] for span in bundle["spans"]}
    assert by_text[lead] == SECTION_UNKNOWN
    assert by_text[S_RESULTS] == SECTION_RESULTS
    assert by_text[S_CONCLUSIONS] == SECTION_CONCLUSIONS


def test_candidate_with_no_text_key_degrades_cleanly(embedder):
    bundle = extract_evidence_bundle(
        CLAIM, [{"doc_id": "doc-empty"}], embedder, section_weight=0.5
    )
    assert bundle["sentence"] == ""
    assert bundle["spans"] == []
    assert bundle["doc_ids"] == []


# ==========================================================================
# 3. Results/Conclusions outrank Background/Objective at comparable relevance
# ==========================================================================

def test_section_aware_promotes_results_over_more_relevant_objective(
    candidates, embedder
):
    """The headline behavior: raw cosine prefers the OBJECTIVE sentence (0.60)
    that parrots the claim, section-aware ranking promotes the RESULTS sentence
    (0.56) that actually resolves it."""
    plain = extract_evidence_bundle(
        CLAIM, candidates, build_embedder(), max_sentences=1, section_weight=0.0
    )
    section_aware = extract_evidence_bundle(
        CLAIM,
        candidates,
        build_embedder(),
        max_sentences=1,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )

    assert rank0(plain)["sentence"] == S_OBJECTIVE
    assert rank0(plain)["section"] == SECTION_BACKGROUND

    assert rank0(section_aware)["sentence"] == S_RESULTS
    assert rank0(section_aware)["section"] == SECTION_RESULTS
    # ...and it won on the adjusted score while still having LOWER raw cosine.
    assert rank0(section_aware)["relevance"] < rank0(plain)["relevance"]


def test_results_wins_an_exact_relevance_tie_against_background(embedder):
    """"when relevance is otherwise comparable" in its strictest form: identical
    cosine, so the section prior is the only thing that can decide."""
    objective = "OBJECTIVE To establish whether mortality falls under treatment."
    results = "RESULTS Mortality rose under treatment in the elderly stratum."
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(objective, axis_vector(0.55, 1))
    emb.register(results, axis_vector(0.55, 2))

    docs = [{"doc_id": "doc-T", "text": " ".join([TITLE, objective, results])}]

    # Tie on raw cosine resolves to the earlier pool position (OBJECTIVE).
    plain = extract_evidence_bundle(
        CLAIM, docs, emb, max_sentences=1, section_weight=0.0
    )
    assert rank0(plain)["sentence"] == objective

    # With the prior, RESULTS wins despite being later in the document.
    section_aware = extract_evidence_bundle(
        CLAIM, docs, emb, max_sentences=1, section_weight=DEFAULT_SECTION_WEIGHT
    )
    assert rank0(section_aware)["sentence"] == results


def test_methods_penalized_less_than_background():
    """The prior is ordinal, not binary: methods (-0.5) must sit between
    background (-1.0) and unknown (0.0)."""
    assert (
        section_prior(SECTION_BACKGROUND)
        < section_prior(SECTION_METHODS)
        < section_prior(SECTION_UNKNOWN)
        < section_prior(SECTION_RESULTS)
    )
    assert section_prior(SECTION_RESULTS) == section_prior(SECTION_CONCLUSIONS)


def test_section_aware_reorders_full_candidate_ranking(candidates):
    """Checks the whole ordering, not just the winner: with the default weight
    the adjusted order is RESULTS > OBJECTIVE > CONCLUSIONS > METHODS, versus
    the raw-cosine order OBJECTIVE > BG_INHERIT > RESULTS > METHODS >
    CONCLUSIONS."""
    bundle = extract_evidence_bundle(
        CLAIM,
        candidates,
        build_embedder(),
        max_sentences=4,
        mmr_lambda=1.0,  # pure relevance, so this reads as a plain ranking
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    by_rank = sorted(bundle["spans"], key=lambda span: span["rank"])
    assert [span["sentence"] for span in by_rank] == [
        S_RESULTS,       # 0.56 + 0.05 = 0.61
        S_OBJECTIVE,     # 0.60 - 0.05 = 0.55
        S_BG_INHERIT,    # 0.58 - 0.05 = 0.53
        S_CONCLUSIONS,   # 0.45 + 0.05 = 0.50
    ]
    # S_METHODS (0.50 - 0.025 = 0.475) is ranked last and falls outside k=4.
    assert S_METHODS not in texts(bundle)


def test_stronger_weight_pulls_conclusions_into_a_two_sentence_bundle(candidates):
    """Arm D with a larger weight: at 0.10 the CONCLUSIONS sentence overtakes
    the OBJECTIVE one for the second slot."""
    at_default = extract_evidence_bundle(
        CLAIM, candidates, build_embedder(), max_sentences=2, section_weight=0.05
    )
    at_stronger = extract_evidence_bundle(
        CLAIM, candidates, build_embedder(), max_sentences=2, section_weight=0.10
    )

    assert texts(at_default) == [S_OBJECTIVE, S_RESULTS]
    assert texts(at_stronger) == [S_RESULTS, S_CONCLUSIONS]


def test_multi2_versus_section_aware_multi2_select_different_bundles(candidates):
    """Ablation arms B and D on the same input. Plain Multi-2 fills the bundle
    with two Background sentences; section-aware Multi-2 leads with RESULTS."""
    arm_b = extract_best_evidence(
        CLAIM, candidates, build_embedder(), evidence_mode="multi", max_sentences=2
    )
    arm_d = extract_best_evidence(
        CLAIM,
        candidates,
        build_embedder(),
        evidence_mode="section",
        max_sentences=2,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )

    assert texts(arm_b) == [S_OBJECTIVE, S_BG_INHERIT]
    assert [span["section"] for span in arm_b["spans"]] == [
        SECTION_BACKGROUND,
        SECTION_BACKGROUND,
    ]

    assert rank0(arm_d)["sentence"] == S_RESULTS
    assert S_RESULTS in texts(arm_d)
    assert arm_b["evidence_mode"] == "multi"
    assert arm_d["evidence_mode"] == "section"


def test_negative_weight_inverts_the_prior(candidates):
    """The anti-prior control arm: a negative weight must favour Background
    over Results, confirming the sign convention is what the docs claim."""
    bundle = extract_evidence_bundle(
        CLAIM, candidates, build_embedder(), max_sentences=1, section_weight=-0.05
    )
    assert rank0(bundle)["sentence"] == S_OBJECTIVE
    assert rank0(bundle)["ranking_score"] == pytest.approx(0.60 + 0.05, abs=1e-5)


# ==========================================================================
# 4. Scoring is interpretable and reported
# ==========================================================================

def test_spans_report_section_prior_and_ranking_score(candidates):
    bundle = extract_evidence_bundle(
        CLAIM,
        candidates,
        build_embedder(),
        max_sentences=4,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    for span in bundle["spans"]:
        assert span["section"] == SECTION_OF[span["sentence"]]
        assert span["section_prior"] == section_prior(span["section"])
        # The documented formula, checked against the reported numbers.
        assert span["ranking_score"] == pytest.approx(
            span["relevance"] + DEFAULT_SECTION_WEIGHT * span["section_prior"],
            abs=1e-6,
        )
        # "relevance" keeps its pre-Task-C meaning: the raw cosine.
        assert span["relevance"] == pytest.approx(
            {
                S_OBJECTIVE: 0.60,
                S_BG_INHERIT: 0.58,
                S_METHODS: 0.50,
                S_RESULTS: 0.56,
                S_CONCLUSIONS: 0.45,
            }[span["sentence"]],
            abs=1e-5,
        )


def test_min_relevance_gates_on_raw_cosine_not_the_boosted_score(candidates):
    """A section bonus must not smuggle a candidate past the relevance floor.
    S_CONCLUSIONS has raw cosine 0.45 (below the 0.48 floor) but an adjusted
    score of 0.50 (above it) -- it must still be excluded."""
    bundle = extract_evidence_bundle(
        CLAIM,
        candidates,
        build_embedder(),
        max_sentences=4,
        min_relevance=0.48,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    assert S_CONCLUSIONS not in texts(bundle)
    assert S_METHODS in texts(bundle)  # raw 0.50 clears the floor
    assert all(span["relevance"] >= 0.48 - 1e-6 for span in bundle["spans"])


def test_evidence_is_still_verbatim_substring_of_its_source(candidates):
    """Section-aware ranking is still selection, never generation: citation
    grounding must survive."""
    bundle = extract_evidence_bundle(
        CLAIM,
        candidates,
        build_embedder(),
        max_sentences=4,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    for span in bundle["spans"]:
        assert span["doc_id"] == "doc-A"
        assert span["sentence"] in DOC_TEXT


@pytest.mark.parametrize("weight", [1.5, -1.5, "0.05", True, None])
def test_invalid_section_weight_rejected(candidates, embedder, weight):
    with pytest.raises(ValueError):
        extract_evidence_bundle(
            CLAIM, candidates, embedder, section_weight=weight
        )


def test_section_weight_range_boundaries_accepted(candidates):
    for weight in (MIN_SECTION_WEIGHT, 0.0, MAX_SECTION_WEIGHT):
        bundle = extract_evidence_bundle(
            CLAIM, candidates, build_embedder(), max_sentences=2, section_weight=weight
        )
        assert len(bundle["spans"]) == 2


def test_section_is_a_registered_evidence_mode():
    assert EVIDENCE_MODES == ("single", "multi", "section")
    assert DEFAULT_EVIDENCE_MODE == "single"
    assert 0.0 < DEFAULT_SECTION_WEIGHT <= MAX_SECTION_WEIGHT


def test_unknown_evidence_mode_still_rejected(candidates, embedder):
    with pytest.raises(ValueError):
        extract_best_evidence(
            CLAIM, candidates, embedder, evidence_mode="section_aware"
        )


# ==========================================================================
# 5. Determinism and tie-breaking
# ==========================================================================

def test_section_mode_is_deterministic_across_repeats(candidates):
    results = [
        extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="section",
            max_sentences=3,
            section_weight=DEFAULT_SECTION_WEIGHT,
        )
        for _ in range(5)
    ]
    first = results[0]
    for other in results[1:]:
        assert other == first


def test_ties_within_the_same_section_resolve_to_lowest_pool_index():
    """Identical relevance AND identical section: the only tie-breaker left is
    source order, which must always pick the earlier sentence."""
    tie_a = "RESULTS Mortality rose in the first reported elderly cohort here."
    tie_b = "Mortality rose in the second reported elderly cohort as well now."
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(tie_a, axis_vector(0.70, 1))
    emb.register(tie_b, axis_vector(0.70, 1))

    docs = [{"doc_id": "doc-T", "text": " ".join([TITLE, tie_a, tie_b])}]

    for _ in range(5):
        bundle = extract_evidence_bundle(
            CLAIM, docs, emb, max_sentences=1, section_weight=DEFAULT_SECTION_WEIGHT
        )
        # tie_b inherits "results" from tie_a, so both priors match too.
        assert rank0(bundle)["section"] == SECTION_RESULTS
        assert rank0(bundle)["sentence"] == tie_a


def test_ties_across_documents_resolve_to_earlier_document():
    sentence = "RESULTS Mortality rose among the elderly patients under study."
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(sentence, axis_vector(0.70, 1))

    docs = [
        {"doc_id": "doc-first", "text": " ".join([TITLE, sentence])},
        {"doc_id": "doc-second", "text": " ".join([TITLE, sentence])},
    ]
    bundle = extract_evidence_bundle(
        CLAIM, docs, emb, max_sentences=1, section_weight=DEFAULT_SECTION_WEIGHT
    )
    assert bundle["doc_id"] == "doc-first"


def test_section_detection_is_pure(candidates):
    """Repeated detection on the same inputs must not drift (no cached or
    mutated state in src/sections.py)."""
    sentences = split_sentences(DOC_TEXT)
    runs = [assign_sections(sentences) for _ in range(5)]
    assert all(run == runs[0] for run in runs)
    assert [detect_section_heading(S_RESULTS) for _ in range(3)] == [
        (SECTION_RESULTS, "RESULTS")
    ] * 3


# ==========================================================================
# 6. Baseline preservation (requirements 1 and 2)
# ==========================================================================

def test_single_mode_return_shape_is_unchanged(candidates, embedder):
    result = extract_best_evidence(CLAIM, candidates, embedder)
    assert set(result.keys()) == {"sentence", "doc_id"}
    assert "section" not in result
    assert "evidence_mode" not in result


def test_single_mode_ignores_section_weight(candidates):
    """Requirement 1: the baseline must be untouched. Even an extreme weight
    must not move it."""
    default = extract_best_evidence(CLAIM, candidates, build_embedder())
    for weight in (0.0, DEFAULT_SECTION_WEIGHT, 1.0, -1.0):
        result = extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="single",
            section_weight=weight,
        )
        assert result == default


def test_single_mode_still_picks_the_raw_cosine_winner(candidates):
    """The sharpest form of the guarantee: on an input where section-aware
    ranking demonstrably picks a different sentence, the baseline keeps picking
    the raw-cosine one."""
    baseline = extract_best_evidence(CLAIM, candidates, build_embedder())
    assert baseline["sentence"] == S_OBJECTIVE
    assert baseline["doc_id"] == "doc-A"

    section_aware = extract_best_evidence(
        CLAIM,
        candidates,
        build_embedder(),
        evidence_mode="section",
        max_sentences=1,
        section_weight=DEFAULT_SECTION_WEIGHT,
    )
    assert rank0(section_aware)["sentence"] == S_RESULTS


def test_multi_mode_ignores_section_weight(candidates):
    """Requirement 2: the Task B MMR implementation stays a separate mode, so
    section_weight must not reach it even when passed explicitly."""
    reference = extract_best_evidence(
        CLAIM, candidates, build_embedder(), evidence_mode="multi", max_sentences=3
    )
    for weight in (0.0, DEFAULT_SECTION_WEIGHT, 1.0, -1.0):
        result = extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="multi",
            max_sentences=3,
            section_weight=weight,
        )
        assert result == reference
    assert reference["evidence_mode"] == "multi"


def test_zero_weight_section_mode_equals_multi_mode_exactly(candidates):
    """section_weight=0.0 is an exact no-op, which is what makes it a valid
    control arm: the bundle must be identical to Multi, right down to the
    reported evidence_mode."""
    for k in (1, 2, 3, 4):
        multi = extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="multi",
            max_sentences=k,
        )
        zeroed = extract_best_evidence(
            CLAIM,
            candidates,
            build_embedder(),
            evidence_mode="section",
            max_sentences=k,
            section_weight=0.0,
        )
        for span in zeroed["spans"]:
            # Section metadata is still reported, it just carries no weight.
            assert span["ranking_score"] == pytest.approx(span["relevance"])
        assert texts(zeroed) == texts(multi)
        assert zeroed["doc_id"] == multi["doc_id"]
        assert zeroed["evidence_mode"] == "multi"


def test_single_mode_unchanged_on_unstructured_documents():
    docs = [{"doc_id": "doc-U", "text": UNSTRUCTURED_DOC}]
    result = extract_best_evidence(CLAIM, docs, unstructured_embedder())
    assert result == {"sentence": UNSTRUCTURED_A, "doc_id": "doc-U"}


def test_wrapper_defaults_keep_baseline_mode():
    """pipeline / orchestration / eval must all still default to "single", and
    the new knob must default to the documented value."""
    import inspect

    import src.orchestration
    import src.pipeline
    import eval.evaluate

    pipeline_sig = inspect.signature(src.pipeline.JurisVerifyPipeline.__init__)
    assert pipeline_sig.parameters["evidence_mode"].default == "single"
    assert pipeline_sig.parameters["max_sentences"].default == DEFAULT_MAX_SENTENCES
    assert pipeline_sig.parameters["section_weight"].default == DEFAULT_SECTION_WEIGHT

    run_sig = inspect.signature(src.orchestration.run_pipeline)
    assert run_sig.parameters["evidence_mode"].default == "single"
    assert run_sig.parameters["section_weight"].default == DEFAULT_SECTION_WEIGHT

    eval_sig = inspect.signature(eval.evaluate.run_eval)
    assert eval_sig.parameters["evidence_mode"].default == "single"
    assert eval_sig.parameters["section_weight"].default == DEFAULT_SECTION_WEIGHT


def test_mmr_lambda_default_untouched():
    """Task C must not have retuned Task B's trade-off."""
    assert DEFAULT_MMR_LAMBDA == 0.7
