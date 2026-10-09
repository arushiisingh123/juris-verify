"""Tests for arm E: the Passive Prosecutor (src/prosecution.py).

CPU-only and offline. No SentenceTransformer and no DeBERTa are loaded: a
deterministic FakeEmbedder supplies the selection geometry (the real
`sentence_transformers.util.cos_sim` still does the similarity math) and a
FakeNLIPipeline supplies scripted class distributions.

Crucially these tests exercise the REAL `JurisVerifyPipeline.predict_claim`.
The pipeline is built with `__new__` and its heavy attributes are injected, so
the actual production control flow runs -- including the prosecution gate --
without instantiating HybridRetriever or NLIVerifier.

The central fixture mirrors the real arm-D failure mode arm H measured: the
claim's strongest section-aware sentence verifies NEUTRAL (so the normalized
prediction is SUPPORT) while a lower-ranked sentence in the same pool is
unambiguously CONTRADICTED. Arm E must find that sentence and must not act on
it.

Run: pytest test_passive_prosecution.py
"""

import inspect
import math

import pytest
import torch

import src.prosecution
import src.verification
from src.extraction import DEFAULT_SECTION_WEIGHT
from src.pipeline import JurisVerifyPipeline
from src.prosecution import (
    CONTRADICTED,
    DEFAULT_PROSECUTION_MODE,
    PARTIALLY_SUPPORTED,
    PROSECUTION_MODES,
    PROSECUTION_RESULT_KEYS,
    SUPPORTED,
    attach_prosecution,
    contradiction_margin,
    nli_results,
    p_contradiction,
    prosecute,
    validate_prosecution_mode,
    verdict_from_top_label,
)
from src.verification import NLIVerifier, normalize_for_scifact_eval


# --------------------------------------------------------------------------
# Deterministic offline doubles
# --------------------------------------------------------------------------

DIM = 6


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def axis_vector(cos_to_claim, axis):
    """Unit vector with exactly `cos_to_claim` on axis 0 and the rest on `axis`.

    Two such vectors on different axes have pairwise cosine c1*c2, keeping the
    MMR redundancy term hand-computable.
    """
    vector = [0.0] * DIM
    vector[0] = cos_to_claim
    vector[axis] = math.sqrt(max(0.0, 1.0 - cos_to_claim * cos_to_claim))
    return vector


class FakeEmbedder:
    def __init__(self):
        self.vectors = {}

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


class FakeNLIPipeline:
    """Stands in for the HF text-classification pipeline on DeBERTa.

    Handles both call shapes production uses: a single dict (how
    NLIVerifier.verify calls it, top-1) and a list with top_k=None (how
    prosecution calls it, full distribution). Records every premise it scored
    so tests can assert on call volume.
    """

    def __init__(self, scripted_contradiction=None, default_p=0.02):
        self.scripted = dict(scripted_contradiction or {})
        self.default_p = default_p
        self.scored_premises = []
        self.calls = 0

    def _distribution(self, premise):
        contra = self.scripted.get(premise, self.default_p)
        entail = 0.01
        neutral = max(0.0, 1.0 - contra - entail)
        return sorted(
            [
                {"label": "contradiction", "score": contra},
                {"label": "neutral", "score": neutral},
                {"label": "entailment", "score": entail},
            ],
            key=lambda item: -item["score"],
        )

    def __call__(self, payload, top_k="", batch_size=None):
        self.calls += 1
        if isinstance(payload, dict):
            self.scored_premises.append(payload["text"])
            dist = self._distribution(payload["text"])
            return dist if top_k is None else [dist[0]]
        out = []
        for item in payload:
            self.scored_premises.append(item["text"])
            dist = self._distribution(item["text"])
            out.append(dist if top_k is None else [dist[0]])
        return out


class FakeVerifier:
    """A real NLIVerifier's public surface, without the 1.6GB model.

    `verify` is the genuine implementation, borrowed from the class, so the
    baseline verdict in these tests is produced by production code.
    """

    def __init__(self, nli_pipeline):
        self.nli_pipeline = nli_pipeline

    def verify(self, claim, evidence, return_dict=False):
        return NLIVerifier.verify(self, claim, evidence, return_dict=return_dict)


class FakeRetriever:
    def __init__(self, documents, embedder):
        self.documents = documents
        self.embedding_model = embedder
        self.retrieve_calls = 0

    def retrieve(self, claim):
        self.retrieve_calls += 1
        return self.documents


# --------------------------------------------------------------------------
# Fixture: arm D picks a neutral Results sentence; the pool holds a refutation
# --------------------------------------------------------------------------
#
#   sentence             cos    section      adj @ w=0.05   scripted NLI
#   S_OBJECTIVE          0.60   background       0.55       neutral
#   S_RESULTS_PRIMARY    0.56   results          0.61       neutral   <- D rank 0
#   S_RESULTS_COUNTER    0.40   results (carry)  0.45       CONTRADICTION 0.95
#   S_METHODS            0.30   methods          0.275      neutral
#
# Arm D (section, k=2) selects {S_RESULTS_PRIMARY, S_OBJECTIVE}; the joined
# premise verifies neutral, so the normalized prediction is SUPPORT. The
# refutation sits at pool position 2, unselected -- a selection failure of
# exactly the kind arm H counted 36 of.

CLAIM = "The treatment reduces mortality in elderly patients."

TITLE = "Folic acid supplementation and mortality in chronic kidney disease."
S_OBJECTIVE = (
    "OBJECTIVE To determine whether the treatment reduces mortality in "
    "elderly patients."
)
S_RESULTS_PRIMARY = (
    "RESULTS The primary composite endpoint favoured the treatment group "
    "overall."
)
S_RESULTS_COUNTER = (
    "Mortality was significantly higher in the elderly treatment subgroup "
    "than in placebo."
)
S_METHODS = (
    "METHODS Participants were randomized across twelve independent study "
    "sites."
)

DOC_TEXT = " ".join(
    [TITLE, S_OBJECTIVE, S_RESULTS_PRIMARY, S_RESULTS_COUNTER, S_METHODS]
)
DOC_ID = "doc-A"

ARM_D = {"evidence_mode": "section", "max_sentences": 2, "section_weight": 0.05}


def build_embedder():
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(S_OBJECTIVE, axis_vector(0.60, 1))
    emb.register(S_RESULTS_PRIMARY, axis_vector(0.56, 2))
    emb.register(S_RESULTS_COUNTER, axis_vector(0.40, 3))
    emb.register(S_METHODS, axis_vector(0.30, 4))
    return emb


def build_pipeline(
    prosecution_mode="off",
    scripted=None,
    documents=None,
    arm=None,
):
    """A real JurisVerifyPipeline with injected doubles (no model loading)."""
    pipeline = JurisVerifyPipeline.__new__(JurisVerifyPipeline)
    docs = documents if documents is not None else [{"doc_id": DOC_ID, "text": DOC_TEXT}]
    pipeline.retriever = FakeRetriever(docs, build_embedder())
    pipeline.verifier = FakeVerifier(
        FakeNLIPipeline(
            scripted if scripted is not None else {S_RESULTS_COUNTER: 0.95}
        )
    )
    config = dict(arm or ARM_D)
    pipeline.evidence_mode = config["evidence_mode"]
    pipeline.max_sentences = config["max_sentences"]
    pipeline.section_weight = config["section_weight"]
    pipeline.prosecution_mode = prosecution_mode
    return pipeline


BASELINE_KEYS = {
    "claim",
    "evidence",
    "evidence_doc_id",
    "verdict",
    "matrix_label",
    "evidence_spans",
    "evidence_doc_ids",
}


def truncated(documents=None):
    docs = documents if documents is not None else [{"doc_id": DOC_ID, "text": DOC_TEXT}]
    return [{"doc_id": d["doc_id"], "text": d["text"][:3000]} for d in docs]


# ==========================================================================
# 0. The fixture really does reproduce the arm-D failure mode
# ==========================================================================

def test_fixture_reproduces_arm_d_failure_mode():
    """Sanity-check the geometry before asserting anything about prosecution:
    arm D must select the neutral Results sentence plus the Objective one, and
    must predict SUPPORT despite a refutation sitting in the pool."""
    result = build_pipeline("off").predict_claim(CLAIM)

    by_rank = sorted(result["evidence_spans"], key=lambda span: span["rank"])
    assert [span["sentence"] for span in by_rank] == [
        S_RESULTS_PRIMARY,
        S_OBJECTIVE,
    ]
    assert result["verdict"] == PARTIALLY_SUPPORTED
    assert normalize_for_scifact_eval(result["verdict"]) == "SUPPORT"
    assert S_RESULTS_COUNTER not in result["evidence"]


# ==========================================================================
# 1-3, 10. Arm E changes no prediction
# ==========================================================================

def test_off_mode_output_has_only_baseline_keys():
    result = build_pipeline("off").predict_claim(CLAIM)
    assert set(result.keys()) == BASELINE_KEYS
    assert not (set(result.keys()) & PROSECUTION_RESULT_KEYS)


def test_passive_mode_preserves_every_baseline_field():
    """Requirement 1/2/3 together: arm E's output is arm D's output plus
    prosecution keys, with no baseline value altered."""
    arm_d = build_pipeline("off").predict_claim(CLAIM)
    arm_e = build_pipeline("passive").predict_claim(CLAIM)

    for key in BASELINE_KEYS:
        assert arm_e[key] == arm_d[key], f"arm E altered baseline field {key!r}"
    assert set(arm_e.keys()) == BASELINE_KEYS | PROSECUTION_RESULT_KEYS


def test_passive_mode_does_not_change_verdict_or_prediction():
    arm_d = build_pipeline("off").predict_claim(CLAIM)
    arm_e = build_pipeline("passive").predict_claim(CLAIM)

    assert arm_e["verdict"] == arm_d["verdict"]
    assert normalize_for_scifact_eval(arm_e["verdict"]) == normalize_for_scifact_eval(
        arm_d["verdict"]
    )
    assert arm_e["matrix_label"] == arm_d["matrix_label"]
    # The discovery happened, and was still not acted on.
    assert arm_e["counter_verdict"] == CONTRADICTED
    assert normalize_for_scifact_eval(arm_e["verdict"]) == "SUPPORT"


def test_passive_mode_does_not_change_evidence_used_for_prediction():
    arm_d = build_pipeline("off").predict_claim(CLAIM)
    arm_e = build_pipeline("passive").predict_claim(CLAIM)
    assert arm_e["evidence"] == arm_d["evidence"]
    assert arm_e["evidence_doc_id"] == arm_d["evidence_doc_id"]
    assert arm_e["evidence_spans"] == arm_d["evidence_spans"]
    # Counter-evidence is held separately and never merged into the premise.
    assert arm_e["counter_evidence"] not in arm_e["evidence"]


@pytest.mark.parametrize(
    "arm",
    [
        {"evidence_mode": "single", "max_sentences": 3, "section_weight": 0.05},
        {"evidence_mode": "multi", "max_sentences": 2, "section_weight": 0.05},
        {"evidence_mode": "section", "max_sentences": 1, "section_weight": 0.05},
        {"evidence_mode": "section", "max_sentences": 2, "section_weight": 0.05},
    ],
)
def test_prediction_invariance_across_every_evidence_mode(arm):
    """Arm E must be prediction-neutral for arms A, B, C and D alike, so the
    passive control can be layered onto any of them."""
    arm_d = build_pipeline("off", arm=arm).predict_claim(CLAIM)
    arm_e = build_pipeline("passive", arm=arm).predict_claim(CLAIM)
    for key in BASELINE_KEYS & set(arm_d):
        assert arm_e[key] == arm_d[key]


def test_prediction_invariance_holds_over_many_claims():
    """Requirement 10, in miniature: the aggregate verdict sequence must be
    identical, which is what makes arm E's metrics equal arm D's."""
    claims = [
        CLAIM,
        "The treatment increases mortality in elderly patients.",
        "Folic acid has no effect on cardiovascular outcomes.",
        "Randomization was stratified by study site.",
        "Mortality fell in every prespecified subgroup.",
    ]
    off = [build_pipeline("off").predict_claim(c) for c in claims]
    passive = [build_pipeline("passive").predict_claim(c) for c in claims]

    assert [r["verdict"] for r in passive] == [r["verdict"] for r in off]
    assert [normalize_for_scifact_eval(r["verdict"]) for r in passive] == [
        normalize_for_scifact_eval(r["verdict"]) for r in off
    ]
    assert [r["evidence"] for r in passive] == [r["evidence"] for r in off]


# ==========================================================================
# 4-6. Counter-evidence discovery, rejection, and provenance
# ==========================================================================

def test_strong_contradictory_candidate_is_recorded_as_counter_evidence():
    result = build_pipeline("passive").predict_claim(CLAIM)

    assert result["prosecution_mode"] == "passive"
    assert result["prosecution_attempted"] is True
    assert result["counter_evidence"] == S_RESULTS_COUNTER
    assert result["counter_verdict"] == CONTRADICTED
    assert result["counter_p_contradiction"] == pytest.approx(0.95)
    assert result["counter_confidence"] == pytest.approx(0.95)
    assert result["counter_score"] == pytest.approx(0.95 - 0.04, abs=1e-6)
    assert result["n_contradictory_candidates"] == 1
    assert result["candidate_pool_size"] == 4
    assert result["max_p_contradiction"] == pytest.approx(0.95)


def test_weak_candidates_never_become_counter_evidence():
    """Requirement 5. Nothing in the pool is argmax-contradiction, so despite
    a non-zero P(contradiction) on every sentence, no counter-evidence is
    reported -- the gate is the model's own verdict, not a tuned threshold."""
    result = build_pipeline("passive", scripted={}).predict_claim(CLAIM)

    assert result["prosecution_attempted"] is True
    assert result["counter_evidence"] is None
    assert result["counter_verdict"] is None
    assert result["counter_confidence"] is None
    assert result["counter_rank"] is None
    assert result["n_contradictory_candidates"] == 0
    assert result["candidate_pool_size"] == 4
    # The pool-wide maximum is still reported for later offline analysis.
    assert result["max_p_contradiction"] == pytest.approx(0.02)


def test_sub_argmax_contradiction_is_not_promoted():
    """A candidate at P(contradiction)=0.45 with neutral at 0.54 is not the
    argmax, so it must not be reported as counter-evidence even though its
    contradiction mass is high."""
    result = build_pipeline(
        "passive", scripted={S_RESULTS_COUNTER: 0.45}
    ).predict_claim(CLAIM)
    assert result["counter_evidence"] is None
    assert result["n_contradictory_candidates"] == 0
    assert result["max_p_contradiction"] == pytest.approx(0.45)


def test_counter_evidence_provenance_is_preserved():
    result = build_pipeline("passive").predict_claim(CLAIM)

    assert result["counter_evidence_doc_id"] == DOC_ID
    # Carry-forward from the RESULTS heading above it.
    assert result["counter_section"] == "results"
    # Selection, not generation: still a verbatim substring of its source.
    assert result["counter_evidence"] in DOC_TEXT
    assert result["counter_pool_index"] == 2
    assert result["counter_rank"] == 0
    assert set(result["counter_distribution"]) == {
        "contradiction",
        "neutral",
        "entailment",
    }


def test_counter_evidence_provenance_across_multiple_documents():
    docs = [
        {"doc_id": "doc-first", "text": " ".join([TITLE, S_OBJECTIVE])},
        {"doc_id": "doc-second", "text": " ".join([TITLE, S_RESULTS_COUNTER])},
    ]
    result = build_pipeline("passive", documents=docs).predict_claim(CLAIM)
    assert result["counter_evidence"] == S_RESULTS_COUNTER
    assert result["counter_evidence_doc_id"] == "doc-second"
    # The prediction still cites the baseline's document, not the prosecutor's.
    assert result["evidence_doc_id"] == "doc-first"


def test_premise_dilution_case_is_flagged():
    """The 8 cases arm H identified: the contradictory sentence WAS in arm D's
    bundle, yet the joined premise still verified neutral. Arm E must mark
    that, because it is a different problem from a selection failure."""
    result = build_pipeline(
        "passive", scripted={S_RESULTS_PRIMARY: 0.95}
    ).predict_claim(CLAIM)

    assert result["counter_evidence"] == S_RESULTS_PRIMARY
    assert result["counter_in_d_selection"] is True
    assert result["counter_selection_rank"] == 0
    # Baseline verdict on the JOINED premise is unaffected and still neutral.
    assert result["verdict"] == PARTIALLY_SUPPORTED


def test_selection_failure_case_is_flagged():
    result = build_pipeline("passive").predict_claim(CLAIM)
    assert result["counter_in_d_selection"] is False
    assert result["counter_selection_rank"] is None


# ==========================================================================
# 7. Determinism and tie-breaking
# ==========================================================================

def test_passive_prosecution_is_deterministic_across_repeats():
    results = [build_pipeline("passive").predict_claim(CLAIM) for _ in range(5)]
    for other in results[1:]:
        assert other == results[0]


def test_ties_resolve_to_lowest_pool_index():
    """Two equally contradictory candidates: the earlier pool position wins,
    every time."""
    scripted = {S_RESULTS_PRIMARY: 0.90, S_RESULTS_COUNTER: 0.90}
    for _ in range(5):
        result = build_pipeline("passive", scripted=scripted).predict_claim(CLAIM)
        assert result["n_contradictory_candidates"] == 2
        assert result["counter_evidence"] == S_RESULTS_PRIMARY
        assert result["counter_pool_index"] == 1


def test_ties_across_documents_resolve_to_earlier_document():
    docs = [
        {"doc_id": "doc-first", "text": " ".join([TITLE, S_RESULTS_COUNTER])},
        {"doc_id": "doc-second", "text": " ".join([TITLE, S_RESULTS_PRIMARY])},
    ]
    result = build_pipeline(
        "passive",
        scripted={S_RESULTS_COUNTER: 0.90, S_RESULTS_PRIMARY: 0.90},
        documents=docs,
    ).predict_claim(CLAIM)
    assert result["counter_evidence_doc_id"] == "doc-first"


def test_counter_rank_reports_pool_wide_contradiction_ordering():
    """counter_rank > 0 means some candidate carried more contradiction mass
    without contradiction being its argmax -- a signal worth keeping."""
    scripted = {S_RESULTS_COUNTER: 0.60, S_OBJECTIVE: 0.49}
    result = build_pipeline("passive", scripted=scripted).predict_claim(CLAIM)
    assert result["counter_evidence"] == S_RESULTS_COUNTER
    assert result["counter_rank"] == 0

    # Now give a non-argmax candidate more contradiction mass than the winner.
    scripted = {S_RESULTS_COUNTER: 0.60, S_OBJECTIVE: 0.80}
    result = build_pipeline("passive", scripted=scripted).predict_claim(CLAIM)
    # S_OBJECTIVE at 0.80 IS argmax-contradiction, so it becomes the counter.
    assert result["counter_evidence"] == S_OBJECTIVE
    assert result["counter_rank"] == 0
    assert result["n_contradictory_candidates"] == 2


# ==========================================================================
# 8. Degradation
# ==========================================================================

def test_empty_candidate_pool_is_handled_safely():
    verifier = FakeVerifier(FakeNLIPipeline())
    record = prosecute(CLAIM, [], verifier)
    assert record["prosecution_attempted"] is True
    assert record["candidate_pool_size"] == 0
    assert record["counter_evidence"] is None
    assert record["max_p_contradiction"] == 0.0
    assert set(record) == PROSECUTION_RESULT_KEYS


def test_candidate_without_text_is_handled_safely():
    verifier = FakeVerifier(FakeNLIPipeline())
    record = prosecute(CLAIM, [{"doc_id": "doc-empty"}], verifier)
    assert record["candidate_pool_size"] == 0
    assert record["counter_evidence"] is None


def test_pipeline_with_empty_document_still_predicts_and_prosecutes():
    docs = [{"doc_id": "doc-empty", "text": ""}]
    arm_d = build_pipeline("off", documents=docs).predict_claim(CLAIM)
    arm_e = build_pipeline("passive", documents=docs).predict_claim(CLAIM)
    assert arm_e["verdict"] == arm_d["verdict"]
    assert arm_e["candidate_pool_size"] == 0
    assert arm_e["counter_evidence"] is None


def test_prosecution_never_called_when_pool_empty_but_record_still_complete():
    record = prosecute(CLAIM, [], FakeVerifier(FakeNLIPipeline()))
    for key in PROSECUTION_RESULT_KEYS:
        assert key in record


# ==========================================================================
# 9. No second model, no second retriever
# ==========================================================================

def test_prosecution_module_imports_no_model_or_retriever():
    """Checked on the AST, not the prose: the module must neither import nor
    construct a verifier, retriever or embedder. Docstrings may of course name
    them."""
    import ast

    tree = ast.parse(inspect.getsource(src.prosecution))
    heavy = {"NLIVerifier", "HybridRetriever", "SentenceTransformer", "CrossEncoder"}

    imported = set()
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)

    assert not (imported & heavy), f"prosecution imports {imported & heavy}"
    assert not (called & heavy), f"prosecution constructs {called & heavy}"
    # Only the extraction pool helpers are imported from the project.
    assert imported == {
        "src.extraction",
        "_collect_candidate_sentences",
        "_normalize_candidates",
    }
    # And nothing from transformers / sentence_transformers at all.
    assert not any(
        name.startswith(("transformers", "sentence_transformers", "chromadb"))
        for name in imported
    )


def test_prosecution_does_not_construct_a_model(monkeypatch):
    """If prosecution ever instantiated DeBERTa, this would raise."""

    def _explode(self, *args, **kwargs):
        raise AssertionError("prosecution must not construct an NLIVerifier")

    monkeypatch.setattr(src.verification.NLIVerifier, "__init__", _explode)

    result = build_pipeline("passive").predict_claim(CLAIM)
    assert result["counter_evidence"] == S_RESULTS_COUNTER


def test_prosecution_reuses_the_injected_verifier_object():
    pipeline = build_pipeline("passive")
    verifier = pipeline.verifier
    pipeline.predict_claim(CLAIM)
    assert pipeline.verifier is verifier
    # Both the baseline verdict and every pool candidate went through the one
    # pipeline object: 1 baseline premise + 4 pool sentences.
    assert verifier.nli_pipeline.scored_premises[0] == pipeline.predict_claim(CLAIM)[
        "evidence"
    ]


def test_prosecution_adds_no_extra_retrieval():
    """Requirement 9: prosecution reuses the candidates the baseline already
    retrieved, so retrieve() is still called exactly once per claim."""
    off = build_pipeline("off")
    off.predict_claim(CLAIM)
    assert off.retriever.retrieve_calls == 1

    passive = build_pipeline("passive")
    passive.predict_claim(CLAIM)
    assert passive.retriever.retrieve_calls == 1


def test_prosecution_scores_exactly_the_pool_once():
    passive = build_pipeline("passive")
    passive.predict_claim(CLAIM)
    premises = passive.verifier.nli_pipeline.scored_premises
    # 1 baseline premise (the joined bundle) + 4 pool sentences, no repeats.
    assert len(premises) == 5
    assert premises[1:] == [
        S_OBJECTIVE,
        S_RESULTS_PRIMARY,
        S_RESULTS_COUNTER,
        S_METHODS,
    ]


# ==========================================================================
# 11-12. Defaults and parameter validation
# ==========================================================================

def test_default_prosecution_mode_is_off_everywhere():
    import eval.evaluate
    import src.pipeline

    assert DEFAULT_PROSECUTION_MODE == "off"
    assert PROSECUTION_MODES == ("off", "passive")

    pipeline_sig = inspect.signature(src.pipeline.JurisVerifyPipeline.__init__)
    assert pipeline_sig.parameters["prosecution_mode"].default == "off"

    eval_sig = inspect.signature(eval.evaluate.run_eval)
    assert eval_sig.parameters["prosecution_mode"].default == "off"


def test_existing_evidence_mode_defaults_untouched():
    """Arm E must not have disturbed the Task B/C knobs."""
    pipeline_sig = inspect.signature(JurisVerifyPipeline.__init__)
    assert pipeline_sig.parameters["evidence_mode"].default == "single"
    assert pipeline_sig.parameters["section_weight"].default == DEFAULT_SECTION_WEIGHT


@pytest.mark.parametrize(
    "mode", ["passive_aggressive", "on", "", None, True, 0, "OFF", "Passive"]
)
def test_invalid_prosecution_mode_rejected(mode):
    with pytest.raises(ValueError):
        validate_prosecution_mode(mode)


@pytest.mark.parametrize("mode", list(PROSECUTION_MODES))
def test_valid_prosecution_modes_accepted(mode):
    assert validate_prosecution_mode(mode) == mode


def test_predict_claim_rejects_an_invalid_mode_set_after_construction():
    pipeline = build_pipeline("off")
    pipeline.prosecution_mode = "sometimes"
    with pytest.raises(ValueError):
        pipeline.predict_claim(CLAIM)


# ==========================================================================
# Scoring primitives and the additive-merge guard
# ==========================================================================

@pytest.mark.parametrize(
    "label, expected",
    [
        ("entailment", SUPPORTED),
        ("contradiction", CONTRADICTED),
        ("neutral", PARTIALLY_SUPPORTED),
        ("something_else", PARTIALLY_SUPPORTED),
    ],
)
def test_verdict_mapping_matches_nliverifier_verify(label, expected):
    """Parity with production: the real NLIVerifier.verify implementation is
    run against a stub pipeline and must agree with the prosecution mapping."""
    assert verdict_from_top_label(label) == expected

    verifier = NLIVerifier.__new__(NLIVerifier)  # no model load
    verifier.nli_pipeline = lambda payload, **kwargs: [
        {"label": label, "score": 0.9}
    ]
    assert verifier.verify("claim", "evidence") == expected


def test_verdict_mapping_is_case_insensitive_like_verify():
    verifier = NLIVerifier.__new__(NLIVerifier)
    verifier.nli_pipeline = lambda payload, **kwargs: [
        {"label": "CONTRADICTION", "score": 0.9}
    ]
    assert verifier.verify("claim", "evidence") == CONTRADICTED
    assert verdict_from_top_label("CONTRADICTION".lower()) == CONTRADICTED


def test_probability_helpers():
    assert p_contradiction({"neutral": 1.0}) == 0.0
    assert p_contradiction({"contradiction": 0.7}) == 0.7
    assert contradiction_margin(
        {"contradiction": 0.7, "neutral": 0.2, "entailment": 0.1}
    ) == pytest.approx(0.5)
    assert contradiction_margin(
        {"contradiction": 0.3, "neutral": 0.6, "entailment": 0.1}
    ) == pytest.approx(-0.3)
    assert contradiction_margin({"contradiction": 0.4}) == pytest.approx(0.4)


def test_nli_results_requires_the_full_distribution():
    """A truncated distribution must raise, not silently corrupt every
    recorded probability."""

    def truncating(payload, top_k=None, batch_size=None):
        return [[{"label": "neutral", "score": 1.0}] for _ in payload]

    with pytest.raises(RuntimeError):
        nli_results(truncating, [("premise", CLAIM)])


def test_nli_results_requests_top_k_none():
    seen = {}

    def spy(payload, top_k=None, batch_size=None):
        seen["top_k"] = top_k
        return [
            [
                {"label": "contradiction", "score": 0.9},
                {"label": "neutral", "score": 0.08},
                {"label": "entailment", "score": 0.02},
            ]
            for _ in payload
        ]

    results = nli_results(spy, [("premise", CLAIM)])
    assert seen["top_k"] is None
    assert results[0][0] == "contradiction"


def test_nli_results_handles_single_and_batch_shapes():
    pipe = FakeNLIPipeline({"a": 0.9})
    single = nli_results(pipe, [("a", CLAIM)])
    assert len(single) == 1 and single[0][0] == "contradiction"
    batch = nli_results(pipe, [("a", CLAIM), ("b", CLAIM)])
    assert len(batch) == 2 and batch[1][0] == "neutral"
    assert nli_results(pipe, []) == []


def test_nli_results_rejects_a_length_mismatch():
    def short(payload, top_k=None, batch_size=None):
        return [
            [
                {"label": "neutral", "score": 0.9},
                {"label": "contradiction", "score": 0.1},
            ]
        ]

    with pytest.raises(RuntimeError):
        nli_results(short, [("a", CLAIM), ("b", CLAIM)])


def test_attach_prosecution_refuses_to_overwrite_baseline_keys():
    with pytest.raises(RuntimeError, match="overwrite"):
        attach_prosecution({"verdict": "SUPPORTED"}, {"verdict": "CONTRADICTED"})


def test_attach_prosecution_rejects_undeclared_keys():
    with pytest.raises(RuntimeError, match="undeclared"):
        attach_prosecution({}, {"counter_evidence": "x", "surprise": 1})


def test_attach_prosecution_is_additive():
    result = {"verdict": SUPPORTED, "evidence": "e"}
    attach_prosecution(result, {"counter_evidence": "c"})
    assert result == {"verdict": SUPPORTED, "evidence": "e", "counter_evidence": "c"}


def test_prosecute_returns_exactly_the_declared_keys():
    record = prosecute(CLAIM, truncated(), FakeVerifier(FakeNLIPipeline({S_RESULTS_COUNTER: 0.95})))
    assert set(record) == PROSECUTION_RESULT_KEYS


# ==========================================================================
# Eval-harness diagnostics (no models, no dataset)
# ==========================================================================

def test_prosecution_diagnostics_aggregate_correctly():
    from eval.evaluate import compute_prosecution_diagnostics

    def rec(gold, counter, p=None, section=None, rank=None, in_sel=False, pool=10):
        return {
            "gold": gold,
            "prosecution_attempted": True,
            "candidate_pool_size": pool,
            "counter_evidence": counter,
            "counter_verdict": CONTRADICTED if counter else None,
            "counter_confidence": p,
            "counter_p_contradiction": p,
            "counter_score": (p - 0.1) if p is not None else None,
            "counter_section": section,
            "counter_rank": rank,
            "counter_in_d_selection": in_sel,
        }

    records = [
        rec("CONTRADICT", "a", 0.95, "results", 0, pool=12),
        rec("CONTRADICT", "b", 0.85, "conclusions", 1, in_sel=True, pool=8),
        rec("CONTRADICT", None, pool=10),
        rec("SUPPORT", "c", 0.75, "results", 0, pool=10),
        rec("SUPPORT", None, pool=10),
        rec("SUPPORT", None, pool=10),
    ]
    diag = compute_prosecution_diagnostics(records)

    assert diag["claims_prosecuted"] == 6
    assert diag["claims_with_counter_evidence"] == 3
    assert diag["counter_evidence_rate"] == 50.0
    assert diag["counter_evidence_by_gold_label"]["CONTRADICT"] == {
        "prosecuted": 3,
        "with_counter_evidence": 2,
        "rate": 66.67,
    }
    assert diag["counter_evidence_by_gold_label"]["SUPPORT"] == {
        "prosecuted": 3,
        "with_counter_evidence": 1,
        "rate": 33.33,
    }
    assert diag["counter_verdict_distribution"] == {CONTRADICTED: 3, "NONE_FOUND": 3}
    assert diag["counter_section_distribution"] == {"results": 2, "conclusions": 1}
    assert diag["mean_candidate_pool_size"] == pytest.approx(10.0)
    assert diag["min_candidate_pool_size"] == 8
    assert diag["max_candidate_pool_size"] == 12
    assert diag["mean_counter_rank"] == pytest.approx(1 / 3, abs=1e-4)
    assert diag["counter_rank_zero_count"] == 2
    assert diag["counter_already_in_baseline_bundle"] == 1
    assert diag["counter_p_contradiction_histogram"]["0.9-1.0"] == 1
    assert diag["counter_p_contradiction_histogram"]["0.8-0.9"] == 1
    assert diag["counter_p_contradiction_histogram"]["0.7-0.8"] == 1
    assert diag["mean_counter_confidence"] == pytest.approx(0.85)


def test_prosecution_diagnostics_handle_no_records():
    from eval.evaluate import compute_prosecution_diagnostics

    diag = compute_prosecution_diagnostics([])
    assert diag["claims_prosecuted"] == 0
    assert diag["claims_with_counter_evidence"] == 0
    assert diag["counter_evidence_rate"] == 0.0
    assert diag["counter_evidence_by_gold_label"] == {}


def test_eval_cli_exposes_prosecution_mode():
    import sys

    import eval.evaluate as E

    argv = sys.argv
    try:
        sys.argv = ["evaluate.py", "--prosecution_mode", "passive"]
        assert E.parse_args().prosecution_mode == "passive"
        sys.argv = ["evaluate.py"]
        assert E.parse_args().prosecution_mode == "off"
    finally:
        sys.argv = argv
