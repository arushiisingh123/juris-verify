"""Tests for arm F: the Adjudicator (src/prosecution.py, prosecution_mode="active").

CPU-only and offline. No DeBERTa and no SentenceTransformer are loaded: a
deterministic FakeEmbedder supplies the selection geometry (the real
`sentence_transformers.util.cos_sim` still does the math) and a DIRECTIONAL
fake NLI pipeline supplies scripted class distributions per (premise,
hypothesis) pair.

Direction matters here in a way it did not for arm E. The Adjudicator's central
gate is SYMMETRY -- a real contradiction holds in both directions, while a
merely topical sentence conflicts in one direction only -- so the fake must be
able to score (counter, claim) differently from (claim, counter). That is the
one capability `test_passive_prosecution.py`'s premise-keyed fake lacks, which
is why this file carries its own.

These tests exercise the REAL `JurisVerifyPipeline.predict_claim`, built via
`__new__` with injected doubles, so production control flow runs end to end --
including the override itself.

Run: pytest test_active_prosecution.py
"""

import inspect
import math

import pytest
import torch

import src.prosecution
import src.verification
from src.pipeline import JurisVerifyPipeline
from src.prosecution import (
    ADJUDICATION_RESULT_KEYS,
    ADJUDICATION_STATUSES,
    CONTRADICTED,
    INSUFFICIENT,
    NOT_CONTRADICTION,
    PARTIALLY_SUPPORTED,
    PROSECUTION_MODES,
    PROSECUTION_RESULT_KEYS,
    SUPPORTED,
    TRUE_CONTRADICTION,
    adjudicate,
    is_finding_grounded,
    prosecute,
)
from src.sections import (
    SECTION_BACKGROUND,
    SECTION_CONCLUSIONS,
    SECTION_METHODS,
    SECTION_RESULTS,
    SECTION_UNKNOWN,
)
from src.verification import NLIVerifier, normalize_for_scifact_eval


# --------------------------------------------------------------------------
# Deterministic offline doubles -- directional
# --------------------------------------------------------------------------

DIM = 6


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def axis_vector(cos_to_claim, axis):
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


class DirectionalNLIPipeline:
    """Scripted NLI that can give (A, B) a different verdict from (B, A).

    `scripted` maps either a (premise, hypothesis) tuple or a bare premise
    string to P(contradiction); the tuple form wins. Unscripted pairs get
    `default_p`, which keeps them neutral.
    """

    def __init__(self, scripted=None, default_p=0.02):
        self.scripted = dict(scripted or {})
        self.default_p = default_p
        self.seen_pairs = []

    def _p(self, premise, hypothesis):
        if (premise, hypothesis) in self.scripted:
            return self.scripted[(premise, hypothesis)]
        if premise in self.scripted:
            return self.scripted[premise]
        return self.default_p

    def _distribution(self, premise, hypothesis):
        contra = self._p(premise, hypothesis)
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
        items = [payload] if isinstance(payload, dict) else payload
        out = []
        for item in items:
            premise, hypothesis = item["text"], item["text_pair"]
            self.seen_pairs.append((premise, hypothesis))
            dist = self._distribution(premise, hypothesis)
            out.append(dist if top_k is None else [dist[0]])
        if isinstance(payload, dict):
            return out[0]
        return out


class EntailingNLIPipeline(DirectionalNLIPipeline):
    """Like the above, but a scripted pair can be forced to ENTAILMENT."""

    def __init__(self, scripted=None, entailing=(), default_p=0.02):
        super().__init__(scripted, default_p)
        self.entailing = set(entailing)

    def _distribution(self, premise, hypothesis):
        if (premise, hypothesis) in self.entailing or premise in self.entailing:
            return sorted(
                [
                    {"label": "entailment", "score": 0.93},
                    {"label": "neutral", "score": 0.06},
                    {"label": "contradiction", "score": 0.01},
                ],
                key=lambda item: -item["score"],
            )
        return super()._distribution(premise, hypothesis)


class FakeVerifier:
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
# Fixture: the arm-D failure mode, with a refutation unselected in the pool
# --------------------------------------------------------------------------

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

# The premise arm D builds: its two spans joined in source order.
PRIMARY_PREMISE = S_OBJECTIVE + " " + S_RESULTS_PRIMARY

ARM_D = {"evidence_mode": "section", "max_sentences": 2, "section_weight": 0.05}

BASELINE_KEYS = {
    "claim",
    "evidence",
    "evidence_doc_id",
    "verdict",
    "matrix_label",
    "evidence_spans",
    "evidence_doc_ids",
}

# A fully symmetric, coherent, finding-grounded contradiction: the counter
# refutes the claim AND the claim refutes the counter.
SYMMETRIC = {
    (S_RESULTS_COUNTER, CLAIM): 0.95,
    (CLAIM, S_RESULTS_COUNTER): 0.93,
}


def build_embedder():
    emb = FakeEmbedder()
    emb.register(CLAIM, axis_vector(1.0, 1))
    emb.register(TITLE, axis_vector(0.10, 1))
    emb.register(S_OBJECTIVE, axis_vector(0.60, 1))
    emb.register(S_RESULTS_PRIMARY, axis_vector(0.56, 2))
    emb.register(S_RESULTS_COUNTER, axis_vector(0.40, 3))
    emb.register(S_METHODS, axis_vector(0.30, 4))
    return emb


def build_pipeline(prosecution_mode="off", pipe=None, documents=None, arm=None):
    pipeline = JurisVerifyPipeline.__new__(JurisVerifyPipeline)
    docs = documents if documents is not None else [{"doc_id": DOC_ID, "text": DOC_TEXT}]
    pipeline.retriever = FakeRetriever(docs, build_embedder())
    pipeline.verifier = FakeVerifier(
        pipe if pipe is not None else DirectionalNLIPipeline(SYMMETRIC)
    )
    config = dict(arm or ARM_D)
    pipeline.evidence_mode = config["evidence_mode"]
    pipeline.max_sentences = config["max_sentences"]
    pipeline.section_weight = config["section_weight"]
    pipeline.prosecution_mode = prosecution_mode
    return pipeline


def truncated(documents=None):
    docs = documents if documents is not None else [{"doc_id": DOC_ID, "text": DOC_TEXT}]
    return [{"doc_id": d["doc_id"], "text": d["text"][:3000]} for d in docs]


def adjudication_for(pipe, primary_verdict=PARTIALLY_SUPPORTED, primary_doc_id=DOC_ID,
                     primary_evidence=PRIMARY_PREMISE, documents=None):
    """Run prosecute + adjudicate directly, bypassing selection."""
    verifier = FakeVerifier(pipe)
    record = prosecute(CLAIM, truncated(documents), verifier)
    return record, adjudicate(
        CLAIM, primary_evidence, primary_verdict, primary_doc_id, record, verifier
    )


# ==========================================================================
# 0. Fixture sanity: arm D really does mispredict this claim
# ==========================================================================

def test_fixture_arm_d_predicts_support_with_a_refutation_in_the_pool():
    result = build_pipeline("off").predict_claim(CLAIM)
    assert result["evidence"] == PRIMARY_PREMISE
    assert result["verdict"] == PARTIALLY_SUPPORTED
    assert normalize_for_scifact_eval(result["verdict"]) == "SUPPORT"
    assert S_RESULTS_COUNTER not in result["evidence"]


# ==========================================================================
# 1, 13. Strong direct contradiction -> override SUPPORT to CONTRADICT
# ==========================================================================

def test_symmetric_grounded_contradiction_overrides_support():
    result = build_pipeline("active").predict_claim(CLAIM)

    assert result["adjudication_status"] == TRUE_CONTRADICTION
    assert result["override_applied"] is True
    assert result["baseline_verdict"] == PARTIALLY_SUPPORTED
    assert result["adjudicated_verdict"] == CONTRADICTED
    assert result["verdict"] == CONTRADICTED
    assert result["matrix_label"] == "CONTRADICT"
    assert normalize_for_scifact_eval(result["verdict"]) == "CONTRADICT"
    assert result["adjudication_reason"] == (
        "symmetric_contradiction_grounded_in_results"
    )
    assert result["adjudication_gates"] == {
        "counter_evidence_present": True,
        "symmetric_contradiction": True,
        "pool_coherent": True,
        "uncontested_support": True,
        "finding_grounded": True,
    }


def test_override_only_ever_happens_on_true_contradiction():
    """Requirement 13, exhaustively: every scripted scenario that does not
    reach TRUE_CONTRADICTION must leave the verdict alone."""
    scenarios = {
        # asymmetric -> NOT_CONTRADICTION
        "asymmetric": DirectionalNLIPipeline({(S_RESULTS_COUNTER, CLAIM): 0.95}),
        # incoherent pool -> NOT_CONTRADICTION
        "incoherent": DirectionalNLIPipeline(
            {**SYMMETRIC, (PRIMARY_PREMISE, S_RESULTS_COUNTER): 0.95}
        ),
        # nothing contradictory at all -> INSUFFICIENT
        "no_counter": DirectionalNLIPipeline({}),
    }
    for name, pipe in scenarios.items():
        arm_d = build_pipeline("off", pipe=pipe).predict_claim(CLAIM)
        arm_f = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
        assert arm_f["adjudication_status"] != TRUE_CONTRADICTION, name
        assert arm_f["override_applied"] is False, name
        assert arm_f["verdict"] == arm_d["verdict"], name
        assert arm_f["matrix_label"] == arm_d["matrix_label"], name


def test_every_override_carries_an_explicit_reason():
    """The critical safety property: there is no implicit
    'counter-evidence exists, therefore override'."""
    result = build_pipeline("active").predict_claim(CLAIM)
    assert result["override_applied"] is True
    assert result["adjudication_status"] == TRUE_CONTRADICTION
    assert result["adjudication_reason"]
    assert isinstance(result["adjudication_reason"], str)
    # And the reason names the grounding section, so it is auditable.
    assert SECTION_RESULTS in result["adjudication_reason"]


# ==========================================================================
# 2. Topically related but non-contradictory -> preserve SUPPORT
# ==========================================================================

def test_asymmetric_contradiction_is_not_a_contradiction():
    """The counter conflicts with the claim but the claim does NOT conflict
    with the counter -- the signature of a sentence that merely carries extra
    scope (a different subgroup, endpoint or comparison). SUPPORT is kept."""
    pipe = DirectionalNLIPipeline({(S_RESULTS_COUNTER, CLAIM): 0.95})
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)

    assert result["counter_evidence"] == S_RESULTS_COUNTER
    assert result["counter_verdict"] == CONTRADICTED  # arm E would have found it
    assert result["reverse_verdict"] == PARTIALLY_SUPPORTED
    assert result["adjudication_status"] == NOT_CONTRADICTION
    assert result["adjudication_reason"] == (
        "asymmetric_contradiction_claim_does_not_refute_counter_evidence"
    )
    assert result["override_applied"] is False
    assert normalize_for_scifact_eval(result["verdict"]) == "SUPPORT"


def test_internally_inconsistent_pool_is_not_a_contradiction():
    """The baseline premise itself refutes the counter-evidence, so the two
    retrieved sentences disagree with each other. That is cross-study
    disagreement, not a refutation of the claim."""
    pipe = DirectionalNLIPipeline(
        {**SYMMETRIC, (PRIMARY_PREMISE, S_RESULTS_COUNTER): 0.95}
    )
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)

    assert result["primary_counter_verdict"] == CONTRADICTED
    assert result["adjudication_status"] == NOT_CONTRADICTION
    assert result["adjudication_reason"] == (
        "primary_evidence_contradicts_counter_evidence_pool_inconsistent"
    )
    assert result["override_applied"] is False
    assert normalize_for_scifact_eval(result["verdict"]) == "SUPPORT"


# ==========================================================================
# 3, 5. Ambiguous / weak counter-evidence -> preserve SUPPORT
# ==========================================================================

def test_no_counter_evidence_yields_insufficient_and_no_override():
    pipe = DirectionalNLIPipeline({})
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)

    assert result["counter_evidence"] is None
    assert result["adjudication_status"] == INSUFFICIENT
    assert result["adjudication_reason"] == "no_counter_evidence"
    assert result["override_applied"] is False
    assert result["adjudication_gates"]["counter_evidence_present"] is False
    assert result["adjudication_gates"]["symmetric_contradiction"] is None


def test_sub_argmax_contradiction_never_reaches_the_adjudicator():
    """A 0.45 contradiction mass under a 0.54 neutral is not counter-evidence
    at all, so no amount of adjudication can act on it."""
    pipe = DirectionalNLIPipeline(
        {(S_RESULTS_COUNTER, CLAIM): 0.45, (CLAIM, S_RESULTS_COUNTER): 0.45}
    )
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
    assert result["counter_evidence"] is None
    assert result["adjudication_status"] == INSUFFICIENT
    assert result["override_applied"] is False


def test_entailing_primary_evidence_blocks_override():
    """When the source positively entails the claim, a contradictory fragment
    elsewhere is not grounds to flip the label. This is the gate that protects
    the confidently-supported population."""
    pipe = EntailingNLIPipeline(SYMMETRIC, entailing={PRIMARY_PREMISE})
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)

    assert result["verdict"] == SUPPORTED or result["baseline_verdict"] == SUPPORTED
    assert result["adjudication_status"] == INSUFFICIENT
    assert result["adjudication_reason"] == "primary_evidence_entails_claim"
    assert result["adjudication_gates"]["uncontested_support"] is False
    assert result["override_applied"] is False
    assert normalize_for_scifact_eval(result["verdict"]) == "SUPPORT"


@pytest.mark.parametrize("section", [SECTION_BACKGROUND, SECTION_METHODS])
def test_non_finding_sections_cannot_ground_an_override(section):
    """A Background sentence may contradict a claim while reporting prior
    literature; a Methods sentence describes design. Neither refutes the claim
    on the study's own evidence."""
    record = {
        "counter_evidence": S_RESULTS_COUNTER,
        "counter_section": section,
        "counter_evidence_doc_id": DOC_ID,
    }
    adjudication = adjudicate(
        CLAIM,
        PRIMARY_PREMISE,
        PARTIALLY_SUPPORTED,
        DOC_ID,
        record,
        FakeVerifier(DirectionalNLIPipeline(SYMMETRIC)),
    )
    assert adjudication["adjudication_status"] == INSUFFICIENT
    assert adjudication["adjudication_reason"] == (
        "counter_evidence_not_finding_grounded_section_" + section
    )
    assert adjudication["adjudication_gates"]["finding_grounded"] is False
    assert adjudication["override_applied"] is False


def test_background_counter_in_the_real_pipeline_is_not_acted_on():
    """End to end: the OBJECTIVE sentence is the contradictory one, and its
    section blocks the override even though the contradiction is symmetric."""
    pipe = DirectionalNLIPipeline(
        {(S_OBJECTIVE, CLAIM): 0.96, (CLAIM, S_OBJECTIVE): 0.94}
    )
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
    assert result["counter_evidence"] == S_OBJECTIVE
    assert result["counter_section"] == SECTION_BACKGROUND
    assert result["adjudication_status"] == INSUFFICIENT
    assert "not_finding_grounded" in result["adjudication_reason"]
    assert result["override_applied"] is False


@pytest.mark.parametrize("section", [SECTION_RESULTS, SECTION_CONCLUSIONS])
def test_finding_sections_ground_an_override(section):
    assert is_finding_grounded(section, "doc-A", "doc-A") is True
    assert is_finding_grounded(section, "doc-B", "doc-A") is True


def test_unknown_section_falls_back_to_provenance():
    """Undecidable from the heading, so allow it only when it comes from the
    document the baseline itself cited."""
    assert is_finding_grounded(SECTION_UNKNOWN, "doc-A", "doc-A") is True
    assert is_finding_grounded(SECTION_UNKNOWN, "doc-B", "doc-A") is False
    assert is_finding_grounded(None, "doc-A", "doc-A") is True
    assert is_finding_grounded(SECTION_UNKNOWN, None, None) is False


def test_cross_document_unknown_section_blocks_override():
    record = {
        "counter_evidence": S_RESULTS_COUNTER,
        "counter_section": SECTION_UNKNOWN,
        "counter_evidence_doc_id": "doc-other",
    }
    adjudication = adjudicate(
        CLAIM,
        PRIMARY_PREMISE,
        PARTIALLY_SUPPORTED,
        DOC_ID,
        record,
        FakeVerifier(DirectionalNLIPipeline(SYMMETRIC)),
    )
    assert adjudication["adjudication_status"] == INSUFFICIENT
    assert "not_finding_grounded" in adjudication["adjudication_reason"]


# ==========================================================================
# 4. Baseline CONTRADICT remains CONTRADICT
# ==========================================================================

def test_baseline_contradict_is_never_downgraded():
    """Arm D already gets 115 CONTRADICT claims right; none may be lost."""
    record = {
        "counter_evidence": S_RESULTS_COUNTER,
        "counter_section": SECTION_RESULTS,
        "counter_evidence_doc_id": DOC_ID,
    }
    adjudication = adjudicate(
        CLAIM,
        PRIMARY_PREMISE,
        CONTRADICTED,
        DOC_ID,
        record,
        FakeVerifier(DirectionalNLIPipeline(SYMMETRIC)),
    )
    assert adjudication["adjudication_status"] == TRUE_CONTRADICTION
    assert adjudication["override_applied"] is False  # nothing to change
    assert adjudication["adjudicated_verdict"] == CONTRADICTED
    assert adjudication["baseline_verdict"] == CONTRADICTED
    assert adjudication["adjudication_reason"].endswith(
        "_baseline_already_contradicted"
    )


@pytest.mark.parametrize(
    "pipe_factory",
    [
        lambda: DirectionalNLIPipeline({}),
        lambda: DirectionalNLIPipeline({(S_RESULTS_COUNTER, CLAIM): 0.95}),
        lambda: EntailingNLIPipeline({}, entailing={S_RESULTS_COUNTER}),
    ],
)
def test_baseline_contradict_survives_every_adjudication_outcome(pipe_factory):
    """Whatever the Prosecutor finds -- nothing, something asymmetric, or
    supporting evidence -- a baseline CONTRADICT stays CONTRADICT."""
    record = prosecute(CLAIM, truncated(), FakeVerifier(pipe_factory()))
    adjudication = adjudicate(
        CLAIM, PRIMARY_PREMISE, CONTRADICTED, DOC_ID, record,
        FakeVerifier(pipe_factory()),
    )
    assert adjudication["adjudicated_verdict"] == CONTRADICTED
    assert adjudication["override_applied"] is False


# ==========================================================================
# 6, 7. Contradiction inside vs outside the baseline bundle
# ==========================================================================

def test_contradiction_outside_the_baseline_bundle_is_adjudicated():
    """The 36 selection-failure cases arm H counted."""
    result = build_pipeline("active").predict_claim(CLAIM)
    assert result["counter_in_d_selection"] is False
    assert result["adjudication_status"] == TRUE_CONTRADICTION
    assert result["override_applied"] is True


def test_contradiction_already_inside_the_baseline_bundle_is_adjudicated():
    """The 8 premise-dilution cases: the contradictory sentence WAS selected,
    but the joined premise still verified neutral. Adjudication must still
    reach it, and must flag that it was already in the bundle."""
    pipe = DirectionalNLIPipeline(
        {
            (S_RESULTS_PRIMARY, CLAIM): 0.95,
            (CLAIM, S_RESULTS_PRIMARY): 0.93,
            # The joined premise stays neutral -- that is the dilution.
        }
    )
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)

    assert result["counter_evidence"] == S_RESULTS_PRIMARY
    assert result["counter_in_d_selection"] is True
    assert result["counter_selection_rank"] == 0
    assert result["verdict"] == CONTRADICTED
    assert result["baseline_verdict"] == PARTIALLY_SUPPORTED
    assert result["override_applied"] is True


# ==========================================================================
# 8. Provenance
# ==========================================================================

def test_provenance_is_preserved_through_an_override():
    result = build_pipeline("active").predict_claim(CLAIM)

    # Baseline citation untouched.
    assert result["evidence"] == PRIMARY_PREMISE
    assert result["evidence_doc_id"] == DOC_ID
    # Counter-evidence provenance recorded separately and verbatim.
    assert result["counter_evidence"] == S_RESULTS_COUNTER
    assert result["counter_evidence"] in DOC_TEXT
    assert result["counter_evidence_doc_id"] == DOC_ID
    assert result["counter_section"] == SECTION_RESULTS
    assert result["counter_pool_index"] == 2


def test_cross_document_provenance_is_recorded():
    docs = [
        {"doc_id": "doc-first", "text": " ".join([TITLE, S_OBJECTIVE, S_RESULTS_PRIMARY])},
        {"doc_id": "doc-second", "text": " ".join([TITLE, "RESULTS " + S_RESULTS_COUNTER])},
    ]
    counter = "RESULTS " + S_RESULTS_COUNTER
    pipe = DirectionalNLIPipeline({(counter, CLAIM): 0.95, (CLAIM, counter): 0.93})
    result = build_pipeline("active", pipe=pipe, documents=docs).predict_claim(CLAIM)

    assert result["counter_evidence"] == counter
    assert result["counter_evidence_doc_id"] == "doc-second"
    assert result["evidence_doc_id"] == "doc-first"
    # Still overridable: a RESULTS section grounds it regardless of document.
    assert result["adjudication_status"] == TRUE_CONTRADICTION


def test_adjudication_records_all_three_nli_directions():
    """Auditability: every probability the decision rests on is recorded."""
    result = build_pipeline("active").predict_claim(CLAIM)

    assert result["counter_p_contradiction"] == pytest.approx(0.95)
    assert result["reverse_p_contradiction"] == pytest.approx(0.93)
    assert result["reverse_verdict"] == CONTRADICTED
    assert result["primary_counter_verdict"] == PARTIALLY_SUPPORTED
    assert result["primary_counter_p_contradiction"] == pytest.approx(0.02)
    assert set(result["primary_verdict_distribution"]) == {
        "contradiction",
        "neutral",
        "entailment",
    }
    assert result["primary_p_contradiction"] == pytest.approx(0.02)


# ==========================================================================
# 9, 10. Empty counter-evidence and determinism
# ==========================================================================

def test_empty_counter_evidence_leaves_baseline_unchanged():
    pipe = DirectionalNLIPipeline({})
    arm_d = build_pipeline("off", pipe=pipe).predict_claim(CLAIM)
    arm_f = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
    for key in BASELINE_KEYS:
        assert arm_f[key] == arm_d[key]


def test_empty_pool_leaves_baseline_unchanged():
    docs = [{"doc_id": "doc-empty", "text": ""}]
    arm_d = build_pipeline("off", documents=docs).predict_claim(CLAIM)
    arm_f = build_pipeline("active", documents=docs).predict_claim(CLAIM)
    for key in BASELINE_KEYS:
        assert arm_f[key] == arm_d[key]
    assert arm_f["adjudication_status"] == INSUFFICIENT
    assert arm_f["override_applied"] is False


def test_missing_primary_evidence_is_handled():
    record = {
        "counter_evidence": S_RESULTS_COUNTER,
        "counter_section": SECTION_RESULTS,
        "counter_evidence_doc_id": DOC_ID,
    }
    adjudication = adjudicate(
        CLAIM, "", PARTIALLY_SUPPORTED, DOC_ID, record,
        FakeVerifier(DirectionalNLIPipeline(SYMMETRIC)),
    )
    assert adjudication["adjudication_reason"] == "no_primary_evidence"
    assert adjudication["override_applied"] is False


def test_active_mode_is_deterministic_across_repeats():
    results = [build_pipeline("active").predict_claim(CLAIM) for _ in range(5)]
    for other in results[1:]:
        assert other == results[0]


def test_adjudication_is_deterministic_for_every_scenario():
    scenarios = [
        DirectionalNLIPipeline(SYMMETRIC),
        DirectionalNLIPipeline({(S_RESULTS_COUNTER, CLAIM): 0.95}),
        DirectionalNLIPipeline({**SYMMETRIC, (PRIMARY_PREMISE, S_RESULTS_COUNTER): 0.95}),
        DirectionalNLIPipeline({}),
    ]
    for pipe_template in scenarios:
        runs = [
            build_pipeline(
                "active",
                pipe=DirectionalNLIPipeline(pipe_template.scripted),
            ).predict_claim(CLAIM)
            for _ in range(3)
        ]
        assert all(run == runs[0] for run in runs)


def test_ties_in_counter_selection_stay_deterministic_under_active_mode():
    pipe = DirectionalNLIPipeline(
        {
            (S_RESULTS_PRIMARY, CLAIM): 0.90,
            (S_RESULTS_COUNTER, CLAIM): 0.90,
            (CLAIM, S_RESULTS_PRIMARY): 0.90,
            (CLAIM, S_RESULTS_COUNTER): 0.90,
        }
    )
    for _ in range(5):
        result = build_pipeline(
            "active", pipe=DirectionalNLIPipeline(pipe.scripted)
        ).predict_claim(CLAIM)
        assert result["counter_evidence"] == S_RESULTS_PRIMARY
        assert result["counter_pool_index"] == 1


# ==========================================================================
# 11, 12. Arm D and Arm E behaviour preserved
# ==========================================================================

def test_off_mode_is_exactly_arm_d():
    result = build_pipeline("off").predict_claim(CLAIM)
    assert set(result.keys()) == BASELINE_KEYS
    assert not (set(result) & PROSECUTION_RESULT_KEYS)
    assert not (set(result) & ADJUDICATION_RESULT_KEYS)


def test_passive_mode_is_still_exactly_arm_d_predictions():
    """Requirement 12: adding arm F must not have perturbed arm E."""
    arm_d = build_pipeline("off").predict_claim(CLAIM)
    arm_e = build_pipeline("passive").predict_claim(CLAIM)

    for key in BASELINE_KEYS:
        assert arm_e[key] == arm_d[key]
    assert arm_e["prosecution_mode"] == "passive"
    # Arm E carries no adjudication fields at all.
    assert not (set(arm_e) & ADJUDICATION_RESULT_KEYS)
    assert arm_e["counter_evidence"] == S_RESULTS_COUNTER  # found, not acted on
    assert normalize_for_scifact_eval(arm_e["verdict"]) == "SUPPORT"


def test_passive_and_active_find_the_same_counter_evidence():
    """The two arms differ only in what they DO with the finding, which is what
    makes E a valid control for F."""
    arm_e = build_pipeline("passive").predict_claim(CLAIM)
    arm_f = build_pipeline("active").predict_claim(CLAIM)
    for key in sorted(PROSECUTION_RESULT_KEYS - {"prosecution_mode"}):
        assert arm_e[key] == arm_f[key], key
    assert arm_e["prosecution_mode"] == "passive"
    assert arm_f["prosecution_mode"] == "active"


@pytest.mark.parametrize(
    "arm",
    [
        {"evidence_mode": "single", "max_sentences": 3, "section_weight": 0.05},
        {"evidence_mode": "multi", "max_sentences": 2, "section_weight": 0.05},
        {"evidence_mode": "section", "max_sentences": 1, "section_weight": 0.05},
        {"evidence_mode": "section", "max_sentences": 2, "section_weight": 0.05},
    ],
)
def test_selection_is_identical_across_all_modes(arm):
    """Requirement 16: arm F changes no retrieval or extraction behaviour, so
    the selected evidence must match arm D's for every evidence mode."""
    arm_d = build_pipeline("off", arm=arm).predict_claim(CLAIM)
    arm_f = build_pipeline("active", arm=arm).predict_claim(CLAIM)
    assert arm_f["evidence"] == arm_d["evidence"]
    assert arm_f["evidence_doc_id"] == arm_d["evidence_doc_id"]
    assert arm_f.get("evidence_spans") == arm_d.get("evidence_spans")
    assert arm_f["baseline_verdict"] == arm_d["verdict"]


# ==========================================================================
# 14, 15. No second model, no second retriever
# ==========================================================================

def test_adjudication_does_not_construct_a_model(monkeypatch):
    def _explode(self, *args, **kwargs):
        raise AssertionError("adjudication must not construct an NLIVerifier")

    monkeypatch.setattr(src.verification.NLIVerifier, "__init__", _explode)
    result = build_pipeline("active").predict_claim(CLAIM)
    assert result["override_applied"] is True


def test_active_mode_adds_no_extra_retrieval():
    pipeline = build_pipeline("active")
    pipeline.predict_claim(CLAIM)
    assert pipeline.retriever.retrieve_calls == 1


def test_active_mode_reuses_the_one_verifier_and_costs_three_extra_passes():
    """Arm F = arm E plus exactly three NLI directions, on the same model."""
    passive = build_pipeline("passive")
    passive.predict_claim(CLAIM)
    active = build_pipeline("active")
    active.predict_claim(CLAIM)

    assert len(active.verifier.nli_pipeline.seen_pairs) == len(
        passive.verifier.nli_pipeline.seen_pairs
    ) + 3
    extra = active.verifier.nli_pipeline.seen_pairs[-3:]
    assert extra == [
        (CLAIM, S_RESULTS_COUNTER),                 # symmetry
        (PRIMARY_PREMISE, S_RESULTS_COUNTER),       # coherence
        (PRIMARY_PREMISE, CLAIM),                   # primary distribution
    ]


def test_no_extra_nli_passes_when_there_is_no_counter_evidence():
    """Adjudication is skipped entirely when there is nothing to adjudicate."""
    pipe = DirectionalNLIPipeline({})
    passive = build_pipeline("passive", pipe=DirectionalNLIPipeline({}))
    passive.predict_claim(CLAIM)
    active = build_pipeline("active", pipe=pipe)
    active.predict_claim(CLAIM)
    assert len(active.verifier.nli_pipeline.seen_pairs) == len(
        passive.verifier.nli_pipeline.seen_pairs
    )


def test_prosecution_module_still_imports_nothing_heavy():
    import ast

    tree = ast.parse(inspect.getsource(src.prosecution))
    heavy = {"NLIVerifier", "HybridRetriever", "SentenceTransformer", "CrossEncoder"}
    imported, called = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert not (imported & heavy)
    assert not (called & heavy)


# ==========================================================================
# Record shape, statuses, modes
# ==========================================================================

def test_active_result_key_set_is_exactly_baseline_plus_both_records():
    result = build_pipeline("active").predict_claim(CLAIM)
    assert set(result) == BASELINE_KEYS | PROSECUTION_RESULT_KEYS | ADJUDICATION_RESULT_KEYS


def test_adjudicate_returns_exactly_the_declared_keys():
    _, adjudication = adjudication_for(DirectionalNLIPipeline(SYMMETRIC))
    assert set(adjudication) == ADJUDICATION_RESULT_KEYS


def test_statuses_and_modes_are_declared():
    assert ADJUDICATION_STATUSES == (
        TRUE_CONTRADICTION,
        NOT_CONTRADICTION,
        INSUFFICIENT,
    )
    assert PROSECUTION_MODES == ("off", "passive", "active")
    assert not (PROSECUTION_RESULT_KEYS & ADJUDICATION_RESULT_KEYS)


def test_active_is_a_valid_pipeline_mode():
    import src.pipeline

    sig = inspect.signature(src.pipeline.JurisVerifyPipeline.__init__)
    assert sig.parameters["prosecution_mode"].default == "off"
    build_pipeline("active").predict_claim(CLAIM)  # does not raise


def test_invalid_mode_still_rejected_in_active_world():
    pipeline = build_pipeline("off")
    pipeline.prosecution_mode = "aggressive"
    with pytest.raises(ValueError):
        pipeline.predict_claim(CLAIM)


# ==========================================================================
# The decision rule is not a probability threshold
# ==========================================================================

def test_decision_is_invariant_to_contradiction_magnitude():
    """The core anti-threshold property: hold the DIRECTIONS fixed and sweep
    the probabilities. Every scripted magnitude that keeps contradiction as the
    argmax produces the same decision, so no cut point on P(contradiction) is
    doing the work."""
    statuses = set()
    for p in (0.52, 0.60, 0.75, 0.90, 0.99):
        pipe = DirectionalNLIPipeline(
            {(S_RESULTS_COUNTER, CLAIM): p, (CLAIM, S_RESULTS_COUNTER): p}
        )
        result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
        statuses.add((result["adjudication_status"], result["override_applied"]))
    assert statuses == {(TRUE_CONTRADICTION, True)}


def test_highest_possible_confidence_still_loses_to_a_failed_gate():
    """And the converse: a maximally confident one-directional contradiction is
    still rejected. Confidence cannot buy an override."""
    pipe = DirectionalNLIPipeline({(S_RESULTS_COUNTER, CLAIM): 0.999})
    result = build_pipeline("active", pipe=pipe).predict_claim(CLAIM)
    assert result["counter_p_contradiction"] == pytest.approx(0.999)
    assert result["adjudication_status"] == NOT_CONTRADICTION
    assert result["override_applied"] is False


def test_adjudicator_compares_no_numbers():
    """Structural guard against the rule quietly becoming a threshold.

    A threshold is, precisely, a comparison against a numeric constant. So the
    check is not "no floats appear" -- `dict.get(key, 0.0)` defaults are fine
    for *recording* a value -- but that no number participates in any
    comparison inside the adjudicator. Every gate must be a label equality or a
    set membership test.
    """
    import ast

    for function in (src.prosecution.adjudicate, src.prosecution.is_finding_grounded):
        tree = ast.parse(inspect.getsource(function).strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                operands = [node.left] + list(node.comparators)
                numeric = [
                    operand.value
                    for operand in operands
                    if isinstance(operand, ast.Constant)
                    and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)
                ]
                assert not numeric, (
                    f"{function.__name__}() compares against {numeric} -- "
                    "that is a threshold"
                )


# ==========================================================================
# Eval-harness adjudication diagnostics (no models, no dataset)
# ==========================================================================

def test_adjudication_diagnostics_report_the_tradeoff():
    from eval.evaluate import compute_adjudication_diagnostics

    def rec(index, gold, baseline, prediction, status, override, reason, gates=None):
        return {
            "claim_index": index,
            "gold": gold,
            "baseline_prediction": baseline,
            "prediction": prediction,
            "adjudication_status": status,
            "adjudication_reason": reason,
            "override_applied": override,
            "adjudication_gates": gates or {},
            "counter_section": SECTION_RESULTS,
            "counter_evidence": "x",
            "counter_evidence_doc_id": "d1",
            "evidence_doc_id": "d1",
        }

    records = [
        # two genuine rescues
        rec(1, "CONTRADICT", "SUPPORT", "CONTRADICT", TRUE_CONTRADICTION, True, "r"),
        rec(2, "CONTRADICT", "SUPPORT", "CONTRADICT", TRUE_CONTRADICTION, True, "r"),
        # one regression
        rec(3, "SUPPORT", "SUPPORT", "CONTRADICT", TRUE_CONTRADICTION, True, "r"),
        # correctly left alone
        rec(4, "SUPPORT", "SUPPORT", "SUPPORT", NOT_CONTRADICTION, False, "asym",
            {"symmetric_contradiction": False}),
        rec(5, "CONTRADICT", "SUPPORT", "SUPPORT", INSUFFICIENT, False, "nog",
            {"finding_grounded": False}),
        rec(6, "CONTRADICT", "CONTRADICT", "CONTRADICT", INSUFFICIENT, False, "noc"),
    ]
    diag = compute_adjudication_diagnostics(records)

    assert diag["claims_adjudicated"] == 6
    assert diag["overrides_applied"] == 3
    assert diag["contradict_rescued"] == 2
    assert diag["support_regressions"] == 1
    assert diag["net_claims_gained"] == 1
    assert diag["override_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert diag["adjudication_status_counts"][TRUE_CONTRADICTION] == 3
    assert diag["adjudication_status_counts"][NOT_CONTRADICTION] == 1
    assert diag["adjudication_status_counts"][INSUFFICIENT] == 2
    assert diag["gate_failure_counts"] == {
        "symmetric_contradiction": 1,
        "finding_grounded": 1,
    }
    # Arm D: claims 1,2,5 wrong-as-SUPPORT, 3,4 right, 6 right -> 3/6
    assert diag["baseline_accuracy"]["correct"] == 3
    # Arm F: 1,2 rescued, 3 regressed -> 4/6
    assert diag["adjudicated_accuracy"]["correct"] == 4
    assert len(diag["overridden_claims"]) == 3
    assert all(c["adjudication_reason"] for c in diag["overridden_claims"])


def test_adjudication_diagnostics_handle_no_records():
    from eval.evaluate import compute_adjudication_diagnostics

    diag = compute_adjudication_diagnostics([])
    assert diag["claims_adjudicated"] == 0
    assert diag["overrides_applied"] == 0
    assert diag["override_precision"] == 0.0
    assert diag["net_claims_gained"] == 0


def test_eval_cli_accepts_all_three_modes():
    import sys

    import eval.evaluate as E

    argv = sys.argv
    try:
        for mode in PROSECUTION_MODES:
            sys.argv = ["evaluate.py", "--prosecution_mode", mode]
            assert E.parse_args().prosecution_mode == mode
    finally:
        sys.argv = argv
