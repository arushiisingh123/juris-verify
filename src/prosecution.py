"""
Passive Prosecutor (Novelty Layer 2, ablation arm E) for Juris-Verify.

WHAT THIS IS
------------
An evidence-finding and verification component, NOT an override engine. It
inspects the same candidate sentence pool arm D ranks over, independently
verifies every candidate with the already-loaded NLI model, identifies the
strongest contradictory candidate, and RECORDS what it found.

It never touches the prediction. `verdict`, `evidence`, `evidence_doc_id` and
`matrix_label` are produced by the unchanged baseline path before prosecution
runs, and `attach_prosecution` refuses to overwrite any of them. A
contradiction discovered on a claim arm D called SUPPORT is recorded and
nothing more.

WHY A PASSIVE ARM EXISTS
------------------------
Arm H (oracle ceiling) established that 158/173 gold CONTRADICT claims
(91.33%) already have a raw-CONTRADICTED candidate in the existing retrieval
pool, and that 44 of arm D's 58 CONTRADICT misses (75.86%) are recoverable
without changing retrieval -- 36 selection failures plus 8 premise-dilution
cases. So the remaining problem is calibration, not retrieval.

Arm H also showed the naive rule is unusable: overriding on any contradictory
candidate scores 54.86% precision and drops SUPPORT accuracy to 60.73%, and the
best raw threshold (tau=0.95) still costs SUPPORT 12.4 points. Raw NLI
contradiction is therefore NOT an acceptable decision rule, which is exactly
why arm E overrides nothing.

Arm E's job is to be the measured control: it must reproduce arm D's metrics
exactly while emitting the per-claim prosecution record that a future
adjudicating arm would consume. If arm E's predictions differ from arm D's by
even one claim, the instrumentation is perturbing the measured path and
anything built on top of it is uninterpretable.

NO TUNING HAPPENS HERE
----------------------
Counter-evidence is gated on the NLI model's own argmax verdict being
CONTRADICTED -- the same definition arm H used, so the two are directly
comparable. There is no threshold and no tunable parameter. The recorded
probabilities and margins exist so a later arm can study thresholds offline
without re-running the model.

REUSE, NOT RECONSTRUCTION
-------------------------
  - candidate pool: `src.extraction._collect_candidate_sentences`, the exact
    deduplicated, section-labelled pool arm D ranks over. Re-deriving it would
    risk prosecuting a pool arm D never saw.
  - verifier: the caller's already-constructed NLIVerifier is passed in and its
    public `nli_pipeline` attribute is read. No model is instantiated here --
    this module does not import NLIVerifier at all.
  - no retriever, no embedder, no second DeBERTa.

WHY THE RAW PIPELINE INSTEAD OF NLIVerifier.verify
--------------------------------------------------
`verify()` returns the argmax label plus that label's score, discarding the
other two class probabilities. Prosecution records the full distribution, so it
calls `nli_pipeline` with `top_k=None`. `src/verification.py` is left untouched.
`verdict_from_top_label` below reproduces `verify`'s label mapping exactly, and
`test_passive_prosecution.py` asserts that parity against the real
NLIVerifier.verify implementation (with a stub pipeline, no model load).
"""

from src.extraction import _collect_candidate_sentences, _normalize_candidates


# --- configuration ----------------------------------------------------------
# "off" is the default everywhere. Arm D must stay reproducible without opting
# out of anything.
DEFAULT_PROSECUTION_MODE = "off"
PROSECUTION_MODES = ("off", "passive")

# Verdict strings, matching src/verification.py.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"

# Batch size for scoring the candidate pool. Prosecution adds one NLI forward
# pass per pool sentence (~10-20 per claim), so batching matters; it does not
# affect which candidate is chosen.
DEFAULT_PROSECUTION_BATCH_SIZE = 16

# Every key `prosecute` returns. The pipeline asserts this set is disjoint from
# the baseline result keys, so prosecution cannot shadow a baseline field even
# if either side gains keys later.
PROSECUTION_RESULT_KEYS = frozenset(
    {
        "prosecution_mode",
        "prosecution_attempted",
        "candidate_pool_size",
        "n_contradictory_candidates",
        "max_p_contradiction",
        "counter_evidence",
        "counter_evidence_doc_id",
        "counter_section",
        "counter_verdict",
        "counter_confidence",
        "counter_p_contradiction",
        "counter_score",
        "counter_distribution",
        "counter_rank",
        "counter_pool_index",
        "counter_in_d_selection",
        "counter_selection_rank",
    }
)


def validate_prosecution_mode(prosecution_mode):
    """Raise ValueError unless `prosecution_mode` is a known mode."""
    if prosecution_mode not in PROSECUTION_MODES:
        raise ValueError(
            "prosecution_mode must be one of " + str(PROSECUTION_MODES)
            + " (got " + repr(prosecution_mode) + ")"
        )
    return prosecution_mode


# ==========================================================================
# NLI scoring against the caller's already-loaded model
# ==========================================================================

def verdict_from_top_label(top_label):
    """Map an NLI label to a verdict, replicating NLIVerifier.verify exactly.

    verify() lower-cases the label then branches entailment -> SUPPORTED,
    contradiction -> CONTRADICTED, anything else -> PARTIALLY_SUPPORTED. The
    trailing `else` is reproduced as an else, so an unexpected label behaves
    here exactly as it would in production.
    """
    if top_label == "entailment":
        return SUPPORTED
    elif top_label == "contradiction":
        return CONTRADICTED
    else:
        return PARTIALLY_SUPPORTED


def nli_results(nli_pipeline, pairs, batch_size=DEFAULT_PROSECUTION_BATCH_SIZE):
    """Score (premise, hypothesis) pairs, returning full class distributions.

    Args:
        nli_pipeline: the HF text-classification pipeline held by NLIVerifier
            (`verifier.nli_pipeline`). Not constructed here.
        pairs: list of (premise, hypothesis). Premise is the candidate sentence
            and hypothesis the claim -- the orientation NLIVerifier.verify uses
            (`{"text": evidence, "text_pair": claim}`).
        batch_size: forwarded to the pipeline.

    Returns:
        list of (top_label, distribution) in input order. `top_label` is the
        lower-cased argmax label, read from the highest-scoring entry -- the
        same entry `verify` reads as `result[0]`. `distribution` maps every
        lower-cased label to its probability.

    Raises RuntimeError on an unexpected output shape rather than degrading
    quietly: a silently truncated distribution would corrupt every recorded
    probability and margin.
    """
    if not pairs:
        return []

    inputs = [
        {"text": premise, "text_pair": hypothesis} for premise, hypothesis in pairs
    ]
    raw = nli_pipeline(inputs, top_k=None, batch_size=batch_size)

    # top_k=None yields a score list per input, sorted descending: list[dict]
    # for a single input, list[list[dict]] for several.
    if len(inputs) == 1 and raw and isinstance(raw[0], dict):
        raw = [raw]

    if len(raw) != len(inputs):
        raise RuntimeError(
            "NLI pipeline returned {} results for {} inputs".format(
                len(raw), len(inputs)
            )
        )

    results = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) < 2 or not isinstance(entry[0], dict):
            raise RuntimeError(
                "expected a multi-class score list per input (top_k=None), got: "
                + repr(entry)[:200]
            )
        distribution = {item["label"].lower(): float(item["score"]) for item in entry}
        results.append((entry[0]["label"].lower(), distribution))
    return results


def p_contradiction(distribution):
    """P(contradiction); 0.0 if the model exposes no such class."""
    return float(distribution.get("contradiction", 0.0))


def contradiction_margin(distribution):
    """P(contradiction) minus the best competing class probability.

    Recorded, not thresholded. A positive margin means contradiction is the
    argmax; the magnitude says how decisively. This is the more separable of
    the two signals a later adjudicating arm could study, which is why it is
    captured now rather than requiring a second full run.
    """
    contra = p_contradiction(distribution)
    others = [
        score for label, score in distribution.items() if label != "contradiction"
    ]
    return contra - max(others) if others else contra


# ==========================================================================
# The passive Prosecutor
# ==========================================================================

def _empty_prosecution(mode, pool_size=0):
    return {
        "prosecution_mode": mode,
        "prosecution_attempted": True,
        "candidate_pool_size": pool_size,
        "n_contradictory_candidates": 0,
        "max_p_contradiction": 0.0,
        "counter_evidence": None,
        "counter_evidence_doc_id": None,
        "counter_section": None,
        "counter_verdict": None,
        "counter_confidence": None,
        "counter_p_contradiction": None,
        "counter_score": None,
        "counter_distribution": None,
        "counter_rank": None,
        "counter_pool_index": None,
        "counter_in_d_selection": False,
        "counter_selection_rank": None,
    }


def prosecute(
    claim,
    candidates,
    verifier,
    selected_spans=None,
    batch_size=DEFAULT_PROSECUTION_BATCH_SIZE,
):
    """Independently search the candidate pool for evidence against `claim`.

    Side-effect free with respect to the prediction: this function receives the
    baseline's inputs, never its outputs, and returns a separate record.

    Args:
        claim: the claim being verified (the NLI hypothesis).
        candidates: the SAME truncated [{"doc_id", "text"}] list the baseline
            extractor received, so the pool is identical to arm D's.
        verifier: the caller's already-constructed NLIVerifier. Only its public
            `nli_pipeline` attribute is read; no model is created.
        selected_spans: the baseline extraction's `spans`, when available. Used
            only to record whether the counter-evidence was already inside the
            baseline bundle -- the premise-dilution case arm H measured.
        batch_size: NLI batch size.

    Returns:
        dict with exactly PROSECUTION_RESULT_KEYS:

        prosecution_mode            "passive"
        prosecution_attempted       True whenever this ran
        candidate_pool_size         pool sentences considered
        n_contradictory_candidates  how many scored raw CONTRADICTED
        max_p_contradiction         highest P(contradiction) in the pool,
                                    regardless of argmax
        counter_evidence            the strongest contradictory sentence,
                                    verbatim from its source doc; None if the
                                    pool contains no CONTRADICTED candidate
        counter_evidence_doc_id     its doc_id
        counter_section             its scientific section (src/sections.py)
        counter_verdict             its raw NLI verdict (CONTRADICTED when found)
        counter_confidence          argmax class probability -- NLIVerifier's
                                    notion of confidence
        counter_p_contradiction     P(contradiction) for that candidate
        counter_score               contradiction margin over the runner-up
        counter_distribution        full 3-way distribution
        counter_rank                its 0-based position when the WHOLE pool is
                                    ordered by P(contradiction) descending. >0
                                    means some candidate carried more
                                    contradiction mass without contradiction
                                    being its argmax
        counter_pool_index          its index in the pool in source order
        counter_in_d_selection      whether the baseline bundle already
                                    contained it (premise dilution)
        counter_selection_rank      its baseline selection rank if selected

        Counter-evidence selection: among candidates whose raw NLI verdict is
        CONTRADICTED, the highest P(contradiction); ties resolve to the lowest
        pool index (earlier document, earlier sentence). Fully deterministic,
        with no threshold.
    """
    pool = _collect_candidate_sentences(_normalize_candidates(candidates))
    if not pool:
        return _empty_prosecution("passive", pool_size=0)

    scored = nli_results(
        verifier.nli_pipeline,
        [(item["sentence"], claim) for item in pool],
        batch_size=batch_size,
    )

    selected_rank_by_text = {}
    for span in selected_spans or []:
        selected_rank_by_text.setdefault(span["sentence"], span.get("rank"))

    candidate_records = []
    for pool_index, (item, (top_label, distribution)) in enumerate(zip(pool, scored)):
        candidate_records.append(
            {
                "pool_index": pool_index,
                "sentence": item["sentence"],
                "doc_id": item["doc_id"],
                "section": item["section"],
                "verdict": verdict_from_top_label(top_label),
                "p_contradiction": p_contradiction(distribution),
                "margin": contradiction_margin(distribution),
                "confidence": float(distribution.get(top_label, 0.0)),
                "distribution": distribution,
            }
        )

    # Pool-wide contradiction ranking, used only to report counter_rank.
    by_contradiction = sorted(
        candidate_records,
        key=lambda record: (-record["p_contradiction"], record["pool_index"]),
    )
    rank_by_pool_index = {
        record["pool_index"]: rank for rank, record in enumerate(by_contradiction)
    }

    contradictory = [r for r in candidate_records if r["verdict"] == CONTRADICTED]
    max_p = max(record["p_contradiction"] for record in candidate_records)

    if not contradictory:
        empty = _empty_prosecution("passive", pool_size=len(pool))
        empty["max_p_contradiction"] = max_p
        return empty

    # Strongest contradiction; ties to the earliest pool position.
    counter = min(
        contradictory,
        key=lambda record: (-record["p_contradiction"], record["pool_index"]),
    )

    return {
        "prosecution_mode": "passive",
        "prosecution_attempted": True,
        "candidate_pool_size": len(pool),
        "n_contradictory_candidates": len(contradictory),
        "max_p_contradiction": max_p,
        "counter_evidence": counter["sentence"],
        "counter_evidence_doc_id": counter["doc_id"],
        "counter_section": counter["section"],
        "counter_verdict": counter["verdict"],
        "counter_confidence": counter["confidence"],
        "counter_p_contradiction": counter["p_contradiction"],
        "counter_score": counter["margin"],
        "counter_distribution": counter["distribution"],
        "counter_rank": rank_by_pool_index[counter["pool_index"]],
        "counter_pool_index": counter["pool_index"],
        "counter_in_d_selection": counter["sentence"] in selected_rank_by_text,
        "counter_selection_rank": selected_rank_by_text.get(counter["sentence"]),
    }


def attach_prosecution(result, prosecution):
    """Merge a prosecution record into a baseline result, in place.

    The guard is the point: prosecution must be additive. If a prosecution key
    ever collides with a baseline key this raises instead of overwriting, so a
    future field rename cannot silently corrupt the measured prediction.
    """
    collisions = sorted(set(prosecution) & set(result))
    if collisions:
        raise RuntimeError(
            "prosecution keys would overwrite baseline result keys: "
            + str(collisions)
        )
    unexpected = sorted(set(prosecution) - PROSECUTION_RESULT_KEYS)
    if unexpected:
        raise RuntimeError(
            "prosecution record has undeclared keys: " + str(unexpected)
        )
    result.update(prosecution)
    return result
