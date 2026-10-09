"""
Prosecutor and Adjudicator (Novelty Layer 2, ablation arms E and F).

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

=========================================================================
ARM F -- THE ADJUDICATOR (prosecution_mode="active")
=========================================================================

WHY A THRESHOLD CANNOT WORK, FROM THE ARM E DATA
------------------------------------------------
Arm E measured counter-evidence on 289/504 claims: 158/173 gold CONTRADICT and
131/331 gold SUPPORT, at a mean counter confidence of 0.9254. Both populations
carry high-confidence counter-evidence, so P(contradiction) is near-constant
across the very distinction a decision rule has to make. No cut point on that
number can separate them -- arm H's sweep confirmed it empirically, with the
best threshold (tau=0.95) still costing SUPPORT 12.4 points.

So the Adjudicator does not read magnitudes. It asks four categorical
questions about the RELATIONSHIP between claim, primary evidence and
counter-evidence, three of which require information the Prosecutor never
computed.

THE FOUR GATES
--------------
Let C = claim, P = primary (baseline) evidence premise, X = counter-evidence.
Arm E only ever computed NLI(premise=X, hypothesis=C). The Adjudicator adds
NLI(premise=C, hypothesis=X) and NLI(premise=P, hypothesis=X).

1. SYMMETRY -- is this a contradiction at all?
       requires verdict(premise=C, hypothesis=X) == CONTRADICTED
   Logical contradiction is symmetric: if X refutes C then C refutes X. A
   sentence that is merely topical, contextual, or contradictory to some OTHER
   statement is typically asymmetric under NLI -- it conflicts in one direction
   because it carries scope the claim never addresses (a different subgroup, a
   different endpoint, a different comparison), and the reverse direction comes
   back neutral. This gate is the direct implementation of "does the
   counter-evidence actually establish a contradiction to the claim".

2. COHERENCE -- does the retrieved evidence agree with itself?
       requires verdict(premise=P, hypothesis=X) != CONTRADICTED
   If the baseline premise contradicts the counter-evidence, the pool is
   internally inconsistent: two retrieved sentences disagree with each other, so
   the counter cannot be taken to represent the source's position on the claim.
   This is the cross-study case -- `retrieve()` returns top_n=2 documents, and
   the second may describe a different study that genuinely disagrees with the
   first. Overriding there would be adjudicating a disagreement between papers,
   not a refutation of the claim.

3. UNCONTESTED SUPPORT -- is there positive support to overturn?
       requires primary_verdict != SUPPORTED
   If the baseline premise ENTAILS the claim, the source positively supports it
   and a contradictory fragment elsewhere in the pool is not grounds to flip the
   label. This gate protects exactly the population that has the most to lose:
   in the N=504 baseline artifact 86 gold-SUPPORT claims verify as SUPPORTED
   against only 3 gold-CONTRADICT, so the gate is strongly asymmetric in favour
   of preserving SUPPORT accuracy.

4. FINDING-GROUNDED -- is the counter-evidence reporting a result?
       requires the counter-evidence section to be results/conclusions, or --
       when no heading is recoverable -- to come from the same document the
       baseline cited.
   A Background sentence can contradict a claim while describing prior
   literature or the open question rather than this study's finding
   ("Previous trials suggested no benefit"). A Methods sentence describes design.
   Neither refutes the claim on the study's own evidence. This reuses
   src/sections.py, whose directional signal the arm C/D ablation already
   validated: inverting the section prior drops CONTRADICT accuracy from 64.16%
   to 59.54%, so section identity carries stance-relevant information rather
   than incidental reranking noise.

STATUS AND OVERRIDE
-------------------
Gates are evaluated in that order and the FIRST failure determines the status
and the recorded reason:

    no counter-evidence          -> INSUFFICIENT
    symmetry fails               -> NOT_CONTRADICTION  (merely topical)
    coherence fails              -> NOT_CONTRADICTION  (inconsistent pool)
    uncontested-support fails    -> INSUFFICIENT       (source entails claim)
    finding-grounded fails       -> INSUFFICIENT       (not a reported finding)
    all four pass                -> TRUE_CONTRADICTION

Only TRUE_CONTRADICTION may change a prediction, and only in one direction:
baseline SUPPORT -> CONTRADICT. A baseline CONTRADICT is never downgraded, so
the 115 CONTRADICT claims arm D already gets right cannot be lost. Every other
status preserves the baseline verdict byte-for-byte.

Every gate outcome is recorded, not just the deciding one, so the 50-claim
diagnostic shows which gate rejects what. There is no implicit
"counter-evidence exists, therefore override": `override_applied` is True only
alongside an explicit `adjudication_status` and `adjudication_reason`.

WHAT ARM F DOES NOT DO
----------------------
No LLM call, no threshold, no tuned constant -- every gate is an argmax label
comparison or a set membership test, so the whole decision is deterministic and
auditable. It costs 3 extra NLI forward passes per claim that has
counter-evidence, batched into one call, on the same already-loaded model.

Arm F deliberately does NOT rewrite `result["evidence"]` on override. The
premise reported to the evaluator stays the baseline premise, because
`eval/evaluate.py` computes faithfulness from that field and changing it would
change the measured methodology. See the README for why that leaves an
overridden claim citing a premise that no longer matches its verdict, and why
fixing that belongs to the deferred Conflict Footnote work.
"""

from src.extraction import _collect_candidate_sentences, _normalize_candidates
from src.sections import (
    SECTION_BACKGROUND,
    SECTION_CONCLUSIONS,
    SECTION_METHODS,
    SECTION_RESULTS,
    SECTION_UNKNOWN,
)


# --- configuration ----------------------------------------------------------
# "off" is the default everywhere. Arm D must stay reproducible without opting
# out of anything.
DEFAULT_PROSECUTION_MODE = "off"
PROSECUTION_MODES = ("off", "passive", "active")

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


# ==========================================================================
# The Adjudicator (arm F)
# ==========================================================================

# Adjudication statuses. Only TRUE_CONTRADICTION may change a prediction.
TRUE_CONTRADICTION = "TRUE_CONTRADICTION"
NOT_CONTRADICTION = "NOT_CONTRADICTION"
INSUFFICIENT = "INSUFFICIENT"

ADJUDICATION_STATUSES = (TRUE_CONTRADICTION, NOT_CONTRADICTION, INSUFFICIENT)

# Sections whose sentences report a finding, and so can ground an override.
FINDING_SECTIONS = frozenset({SECTION_RESULTS, SECTION_CONCLUSIONS})
# Sections whose sentences describe prior literature, the open question, or
# study design. These can contradict a claim without the study's own evidence
# refuting it, so they cannot ground an override.
NON_FINDING_SECTIONS = frozenset({SECTION_BACKGROUND, SECTION_METHODS})

ADJUDICATION_RESULT_KEYS = frozenset(
    {
        "adjudication_status",
        "adjudication_reason",
        "adjudication_gates",
        "baseline_verdict",
        "adjudicated_verdict",
        "override_applied",
        "primary_verdict_distribution",
        "primary_p_contradiction",
        "primary_confidence",
        "reverse_verdict",
        "reverse_p_contradiction",
        "primary_counter_verdict",
        "primary_counter_p_contradiction",
    }
)


def _empty_adjudication(baseline_verdict, reason):
    return {
        "adjudication_status": INSUFFICIENT,
        "adjudication_reason": reason,
        "adjudication_gates": {
            "counter_evidence_present": False,
            "symmetric_contradiction": None,
            "pool_coherent": None,
            "uncontested_support": None,
            "finding_grounded": None,
        },
        "baseline_verdict": baseline_verdict,
        "adjudicated_verdict": baseline_verdict,
        "override_applied": False,
        "primary_verdict_distribution": None,
        "primary_p_contradiction": None,
        "primary_confidence": None,
        "reverse_verdict": None,
        "reverse_p_contradiction": None,
        "primary_counter_verdict": None,
        "primary_counter_p_contradiction": None,
    }


def is_finding_grounded(counter_section, counter_doc_id, primary_doc_id):
    """Gate 4: can this counter-evidence section ground an override?

    results/conclusions   -> yes, it reports a finding.
    background/methods    -> no, it reports prior work, the open question, or
                             study design.
    unknown               -> undecidable from the heading, so fall back to
                             provenance: allow it only when it comes from the
                             document the baseline itself cited. A cross-document
                             counter whose section cannot even be read is the
                             weakest possible ground for flipping a label.
    """
    if counter_section in FINDING_SECTIONS:
        return True
    if counter_section in NON_FINDING_SECTIONS:
        return False
    return counter_doc_id is not None and counter_doc_id == primary_doc_id


def adjudicate(
    claim,
    primary_evidence,
    primary_verdict,
    primary_doc_id,
    prosecution,
    verifier,
    batch_size=DEFAULT_PROSECUTION_BATCH_SIZE,
):
    """Decide whether the Prosecutor's counter-evidence actually refutes `claim`.

    Deterministic: every gate is an argmax label comparison or a set membership
    test. No threshold, no tuned constant, no sampling.

    Args:
        claim: the claim under test.
        primary_evidence: the baseline premise (the joined bundle arm D built).
        primary_verdict: the baseline NLI verdict on that premise.
        primary_doc_id: doc_id the baseline cited, for the provenance fallback.
        prosecution: the record returned by `prosecute`.
        verifier: the caller's already-constructed NLIVerifier. Only its public
            `nli_pipeline` is read; no model is created.
        batch_size: NLI batch size.

    Returns:
        dict with exactly ADJUDICATION_RESULT_KEYS. `override_applied` is True
        only when the status is TRUE_CONTRADICTION and the baseline verdict is
        not already CONTRADICTED; `adjudicated_verdict` is the baseline verdict
        in every other case.
    """
    counter_evidence = prosecution.get("counter_evidence")
    if not counter_evidence:
        return _empty_adjudication(primary_verdict, "no_counter_evidence")
    if not primary_evidence:
        return _empty_adjudication(primary_verdict, "no_primary_evidence")

    # Three directions arm E never computed, in one batched call:
    #   reverse   : does the claim refute the counter-evidence? (symmetry)
    #   coherence : does the baseline premise refute it?        (consistency)
    #   primary   : the baseline premise's own distribution     (auditability)
    scored = nli_results(
        verifier.nli_pipeline,
        [
            (claim, counter_evidence),
            (primary_evidence, counter_evidence),
            (primary_evidence, claim),
        ],
        batch_size=batch_size,
    )
    (reverse_label, reverse_dist) = scored[0]
    (coherence_label, coherence_dist) = scored[1]
    (primary_label, primary_dist) = scored[2]

    reverse_verdict = verdict_from_top_label(reverse_label)
    primary_counter_verdict = verdict_from_top_label(coherence_label)

    gates = {
        "counter_evidence_present": True,
        # 1. Contradiction holds in BOTH directions, so it is not merely topical.
        "symmetric_contradiction": reverse_verdict == CONTRADICTED,
        # 2. The baseline premise does not itself refute the counter-evidence.
        "pool_coherent": primary_counter_verdict != CONTRADICTED,
        # 3. The source does not positively entail the claim.
        "uncontested_support": primary_verdict != SUPPORTED,
        # 4. The counter-evidence reports a finding.
        "finding_grounded": is_finding_grounded(
            prosecution.get("counter_section"),
            prosecution.get("counter_evidence_doc_id"),
            primary_doc_id,
        ),
    }

    # First failure decides, in order of what the gate actually establishes.
    if not gates["symmetric_contradiction"]:
        status, reason = (
            NOT_CONTRADICTION,
            "asymmetric_contradiction_claim_does_not_refute_counter_evidence",
        )
    elif not gates["pool_coherent"]:
        status, reason = (
            NOT_CONTRADICTION,
            "primary_evidence_contradicts_counter_evidence_pool_inconsistent",
        )
    elif not gates["uncontested_support"]:
        status, reason = INSUFFICIENT, "primary_evidence_entails_claim"
    elif not gates["finding_grounded"]:
        section = prosecution.get("counter_section") or SECTION_UNKNOWN
        status, reason = (
            INSUFFICIENT,
            "counter_evidence_not_finding_grounded_section_" + str(section),
        )
    else:
        section = prosecution.get("counter_section") or SECTION_UNKNOWN
        status, reason = (
            TRUE_CONTRADICTION,
            "symmetric_contradiction_grounded_in_" + str(section),
        )

    # Only a validated contradiction may flip a prediction, and only upward:
    # a baseline CONTRADICT is never downgraded.
    override_applied = status == TRUE_CONTRADICTION and primary_verdict != CONTRADICTED
    if status == TRUE_CONTRADICTION and primary_verdict == CONTRADICTED:
        reason = reason + "_baseline_already_contradicted"

    return {
        "adjudication_status": status,
        "adjudication_reason": reason,
        "adjudication_gates": gates,
        "baseline_verdict": primary_verdict,
        "adjudicated_verdict": CONTRADICTED if override_applied else primary_verdict,
        "override_applied": override_applied,
        "primary_verdict_distribution": primary_dist,
        "primary_p_contradiction": p_contradiction(primary_dist),
        "primary_confidence": float(primary_dist.get(primary_label, 0.0)),
        "reverse_verdict": reverse_verdict,
        "reverse_p_contradiction": p_contradiction(reverse_dist),
        "primary_counter_verdict": primary_counter_verdict,
        "primary_counter_p_contradiction": p_contradiction(coherence_dist),
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
    allowed = PROSECUTION_RESULT_KEYS | ADJUDICATION_RESULT_KEYS
    unexpected = sorted(set(prosecution) - allowed)
    if unexpected:
        raise RuntimeError(
            "prosecution record has undeclared keys: " + str(unexpected)
        )
    result.update(prosecution)
    return result
