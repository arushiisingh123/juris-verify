"""
End-to-end pipeline for Juris-Verify Phase 1.

This is the VALIDATED baseline (88% accuracy, see eval/evaluate.py):

    Claim -> HybridRetriever -> extract_best_evidence -> NLIVerifier -> label

evidence_mode defaults to "single", which is exactly that baseline. Passing
evidence_mode="multi" opts into the experimental multi-sentence evidence
bundle, and evidence_mode="section" into the experimental section-aware
ranking on top of it (see src/extraction.py). Nothing enables either
implicitly.

prosecution_mode defaults to "off". Passing prosecution_mode="passive" opts
into the Passive Prosecutor (ablation arm E, see src/prosecution.py), which
searches the same candidate pool for counter-evidence and RECORDS what it
finds. It never alters verdict, evidence, evidence_doc_id or matrix_label --
every prediction-bearing field is computed by the unchanged baseline path
before prosecution runs, and attach_prosecution refuses to overwrite any of
them. Arm E must therefore reproduce arm D's metrics exactly.
"""

from src.retrieval import HybridRetriever
from src.extraction import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_MAX_SENTENCES,
    DEFAULT_SECTION_WEIGHT,
    extract_best_evidence,
)
from src.prosecution import (
    DEFAULT_PROSECUTION_MODE,
    attach_prosecution,
    prosecute,
    validate_prosecution_mode,
)
from src.verification import NLIVerifier, map_to_matrix_label


class JurisVerifyPipeline:
    def __init__(
        self,
        documents,
        evidence_mode=DEFAULT_EVIDENCE_MODE,
        max_sentences=DEFAULT_MAX_SENTENCES,
        section_weight=DEFAULT_SECTION_WEIGHT,
        prosecution_mode=DEFAULT_PROSECUTION_MODE,
    ):
        """
        Args:
            documents: list of {"doc_id": ..., "text": ...} dicts
            evidence_mode: "single" (validated baseline, default), "multi", or
                "section"
            max_sentences: bundle size cap for "multi"/"section" modes only
            section_weight: section-prior magnitude for evidence_mode="section"
                only; ignored by "single" and "multi"
            prosecution_mode: "off" (default) or "passive". "passive" records
                counter-evidence without changing any prediction.
        """
        self.retriever = HybridRetriever(documents)
        self.verifier = NLIVerifier()
        self.evidence_mode = evidence_mode
        self.max_sentences = max_sentences
        self.section_weight = section_weight
        self.prosecution_mode = validate_prosecution_mode(prosecution_mode)

    def predict_claim(self, claim):
        """Run the full pipeline on a single claim.

        Returns:
            dict with claim, evidence, evidence_doc_id, verdict, and matrix_label.
            With evidence_mode="multi"/"section", also evidence_spans and
            evidence_doc_ids. With prosecution_mode="passive", also the
            prosecution record documented in src/prosecution.prosecute -- which
            is purely additive and leaves every field above untouched.
        """
        validate_prosecution_mode(self.prosecution_mode)
        candidate_docs = self.retriever.retrieve(claim)

        truncated_candidates = [
            {"doc_id": doc["doc_id"], "text": doc["text"][:3000]}
            for doc in candidate_docs
        ]

        extraction = extract_best_evidence(
            claim,
            truncated_candidates,
            self.retriever.embedding_model,
            evidence_mode=self.evidence_mode,
            max_sentences=self.max_sentences,
            section_weight=self.section_weight,
        )
        evidence = extraction["sentence"]
        evidence_doc_id = extraction["doc_id"]

        verdict = self.verifier.verify(claim, evidence)
        matrix_label = map_to_matrix_label(verdict)

        result = {
            "claim": claim,
            "evidence": evidence,
            "evidence_doc_id": evidence_doc_id,
            "verdict": verdict,
            "matrix_label": matrix_label,
        }

        # Extra provenance only in the experimental modes, so baseline output
        # keeps exactly the keys it had before.
        if "spans" in extraction:
            result["evidence_spans"] = extraction["spans"]
            result["evidence_doc_ids"] = extraction["doc_ids"]

        # Passive prosecution runs only after every prediction-bearing field
        # above is final, and can only add keys (attach_prosecution raises on
        # any collision). With prosecution_mode="off" nothing below executes,
        # so the baseline result is identical to its pre-arm-E state.
        if self.prosecution_mode == "passive":
            attach_prosecution(
                result,
                prosecute(
                    claim,
                    truncated_candidates,
                    self.verifier,
                    selected_spans=extraction.get("spans"),
                ),
            )

        return result
