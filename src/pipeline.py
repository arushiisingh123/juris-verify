"""
End-to-end pipeline for Juris-Verify Phase 1.

This is the VALIDATED baseline (88% accuracy, see eval/evaluate.py):

    Claim -> HybridRetriever -> extract_best_evidence -> NLIVerifier -> label

evidence_mode defaults to "single", which is exactly that baseline. Passing
evidence_mode="multi" opts into the experimental multi-sentence evidence
bundle (see src/extraction.py); nothing enables it implicitly.
"""

from src.retrieval import HybridRetriever
from src.extraction import (
    DEFAULT_EVIDENCE_MODE,
    DEFAULT_MAX_SENTENCES,
    extract_best_evidence,
)
from src.verification import NLIVerifier, map_to_matrix_label


class JurisVerifyPipeline:
    def __init__(
        self,
        documents,
        evidence_mode=DEFAULT_EVIDENCE_MODE,
        max_sentences=DEFAULT_MAX_SENTENCES,
    ):
        """
        Args:
            documents: list of {"doc_id": ..., "text": ...} dicts
            evidence_mode: "single" (validated baseline, default) or "multi"
            max_sentences: bundle size cap for evidence_mode="multi" only
        """
        self.retriever = HybridRetriever(documents)
        self.verifier = NLIVerifier()
        self.evidence_mode = evidence_mode
        self.max_sentences = max_sentences

    def predict_claim(self, claim):
        """Run the full pipeline on a single claim.

        Returns:
            dict with claim, evidence, evidence_doc_id, verdict, and matrix_label.
            With evidence_mode="multi", also evidence_spans and evidence_doc_ids.
        """
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

        # Extra provenance only in the experimental mode, so baseline output
        # keeps exactly the keys it had before.
        if "spans" in extraction:
            result["evidence_spans"] = extraction["spans"]
            result["evidence_doc_ids"] = extraction["doc_ids"]

        return result
