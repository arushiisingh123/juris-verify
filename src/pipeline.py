"""
End-to-end pipeline for Juris-Verify Phase 1.

This is the VALIDATED baseline (88% accuracy, see eval/evaluate.py):

    Claim -> HybridRetriever -> extract_best_evidence -> NLIVerifier -> label
"""

from src.retrieval import HybridRetriever
from src.extraction import extract_best_evidence
from src.verification import NLIVerifier, map_to_matrix_label


class JurisVerifyPipeline:
    def __init__(self, documents):
        self.retriever = HybridRetriever(documents)
        self.verifier = NLIVerifier()

    def predict_claim(self, claim):
        """Run the full pipeline on a single claim.

        Returns:
            dict with claim, evidence, evidence_doc_id, verdict, and matrix_label
        """
        candidate_docs = self.retriever.retrieve(claim)

        truncated_candidates = [
            {"doc_id": doc["doc_id"], "text": doc["text"][:3000]}
            for doc in candidate_docs
        ]

        extraction = extract_best_evidence(
            claim, truncated_candidates, self.retriever.embedding_model
        )
        evidence = extraction["sentence"]
        evidence_doc_id = extraction["doc_id"]

        verdict = self.verifier.verify(claim, evidence)
        matrix_label = map_to_matrix_label(verdict)

        return {
            "claim": claim,
            "evidence": evidence,
            "evidence_doc_id": evidence_doc_id,
            "verdict": verdict,
            "matrix_label": matrix_label,
        }
