"""
End-to-end pipeline for Juris-Verify Phase 1.

This is the VALIDATED baseline (88% accuracy, see eval/evaluate.py):

    Claim -> HybridRetriever -> extract_best_evidence -> NLIVerifier -> label

NOT YET BUILT (this is the actual next-phase work, see README.md):
- LangGraph orchestration wiring this into a real Synthesizer <-> Prosecutor
  <-> Adjudicator loop (right now this is a single-pass classifier, not
  an adversarial multi-agent system)
- Adjudicator: when verdict is CONTRADICT, feed the evidence span back to
  a Synthesizer LLM and force a rewrite with a "Conflict Footnote"
- Persisted ContradictionMatrix records (see matrix.py stub)
- Citation URIs (doc_id + span) attached to every correction
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
            dict with claim, evidence, verdict, and matrix_label
        """
        document = self.retriever.retrieve(claim)

        evidence = extract_best_evidence(
            claim, document[:3000], self.retriever.embedding_model
        )

        verdict = self.verifier.verify(claim, evidence)
        matrix_label = map_to_matrix_label(verdict)

        return {
            "claim": claim,
            "evidence": evidence,
            "verdict": verdict,
            "matrix_label": matrix_label,
        }
