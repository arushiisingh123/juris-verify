"""
Verification (the "Prosecutor" building block) for Juris-Verify.

IMPORTANT: Two prior verifiers were tried and rejected:
- FLAN-T5 zero-shot classification into AGREE/CONTRADICT/SCOPE-MISMATCH
  -- too weak to be reliable, notebook flagged it as "bottleneck".
- Mistral-7B-Instruct and Qwen2.5-7B-Instruct (prompted to output
  SUPPORTED/CONTRADICTED/PARTIALLY_SUPPORTED) -- both over-predicted
  "partially supported" on claims that were clearly one-sided.

DeBERTa-v3-large-mnli-fever-anli-ling-wanli (a dedicated NLI model,
not a generative LLM) is the validated verifier: 88% accuracy on a
50-claim held-out SciFact eval. This is the one to keep using --
do not swap back to a generative verifier without re-running the
eval harness in eval/evaluate.py.
"""

from transformers import pipeline

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class NLIVerifier:
    def __init__(self, device=0):
        self.nli_pipeline = pipeline(
            "text-classification", model=NLI_MODEL_NAME, device=device
        )

    def verify(self, claim, evidence):
        """Run NLI: premise=evidence, hypothesis=claim.

        Returns:
            str: one of "SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED"
        """
        result = self.nli_pipeline({"text": evidence, "text_pair": claim})
        if isinstance(result, list):
            result = result[0]

        label = result["label"].lower()

        if label == "entailment":
            return "SUPPORTED"
        elif label == "contradiction":
            return "CONTRADICTED"
        else:
            return "PARTIALLY_SUPPORTED"


def map_to_matrix_label(verdict):
    """Map the verifier's SUPPORTED/CONTRADICTED/PARTIALLY_SUPPORTED
    output onto the project's AGREE/CONTRADICT/SCOPE-MISMATCH matrix.
    """
    if verdict == "SUPPORTED":
        return "AGREE"
    elif verdict == "CONTRADICTED":
        return "CONTRADICT"
    else:
        return "SCOPE-MISMATCH"


def normalize_for_scifact_eval(verdict):
    """Map verifier output onto SciFact's own gold label space
    (SUPPORT / CONTRADICT / NEI) for accuracy scoring against
    evidence_label in the dataset.
    """
    if verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
        return "SUPPORT"
    elif verdict == "CONTRADICTED":
        return "CONTRADICT"
    else:
        return "NEI"
