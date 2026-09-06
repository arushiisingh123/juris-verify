# NOTE: This test script uses Mock objects for the ML libraries (HybridRetriever, NLIVerifier) to test the graph routing logic in isolation. It does NOT exercise real retrieval or verification.
import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies so we can run the logic locally without a GPU/environment
sys.modules['transformers'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['rank_bm25'] = MagicMock()

# Now we can safely import our modules
from src.orchestration import run_pipeline, _verifier, get_verifier

# Mock the verifier directly for this test
class MockVerifier:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        
    def verify(self, claim, evidence, return_dict=False):
        resp = self.responses[self.calls] if self.calls < len(self.responses) else self.responses[-1]
        self.calls += 1
        if return_dict:
            return {"verdict": resp, "confidence": 0.99}
        return resp

# We also need to mock HybridRetriever and extract_best_evidence to return predictable strings
# instead of calling the mocked SentenceTransformers which would return MagicMocks.
import src.orchestration
from src.retrieval import HybridRetriever

class MockRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.embedding_model = "mock_model"
    def retrieve(self, claim):
        return self.documents

def mock_extract(claim, candidates, model):
    if not candidates:
        return {"sentence": "", "doc_id": None}
    first_doc = candidates[0]
    return {
        "sentence": first_doc["text"].split(".")[0] + ".",
        "doc_id": first_doc.get("doc_id")
    }

src.orchestration.HybridRetriever = MockRetriever
src.orchestration.extract_best_evidence = mock_extract
src.orchestration._verifier = MockVerifier(["CONTRADICTED", "SUPPORTED"])


def test_path(name, claim_text, responses, doc_id=None, max_retries=3):
    print(f"--- Testing {name} ---")
    src.orchestration._verifier = MockVerifier(responses)
    
    app = src.orchestration.build_graph()
    initial_state = {
        "run_id": f"run-{name.split()[0].lower()}",
        "claim": claim_text,
        "evidence": "Test evidence text.",
        "evidence_doc_id": doc_id,
        "retries": 0,
        "history": [],
        "confidence": None
    }
    
    result = app.invoke(
        initial_state,
        config={"configurable": {"max_retries": max_retries}}
    )
    
    print(f"Final Status: {result.get('status')}")
    print(f"Final Retries: {result.get('retries')}")
    print("History:")
    for idx, h in enumerate(result.get("history", [])):
        print(f"  [{idx}] Verdict: {h['verdict']}, Feedback: {h['feedback']}")
    print()

test_path("CONTRADICTED path with doc_id", "Test claim - contradicted", ["CONTRADICTED", "SUPPORTED"], doc_id="doc-123")
test_path("PARTIALLY_SUPPORTED path without doc_id", "Test claim - partially supported", ["PARTIALLY_SUPPORTED", "SUPPORTED"])
test_path("EXHAUSTED path", "Test claim - exhausted", ["CONTRADICTED", "CONTRADICTED", "CONTRADICTED", "CONTRADICTED"], max_retries=3)

print("--- Testing run_pipeline with MockRetriever & MockExtractor ---")
src.orchestration._verifier = MockVerifier(["CONTRADICTED", "SUPPORTED"])
dummy_docs = [{"doc_id": "doc-999", "text": "This is a sentence. This is another sentence."}]
res = run_pipeline("Some claim", dummy_docs, max_retries=2)
print(f"run_pipeline evidence: {res.get('evidence')}")
print(f"run_pipeline evidence_doc_id: {res.get('evidence_doc_id')}")
assert res.get("evidence_doc_id") == "doc-999", f"Expected doc-999, got {res.get('evidence_doc_id')}"
print("Assertion passed: evidence_doc_id correctly propagated!\n")

