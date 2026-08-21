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
        
    def verify(self, claim, evidence):
        resp = self.responses[self.calls] if self.calls < len(self.responses) else self.responses[-1]
        self.calls += 1
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
        return "\n\n".join(doc["text"] for doc in self.documents)

def mock_extract(claim, doc_text, model):
    # Simply return the first sentence as mock evidence
    return doc_text.split(".")[0] + "."

src.orchestration.HybridRetriever = MockRetriever
src.orchestration.extract_best_evidence = mock_extract
src.orchestration._verifier = MockVerifier(["CONTRADICTED", "SUPPORTED"])


print("--- Testing End-to-End run_pipeline ---")
dummy_documents = [
    {"doc_id": "1", "text": "This is a dummy document. The Earth is actually flat."},
    {"doc_id": "2", "text": "This is another document. The Earth revolves around the sun and is spherical in nature."}
]

result = src.orchestration.run_pipeline("The Earth is flat.", dummy_documents, max_retries=2)

print(f"Final Status: {result.get('status')}")
print(f"Final Evidence extracted: {result.get('evidence')}")
print("History:")
for idx, h in enumerate(result.get("history", [])):
    print(f"  [{idx}] Summary: {h['summary']}\n      Verdict: {h['verdict']}, Feedback: {h['feedback']}")
print()

