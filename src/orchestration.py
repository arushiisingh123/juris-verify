"""
Orchestration layer for Juris-Verify.

Implements the Synthesizer -> Prosecutor -> Adjudicator loop using LangGraph.
"""

import hashlib
import uuid
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables.config import RunnableConfig

from src.verification import NLIVerifier, map_to_matrix_label
from src.retrieval import HybridRetriever
from src.extraction import extract_best_evidence
from src.matrix import insert_record

class AgentState(TypedDict):
    run_id: str
    claim: str
    evidence: str
    evidence_doc_id: Optional[str]
    summary: Optional[str]
    verdict: Optional[str]
    confidence: Optional[float]
    retries: int  # Number of retries after the first attempt (e.g. max_retries=3 means 4 total attempts)
    feedback: Optional[str]
    status: Optional[str]
    history: List[Dict[str, Any]]


# Global verifier instance to avoid reloading during the loop
_verifier = None

def get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = NLIVerifier()
    return _verifier


def generate_summary(claim: str, evidence: str, feedback: Optional[str]) -> str:
    """Stub Synthesizer LLM."""
    if feedback:
        return f"{claim} [Revised based on feedback: {feedback}]"
    return f"{claim} [Based on: {evidence[:50]}...]"


def synthesizer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Synthesizer node: generates or revises the summary."""
    summary = generate_summary(state["claim"], state["evidence"], state.get("feedback"))
    return {"summary": summary}


def prosecutor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Prosecutor node: checks the summary against the evidence."""
    verifier = get_verifier()
    # verify() takes (claim, evidence) -- we pass the summary as the claim to verify it.
    res = verifier.verify(state["summary"], state["evidence"], return_dict=True)
    return {"verdict": res["verdict"], "confidence": res["confidence"]}


def adjudicator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Adjudicator node: evaluates verdict, generates feedback, manages retries."""
    verdict = state["verdict"]
    retries = state["retries"]
    max_retries = config.get("configurable", {}).get("max_retries", 3)
    
    evidence = state["evidence"]
    evidence_doc_id = state.get("evidence_doc_id")
    
    feedback = None
    status = None
    
    if verdict == "SUPPORTED":
        status = "RESOLVED"
    elif verdict in ["CONTRADICTED", "PARTIALLY_SUPPORTED"]:
        if retries < max_retries:
            # Generate feedback based on whether we have a doc_id to cite
            if evidence_doc_id:
                citation = f"citing {evidence_doc_id}"
            else:
                citation = "from the source"
            
            if verdict == "CONTRADICTED":
                feedback = (
                    f"Include a Conflict Footnote {citation}. "
                    f"The source contradicts the claim with this evidence: '{evidence}'"
                )
            else:
                feedback = (
                    f"Narrow or qualify the claim based on evidence {citation}. "
                    f"The source only partially supports it with this evidence: '{evidence}'"
                )
        else:
            status = "EXHAUSTED"
            
    # Persist the current pass to the Contradiction Matrix
    claim_id = hashlib.sha256(state["claim"].encode('utf-8')).hexdigest()
    matrix_label = map_to_matrix_label(verdict)
    insert_record(
        run_id=state["run_id"],
        claim_id=claim_id,
        source_doc_id=evidence_doc_id,
        label=matrix_label,
        confidence=state.get("confidence"),
        evidence_span=evidence
    )
    
    # Append to history on every pass
    new_history = list(state.get("history", []))
    new_history.append({
        "summary": state["summary"],
        "verdict": verdict,
        "feedback": feedback
    })
    
    return {
        "retries": retries + 1 if verdict != "SUPPORTED" else retries,
        "feedback": feedback,
        "status": status,
        "history": new_history
    }


def should_continue(state: AgentState) -> str:
    """Routing function for the adjudicator conditional edge."""
    if state.get("status") in ["RESOLVED", "EXHAUSTED"]:
        return END
    return "synthesizer_node"


def build_graph():
    """Builds and returns the compiled LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("synthesizer_node", synthesizer_node)
    workflow.add_node("prosecutor_node", prosecutor_node)
    workflow.add_node("adjudicator_node", adjudicator_node)
    
    workflow.set_entry_point("synthesizer_node")
    workflow.add_edge("synthesizer_node", "prosecutor_node")
    workflow.add_edge("prosecutor_node", "adjudicator_node")
    workflow.add_conditional_edges("adjudicator_node", should_continue)
    
    return workflow.compile()


def run_pipeline(claim: str, documents: list, max_retries: int = 3) -> dict:
    """Wires the LangGraph loop into the existing retriever + extractor."""
    retriever = HybridRetriever(documents)
    retrieved_docs_text = retriever.retrieve(claim)
    
    evidence = extract_best_evidence(
        claim, retrieved_docs_text[:3000], retriever.embedding_model
    )
    
    # TODO(README #4): populate real doc_id once retrieval.py returns per-document ids
    evidence_doc_id = None
    
    app = build_graph()
    initial_state = {
        "run_id": str(uuid.uuid4()),
        "claim": claim,
        "evidence": evidence,
        "evidence_doc_id": evidence_doc_id,
        "retries": 0,
        "history": [],
        "summary": None,
        "verdict": None,
        "confidence": None,
        "feedback": None,
        "status": None
    }
    
    return app.invoke(
        initial_state,
        config={"configurable": {"max_retries": max_retries}}
    )


if __name__ == "__main__":
    import json
    
    # Simple Mocking to test the graph logic without loading heavy models
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

    # Override get_verifier for testing
    original_get_verifier = get_verifier
    
    def test_path(name, claim_text, responses, doc_id=None, max_retries=3):
        print(f"--- Testing {name} ---")
        global _verifier
        _verifier = MockVerifier(responses)
        
        app = build_graph()
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

    # 1. CONTRADICTED path (resolves after 1 retry)
    test_path("CONTRADICTED path with doc_id", "Test claim - contradicted", ["CONTRADICTED", "SUPPORTED"], doc_id="doc-123")
    
    # 2. PARTIALLY_SUPPORTED path (resolves after 1 retry, no doc_id)
    test_path("PARTIALLY_SUPPORTED path without doc_id", "Test claim - partially supported", ["PARTIALLY_SUPPORTED", "SUPPORTED"])
    
    # 3. Retries-exhausted path
    test_path("EXHAUSTED path", "Test claim - exhausted", ["CONTRADICTED", "CONTRADICTED", "CONTRADICTED", "CONTRADICTED"], max_retries=3)
    
    # 4. End-to-End wiring via run_pipeline
    print("--- Testing End-to-End run_pipeline ---")
    dummy_documents = [
        {"doc_id": "1", "text": "This is a dummy document. The Earth is actually flat."},
        {"doc_id": "2", "text": "This is another document. The Earth revolves around the sun and is spherical in nature."}
    ]
    
    # We explicitly want to test the REAL HybridRetriever and NLIVerifier here.
    # Note: this will take a while on CPU.
    # Ensure the verifier is reset to None so it loads the real NLIVerifier
    _verifier = None
    
    result = run_pipeline("The Earth is flat.", dummy_documents, max_retries=2)
    print(f"Final Status: {result.get('status')}")
    print(f"Final Evidence extracted: {result.get('evidence')}")
    print("History:")
    for idx, h in enumerate(result.get("history", [])):
        print(f"  [{idx}] Summary: {h['summary']}\n      Verdict: {h['verdict']}, Feedback: {h['feedback']}")
    print()
