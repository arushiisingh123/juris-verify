import sys
from src.orchestration import run_pipeline
from src.data_loader import load_scifact, get_valid_claims, build_document_corpus

def main():
    print("Loading SciFact dataset...")
    claims_ds, corpus_ds = load_scifact()
    valid_claims = get_valid_claims(claims_ds)
    
    # Find a CONTRADICT claim
    contradict_claim_item = None
    for item in valid_claims:
        if item["evidence_label"] == "CONTRADICT":
            contradict_claim_item = item
            break
            
    if not contradict_claim_item:
        print("No CONTRADICT claim found.")
        return
        
    print(f"\nFound CONTRADICT claim: {contradict_claim_item['claim']}")
    
    # Get the corresponding document
    gold_doc_ids = contradict_claim_item["cited_doc_ids"]
    all_docs = build_document_corpus(corpus_ds)
    gold_docs = [doc for doc in all_docs if doc["doc_id"] in gold_doc_ids]
    
    # Run the pipeline with just the gold doc so indexing is instant
    print("\n--- Running End-to-End run_pipeline ---")
    result = run_pipeline(contradict_claim_item["claim"], gold_docs, max_retries=3)
    
    print(f"\nFinal Status: {result.get('status')}")
    print(f"Final Retries: {result.get('retries')}")
    print(f"Final Evidence extracted: {result.get('evidence')}")
    print("History:")
    for idx, h in enumerate(result.get("history", [])):
        print(f"  [{idx}] Summary: {h['summary']}\n      Verdict: {h['verdict']}\n      Feedback: {h['feedback']}")
    print()

if __name__ == "__main__":
    # Remove the mock from src.orchestration if it was left over
    import src.orchestration
    src.orchestration._verifier = None
    main()
