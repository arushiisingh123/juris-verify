import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import (
    load_scifact,
    get_valid_claims,
    get_unique_claims,
    build_document_corpus,
)
from src.retrieval import HybridRetriever
from src.extraction import extract_best_evidence

def inspect_retrieval_paths():
    claims_dataset, corpus_dataset = load_scifact()
    valid_claims = get_valid_claims(claims_dataset)
    unique_claims = get_unique_claims(valid_claims)
    documents = build_document_corpus(corpus_dataset)

    retriever = HybridRetriever(documents)
    embedder = retriever.embedding_model

    semantic_only_hits = []
    bm25_only_hits = []
    both_hits = []

    for idx, item in enumerate(unique_claims[:50]):
        claim = item["claim"]
        
        # 1. Inspect semantic search alone
        query_embedding = retriever.embedding_model.encode(claim).tolist()
        semantic_results = retriever.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"],
        )
        semantic_doc_ids = [str(meta["doc_id"]) for meta in semantic_results["metadatas"][0]]
        
        # 2. Inspect BM25 search alone
        tokenized_query = claim.lower().split()
        bm25_results = retriever.bm25.get_top_n(tokenized_query, retriever.documents, n=3)
        bm25_doc_ids = [str(doc["doc_id"]) for doc in bm25_results]
        
        # 3. Full hybrid retrieval
        candidate_docs = retriever.retrieve(claim)
        
        # 4. Extract best evidence
        extraction = extract_best_evidence(claim, candidate_docs, embedder)
        winning_doc_id = str(extraction["doc_id"]) if extraction["doc_id"] is not None else None
        
        in_semantic = winning_doc_id in semantic_doc_ids
        in_bm25 = winning_doc_id in bm25_doc_ids
        
        info = {
            "claim_idx": idx,
            "claim": claim,
            "winning_doc_id": winning_doc_id,
            "evidence": extraction["sentence"],
            "semantic_doc_ids": semantic_doc_ids,
            "bm25_doc_ids": bm25_doc_ids,
            "in_semantic": in_semantic,
            "in_bm25": in_bm25
        }
        
        if in_semantic and not in_bm25:
            semantic_only_hits.append(info)
        elif in_bm25 and not in_semantic:
            bm25_only_hits.append(info)
        elif in_semantic and in_bm25:
            both_hits.append(info)

    print("================ RETRIEVAL PATH ANALYSIS ================")
    print(f"Total evaluated claims: 50")
    print(f"Evidence from Semantic-only hits: {len(semantic_only_hits)}")
    print(f"Evidence from BM25-only hits:     {len(bm25_only_hits)}")
    print(f"Evidence from Both paths:         {len(both_hits)}")
    print("=========================================================\n")

    print("--- SAMPLE SEMANTIC-ONLY RETRIEVAL HIT ---")
    if semantic_only_hits:
        s = semantic_only_hits[0]
        print(f"Claim: {s['claim']}")
        print(f"Winning Doc ID: {s['winning_doc_id']}")
        print(f"Evidence Sentence: {s['evidence']}")
        print(f"Semantic Candidate IDs: {s['semantic_doc_ids']}")
        print(f"BM25 Candidate IDs: {s['bm25_doc_ids']}")
    else:
        print("None in sample")

    print("\n--- SAMPLE BM25-ONLY RETRIEVAL HIT ---")
    if bm25_only_hits:
        b = bm25_only_hits[0]
        print(f"Claim: {b['claim']}")
        print(f"Winning Doc ID: {b['winning_doc_id']}")
        print(f"Evidence Sentence: {b['evidence']}")
        print(f"Semantic Candidate IDs: {b['semantic_doc_ids']}")
        print(f"BM25 Candidate IDs: {b['bm25_doc_ids']}")
    else:
        print("None in sample")

    print("\n--- SAMPLE BOTH PATHS HIT ---")
    if both_hits:
        h = both_hits[0]
        print(f"Claim: {h['claim']}")
        print(f"Winning Doc ID: {h['winning_doc_id']}")
        print(f"Evidence Sentence: {h['evidence']}")

if __name__ == "__main__":
    inspect_retrieval_paths()
