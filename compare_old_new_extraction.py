import sys
import os
import re
from sentence_transformers import util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.data_loader import (
    load_scifact,
    get_valid_claims,
    get_unique_claims,
    build_document_corpus,
)
from src.retrieval import HybridRetriever
from src.verification import NLIVerifier, normalize_for_scifact_eval

def old_split_sentences(document):
    sentences = re.split(r"(?<=[.!?])\s+", document)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) > 1:
        sentences = sentences[1:]  # drop title
    return sentences

def old_extract_best_evidence(claim, document, embedder):
    sentences = old_split_sentences(document)
    if not sentences:
        return ""
    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax().item()
    return sentences[best_idx]

from src.extraction import extract_best_evidence as new_extract_best_evidence

def main():
    claims_dataset, corpus_dataset = load_scifact()
    valid_claims = get_valid_claims(claims_dataset)
    unique_claims = get_unique_claims(valid_claims)
    documents = build_document_corpus(corpus_dataset)

    retriever = HybridRetriever(documents)
    verifier = NLIVerifier()

    eval_claims = unique_claims[:50]

    differences = []
    old_correct = 0
    new_correct = 0

    old_wrong = []
    new_wrong = []

    for idx, item in enumerate(eval_claims):
        claim_text = item["claim"]
        gold = item["evidence_label"]

        # 1. Retrieve candidate docs
        candidate_docs = retriever.retrieve(claim_text)

        # Old way: joined top_docs string sliced to [:3000]
        old_joined_doc = "\n\n".join(doc["text"] for doc in candidate_docs)[:3000]
        old_evidence = old_extract_best_evidence(claim_text, old_joined_doc, retriever.embedding_model)
        old_verdict = verifier.verify(claim_text, old_evidence)
        old_pred = normalize_for_scifact_eval(old_verdict)

        # New way: truncated candidates list
        truncated_candidates = [
            {"doc_id": doc["doc_id"], "text": doc["text"][:3000]}
            for doc in candidate_docs
        ]
        new_extraction = new_extract_best_evidence(claim_text, truncated_candidates, retriever.embedding_model)
        new_evidence = new_extraction["sentence"]
        new_doc_id = new_extraction["doc_id"]
        new_verdict = verifier.verify(claim_text, new_evidence)
        new_pred = normalize_for_scifact_eval(new_verdict)

        if old_pred == gold:
            old_correct += 1
        else:
            old_wrong.append((idx, claim_text, gold, old_pred, old_evidence))

        if new_pred == gold:
            new_correct += 1
        else:
            new_wrong.append((idx, claim_text, gold, new_pred, new_evidence, new_doc_id))

        if old_pred != new_pred or old_evidence != new_evidence:
            differences.append({
                "idx": idx,
                "claim": claim_text,
                "gold": gold,
                "old_evidence": old_evidence,
                "old_verdict": old_verdict,
                "old_pred": old_pred,
                "new_evidence": new_evidence,
                "new_doc_id": new_doc_id,
                "new_verdict": new_verdict,
                "new_pred": new_pred,
            })

    print(f"Old accuracy: {old_correct}/50 ({round(old_correct/50*100, 2)}%)")
    print(f"New accuracy: {new_correct}/50 ({round(new_correct/50*100, 2)}%)")
    print(f"\nTotal claims with difference in evidence or prediction: {len(differences)}")
    for d in differences:
        print("\n" + "="*80)
        print(f"Claim #{d['idx']}: {d['claim']}")
        print(f"Gold Label: {d['gold']}")
        print(f"Old Pred: {d['old_pred']} ({d['old_verdict']})")
        print(f"Old Evidence: {d['old_evidence']}")
        print(f"New Pred: {d['new_pred']} ({d['new_verdict']}) [Doc ID: {d['new_doc_id']}]")
        print(f"New Evidence: {d['new_evidence']}")

if __name__ == "__main__":
    main()
