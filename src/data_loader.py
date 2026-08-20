"""
Data loading for Juris-Verify Phase 1 (Scientific track).

Extracted and cleaned from the original Colab exploration notebook
(Juris_Verify_MVP.ipynb). Loads SciFact claims + corpus and prepares
them for the retrieval pipeline.
"""

from datasets import load_dataset


def load_scifact():
    """Load SciFact claims and corpus splits.

    Returns:
        claims_dataset, corpus_dataset (HuggingFace DatasetDict objects)
    """
    claims_dataset = load_dataset("allenai/scifact", "claims")
    corpus_dataset = load_dataset("allenai/scifact", "corpus")
    return claims_dataset, corpus_dataset


def get_valid_claims(claims_dataset):
    """Filter claims that have a non-empty evidence label."""
    claims_train = claims_dataset["train"]
    return [item for item in claims_train if item["evidence_label"] != ""]


def get_unique_claims(labeled_claims):
    """Dedupe claims by claim text, keeping first occurrence."""
    unique = {}
    for item in labeled_claims:
        claim = item["claim"]
        if claim not in unique:
            unique[claim] = item
    return list(unique.values())


def build_document_corpus(corpus_dataset):
    """Flatten the SciFact corpus into {doc_id, text} records.

    text = title + " " + joined abstract sentences.
    """
    documents = []
    corpus_train = corpus_dataset["train"]

    for i in range(len(corpus_train)):
        title = corpus_train[i]["title"]
        abstract_sentences = corpus_train[i]["abstract"]
        abstract_text = " ".join(abstract_sentences)
        full_text = title + " " + abstract_text

        documents.append({
            "doc_id": corpus_train[i]["doc_id"],
            "text": full_text,
        })

    return documents


def get_gold_documents(claim_item, documents):
    """Look up the ground-truth cited documents for a claim.

    Useful for retrieval-quality evaluation (Recall@k, nDCG) since
    SciFact provides cited_doc_ids per claim.
    """
    gold_doc_ids = claim_item["cited_doc_ids"]
    return [doc["text"] for doc in documents if doc["doc_id"] in gold_doc_ids]
