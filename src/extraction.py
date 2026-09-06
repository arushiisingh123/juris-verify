"""
Evidence extraction for Juris-Verify.

IMPORTANT: A generative extractor (Qwen2.5-7B-Instruct, prompted to
"copy the sentence verbatim") was tried first and rejected -- it
paraphrased/generated from its own knowledge instead of extracting,
which silently breaks citation-grounding (the extracted "evidence"
would not actually appear in the source document).

This embedding-based sentence-ranking extractor replaced it and is
the validated approach: split the doc into sentences, embed each,
cosine-sim against the claim, take the top-scoring sentence. Because
it's a selection over existing text (not generation), the output is
always a substring of the document -- which is exactly what
citation-grounding requires.
"""

import re
from sentence_transformers import util


def split_sentences(document):
    """Split a document into sentences, dropping short fragments and
    the leading title (SciFact docs are 'title + abstract').
    
    NOTE: Title stripping is applied per-document. In the legacy
    single-string implementation, only Doc 1's title was dropped,
    allowing Doc 2's title to leak into sentence candidate pools (e.g.
    Claim #36 where Doc 2's title was previously extracted).
    """
    sentences = re.split(r"(?<=[.!?])\s+", document)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 1:
        sentences = sentences[1:]  # drop title

    return sentences


def extract_best_evidence(claim, candidates, embedder):
    """Select the single sentence across candidate documents most similar to `claim`.

    Args:
        claim: the claim string to find evidence for
        candidates: list of {"doc_id": ..., "text": ...} dicts (or a single candidate dict)
        embedder: a SentenceTransformer instance (reuse the retriever's
            embedding_model to avoid loading the model twice)

    Returns:
        dict: {"sentence": str, "doc_id": Any} where sentence is guaranteed to be a
        verbatim substring of the source document with the given doc_id.
    """
    if isinstance(candidates, dict):
        candidates = [candidates]
    elif not candidates:
        return {"sentence": "", "doc_id": None}

    all_sentences = []
    all_doc_ids = []

    for candidate in candidates:
        text = candidate.get("text", "")
        doc_id = candidate.get("doc_id")

        sentences = split_sentences(text)
        for s in sentences:
            all_sentences.append(s)
            all_doc_ids.append(doc_id)

    if not all_sentences:
        return {"sentence": "", "doc_id": None}

    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(all_sentences, convert_to_tensor=True)

    scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax().item()

    return {
        "sentence": all_sentences[best_idx],
        "doc_id": all_doc_ids[best_idx],
    }
