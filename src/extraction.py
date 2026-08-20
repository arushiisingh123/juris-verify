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
    the leading title (SciFact docs are 'title + abstract')."""
    sentences = re.split(r"(?<=[.!?])\s+", document)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) > 1:
        sentences = sentences[1:]  # drop title

    return sentences


def extract_best_evidence(claim, document, embedder):
    """Select the single sentence in `document` most similar to `claim`.

    Args:
        embedder: a SentenceTransformer instance (reuse the retriever's
            embedding_model to avoid loading the model twice)

    Returns:
        str: the extracted sentence, guaranteed to be a verbatim
        substring of `document`
    """
    sentences = split_sentences(document)
    if not sentences:
        return ""

    claim_embedding = embedder.encode(claim, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
    best_idx = scores.argmax().item()

    return sentences[best_idx]
