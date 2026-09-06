"""
Hybrid retrieval for Juris-Verify.

Pipeline: semantic search (SentenceTransformer + ChromaDB) + BM25,
deduplicated, then reranked with a CrossEncoder. This is the exact
approach validated in the original notebook -- do not swap components
without re-running the eval harness in eval/evaluate.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRetriever:
    def __init__(self, documents, collection_name="scifact_collection"):
        """
        Args:
            documents: list of {"doc_id": ..., "text": ...} dicts
        """
        self.documents = documents
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        self._build_semantic_index(collection_name)
        self._build_bm25_index()

    def _build_semantic_index(self, collection_name):
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, batch_size=32
        )

        client = chromadb.Client()
        self.collection = client.create_collection(name=collection_name)

        ids = [str(i) for i in range(len(self.documents))]
        metadatas = [{"doc_id": str(doc["doc_id"])} for doc in self.documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def _build_bm25_index(self):
        tokenized_corpus = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, claim, semantic_k=3, bm25_k=3, top_n=2):
        """Retrieve and rerank documents for a claim.

        Returns:
            list: list of {"doc_id": ..., "text": ...} dicts for the top_n reranked documents
        """
        # Semantic
        query_embedding = self.embedding_model.encode(claim).tolist()
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=semantic_k,
            include=["documents", "metadatas"],
        )
        semantic_docs_text = semantic_results["documents"][0]
        semantic_metadatas = semantic_results["metadatas"][0]

        semantic_candidates = [
            {"doc_id": meta["doc_id"], "text": text}
            for text, meta in zip(semantic_docs_text, semantic_metadatas)
        ]

        # BM25
        tokenized_query = claim.lower().split()
        bm25_results = self.bm25.get_top_n(tokenized_query, self.documents, n=bm25_k)

        # Merge, dedupe by doc_id
        seen_doc_ids = set()
        hybrid_docs = []

        for doc in semantic_candidates:
            doc_id_str = str(doc["doc_id"])
            if doc_id_str not in seen_doc_ids:
                seen_doc_ids.add(doc_id_str)
                hybrid_docs.append(doc)

        for doc in bm25_results:
            doc_id_str = str(doc["doc_id"])
            if doc_id_str not in seen_doc_ids:
                seen_doc_ids.add(doc_id_str)
                hybrid_docs.append(doc)

        # Rerank
        pairs = [(claim, doc["text"]) for doc in hybrid_docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(hybrid_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [ranked[i][0] for i in range(min(top_n, len(ranked)))]

        return top_docs
