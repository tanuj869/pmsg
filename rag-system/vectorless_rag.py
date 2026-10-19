import os
import re
import json
from rank_bm25 import BM25Okapi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "rag-system", "corpus", "corpus.jsonl")

def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer with lowercase normalization."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

class VectorlessBM25Retriever:
    """
    Vectorless RAG engine utilizing BM25 (Best Matching 25) lexical ranking 
    paired with strict metadata filtering. Operates with zero embedding vectors, 
    zero GPU dependencies, and ultra-low latency.
    """
    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._load_corpus()

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.documents.append(item)
                    tokens = tokenize(f"{item.get('title', '')} {item.get('text', '')}")
                    self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5, metadata_filters: dict = None) -> list[dict]:
        """
        Executes BM25 search with optional metadata filtering (e.g. source, category, year).
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Rank all documents
        scored_docs = list(enumerate(doc_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_docs:
            doc = self.documents[idx]
            # Extract metadata from top-level keys or nested dict
            meta = doc.get("metadata", {}).copy()
            for key in ["id", "source", "section", "title"]:
                if key in doc:
                    meta[key] = doc[key]

            # Apply metadata filters if provided
            if metadata_filters:
                match = True
                for k, v in metadata_filters.items():
                    val = meta.get(k)
                    if isinstance(v, list):
                        if val not in v:
                            match = False
                            break
                    elif str(val).lower() != str(v).lower():
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "text": doc.get("text"),
                "bm25_score": round(float(score), 4),
                "metadata": meta
            })

            if len(results) >= top_k:
                break

        return results

# Singleton instance
_vectorless_retriever = None

def get_vectorless_retriever() -> VectorlessBM25Retriever:
    global _vectorless_retriever
    if _vectorless_retriever is None:
        _vectorless_retriever = VectorlessBM25Retriever()
    return _vectorless_retriever

if __name__ == "__main__":
    print(f"Loading Vectorless BM25 RAG index from: {CORPUS_PATH}")
    retriever = get_vectorless_retriever()
    print(f"Indexed {len(retriever.documents)} chunks successfully.")

    # Test Query 1: Empirical Subsidy Paradox
    q1 = "What is the subsidy paradox in rooftop solar?"
    print(f"\n--- Testing BM25 Search: '{q1}' ---")
    hits1 = retriever.search(q1, top_k=3)
    for i, h in enumerate(hits1, 1):
        print(f"  [{i}] Score: {h['bm25_score']} | ID: {h['id']} | Title: {h['title']}")

    # Test Query 2: With Metadata Filter (Literature only)
    q2 = "machine learning customer acquisition"
    print(f"\n--- Testing Filtered BM25: '{q2}' (source=litreview) ---")
    hits2 = retriever.search(q2, top_k=2, metadata_filters={"source": "litreview"})
    for i, h in enumerate(hits2, 1):
        print(f"  [{i}] Score: {h['bm25_score']} | ID: {h['id']} | Title: {h['title']}")
