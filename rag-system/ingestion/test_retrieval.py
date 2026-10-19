import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DB_PATH = os.path.join(PROJECT_ROOT, "rag-system", "chroma_db")

TEST_QUERIES = [
    "What predicts PM Surya Ghar adoption best?",
    "What is the subsidy paradox?",
    "Which states are high-potential, low-adoption?",
    "Why does Random Forest underperform the linear model?",
    "What machine learning methods have been used for solar adoption prediction?"
]

def run_retrieval_tests():
    print(f"Connecting to ChromaDB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("pmsg_corpus")
    print(f"Collection count: {collection.count()} chunks")

    print("Loading embedding model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for i, query in enumerate(TEST_QUERIES, 1):
        print("\n" + "=" * 70)
        print(f"TEST QUERY {i}: {query}")
        print("=" * 70)
        
        q_emb = model.encode([query], normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=q_emb,
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        for rank in range(len(results["ids"][0])):
            doc_id = results["ids"][0][rank]
            dist = results["distances"][0][rank] if "distances" in results else None
            meta = results["metadatas"][0][rank]
            doc = results["documents"][0][rank]
            # Convert cosine distance to similarity score
            sim = 1.0 - dist if dist is not None else None
            
            print(f"Rank {rank+1} [Sim: {sim:.4f} | ID: {doc_id} | Source: {meta.get('source')} | Section: {meta.get('section')}]")
            print(f"Title: {meta.get('title')}")
            print(f"Snippet: {doc[:220]}...\n")

if __name__ == "__main__":
    run_retrieval_tests()
