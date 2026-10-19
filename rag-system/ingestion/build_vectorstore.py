import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "rag-system", "corpus", "corpus.jsonl")
DB_PATH = os.path.join(PROJECT_ROOT, "rag-system", "chroma_db")

def build_vectorstore():
    print(f"Loading corpus from: {CORPUS_PATH}")
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    ids = []
    documents = []
    metadatas = []

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ids.append(item["id"])
            documents.append(item["text"])
            metadatas.append({
                "source": item.get("source", ""),
                "section": item.get("section", ""),
                "title": item.get("title", "")
            })

    total_chunks = len(ids)
    print(f"Loaded {total_chunks} chunks from corpus.")

    print("Initializing SentenceTransformer('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    embeddings_list = embeddings.tolist()

    print(f"Connecting to persistent ChromaDB at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Reset or get collection
    try:
        client.delete_collection("pmsg_corpus")
        print("Existing 'pmsg_corpus' collection deleted for fresh rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name="pmsg_corpus",
        metadata={"hnsw:space": "cosine"}
    )

    print("Adding chunks to Chroma collection...")
    # Add in batches to stay within safe chunk limits
    batch_size = 50
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings_list[i:end],
            metadatas=metadatas[i:end]
        )

    count = collection.count()
    print(f"Vector store build complete! Collection count: {count} (matches corpus: {count == total_chunks})")
    assert count == total_chunks, f"Collection count ({count}) does not match corpus line count ({total_chunks})!"
    return count

if __name__ == "__main__":
    build_vectorstore()
