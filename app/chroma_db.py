import os
import json
import chromadb
from chromadb.config import Settings

# ----------------------------
# Paths (robust for Streamlit Cloud)
# ----------------------------
# BASE_DIR points to the repo root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB_PATH = os.path.join(BASE_DIR, 'processed', 'embeddings.json')
CHROMA_PATH = os.path.join(BASE_DIR, 'processed', 'chroma_db')
COLLECTION_NAME = 'music_album_chunks'

# ----------------------------
# Global Chroma client
# ----------------------------
_client = None

def get_client():
    global _client
    if _client is None:
        # Ensure the CHROMA_PATH folder exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

# ----------------------------
# Initialize Chroma DB
# ----------------------------
def initialize_db():
    client = get_client()
    coll = client.get_or_create_collection(COLLECTION_NAME)

    # Only add data if collection is empty
    if coll.count() == 0:
        if not os.path.exists(EMB_PATH):
            raise FileNotFoundError(f"Embedding file not found: {EMB_PATH}")

        with open(EMB_PATH, encoding='utf-8') as f:
            data = json.load(f)

        ids = [str(i) for i in range(len(data))]
        embeddings = [d['embedding'] for d in data]
        metas = [{"sentence": d['sentence']} for d in data]

        coll.add(ids=ids, embeddings=embeddings, metadatas=metas)
        print("Chroma DB initialized with chunks.")
    else:
        print("Chroma DB already populated.")

    return coll

# ----------------------------
# Query Chroma DB
# ----------------------------
def query_db(query_embedding, top_k=3):
    client = get_client()
    coll = client.get_collection(COLLECTION_NAME)
    results = coll.query(query_embeddings=[query_embedding], n_results=top_k)
    return [r['sentence'] for r in results['metadatas'][0]]

# ----------------------------
# Utility: Get Chroma Collection (for Streamlit app)
# ----------------------------
def get_chroma_collection():
    """Return an existing or initialized Chroma collection."""
    client = get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = initialize_db()
    return collection