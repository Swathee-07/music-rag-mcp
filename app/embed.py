# app/embed.py
from sentence_transformers import SentenceTransformer
import json
import os

# -----------------------------
# Global model load (used by both functions)
# -----------------------------
_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Paths
# -----------------------------
CHUNKS_PATH = os.path.join("processed", "chunks.txt")
OUTPUT_JSON = os.path.join("processed", "embeddings.json")

# -----------------------------
# Function: Embed all chunks and save to JSON
# -----------------------------
def embed_chunks():
    """Create embeddings for all text chunks and save them."""
    if not os.path.exists(CHUNKS_PATH):
        print(f"⚠️ No chunks file found at {CHUNKS_PATH}")
        return

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    embeddings = _model.encode(sentences, show_progress_bar=True)

    to_save = [
        {"sentence": s, "embedding": emb.tolist()}
        for s, emb in zip(sentences, embeddings)
    ]

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(to_save, out, indent=2)

    print(f"✅ Embeddings saved in: {OUTPUT_JSON}")


# -----------------------------
# Function: Single embedding (used in mcp_server.py)
# -----------------------------
def get_embedding(text: str):
    """Generate an embedding for the given input text."""
    if not text or not text.strip():
        return []
    return _model.encode(text).tolist()


# -----------------------------
# Run as script
# -----------------------------
if __name__ == "__main__":
    embed_chunks()
