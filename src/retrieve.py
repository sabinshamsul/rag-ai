# Loads the saved FAISS index & chunks, embeds a question with the same model,
# and returns the top-k most semantically similar chunks (nearest neighbours).

import pickle
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_NAME = "all-MiniLM-L6-v2" # Must match the model in embed_index.py

# Load once when imported
_model = None
_index = None
_chunks = None

def _load(): # Load the model, index and chunks the first time retrieval
    global _model, _index, _chunks
    if _model is None:
        print("Loading model and index...")
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(ARTIFACTS_DIR / "index.faiss"))
        with open(ARTIFACTS_DIR / "chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)
        print(f"Ready: {_index.ntotal} vectors loaded.")

def retrieve(query: str, k: int = 5): # Returns the top-k chunks most similar in meaning to query
    _load()

    # Embed the question with the same model & normalisation    
    q_vec = _model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    # FAISS returns similarity scores & positions of nearest vectors
    scores, positions = _index.search(q_vec, k)

    # Map each position back to its real chunk (text & source metadata)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        chunk = _chunks[pos]
        results.append({
            "score": float(score),
            "text": chunk["text"],
            "doc_name": chunk["doc_name"],
            "authority": chunk["authority"],
        })
    return results

# Manual test
if __name__ == "__main__":
    question = "How is a high-risk AI system defined?"
    print(f"\nQUESTION: {question}\n" + "=" * 70)
    for i, r in enumerate(retrieve(question), 1):
        print(f"\n[{i}] score={r['score']:.3f} | {r['doc_name']} ({r['authority']})")
        print(r["text"][:300], "...")