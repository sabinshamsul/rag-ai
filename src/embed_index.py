# Embedding - turns each text chunk into a vector (numbers) and stores all vectors
# in a FAISS index for fast nearest neighbour search.

import pickle
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from ingest import build_chunks

# Where the index and chunk metadata is saved
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

def build_index():
    # Get clean chunks from ingest.py
    chunks = build_chunks()
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME} ...")

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Embed all chunks
    embeddings = model.encode(
        texts,
        show_progress_bar=True, # Watch it work
        normalize_embeddings=True, # To pair cleanly with FAISS index type 
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")
    print("Embedding matrix shape:", embeddings.shape)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # How close in meaning
    index.add(embeddings)
    print("Vectors in index:", index.ntotal)

    # Save the index and chunk metadata together, by order
    faiss.write_index(index, str(ARTIFACTS_DIR / "index.faiss"))
    with open (ARTIFACTS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index + {len(chunks)} chunks to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    build_index()