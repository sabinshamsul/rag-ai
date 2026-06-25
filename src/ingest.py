# Loads the data, joins segment text with document metadata, filters non-useful segments,
# and returns clean text chunks.

import pandas as pd

from pathlib import Path

# Point at the data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_data(): 
    # Encoding UTF-8 removes hiddem BOM (Byte Order Mark) character in CSV files, just in case
    segments = pd.read_csv(DATA_DIR / "segments.csv", encoding="utf-8-sig") # Text is inside segments
    documents = pd.read_csv(DATA_DIR / "documents.csv", encoding="utf-8-sig") # Metada is inside documents
    return segments, documents

def build_chunks():
    segments, documents = load_data()

    print(f"Loaded {len(segments)} segments and {len(documents)} documents.")

    # --- FILTER ---
    # Drop the junk segments
    before = len(segments)
    segments = segments[segments["Non-operative"].astype(str).str.lower() != "true"] # Boilerplate text (cut and paste text)
    segments = segments[segments["Not AI-related"].astype(str).str.lower() != "true"] # Off topic
    segments = segments[segments["Text"].astype(str).str.strip() != ""] # Drop empty & whitespace-only text
    print(f"Filtered {before - len(segments)} junk segments; {len(segments)} remain.")

    # --- JOIN ---
    # Join each segment with the parent-document metadata
    meta_cols = ["AGORA ID", "Official name", "Authority", "Collections"] # Useful metadata columns for citation
    documents_small = documents[meta_cols]

    # segments.csv 'Document ID' = documents.csv 'AGORA ID'
    merged = segments.merge(
        documents_small,
        left_on="Document ID",
        right_on="AGORA ID",
        how="left", # Keep every segment even if metadata is missing
    )

    # --- BUILD ---
    # Each chunk = text + metadata as source citation
    chunks = []
    for _, row in merged.iterrows():
        chunks.append({
            "text": str(row["Text"]).strip(),
            "doc_name": row.get("Official name", "Unknown document"),
            "authority": row.get("Authority", "Unknown authority"),
            "document_id": row.get("Document ID"),
            "segment_position": row.get("Segment position", None),
        })
    print(f"Built {len(chunks)} clean chunks ready for embedding.")
    return chunks

# Sanity check the output
if __name__ == "__main__":
    chunks = build_chunks()
    print("\n--- Sample chunk ---")
    sample = chunks[0]
    print("Source:", sample["doc_name"], "|", sample["authority"]) # Shows law name & authority (citation)
    print("Text preview:", sample["text"][:300], "...") #Shows sample text