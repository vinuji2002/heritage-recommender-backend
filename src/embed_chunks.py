import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.jsonl"
INDEX_PATH = "models/faiss_index.bin"
MAP_PATH = "models/id_map.json"

def main():
    # 1. Load embedding model (small & fast, good for demo)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Read chunks from preprocessing
    texts, meta = [], []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            meta.append(obj)

    print("Loaded", len(texts), "chunks")

    # 3. Create embeddings
    print("Encoding text chunks into embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 4. Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)

    # 5. Save index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved metadata to {MAP_PATH}")
    print("Embedding pipeline complete!")

if __name__ == "__main__":
    main()
