import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

CSV_PATH = "data/events.csv"
CHUNKS_OUT = "data/chunks.jsonl"
TOKENIZER_OUT = "models/tokenizer.pkl"
SEQUENCES_OUT = "models/padded_sequences.npy"
PADDED_SHAPE_OUT = "models/padded_shape.txt"

def sentence_chunking(text, max_sentences=2):
    if not text or str(text).strip() == "":
        return []
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    chunks = []
    for i in range(0, len(sents), max_sentences):
        chunk = " ".join([s for s in sents[i:i+max_sentences] if s])
        if chunk:
            chunks.append(chunk)
    return chunks

def main():
    assert os.path.exists(CSV_PATH), f"{CSV_PATH} not found. Create it first."

    # Load dataset
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print("Loaded rows:", len(df))

    # Chunk dataset
    chunks = []
    chunk_id = 0
    for _, r in df.iterrows():
        desc = r.get('Description', '')
        place = r.get('site', '')
        event = r.get('name', '')
        year = f"{r.get('start_year', '')}-{r.get('end_year', '')}"
        c_list = sentence_chunking(desc, max_sentences=2)
        if not c_list:
            c_list = [desc]
        for c in c_list:
            chunks.append({
                "id": chunk_id,
                "place": place,
                "event": event,
                "year": year,
                "text": c
            })
            chunk_id += 1

    print("Total chunks created:", len(chunks))

    # Save chunks
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Tokenizer + Sequences
    texts = [c['text'] for c in chunks]
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding='post')

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    with open(TOKENIZER_OUT, "wb") as f:
        pickle.dump(tokenizer, f)
    np.save(SEQUENCES_OUT, padded)
    with open(PADDED_SHAPE_OUT, "w") as f:
        f.write(str(padded.shape))

    print("Preprocessing done.")
    print("Tokenizer saved to", TOKENIZER_OUT)
    print("Sequences saved to", SEQUENCES_OUT)
    print("Padded shape:", padded.shape)

if __name__ == "__main__":
    main()
