"""
indexer.py — run ONCE to build the ChromaDB vector index from plot summaries.
Uses sentence-transformers locally (no API key, no cost).
"""
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

from data.loader import load_movies

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "movie_plots"
BATCH_SIZE = 64
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_index():
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    movies = load_movies()
    movies = movies.dropna(subset=["plot"])
    movies = movies[movies["plot"].str.strip() != ""]
    print(f"Indexing {len(movies)} movies with plots...")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(movies)
    for start in range(0, total, BATCH_SIZE):
        batch = movies.iloc[start: start + BATCH_SIZE]

        # Truncate plots to keep memory reasonable
        texts = [str(p)[:2000] for p in batch["plot"].tolist()]
        ids = batch["movieid"].astype(str).tolist()
        metadatas = [
            {
                "title": str(r.get("title", "")),
                "year": str(r.get("year", "")),
                "genres": str(r.get("genres", "")),
                "movieid": int(r["movieid"]),
            }
            for _, r in batch.iterrows()
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(start + BATCH_SIZE, total)
        print(f"  {done}/{total} indexed...")

    print(f"\nDone! {collection.count()} movies indexed.")
    print(f"Index saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()