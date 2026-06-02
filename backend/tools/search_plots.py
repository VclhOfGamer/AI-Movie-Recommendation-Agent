"""
    Semantic search over plot summaries.
    Args:
        query: Natural language description (e.g. "dark thriller with a twist ending")
        n_results: How many results to return 
        genre_filter: Optional genre string to narrow results (e.g. "Drama")
    Returns: List of movies with title, genres, plot snippet, similarity score
    """
from pathlib import Path
from functools import lru_cache
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "movie_plots"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_collection = None
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def search_plots(query: str, n_results: int = 5, genre_filter: str | None = None) -> list[dict]:
    
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()[0]

    where = None
    if genre_filter:
        where = {"genres": {"$contains": genre_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["metadatas", "documents", "distances"],
    )

    output = []
    for i, meta in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 3)
        # 180 chars ≈ 2-3 sentences — enough for the model to judge relevance
        plot_snippet = results["documents"][0][i][:180] + "..."

        output.append({
            "movieid": meta.get("movieid"),
            "title": meta.get("title"),
            "genres": meta.get("genres"),
            "plot_snippet": plot_snippet,
            "similarity_score": similarity,
        })

    return output