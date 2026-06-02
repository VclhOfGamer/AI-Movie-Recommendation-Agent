"""
    Returns a user's rating history plus derived taste signals.
        {
          total_ratings, avg_rating,
          top_rated_movies, recently_rated_movies,
          genre_profile: {genre: {count, avg_rating}},
          blind_spots: [genres not rated],
          top_tags: [user's most applied tags],
        }

"""
import pandas as pd
from collections import defaultdict
from data.loader import load_ratings, load_movies, load_tags, get_rating_matrix


def get_user_history(user_id: int, limit: int = 10) -> dict:
    
    ratings = load_ratings()
    movies = load_movies()
    tags = load_tags()

    user_ratings = ratings[ratings["userid"] == user_id].copy()
    if user_ratings.empty:
        return {"error": f"User {user_id} not found in dataset."}

    user_ratings = user_ratings.sort_values("timestamp", ascending=False)

    merged = user_ratings.merge(movies, on="movieid", how="left")

    # Top rated — stripped to minimal fields (no genres, already in genre_profile)
    top_rated = (
        merged.sort_values("rating", ascending=False)
        .head(limit)[["movieid", "title", "rating"]]
        .to_dict("records")
    )

    recent = (
        merged.head(5)[["movieid", "title", "rating"]]
        .to_dict("records")
    )

    # Genre profile
    genre_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_rating": 0.0})
    for _, row in merged.iterrows():
        genres_str = row.get("genres", "") or ""
        for g in genres_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                genre_stats[g]["count"] += 1
                genre_stats[g]["total_rating"] += row["rating"]

    genre_profile = {}
    for g, stats in sorted(genre_stats.items(), key=lambda x: -x[1]["count"]):
        genre_profile[g] = {
            "count": stats["count"],
            "avg_rating": round(stats["total_rating"] / stats["count"], 2),
        }

    # Blind spots — cap at 8 (the model only needs a few examples)
    all_genres = set()
    for g_str in movies["genres"].dropna():
        for g in g_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                all_genres.add(g)
    blind_spots = sorted(all_genres - set(genre_profile.keys()))[:8]

    # User tags — already capped at 10, keep as-is
    user_tags = tags[tags["userid"] == user_id]["tag"].value_counts().head(10).index.tolist()

    return {
        "user_id": user_id,
        "total_ratings": len(user_ratings),
        "avg_rating": round(user_ratings["rating"].mean(), 2),
        "top_rated_movies": top_rated,
        "recently_rated_movies": recent,
        "genre_profile": genre_profile,
        "blind_spots": blind_spots,
        "top_tags": user_tags,
    }


def get_user_taste_summary(user_id: int) -> str:
    """
    Returns a concise human-readable taste summary for use in system prompts.
    Unchanged — already compact and used only in the (cached) system prompt.
    """
    history = get_user_history(user_id, limit=10)
    if "error" in history:
        return f"Unknown user {user_id}."

    top_genres = list(history["genre_profile"].keys())[:5]
    top_movies = [m["title"] for m in history["top_rated_movies"][:5]]
    avg = history["avg_rating"]
    blind = history["blind_spots"][:5]

    return (
        f"User {user_id} has rated {history['total_ratings']} movies (avg {avg}/5). "
        f"Favorite genres: {', '.join(top_genres)}. "
        f"Top-rated movies include: {', '.join(top_movies)}. "
        f"Genres they haven't explored: {', '.join(blind)}."
    )