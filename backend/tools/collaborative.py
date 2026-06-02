"""
collaborative.py — User-based collaborative filtering + cold start fallback.
"""
import numpy as np
import pandas as pd
from data.loader import get_rating_matrix, get_movie_id_to_title, load_ratings, load_movies


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return 0.0
    a_m, b_m = a[mask], b[mask]
    denom = np.linalg.norm(a_m) * np.linalg.norm(b_m)
    if denom == 0:
        return 0.0
    return float(np.dot(a_m, b_m) / denom)


def find_similar_users(user_id: int, top_k: int = 10) -> list[dict]:
    matrix = get_rating_matrix()
    if user_id not in matrix.index:
        return []
    target = matrix.loc[user_id].values
    similarities = []
    for uid in matrix.index:
        if uid == user_id:
            continue
        other = matrix.loc[uid].values
        sim = _cosine_similarity(target, other)
        if sim > 0:
            mask = ~(np.isnan(target) | np.isnan(other))
            similarities.append({
                "user_id": int(uid),
                "similarity": round(sim, 4),
                "n_common_movies": int(mask.sum()),
            })
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


def get_top_rated_movies(
    min_ratings: int = 20,
    top_k: int = 10,
    genre_filter: str | None = None,
) -> list[dict]:
    """
    Global top-rated movies by Bayesian average (balances avg rating vs popularity).
    Used as cold-start fallback for new users with no rating history.

    Args:
        min_ratings: Minimum number of ratings required (filters out obscure movies)
        top_k: How many movies to return
        genre_filter: Optional genre to filter (e.g. "Horror", "Comedy")

    Returns:
        List of {movieid, title, year, genres, avg_rating, n_ratings, bayesian_score}
    """
    ratings = load_ratings()
    movies = load_movies()

    # Compute per-movie stats
    stats = ratings.groupby("movieid")["rating"].agg(["mean", "count"]).reset_index()
    stats.columns = ["movieid", "avg_rating", "n_ratings"]

    # Bayesian average: score = (C * m + n * avg) / (C + n)
    # C = global mean, m = minimum vote threshold
    C = stats["avg_rating"].mean()
    m = min_ratings
    stats["bayesian_score"] = (C * m + stats["n_ratings"] * stats["avg_rating"]) / (m + stats["n_ratings"])

    # Filter by minimum ratings
    stats = stats[stats["n_ratings"] >= min_ratings]

    # Merge with movie info
    merged = stats.merge(movies[["movieid", "title", "year", "genres"]], on="movieid", how="left")

    # Optional genre filter
    if genre_filter:
        merged = merged[merged["genres"].str.contains(genre_filter, case=False, na=False)]

    merged = merged.sort_values("bayesian_score", ascending=False).head(top_k)

    return merged[["movieid", "title", "year", "genres", "avg_rating", "n_ratings", "bayesian_score"]].round(2).to_dict("records")


def get_cf_recommendations(
    user_id: int,
    top_k_users: int = 20,
    top_k_movies: int = 10,
    min_rating: float = 3.5,
) -> list[dict]:
    """
    CF recommendations. Falls back to top-rated movies for new/cold-start users.
    """
    matrix = get_rating_matrix()
    id_to_title = get_movie_id_to_title()

    # Cold start: user not in dataset
    if user_id not in matrix.index:
        top = get_top_rated_movies(min_ratings=20, top_k=top_k_movies)
        return [{**m, "source": "top_rated_fallback", "note": "New user — showing globally top-rated movies"} for m in top]

    similar = find_similar_users(user_id, top_k=top_k_users)
    if not similar:
        top = get_top_rated_movies(min_ratings=20, top_k=top_k_movies)
        return [{**m, "source": "top_rated_fallback"} for m in top]

    target_rated = set(matrix.loc[user_id].dropna().index)
    scores: dict[int, list] = {}
    for sim_user in similar:
        uid = sim_user["user_id"]
        weight = sim_user["similarity"]
        for movie_id, rating in matrix.loc[uid].dropna().items():
            if movie_id in target_rated or rating < min_rating:
                continue
            if movie_id not in scores:
                scores[movie_id] = []
            scores[movie_id].append(rating * weight)

    results = []
    for movie_id, weighted_ratings in scores.items():
        if len(weighted_ratings) < 2:
            continue
        predicted = sum(weighted_ratings) / len(weighted_ratings)
        results.append({
            "movieid": int(movie_id),
            "title": id_to_title.get(movie_id, f"Movie {movie_id}"),
            "predicted_score": round(predicted, 2),
            "n_similar_users_rated": len(weighted_ratings),
            "source": "collaborative_filtering",
        })

    results.sort(key=lambda x: (x["n_similar_users_rated"], x["predicted_score"]), reverse=True)
    return results[:top_k_movies]


def get_similar_users_opinion(user_id: int, movie_id: int, top_k_users: int = 20) -> dict:
    matrix = get_rating_matrix()
    id_to_title = get_movie_id_to_title()
    ratings = load_ratings()

    similar = find_similar_users(user_id, top_k=top_k_users) if user_id in matrix.index else []
    similar_ids = {s["user_id"]: s["similarity"] for s in similar}

    movie_ratings = ratings[ratings["movieid"] == movie_id]

    seen_by_target = (
        user_id in matrix.index
        and movie_id in matrix.columns
        and not pd.isna(matrix.loc[user_id, movie_id])
    ) if user_id in matrix.index else False

    target_rating = float(matrix.loc[user_id, movie_id]) if seen_by_target else None

    # All users who rated this movie (not just similar users)
    all_raters = movie_ratings.copy()
    n_total = len(all_raters)
    global_avg = round(all_raters["rating"].mean(), 2) if n_total > 0 else None

    # Similar users who rated it (only relevant for existing users)
    if similar_ids:
        similar_ratings = movie_ratings[movie_ratings["userid"].isin(similar_ids)]
        detail = [
            {
                "user_id": int(row["userid"]),
                "rating": float(row["rating"]),
                "similarity_to_you": round(similar_ids.get(int(row["userid"]), 0), 3),
            }
            for _, row in similar_ratings.iterrows()
        ]
        detail.sort(key=lambda x: x["similarity_to_you"], reverse=True)
        similar_avg = round(sum(d["rating"] for d in detail) / len(detail), 2) if detail else None
    else:
        detail = []
        similar_avg = None

    # Rating distribution
    if n_total > 0:
        dist = all_raters["rating"].value_counts().sort_index()
        rating_distribution = {str(k): int(v) for k, v in dist.items()}
    else:
        rating_distribution = {}

    return {
        "movie_title": id_to_title.get(movie_id, f"Movie {movie_id}"),
        "movieid": movie_id,
        # Global opinion (always available)
        "global_avg_rating": global_avg,
        "total_ratings_in_dataset": n_total,
        "rating_distribution": rating_distribution,
        # Similar-user opinion (only if user has history)
        "avg_rating_similar_users": similar_avg,
        "n_similar_users_rated": len(detail),
        "top_similar_user_ratings": detail[:5],
        # Target user
        "seen_by_target_user": seen_by_target,
        "target_user_rating": target_rating,
    }


def get_general_movie_opinion(movie_id: int) -> dict:
    """
    Get community opinion on a movie without needing a specific user context.
    Returns global rating stats + rating distribution.
    Used when user asks 'what do people think about X?' without CF context.
    """
    id_to_title = get_movie_id_to_title()
    ratings = load_ratings()
    tags_df = None
    try:
        from data.loader import load_tags
        tags_df = load_tags()
    except Exception:
        pass

    movie_ratings = ratings[ratings["movieid"] == movie_id]

    if movie_ratings.empty:
        return {
            "movie_title": id_to_title.get(movie_id, f"Movie {movie_id}"),
            "movieid": movie_id,
            "message": "This movie has no ratings in the dataset.",
        }

    avg = round(movie_ratings["rating"].mean(), 2)
    n = len(movie_ratings)
    dist = movie_ratings["rating"].value_counts().sort_index()
    pct_positive = round(len(movie_ratings[movie_ratings["rating"] >= 3.5]) / n * 100, 1)

    result = {
        "movie_title": id_to_title.get(movie_id, f"Movie {movie_id}"),
        "movieid": movie_id,
        "avg_rating": avg,
        "n_ratings": n,
        "pct_positive": pct_positive,
        "rating_distribution": {str(k): int(v) for k, v in dist.items()},
    }

    if tags_df is not None:
        movie_tags = tags_df[tags_df["movieid"] == movie_id]["tag"].value_counts().head(8)
        result["community_tags"] = movie_tags.index.tolist()

    return result