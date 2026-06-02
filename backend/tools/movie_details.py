"""
Get full details for a movie by ID or fuzzy title search.
Returns: Movie details including plot, genres, rating stats, tags
"""
from data.loader import get_movie_by_id, load_ratings, load_tags, search_movie_by_title


def get_movie_details(movie_id: int | None = None, title_query: str | None = None) -> dict:
    if movie_id is None and title_query:
        candidates = search_movie_by_title(title_query)
        if not candidates:
            return {"error": f"No movie found matching '{title_query}'"}
        # Use the first match
        movie_id = candidates[0]["movieid"]

    if movie_id is None:
        return {"error": "Provide either movie_id or title_query"}

    movie = get_movie_by_id(int(movie_id))
    if not movie:
        return {"error": f"Movie ID {movie_id} not found"}

    # Rating stats
    ratings = load_ratings()
    movie_ratings = ratings[ratings["movieid"] == int(movie_id)]
    rating_stats = {}
    if not movie_ratings.empty:
        rating_stats = {
            "n_ratings": len(movie_ratings),
            "avg_rating": round(movie_ratings["rating"].mean(), 2),
            "rating_distribution": movie_ratings["rating"].value_counts().sort_index().to_dict(),
        }

    # Tags for this movie
    tags_df = load_tags()
    movie_tags = tags_df[tags_df["movieid"] == int(movie_id)]["tag"].value_counts().head(10)
    tag_list = movie_tags.index.tolist()

    return {
        **movie,
        "rating_stats": rating_stats,
        "tags": tag_list,
    }