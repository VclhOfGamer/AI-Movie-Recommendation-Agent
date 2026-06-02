"""
Data loader: reads CSVs once at startup, exposes shared dataframes and the
pre-computed user-item rating matrix used by collaborative filtering.
"""
import os
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path

from sympy import re

DATA_DIR = Path(os.getenv("DATA_DIR", "../../data/ml-latest-small-filtered"))


@lru_cache(maxsize=1)
def load_movies() -> pd.DataFrame:
    """5135 movies with title, year, genres, plot."""
    df = pd.read_csv(DATA_DIR / "movies_with_plots.csv")
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@lru_cache(maxsize=1)
def load_ratings() -> pd.DataFrame:
    """74k ratings: userId, movieId, rating, timestamp."""
    df = pd.read_csv(DATA_DIR / "ratings.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@lru_cache(maxsize=1)
def load_tags() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "tags.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@lru_cache(maxsize=1)
def get_rating_matrix() -> pd.DataFrame:
    """
    Sparse user × movie matrix (NaN for unseen).
    Rows = userId, columns = movieId.
    Used for cosine-similarity CF.
    """
    ratings = load_ratings()
    matrix = ratings.pivot_table(
        index="userid", columns="movieid", values="rating"
    )
    return matrix


@lru_cache(maxsize=1)
def get_movie_id_to_title() -> dict:
    movies = load_movies()
    return dict(zip(movies["movieid"].astype(int), movies["title"]))


@lru_cache(maxsize=1)
def get_title_to_movie_id() -> dict:
    movies = load_movies()
    return {t.lower(): mid for t, mid in zip(movies["title"], movies["movieid"].astype(int))}


def get_movie_by_id(movie_id: int) -> dict | None:
    movies = load_movies()
    row = movies[movies["movieid"] == movie_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "movieid": int(r["movieid"]),
        "title": r.get("title", ""),
        "year": r.get("year", ""),
        "genres": r.get("genres", ""),
        "plot": r.get("plot", ""),
    }


def search_movie_by_title(title_query: str) -> list[dict]:
    movies = load_movies()

    pattern = re.escape(title_query.lower())

    mask = movies["title"].str.lower().str.contains(
        pattern,
        na=False
    )

    results = movies[mask].head(5)

    return results[
        ["movieid", "title", "year", "genres"]
    ].to_dict("records")