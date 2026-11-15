"""Content-based recommender: TF-IDF course vectors + rating-weighted user profiles (Step 4)."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from uniguide.config import load_config
from uniguide.models.base import BaseRecommender
from uniguide.models.popularity import PopularityRecommender


class ContentBasedRecommender(BaseRecommender):
    """Recommend the courses whose text is closest to what the user liked.

    1. Each course text is turned into a TF-IDF vector (L2-normalised).
    2. The profile of a user is the rating-weighted mean of the vectors of the
       courses they liked (rating >= ``liked_threshold``) in the training data.
    3. The score of a course is the cosine similarity between the profile and the course.

    Users without a usable profile (unknown user, or no liked course) fall back to the
    popularity baseline, as required by the interface contract.
    """

    name = "content_based"

    def __init__(
        self,
        liked_threshold: float | None = None,
        rating_offset: float = 0.0,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 2,
        max_df: int | float = 0.5,
        max_features: int | None = 20000,
        sublinear_tf: bool = True,
    ) -> None:
        super().__init__()
        if liked_threshold is None:
            liked_threshold = load_config()["data"]["relevance_threshold"]
        if rating_offset >= liked_threshold:
            raise ValueError("rating_offset must be lower than liked_threshold")
        self.liked_threshold = liked_threshold
        self.rating_offset = rating_offset
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.item_index: dict[str, int] = {}
        self.item_vectors = None          # sparse (n_items, n_terms), rows L2-normalised
        self._liked: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._popularity_vector = np.array([])
        self._cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, courses: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit TF-IDF on the course texts and store each user's liked courses."""
        if not {"user_id", "course_id", "rating"}.issubset(train.columns):
            raise ValueError("train must contain user_id, course_id and rating")
        if not {"course_id", "text"}.issubset(courses.columns):
            raise ValueError("courses must contain course_id and text")
        if courses["course_id"].duplicated().any():
            raise ValueError("courses must not contain duplicated course_id")

        self._register_data(train, courses)
        self.item_index = {c: i for i, c in enumerate(self.catalog)}
        self.item_vectors = self._build_item_vectors(courses)
        self._liked = {}
        for user_id, group in train.groupby("user_id"):
            liked = self._liked_arrays(dict(zip(group["course_id"], group["rating"])))
            if liked is not None:
                self._liked[user_id] = liked

        # Fallback for users without a profile (also gives NaN to unrated courses).
        self._popularity_vector = PopularityRecommender().fit(train, courses).score(-1, self.catalog)
        self._cache = {}
        return self

    def _liked_arrays(self, ratings: dict[str, float]):
        """(item indices, weights) of the liked courses of a user, or None if there are none."""
        idx, weights = [], []
        for course_id, rating in ratings.items():
            if rating >= self.liked_threshold and course_id in self.item_index:
                idx.append(self.item_index[course_id])
                weights.append(rating - self.rating_offset)
        if not idx:
            return None
        return np.asarray(idx), np.asarray(weights, dtype=float)

    def _build_item_vectors(self, courses: pd.DataFrame):
        """Course vectors, rows L2-normalised. TF-IDF here; overridden by the embedding model."""
        return self.vectorizer.fit_transform(courses["text"].fillna("").tolist())

    # -------------------------------------------------------------- scoring
    def _cosine_scores(self, idx: np.ndarray, weights: np.ndarray):
        """Cosine between the profile and every course; None if the profile is empty."""
        weights = weights / weights.sum()                              # weighted *mean*
        profile = csr_matrix(weights) @ self.item_vectors[idx]         # (1, n_terms)
        norm = np.sqrt(profile.multiply(profile).sum())
        if norm == 0:
            return None
        # Course vectors have unit norm, so dot product / ||profile|| is the cosine.
        return (self.item_vectors @ profile.T).toarray().ravel() / norm

    def _user_scores(self, user_id: int) -> np.ndarray:
        """Scores over the whole catalog for one user (cached)."""
        if user_id not in self._cache:
            scores = None
            if user_id in self._liked:
                scores = self._cosine_scores(*self._liked[user_id])
            self._cache[user_id] = self._popularity_vector if scores is None else scores
        return self._cache[user_id]

    def _pick(self, scores: np.ndarray, item_ids: list[str]) -> np.ndarray:
        idx = np.array([self.item_index.get(i, -1) for i in item_ids], dtype=int)
        out = np.full(len(idx), np.nan)
        known = idx >= 0
        out[known] = scores[idx[known]]
        return out

    def score(self, user_id: int, item_ids: list[str]) -> np.ndarray:
        """Cosine similarity between the user profile and each course (NaN if unknown course)."""
        return self._pick(self._user_scores(user_id), item_ids)

    def score_from_ratings(self, ratings: dict[str, float], item_ids: list[str]) -> np.ndarray:
        """Same as ``score`` for an ad-hoc history {course_id: rating} (e.g. a new user in the demo)."""
        liked = self._liked_arrays(ratings)
        scores = None if liked is None else self._cosine_scores(*liked)
        return self._pick(self._popularity_vector if scores is None else scores, item_ids)