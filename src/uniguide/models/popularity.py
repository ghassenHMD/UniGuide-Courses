"""Popularity baseline based on the IMDb weighted rating (Step 3)."""
import numpy as np
import pandas as pd

from uniguide.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline: every user gets the same ranking (minus seen courses).

    Weighted rating (IMDb formula)::

        WR = v / (v + m) * R + m / (v + m) * C

    v = number of training ratings of the course, R = its mean rating,
    C = mean rating over all training interactions, m = minimum number of votes
    required to be taken "at face value".

    A course with few ratings is pulled towards C, so a 5.0 average based on 3 reviews
    cannot beat a 4.8 average based on 3000 reviews.
    """

    name = "popularity"

    def __init__(self, m_quantile: float = 0.9, min_votes: float | None = None) -> None:
        super().__init__()
        if not 0.0 <= m_quantile <= 1.0:
            raise ValueError("m_quantile must be between 0 and 1")
        if min_votes is not None and min_votes < 0:
            raise ValueError("min_votes must be >= 0")
        self.m_quantile = m_quantile
        self.min_votes = min_votes
        self.global_mean = np.nan  # C
        self.m = np.nan            # minimum votes
        self.stats = pd.DataFrame(columns=["v", "R", "wr"])  # one row per rated course
        self._wr: dict[str, float] = {}

    def fit(self, train: pd.DataFrame, courses: pd.DataFrame) -> "PopularityRecommender":
        """Compute v, R and WR for every course using training data only."""
        if not {"user_id", "course_id", "rating"}.issubset(train.columns):
            raise ValueError("train must contain user_id, course_id and rating")
        if "course_id" not in courses.columns:
            raise ValueError("courses must contain course_id")

        self._register_data(train, courses)
        data = train[train["course_id"].isin(set(self.catalog))]
        if data.empty:
            self.stats = pd.DataFrame(columns=["v", "R", "wr"])
            self._wr = {}
            return self

        stats = data.groupby("course_id")["rating"].agg(v="size", R="mean")
        self.global_mean = float(data["rating"].mean())
        if self.min_votes is not None:
            self.m = float(self.min_votes)
        else:
            self.m = float(stats["v"].quantile(self.m_quantile))

        v, r, c, m = stats["v"], stats["R"], self.global_mean, self.m
        stats["wr"] = v / (v + m) * r + m / (v + m) * c
        self.stats = stats
        self._wr = stats["wr"].to_dict()
        return self

    def score(self, user_id: int, item_ids: list[str]) -> np.ndarray:
        """Weighted rating of each course (the user is ignored).

        Courses without any training rating (new courses) cannot be scored: NaN.
        """
        return np.array([self._wr.get(i, np.nan) for i in item_ids], dtype=float)