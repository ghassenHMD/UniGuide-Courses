
"""
Hybrid recommendation model.

Combines Content-Based and Collaborative Filtering scores.
"""

import numpy as np
import pandas as pd

from uniguide.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining CB and CF scores."""

    name = "hybrid"

    def __init__(
        self,
        content_model,
        collaborative_model,
        alpha=0.5,
    ):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

        self.popularity_scores = {}

    def fit(self, train, courses):
        """Fit both component models using training data only."""

        if not {"user_id", "course_id"}.issubset(train.columns):
            raise ValueError(
                "train must contain user_id and course_id"
            )

        if "course_id" not in courses.columns:
            raise ValueError("courses must contain course_id")

        # Register the shared catalog and user histories.
        self._register_data(train, courses)

        # Popularity fallback for users without history.
        counts = train["course_id"].value_counts()
        max_count = counts.max() if not counts.empty else 0

        if max_count > 0:
            self.popularity_scores = {
                item: float(counts.get(item, 0) / max_count)
                for item in self.catalog
            }
        else:
            self.popularity_scores = {}

        # Fit both component recommenders.
        self.content_model.fit(train, courses)
        self.collaborative_model.fit(train, courses)

        return self

    @staticmethod
    def _normalize(scores):
        """
        Apply min-max normalization to valid scores.

        If all valid scores are equal, return 0.0.
        NaN values remain NaN.
        """

        scores = np.asarray(scores, dtype=float)

        normalized = np.full(scores.shape, np.nan)

        valid = np.isfinite(scores)

        if not valid.any():
            return normalized

        minimum = scores[valid].min()
        maximum = scores[valid].max()

        if maximum == minimum:
            normalized[valid] = 0.0
        else:
            normalized[valid] = (
                (scores[valid] - minimum)
                / (maximum - minimum)
            )

        return normalized

    def score(self, user_id, item_ids):
        """
        Combine normalized CB and CF scores.

        alpha is the Content-Based weight.
        (1 - alpha) is the Collaborative weight.

        When CF cannot score a new course, CB is used alone.
        """

        item_ids = list(item_ids)

        # Unknown courses must return NaN.
        known_items = [
            item in self.catalog
            for item in item_ids
        ]

        if not item_ids:
            return np.array([], dtype=float)

        # Unknown users or users without history use popularity.
        if not self.seen_items(user_id):
            return np.asarray([
                self.popularity_scores.get(item, np.nan)
                if known else np.nan
                for item, known in zip(item_ids, known_items)
            ], dtype=float)

        cb_scores = np.asarray(
            self.content_model.score(user_id, item_ids),
            dtype=float,
        )

        cf_scores = np.asarray(
            self.collaborative_model.score(user_id, item_ids),
            dtype=float,
        )

        if len(cb_scores) != len(item_ids):
            raise ValueError(
                "Content model returned an invalid number of scores"
            )

        if len(cf_scores) != len(item_ids):
            raise ValueError(
                "Collaborative model returned an invalid number of scores"
            )

        cb_scores = self._normalize(cb_scores)
        cf_scores = self._normalize(cf_scores)

        final_scores = []

        for item, known, cb, cf in zip(
            item_ids,
            known_items,
            cb_scores,
            cf_scores,
        ):

            if not known:
                final_scores.append(np.nan)
                continue

            cb_valid = np.isfinite(cb)
            cf_valid = np.isfinite(cf)

            if cb_valid and cf_valid:
                value = (
                    self.alpha * cb
                    + (1 - self.alpha) * cf
                )

            elif cb_valid:
                # New course: use Content-Based only.
                value = cb

            elif cf_valid:
                # Content score unavailable: use CF only.
                value = cf

            else:
                value = np.nan

            final_scores.append(value)

        return np.asarray(final_scores, dtype=float)

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Generate Top-K hybrid recommendations."""

        if k <= 0:
            raise ValueError("k must be greater than 0")

        candidates = list(self.catalog)

        if exclude_seen:
            seen = self.seen_items(user_id)
            candidates = [
                item for item in candidates
                if item not in seen
            ]

        if not candidates:
            return pd.DataFrame(columns=["course_id", "score"])

        scores = self.score(user_id, candidates)

        results = pd.DataFrame({
            "course_id": candidates,
            "score": scores,
        })

        results = results.dropna(subset=["score"])

        results = results.sort_values(
            "score",
            ascending=False,
            kind="mergesort",
        )

        return results.head(k).reset_index(drop=True)
