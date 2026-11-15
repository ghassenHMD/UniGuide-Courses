"""Hybrid recommendation model: weighted CB + CF combination with history-based switching."""
import numpy as np
import pandas as pd

from uniguide.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining content-based (CB) and collaborative (CF) scores.

    ``Score(u, i) = alpha * CB(u, i) + (1 - alpha) * CF(u, i)`` where both scores are min-max
    normalised per user over the candidate courses.

    Switching (by history size):
      * 0 training interactions  -> popularity only;
      * ``alpha_schedule``       -> alpha depends on the number of training interactions of the
        user, e.g. ``[(2, 0.9), (5, 0.7)]`` means alpha=0.9 up to 2 interactions, 0.7 up to 5,
        and the default ``alpha`` beyond;
      * new course without CF score -> CB only (weights renormalised).
    """

    name = "hybrid"
    COLUMNS = ["course_id", "cb", "cf", "w_cb", "w_cf", "score", "mode"]

    def __init__(self, content_model, collaborative_model, alpha=0.5, alpha_schedule=None):
        super().__init__()

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        schedule = sorted((int(n), float(a)) for n, a in (alpha_schedule or []))
        if any(not 0 <= a <= 1 for _, a in schedule):
            raise ValueError("every alpha in alpha_schedule must be between 0 and 1")

        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha
        self.alpha_schedule = schedule
        self.popularity_scores = {}
        self._catalog_set = set()

    # ------------------------------------------------------------------ fit
    def fit(self, train, courses):
        """Fit both component models using training data only."""
        if not {"user_id", "course_id"}.issubset(train.columns):
            raise ValueError("train must contain user_id and course_id")
        if "course_id" not in courses.columns:
            raise ValueError("courses must contain course_id")

        self._register_data(train, courses)
        self._catalog_set = set(self.catalog)

        # Popularity fallback for users without history.
        counts = train["course_id"].value_counts()
        max_count = counts.max() if not counts.empty else 0
        self.popularity_scores = (
            {item: float(counts.get(item, 0) / max_count) for item in self.catalog}
            if max_count > 0
            else {}
        )

        self.content_model.fit(train, courses)
        self.collaborative_model.fit(train, courses)
        return self

    # ------------------------------------------------------------- switching
    def alpha_for(self, n_history: int) -> float:
        """CB weight for a user with ``n_history`` training interactions."""
        for max_history, alpha in self.alpha_schedule:
            if n_history <= max_history:
                return alpha
        return self.alpha

    @staticmethod
    def _normalize(scores):
        """Min-max normalisation of the finite scores (all equal -> 0.0, NaN stays NaN)."""
        scores = np.asarray(scores, dtype=float)
        normalized = np.full(scores.shape, np.nan)
        valid = np.isfinite(scores)
        if not valid.any():
            return normalized
        minimum, maximum = scores[valid].min(), scores[valid].max()
        if maximum == minimum:
            normalized[valid] = 0.0
        else:
            normalized[valid] = (scores[valid] - minimum) / (maximum - minimum)
        return normalized

    # ------------------------------------------------------------ components
    def _popularity_frame(self, item_ids):
        pop = np.array(
            [self.popularity_scores.get(i, np.nan) if i in self._catalog_set else np.nan
             for i in item_ids],
            dtype=float,
        )
        return pd.DataFrame({
            "course_id": item_ids, "cb": np.nan, "cf": np.nan, "w_cb": 0.0, "w_cf": 0.0,
            "score": pop, "mode": "popularity",
        })[self.COLUMNS]

    def _combine(self, item_ids, cb_raw, cf_raw, n_history):
        """Normalise the raw CB / CF scores and mix them (see class docstring)."""
        cb_raw = np.asarray(cb_raw, dtype=float)
        cf_raw = np.asarray(cf_raw, dtype=float)
        if len(cb_raw) != len(item_ids):
            raise ValueError("Content model returned an invalid number of scores")
        if len(cf_raw) != len(item_ids):
            raise ValueError("Collaborative model returned an invalid number of scores")

        known = np.array([i in self._catalog_set for i in item_ids], dtype=bool)
        cb, cf = self._normalize(cb_raw), self._normalize(cf_raw)
        cb_ok = np.isfinite(cb) & known
        cf_ok = np.isfinite(cf) & known
        alpha = self.alpha_for(n_history)

        w_cb = np.where(cb_ok & cf_ok, alpha, np.where(cb_ok, 1.0, 0.0))
        w_cf = np.where(cb_ok & cf_ok, 1.0 - alpha, np.where(cf_ok, 1.0, 0.0))
        score = np.where(cb_ok, w_cb * np.nan_to_num(cb), 0.0) + np.where(
            cf_ok, w_cf * np.nan_to_num(cf), 0.0
        )
        score = np.where(cb_ok | cf_ok, score, np.nan)
        mode = np.select(
            [cb_ok & cf_ok, cb_ok, cf_ok],
            ["hybrid", "content_only", "collaborative_only"],
            default="unscorable",
        )
        return pd.DataFrame({
            "course_id": item_ids, "cb": cb, "cf": cf, "w_cb": w_cb, "w_cf": w_cf,
            "score": score, "mode": mode,
        })[self.COLUMNS]

    def components(self, user_id, item_ids):
        """Per-course breakdown of the hybrid score (used by the explanations).

        Columns: course_id, cb, cf (normalised in [0, 1], NaN if unavailable), w_cb, w_cf
        (effective weights after renormalisation), score, mode. ``score`` is exactly what
        ``score`` returns. Normalisation is done over ``item_ids``: pass the same candidate
        list as ``recommend`` (all unseen courses) to reproduce its scores.
        """
        item_ids = list(item_ids)
        n_history = len(self.seen_items(user_id))
        if n_history == 0:
            return self._popularity_frame(item_ids)
        return self._combine(
            item_ids,
            self.content_model.score(user_id, item_ids),
            self.collaborative_model.score(user_id, item_ids),
            n_history,
        )

    def components_from_ratings(self, ratings, item_ids):
        """Same as ``components`` for an ad-hoc history {course_id: rating} (new user in the demo)."""
        item_ids = list(item_ids)
        ratings = {c: r for c, r in ratings.items() if c in self._catalog_set}
        if not ratings:
            return self._popularity_frame(item_ids)
        return self._combine(
            item_ids,
            self.content_model.score_from_ratings(ratings, item_ids),
            self.collaborative_model.score_from_ratings(ratings, item_ids),
            len(ratings),
        )

    # --------------------------------------------------------------- scoring
    def score(self, user_id, item_ids):
        """Hybrid score of each course (NaN for unknown courses)."""
        item_ids = list(item_ids)
        if not item_ids:
            return np.array([], dtype=float)
        return self.components(user_id, item_ids)["score"].to_numpy(dtype=float)

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Generate Top-K hybrid recommendations."""
        if k <= 0:
            raise ValueError("k must be greater than 0")

        candidates = list(self.catalog)
        if exclude_seen:
            seen = self.seen_items(user_id)
            candidates = [item for item in candidates if item not in seen]
        if not candidates:
            return pd.DataFrame(columns=["course_id", "score"])

        results = pd.DataFrame({"course_id": candidates, "score": self.score(user_id, candidates)})
        results = results.dropna(subset=["score"]).sort_values(
            "score", ascending=False, kind="mergesort"
        )
        return results.head(k).reset_index(drop=True)