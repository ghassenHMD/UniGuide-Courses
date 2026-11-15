"""Explanations built from real model data (Step 7).

Each recommendation is explained with:
  1. the contribution of each component to the hybrid score (content, collaborative, popularity);
  2. "Because you liked...": the user's well-rated courses closest to the recommended course;
  3. the highest-weighted TF-IDF keywords of the recommended course.

The explainer never chooses a course and never changes a score: it only describes the ranking
produced by the hybrid model (an optional LLM could rephrase ``Explanation.full_text``, nothing more).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from uniguide.config import load_config
from uniguide.models.hybrid import HybridRecommender


@dataclass
class Explanation:
    """Everything the interface needs to display one recommendation."""

    rank: int
    course_id: str
    name: str
    institution: str
    url: str
    score: float
    mode: str                   # hybrid | content_only | collaborative_only | popularity
    alpha: float | None         # CB weight actually used (None for popularity)
    cb_score: float | None      # normalised CB score in [0, 1]
    cf_score: float | None      # normalised CF score in [0, 1]
    cb_contribution: float      # w_cb * cb_score (part of the final score)
    cf_contribution: float      # w_cf * cf_score
    pop_contribution: float     # whole score when the user has no history
    because: list[dict] = field(default_factory=list)   # liked courses close to this one
    keywords: list[str] = field(default_factory=list)
    text: str = ""              # headline: how the score was obtained

    @property
    def full_text(self) -> str:
        """Headline + reasons + keywords as one paragraph (for logs, or to be rephrased by an LLM)."""
        parts = [self.text]
        if self.because:
            liked = ", ".join(f"{b['name']} (you gave {b['rating']:.0f}/5)" for b in self.because)
            parts.append(f"Because you liked: {liked}.")
        if self.keywords:
            parts.append("Key topics: " + ", ".join(self.keywords) + ".")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_tfidf_model(model):
    """Return the first TF-IDF content model inside ``model`` (fusion models are searched)."""
    if getattr(model, "vectorizer", None) is not None and getattr(model, "item_vectors", None) is not None:
        return model
    for sub in getattr(model, "models", []):
        found = _find_tfidf_model(sub)
        if found is not None:
            return found
    return None


class Explainer:
    """Produce explained Top-K lists from a fitted :class:`HybridRecommender`.

    Args:
        hybrid: fitted hybrid model (its candidate set and scores are reused as they are).
        courses: DataFrame with course_id, name, institution, url.
        history: training interactions (user_id, course_id, rating) used to fit ``hybrid``.
        tfidf_model: fitted ``ContentBasedRecommender`` (TF-IDF) used for keywords and for the
            "because you liked" similarities. Looked up inside the hybrid when omitted.
        n_because / n_keywords: number of liked courses and keywords shown per course.
    """

    def __init__(
        self,
        hybrid: HybridRecommender,
        courses: pd.DataFrame,
        history: pd.DataFrame,
        tfidf_model=None,
        n_because: int = 3,
        n_keywords: int = 8,
        liked_threshold: float | None = None,
    ) -> None:
        self.hybrid = hybrid
        self.courses = courses.drop_duplicates("course_id").set_index("course_id")
        self.tfidf = tfidf_model or _find_tfidf_model(hybrid.content_model)
        if self.tfidf is None:
            raise ValueError("A fitted TF-IDF ContentBasedRecommender is required for keywords")
        self.n_because = n_because
        self.n_keywords = n_keywords
        self.liked_threshold = (
            load_config()["data"]["relevance_threshold"] if liked_threshold is None else liked_threshold
        )
        self._terms = np.asarray(self.tfidf.vectorizer.get_feature_names_out())
        self._by_user = history.groupby("user_id")
        self._known_users = set(history["user_id"].unique())
        self._catalog = set(hybrid.catalog)

    # ---------------------------------------------------------------- inputs
    def history_of(self, user_id) -> dict[str, float]:
        """Training ratings {course_id: rating} of a known user ({} if unknown)."""
        if user_id not in self._known_users:
            return {}
        group = self._by_user.get_group(user_id)
        return dict(zip(group["course_id"], group["rating"]))

    def _candidates(self, seen: set[str]) -> list[str]:
        return [c for c in self.hybrid.catalog if c not in seen]

    # ----------------------------------------------------------- public API
    def explain_user(self, user_id, k: int = 10) -> list[Explanation]:
        """Explained Top-K for a user of the training data (same ranking as ``hybrid.recommend``)."""
        ratings = self.history_of(user_id)
        candidates = self._candidates(self.hybrid.seen_items(user_id))
        n_history = len(self.hybrid.seen_items(user_id))
        frame = self.hybrid.components(user_id, candidates)
        return self._explain(frame, ratings, n_history, k)

    def explain_ratings(self, ratings: dict[str, float], k: int = 10) -> list[Explanation]:
        """Explained Top-K for an ad-hoc history {course_id: rating} (new user in the demo)."""
        ratings = {c: r for c, r in ratings.items() if c in self._catalog}
        candidates = self._candidates(set(ratings))
        frame = self.hybrid.components_from_ratings(ratings, candidates)
        return self._explain(frame, ratings, len(ratings), k)

    # ------------------------------------------------------------- internals
    def _explain(self, frame: pd.DataFrame, ratings: dict, n_history: int, k: int) -> list[Explanation]:
        top = (
            frame.dropna(subset=["score"])
            .sort_values("score", ascending=False, kind="mergesort")
            .head(k)
        )
        return [
            self._explain_row(rank, row, ratings, n_history)
            for rank, row in enumerate(top.itertuples(index=False), start=1)
        ]

    def _explain_row(self, rank: int, row, ratings: dict, n_history: int) -> Explanation:
        info = self.courses.loc[row.course_id] if row.course_id in self.courses.index else None
        cb_part = 0.0 if np.isnan(row.cb) else row.w_cb * row.cb
        cf_part = 0.0 if np.isnan(row.cf) else row.w_cf * row.cf
        popularity_part = float(row.score) if row.mode == "popularity" else 0.0

        because = []
        if row.mode != "popularity":
            because = self._because(row.course_id, ratings, prefer_cf=cf_part > cb_part)

        explanation = Explanation(
            rank=rank,
            course_id=row.course_id,
            name="" if info is None else str(info.get("name", "")),
            institution="" if info is None else str(info.get("institution", "")),
            url="" if info is None else str(info.get("url", "")),
            score=float(row.score),
            mode=row.mode,
            alpha=None if row.mode == "popularity" else float(self.hybrid.alpha_for(n_history)),
            cb_score=None if np.isnan(row.cb) else float(row.cb),
            cf_score=None if np.isnan(row.cf) else float(row.cf),
            cb_contribution=float(cb_part),
            cf_contribution=float(cf_part),
            pop_contribution=popularity_part,
            because=because,
            keywords=self._keywords(row.course_id),
        )
        explanation.text = self._compose_text(explanation)
        return explanation

    def _keywords(self, course_id: str) -> list[str]:
        """Highest TF-IDF weighted terms of the course text."""
        idx = self.tfidf.item_index.get(course_id)
        if idx is None:
            return []
        row = self.tfidf.item_vectors[idx]
        if issparse(row):
            weights, cols = row.data, row.indices
        else:
            cols = np.flatnonzero(row)
            weights = np.asarray(row)[cols]
        order = np.lexsort((cols, -weights))          # weight desc, then term index (deterministic)
        keywords, covered = [], set()
        for i in order:
            term = str(self._terms[cols[i]])
            words = set(term.split())
            if words <= covered:                       # skip a term whose words are already shown
                continue
            keywords.append(term)
            covered |= words
            if len(keywords) == self.n_keywords:
                break
        return keywords

    def _because(self, course_id: str, ratings: dict, prefer_cf: bool) -> list[dict]:
        """Courses the user liked that are closest to ``course_id``.

        Closeness is the co-rating similarity of the item-item CF model when CF drives the
        score (and the model is kNN), otherwise the TF-IDF cosine similarity.
        """
        liked = [c for c, r in ratings.items() if r >= self.liked_threshold and c != course_id]
        if not liked:
            return []

        similarities, source = None, "content"
        if prefer_cf:
            similarities = self._cf_similarities(course_id, liked)
            source = "collaborative"
        if similarities is None or not np.any(similarities > 0):
            similarities, source = self._content_similarities(course_id, liked), "content"
        if similarities is None:
            return []

        order = sorted(
            (i for i in range(len(liked)) if similarities[i] > 0),
            key=lambda i: (-similarities[i], -ratings[liked[i]], liked[i]),
        )[: self.n_because]
        return [
            {
                "course_id": liked[i],
                "name": str(self.courses.loc[liked[i], "name"]) if liked[i] in self.courses.index else liked[i],
                "similarity": float(similarities[i]),
                "rating": float(ratings[liked[i]]),
                "source": source,
            }
            for i in order
        ]

    def _content_similarities(self, course_id: str, liked: list[str]):
        index = self.tfidf.item_index
        if course_id not in index:
            return None
        sims = np.zeros(len(liked))
        known = [(pos, index[c]) for pos, c in enumerate(liked) if c in index]
        if not known:
            return None
        vectors = self.tfidf.item_vectors
        target = vectors[index[course_id]]
        values = vectors[[j for _, j in known]] @ target.T
        values = values.toarray().ravel() if issparse(values) else np.ravel(values)
        for (pos, _), value in zip(known, values):
            sims[pos] = value
        return sims

    def _cf_similarities(self, course_id: str, liked: list[str]):
        cf = self.hybrid.collaborative_model
        matrix = getattr(cf, "item_similarity", None)
        if matrix is None or course_id not in cf.item_to_idx:
            return None
        row = matrix.getrow(cf.item_to_idx[course_id]).toarray().ravel()
        return np.array([row[cf.item_to_idx[c]] if c in cf.item_to_idx else 0.0 for c in liked])

    @staticmethod
    def _compose_text(e: Explanation) -> str:
        """Headline sentence built only from the numbers above (how the score was obtained)."""
        if e.mode == "popularity":
            return (
                "We do not know your tastes yet, so this course is suggested because it is "
                "among the most popular on the platform."
            )
        if e.mode == "content_only":
            return (
                "This course has no ratings from other learners yet, so it is ranked only on how "
                f"close its content is to what you liked (content score {e.cb_score:.2f})."
            )
        if e.mode == "collaborative_only":
            return (
                "It is ranked only on the ratings of learners similar to you "
                f"(collaborative score {e.cf_score:.2f})."
            )
        total = e.cb_contribution + e.cf_contribution
        cb_share = e.cb_contribution / total if total > 0 else 0.0
        main = "content similarity" if cb_share >= 0.5 else "the ratings of similar learners"
        return (
            f"Score {e.score:.2f} = {e.alpha:.2f} x content ({e.cb_score:.2f}) + "
            f"{1 - e.alpha:.2f} x collaborative ({e.cf_score:.2f}); mostly driven by {main}."
        )