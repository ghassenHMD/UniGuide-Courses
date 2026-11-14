"""Common interface for all recommenders (see docs/contracts.md, section 6)."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Base class shared by popularity, content-based, collaborative and hybrid models."""

    name: str = "base"

    def __init__(self) -> None:
        self.catalog: list[str] = []
        self._seen: dict[int, set[str]] = {}

    def _register_data(self, train: pd.DataFrame, courses: pd.DataFrame) -> None:
        """Store the catalog and each user's training history (used by exclude_seen)."""
        self.catalog = courses["course_id"].tolist()
        self._seen = train.groupby("user_id")["course_id"].apply(set).to_dict()

    def seen_items(self, user_id: int) -> set[str]:
        """Courses the user interacted with in the training data."""
        return self._seen.get(user_id, set())

    @abstractmethod
    def fit(self, train: pd.DataFrame, courses: pd.DataFrame) -> "BaseRecommender":
        """Learn from training data only. Must return ``self``."""

    @abstractmethod
    def score(self, user_id: int, item_ids: list[str]) -> np.ndarray:
        """Return one raw score per item (same order); ``np.nan`` if not scorable."""

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        """Return the top-k courses as a DataFrame ['course_id', 'score'], best first."""
        items = self.catalog
        if exclude_seen:
            seen = self.seen_items(user_id)
            items = [i for i in items if i not in seen]
        scores = np.asarray(self.score(user_id, items), dtype=float)
        df = pd.DataFrame({"course_id": items, "score": scores}).dropna()
        return df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
