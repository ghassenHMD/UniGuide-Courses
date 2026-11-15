
"""
Collaborative Filtering module for UniGuide-Courses.

Person B:
- Item-Item KNN
- Truncated SVD

Interface:
fit(train, courses)
score(user_id, item_ids)
recommend(user_id, k=10, exclude_seen=True)
"""

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from uniguide.models.base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """Collaborative Filtering using Item-Item KNN or Truncated SVD."""

    name = "collaborative"

    def __init__(
        self,
        method="knn",
        n_neighbors=20,
        n_components=20,
        random_state=42,
    ):
        super().__init__()

        if method not in ("knn", "svd"):
            raise ValueError("method must be 'knn' or 'svd'")

        if n_neighbors <= 0 or n_components <= 0:
            raise ValueError("Model parameters must be positive")

        self.method = method
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.random_state = random_state

        self.users = []
        self.items = []

        self.user_to_idx = {}
        self.item_to_idx = {}

        self.user_ratings = {}

        self.matrix = None
        self.item_similarity = None
        self.svd = None

        self.popularity_scores = {}

    def fit(self, train, courses):
        """Fit the collaborative model using training data only."""

        required_train = {"user_id", "course_id", "rating"}
        required_courses = {"course_id"}

        if not required_train.issubset(train.columns):
            raise ValueError(
                f"train must contain {required_train}"
            )

        if not required_courses.issubset(courses.columns):
            raise ValueError("courses must contain course_id")

        # Register the shared catalog and user histories.
        self._register_data(train, courses)

        # Keep only interactions belonging to the catalog.
        data = train[
            train["course_id"].isin(self.catalog)
        ].copy()

        # Aggregate duplicate user-course interactions.
        data = (
            data.groupby(["user_id", "course_id"], as_index=False)
            ["rating"].mean()
        )

        self.users = sorted(data["user_id"].unique().tolist())

        observed_items = set(data["course_id"])

        self.items = [
            item for item in self.catalog
            if item in observed_items
        ]

        self.user_to_idx = {
            user: idx
            for idx, user in enumerate(self.users)
        }

        self.item_to_idx = {
            item: idx
            for idx, item in enumerate(self.items)
        }

        self.user_ratings = {
            user: dict(zip(group["course_id"], group["rating"]))
            for user, group in data.groupby("user_id")
        }

        # Popularity scores for users without training history.
        counts = data.groupby("course_id").size()
        max_count = counts.max() if not counts.empty else 0

        if max_count > 0:
            self.popularity_scores = {
                item: float(counts.get(item, 0) / max_count)
                for item in self.items
            }
        else:
            self.popularity_scores = {}

        # Build the sparse user-item rating matrix.
        rows = data["user_id"].map(self.user_to_idx).to_numpy()
        cols = data["course_id"].map(self.item_to_idx).to_numpy()
        values = data["rating"].to_numpy(dtype=float)

        self.matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )

        self.item_similarity = None
        self.svd = None

        if self.matrix.nnz == 0:
            return self

        if self.method == "knn":
            self._fit_knn()
        else:
            self._fit_svd()

        return self

    def _fit_knn(self):
        """Calculate Item-Item cosine similarities."""

        item_user_matrix = self.matrix.T.tocsr()

        similarity = cosine_similarity(
            item_user_matrix,
            dense_output=False,
        ).tocsr()

        similarity.setdiag(0)
        similarity.eliminate_zeros()

        rows = []
        cols = []
        values = []

        for item_idx in range(similarity.shape[0]):

            start = similarity.indptr[item_idx]
            end = similarity.indptr[item_idx + 1]

            indices = similarity.indices[start:end]
            scores = similarity.data[start:end]

            valid = scores > 0
            indices = indices[valid]
            scores = scores[valid]

            if len(scores) > self.n_neighbors:
                # Deterministic selection of the strongest neighbors.
                order = np.lexsort((indices, -scores))
                selected = order[:self.n_neighbors]

                indices = indices[selected]
                scores = scores[selected]

            rows.extend([item_idx] * len(indices))
            cols.extend(indices.tolist())
            values.extend(scores.tolist())

        self.item_similarity = csr_matrix(
            (values, (rows, cols)),
            shape=similarity.shape,
        )

    def _fit_svd(self):
        """Train a Truncated SVD model."""

        n_users, n_items = self.matrix.shape

        if n_items < 2 or n_users < 2:
            return

        components = min(
            self.n_components,
            n_items - 1,
            n_users - 1,
        )

        if components < 1:
            return

        self.svd = TruncatedSVD(
            n_components=components,
            random_state=self.random_state,
        )

        self.svd.fit(self.matrix)

    def _popularity_score(self, item_id):
        """Return the normalized popularity score."""

        return self.popularity_scores.get(item_id, np.nan)

    def _predict_all(self, vector):
        """Predicted rating of every trained item for one rating vector (0 = unrated).

        kNN: similarity-weighted mean of the user's ratings over the neighbours of each item
        (user mean when no neighbour was rated). SVD: reconstruction of the rating row.
        """
        rated = vector > 0
        mean_rating = float(vector[rated].mean()) if rated.any() else 0.0

        if self.method == "knn":
            if self.item_similarity is None:
                return np.full(len(self.items), mean_rating)
            numerator = self.item_similarity @ vector
            denominator = self.item_similarity @ rated.astype(float)
            safe = np.where(denominator > 0, denominator, 1.0)
            return np.where(denominator > 0, numerator / safe, mean_rating)

        if self.svd is None:
            return np.full(len(self.items), mean_rating)
        latent = self.svd.transform(csr_matrix(vector.reshape(1, -1)))
        return (latent @ self.svd.components_).ravel()

    def _pick(self, values, item_ids):
        """Values of the requested items (NaN for items the model was not trained on)."""
        out = np.full(len(item_ids), np.nan)
        for pos, item_id in enumerate(item_ids):
            idx = self.item_to_idx.get(item_id)
            if idx is not None:
                out[pos] = values[idx]
        return out

    def score(self, user_id, item_ids):
        """
        Return predicted scores for a list of course IDs.

        Unknown or collaborative-untrained items return NaN.
        Unknown users use popularity fallback.
        """
        item_ids = list(item_ids)

        if user_id not in self.user_to_idx or self.matrix is None or self.matrix.nnz == 0:
            return np.array(
                [self._popularity_score(i) if i in self.item_to_idx else np.nan for i in item_ids],
                dtype=float,
            )

        vector = self.matrix.getrow(self.user_to_idx[user_id]).toarray().ravel()
        return self._pick(self._predict_all(vector), item_ids)

    def score_from_ratings(self, ratings, item_ids):
        """Same as ``score`` for an ad-hoc history {course_id: rating} (e.g. a new user in the demo)."""
        item_ids = list(item_ids)
        vector = np.zeros(len(self.items))
        for course_id, rating in ratings.items():
            idx = self.item_to_idx.get(course_id)
            if idx is not None:
                vector[idx] = float(rating)

        if not (vector > 0).any() or self.matrix is None or self.matrix.nnz == 0:
            return np.array(
                [self._popularity_score(i) if i in self.item_to_idx else np.nan for i in item_ids],
                dtype=float,
            )
        return self._pick(self._predict_all(vector), item_ids)

    def predict(self, user_id, course_id):
        """Compatibility helper for predicting one course."""

        return float(self.score(user_id, [course_id])[0])

    def recommend(self, user_id, k=10, exclude_seen=True):
        """Generate Top-K recommendations."""

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
