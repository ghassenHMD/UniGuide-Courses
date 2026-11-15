"""Beyond-accuracy metrics: catalogue coverage, intra-list diversity, novelty (Step 6)."""
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import normalize


class ItemSimilarity:
    """Dense course-course cosine similarity with a course_id lookup."""

    def __init__(self, vectors, item_ids: list[str]) -> None:
        vectors = normalize(vectors)  # works for sparse and dense, L2 per row
        matrix = vectors @ vectors.T
        self.matrix = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
        self.index = {c: i for i, c in enumerate(item_ids)}

    @classmethod
    def from_content_model(cls, model) -> "ItemSimilarity":
        """Build from a fitted content model (TF-IDF or embeddings)."""
        return cls(model.item_vectors, model.catalog)


def popularity_from_train(train: pd.DataFrame, catalog: list[str]) -> dict[str, float]:
    """Share of training users who interacted with each course (at least 1 user, never 0)."""
    n_users = train["user_id"].nunique()
    counts = train.groupby("course_id")["user_id"].nunique()
    return {c: max(int(counts.get(c, 0)), 1) / n_users for c in catalog}


def catalog_coverage(all_recommendations: list[list[str]], catalog: set[str]) -> float:
    """Share of the catalogue recommended at least once (to anybody)."""
    if not catalog:
        return float("nan")
    recommended = {c for recs in all_recommendations for c in recs}
    return len(recommended & set(catalog)) / len(catalog)


def intra_list_diversity(recommended: list[str], similarity: ItemSimilarity) -> float:
    """1 - mean pairwise cosine similarity inside one list (NaN if fewer than 2 known courses)."""
    idx = [similarity.index[c] for c in recommended if c in similarity.index]
    n = len(idx)
    if n < 2:
        return float("nan")
    sub = similarity.matrix[np.ix_(idx, idx)]
    mean_sim = (sub.sum() - np.trace(sub)) / (n * (n - 1))
    return float(1.0 - mean_sim)


def novelty(recommended: list[str], popularity: dict[str, float]) -> float:
    """Mean self-information -log2(popularity) of the recommended courses (higher = more novel).

    `popularity` maps a course to the share of users who interacted with it, in (0, 1].
    Courses missing from the dictionary are ignored.
    """
    values = [-np.log2(popularity[c]) for c in recommended if c in popularity]
    return float(np.mean(values)) if values else float("nan")


def evaluation_users(test: pd.DataFrame) -> list:
    """The users of the accuracy protocol: those with at least one relevant test course."""
    return sorted(test.loc[test["is_relevant"], "user_id"].unique())


def evaluate_beyond_accuracy(
    model,
    users,
    similarity: ItemSimilarity,
    popularity: dict[str, float],
    catalog: set[str],
    k_values=(5, 10),
    experiment: str = "main",
    seed: int = 42,
) -> pd.DataFrame:
    """Same long format as the accuracy results: method, experiment, seed, k, metric, value.

    * `diversity` and `novelty`: one row per user (so mean and std can be computed).
    * `coverage`: one global row per K (computed on the lists of all users).
    """
    max_k = max(k_values)
    lists = {
        u: model.recommend(u, k=max_k, exclude_seen=True)["course_id"].tolist() for u in users
    }
    rows = []
    for k in k_values:
        top = {u: recs[:k] for u, recs in lists.items()}  # lists are sorted: top-k = first k
        base = {"method": model.name, "experiment": experiment, "seed": seed, "k": k}
        rows.append({**base, "metric": "coverage",
                     "value": catalog_coverage(list(top.values()), catalog)})
        for recs in top.values():
            for metric, value in (("diversity", intra_list_diversity(recs, similarity)),
                                  ("novelty", novelty(recs, popularity))):
                if not np.isnan(value):
                    rows.append({**base, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["method", "experiment", "seed", "k", "metric", "value"])