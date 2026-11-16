
"""
Evaluation metrics for UniGuide-Courses.
Person B.
"""


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        recommended: Ordered list of recommended course IDs.
        relevant: Set of relevant course IDs.
        k: Number of recommendations to consider.

    Returns:
        Proportion of the top-K recommendations that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = recommended[:k]
    hits = sum(1 for course in top_k if course in relevant)

    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        recommended: Ordered list of recommended course IDs.
        relevant: Set of relevant course IDs.
        k: Number of recommendations to consider.

    Returns:
        Proportion of all relevant courses found in the top-K recommendations.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for course in top_k if course in relevant)

    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: Ordered list of recommended course IDs.
        relevant: Set of relevant course IDs.
        k: Number of recommendations to consider.

    Returns:
        NDCG score between 0 and 1.
    """
    import math

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant:
        return 0.0

    top_k = recommended[:k]

    dcg = sum(
        1 / math.log2(position + 2)
        for position, course in enumerate(top_k)
        if course in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / math.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Calculate Hit Rate@K.

    Args:
        recommended: Ordered list of recommended course IDs.
        relevant: Set of relevant course IDs.
        k: Number of recommendations to consider.

    Returns:
        1.0 if at least one relevant course is recommended, otherwise 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = recommended[:k]

    return float(any(course in relevant for course in top_k))

def rating_errors(model, test_data) -> dict:
    """MAE and RMSE of a model whose ``score`` is a predicted rating (e.g. collaborative).

    Only observed test ratings that the model can score are used; the number of scored
    interactions and the coverage (share of test interactions scored) are returned too.
    The hybrid score lives in [0, 1], not on the rating scale, so it cannot be used here.
    """
    import numpy as np

    errors = []
    for user_id, group in test_data.groupby("user_id"):
        predicted = np.asarray(model.score(user_id, group["course_id"].tolist()), dtype=float)
        valid = np.isfinite(predicted)
        errors.extend((predicted[valid] - group["rating"].to_numpy(dtype=float)[valid]).tolist())

    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_scored": 0, "coverage": 0.0}
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "n_scored": int(errors.size),
        "coverage": float(errors.size / len(test_data)),
    }