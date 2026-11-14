
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
