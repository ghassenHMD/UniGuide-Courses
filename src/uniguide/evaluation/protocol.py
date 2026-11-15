
"""Common evaluation protocol for recommendation models."""

import pandas as pd

from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
)

METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}


def evaluate_model(
    model,
    test_data,
    k_values=(5, 10),
    experiment="default",
    seed=42,
):
    """Evaluate a fitted model using the same test users and K values."""

    required = {"user_id", "course_id", "is_relevant"}
    if not required.issubset(test_data.columns):
        raise ValueError(f"test_data must contain {required}")

    relevant_data = test_data[test_data["is_relevant"] == True]
    relevant_by_user = relevant_data.groupby("user_id")["course_id"].apply(set)
    users = sorted(relevant_by_user.index)
    max_k = max(k_values)

    rows = []

    for user_id in users:
        relevant_items = relevant_by_user[user_id]

        # One call per user: the top-k list is the first k items of the top-max_k list.
        recommendations = model.recommend(user_id, k=max_k, exclude_seen=True)
        recommended_items = (
            recommendations["course_id"].tolist()
            if not recommendations.empty
            else []
        )

        for k in k_values:
            for metric_name, metric_function in METRICS.items():
                rows.append({
                    "method": model.name,
                    "experiment": experiment,
                    "seed": seed,
                    "k": k,
                    "metric": metric_name,
                    "value": metric_function(recommended_items, relevant_items, k),
                })

    results = pd.DataFrame(
        rows,
        columns=[
            "method", "experiment", "seed",
            "k", "metric", "value"
        ],
    )

    results.attrs["evaluated_users"] = len(users)
    results.attrs["excluded_users_without_relevant_items"] = (
        test_data["user_id"].nunique() - len(users)
    )

    return results


def evaluate_cold_start(model, test_data, k_values=(5, 10)):
    """Evaluate recommendations for test courses absent from CF training."""

    collaborative = getattr(model, "collaborative_model", model)
    known_items = set(getattr(collaborative, "item_to_idx", {}).keys())

    cold_items = set(test_data["course_id"].unique()) - known_items
    cold_test = test_data[test_data["course_id"].isin(cold_items)]

    if cold_test.empty:
        return pd.DataFrame(
            columns=[
                "method", "experiment", "seed",
                "k", "metric", "value"
            ]
        )

    return evaluate_model(
        model,
        cold_test,
        k_values=k_values,
        experiment="cold_start",
    )
