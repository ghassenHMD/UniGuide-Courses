
"""Experiments and hyperparameter selection."""

import copy

import numpy as np
import pandas as pd

from uniguide.models.hybrid import HybridRecommender
from .protocol import evaluate_model


def select_best_alpha(
    content_model,
    collaborative_model,
    train,
    courses,
    validation,
    alphas=None,
    k=10,
):
    """
    Select the best Hybrid alpha using validation NDCG@k.

    Alpha is the weight assigned to the Content-Based model.
    The Collaborative Filtering weight is (1 - alpha).
    """

    if k <= 0:
        raise ValueError("k must be strictly positive.")

    if validation.empty:
        raise ValueError("Validation dataset is empty.")

    if alphas is None:
        alphas = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ]

    alphas = [float(alpha) for alpha in alphas]

    if not alphas:
        raise ValueError("The alpha list cannot be empty.")

    if any(not np.isfinite(a) or a < 0 or a > 1 for a in alphas):
        raise ValueError("Every alpha must be between 0 and 1.")

    if len(alphas) != len(set(alphas)):
        raise ValueError("The alpha list must not contain duplicates.")

    experiments = []

    # alpha only changes how the two normalised scores are mixed: fit the components once.
    model = HybridRecommender(
        copy.deepcopy(content_model),
        copy.deepcopy(collaborative_model),
        alpha=alphas[0],
    )
    model.fit(train, courses)

    for alpha in alphas:
        model.alpha = alpha

        results = evaluate_model(
            model,
            validation,
            k_values=(k,),
            experiment=f"alpha_{alpha:.1f}",
        )

        ndcg_values = results.loc[results["metric"] == "ndcg", "value"]
        mean_ndcg = float(ndcg_values.mean()) if not ndcg_values.empty else np.nan

        experiments.append({"alpha": alpha, f"ndcg@{k}": mean_ndcg})

    comparison = pd.DataFrame(experiments)

    score_column = f"ndcg@{k}"

    if comparison[score_column].isna().all():
        raise ValueError(
            "Cannot select alpha: no valid NDCG results were obtained."
        )

    best_index = comparison[score_column].idxmax()
    best_alpha = float(comparison.loc[best_index, "alpha"])

    return best_alpha, comparison

def select_alpha_schedule(
    content_model,
    collaborative_model,
    train,
    courses,
    validation,
    cold_users,
    default_alpha,
    alphas=None,
    k=10,
):
    """Tune the switching rule: one alpha per history size (cold-start groups of the split).

    The training history of each cold-start group (sizes in ``cold_users``, e.g. 1, 3, 5) is
    truncated with ``apply_cold_start``; the hybrid is fitted once and each group is evaluated on
    the validation interactions of its own users. Group 0 is skipped (popularity is used).

    Returns (schedule, table): ``schedule`` is ``[(max_history, alpha), ...]`` for
    ``HybridRecommender(alpha_schedule=...)`` (``default_alpha`` applies beyond the largest group)
    and ``table`` shows NDCG@k for every (history size, alpha).
    """
    from uniguide.data.splits import apply_cold_start

    alphas = [round(a / 10, 1) for a in range(11)] if alphas is None else list(alphas)
    truncated = apply_cold_start(train, cold_users)

    model = HybridRecommender(
        copy.deepcopy(content_model),
        copy.deepcopy(collaborative_model),
        alpha=default_alpha,
    ).fit(truncated, courses)

    rows = []
    for size, users in cold_users.items():
        if int(size) == 0:
            continue
        group_validation = validation[validation["user_id"].isin(users)]
        if group_validation.empty:
            continue
        for alpha in alphas:
            model.alpha = alpha
            res = evaluate_model(model, group_validation, k_values=(k,), experiment=f"history_{size}")
            ndcg = res.loc[res["metric"] == "ndcg", "value"]
            rows.append({"history": int(size), "alpha": alpha,
                         f"ndcg@{k}": float(ndcg.mean()) if len(ndcg) else np.nan})

    table = pd.DataFrame(rows)
    schedule = []
    for size, group in table.groupby("history"):
        best = group.loc[group[f"ndcg@{k}"].idxmax(), "alpha"]
        schedule.append((int(size), float(best)))
    return sorted(schedule), table