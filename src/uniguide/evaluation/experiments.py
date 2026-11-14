
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

    for alpha in alphas:
        model = HybridRecommender(
            copy.deepcopy(content_model),
            copy.deepcopy(collaborative_model),
            alpha=alpha,
        )

        model.fit(train, courses)

        results = evaluate_model(
            model,
            validation,
            k_values=(k,),
            experiment=f"alpha_{alpha:.1f}",
        )

        ndcg_values = results.loc[
            results["metric"] == "ndcg", "value"
        ]

        mean_ndcg = (
            float(ndcg_values.mean())
            if not ndcg_values.empty
            else np.nan
        )

        experiments.append({
            "alpha": alpha,
            f"ndcg@{k}": mean_ndcg,
        })

    comparison = pd.DataFrame(experiments)

    score_column = f"ndcg@{k}"

    if comparison[score_column].isna().all():
        raise ValueError(
            "Cannot select alpha: no valid NDCG results were obtained."
        )

    best_index = comparison[score_column].idxmax()
    best_alpha = float(comparison.loc[best_index, "alpha"])

    return best_alpha, comparison
