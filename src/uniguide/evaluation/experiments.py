
"""Experiments and hyperparameter selection."""

import copy

import numpy as np
import pandas as pd

from uniguide.models.base import BaseRecommender
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

# --------------------------------------------------------------------------- helpers
def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    """Per-user rows -> one row per (method, experiment, seed, k, metric) with the number of users."""
    keys = ["method", "experiment", "seed", "k", "metric"]
    return results.groupby(keys, as_index=False).agg(value=("value", "mean"), n_users=("value", "size"))


def evaluate_models(models, test, k_values=(5, 10), experiment="main", seed=42, per_user=False):
    """Evaluate several fitted models on the same users and K values.

    ``models`` is a dict {name: fitted model}; the dict key is used as the method name.
    Returns aggregated rows (or per-user rows with ``per_user=True``).
    """
    frames = []
    for name, model in models.items():
        previous, model.name = model.name, name
        try:
            frames.append(evaluate_model(model, test, k_values=k_values, experiment=experiment, seed=seed))
        finally:
            model.name = previous
    results = pd.concat(frames, ignore_index=True)
    return results if per_user else aggregate(results)


def summarize(per_user_results: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation over users and 95% confidence interval of the mean."""
    grouped = per_user_results.groupby(["method", "experiment", "k", "metric"])["value"]
    out = grouped.agg(mean="mean", std="std", n_users="size").reset_index()
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["n_users"].clip(lower=1))
    return out


def across_seeds(aggregated: pd.DataFrame, extra_keys=()) -> pd.DataFrame:
    """Mean and standard deviation across seeds of already aggregated results."""
    keys = ["method", "experiment", "k", "metric", *extra_keys]
    return aggregated.groupby(keys)["value"].agg(mean="mean", std="std", n_seeds="size").reset_index()


# --------------------------------------------------------------------------- cold-start users
def cold_user_experiment(factories, history, test, courses, cold_users, k_values=(5, 10), seed=42):
    """Cold-start users: performance with 0, 1, 3, 5... training interactions, per method.

    ``factories`` is a dict {name: callable returning a new unfitted model}. Each model is fitted
    once on ``history`` truncated with ``apply_cold_start`` and evaluated on the test interactions
    of each group of cold users (experiment name ``cold_users_<size>``).
    """
    from uniguide.data.splits import apply_cold_start

    truncated = apply_cold_start(history, cold_users)
    frames = []
    for name, factory in factories.items():
        model = factory().fit(truncated, courses)
        for size, users in cold_users.items():
            group_test = test[test["user_id"].isin(users)]
            if group_test.empty:
                continue
            frames.append(evaluate_models(
                {name: model}, group_test, k_values, experiment=f"cold_users_{size}", seed=seed))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- cold-start courses
class RestrictedRecommender(BaseRecommender):
    """Wrap a fitted model so that it only ranks a given set of candidate courses."""

    def __init__(self, model, candidates):
        super().__init__()
        self.model = model
        self.candidates = list(candidates)
        self.name = model.name

    def fit(self, train, courses):
        raise NotImplementedError("RestrictedRecommender wraps an already fitted model")

    def seen_items(self, user_id):
        return self.model.seen_items(user_id)

    def score(self, user_id, item_ids):
        return self.model.score(user_id, item_ids)

    def recommend(self, user_id, k=10, exclude_seen=True):
        seen = self.model.seen_items(user_id) if exclude_seen else set()
        items = [c for c in self.candidates if c not in seen]
        if not items:
            return pd.DataFrame(columns=["course_id", "score"])
        frame = pd.DataFrame({"course_id": items, "score": self.model.score(user_id, items)})
        frame = frame.dropna(subset=["score"]).sort_values("score", ascending=False, kind="mergesort")
        return frame.head(k).reset_index(drop=True)


def cold_course_experiment(factories, history, cold_course_test, courses, cold_courses,
                           k_values=(5, 10), seed=42):
    """Cold-start courses: rank the *new* courses (absent from every training set) per method.

    ``courses`` must be the full catalogue (cold courses included, with name + institution text).
    Models are fitted on ``history`` (cold courses have no interaction in it) and rank only the
    cold courses, so the question is "can the method find the new courses a user liked?".
    Collaborative filtering and popularity cannot score them (they return no recommendation,
    hence 0); content-based and hybrid rely on the course text.
    """
    candidates = sorted(set(cold_courses))
    frames = []
    for name, factory in factories.items():
        model = RestrictedRecommender(factory().fit(history, courses), candidates)
        frames.append(evaluate_models(
            {name: model}, cold_course_test, k_values, experiment="cold_courses", seed=seed))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- sensitivity
def sparsify(history: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Randomly drop ``fraction`` of the training interactions (each user keeps its latest one)."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    ordered = history.sort_values(["user_id", "timestamp", "course_id"])
    is_last = ~ordered.duplicated("user_id", keep="last")
    drop = (rng.random(len(ordered)) < fraction) & ~is_last.to_numpy()
    return ordered[~drop]


def sparsity_experiment(factories, history, test, courses, fractions=(0.0, 0.25, 0.5),
                        seeds=(42, 43, 44), k_values=(10,)):
    """Effect of sparsity: refit every method after removing a fraction of training interactions."""
    frames = []
    for fraction in fractions:
        for seed in (seeds if fraction > 0 else seeds[:1]):
            sparse = sparsify(history, fraction, seed) if fraction > 0 else history
            for name, factory in factories.items():
                model = factory().fit(sparse, courses)
                frames.append(evaluate_models(
                    {name: model}, test, k_values, experiment=f"sparsity_{int(fraction * 100)}", seed=seed))
    return pd.concat(frames, ignore_index=True)


def neighbors_experiment(build_cf, build_hybrid, history, validation, courses,
                         n_neighbors=(5, 10, 20, 50, 100), k=10):
    """Effect of the number of neighbours of item-kNN, alone and inside the hybrid (validation).

    ``build_cf`` / ``build_hybrid`` are the factories of ``uniguide.pipeline`` (config overrides).
    """
    frames = []
    for n in n_neighbors:
        cf = build_cf(cf_method="knn", cf_n_neighbors=n).fit(history, courses)
        hybrid = build_hybrid(cf_method="knn", cf_n_neighbors=n).fit(history, courses)
        frames.append(evaluate_models(
            {"collaborative": cf, "hybrid": hybrid}, validation, (k,), experiment=f"neighbors_{n}"))
    return pd.concat(frames, ignore_index=True)


def ablation_models(fitted_hybrid: HybridRecommender) -> dict:
    """Hybrid variants sharing the components of a fitted hybrid (no refit needed).

    full: tuned hybrid with switching; no_switching: constant alpha; no_cb: CF only (CB is kept
    only for courses CF cannot score); no_cf: CB only.
    """
    def variant(alpha, schedule):
        clone = copy.copy(fitted_hybrid)
        clone.alpha = alpha
        clone.alpha_schedule = schedule
        return clone

    return {
        "hybrid_full": fitted_hybrid,
        "hybrid_no_switching": variant(fitted_hybrid.alpha, []),
        "hybrid_no_cb": variant(0.0, []),
        "hybrid_no_cf": variant(1.0, []),
    }