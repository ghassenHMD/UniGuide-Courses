import numpy as np
import pandas as pd
import pytest

from uniguide.evaluation.experiments import (
    RestrictedRecommender, ablation_models, sparsify, summarize)
from uniguide.evaluation.metrics import rating_errors
from uniguide.models.collaborative import CollaborativeRecommender
from uniguide.models.content_based import ContentBasedRecommender
from uniguide.models.hybrid import HybridRecommender, RRFHybridRecommender


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    courses = pd.DataFrame({"course_id": [f"c{i}" for i in range(22)]})
    courses["text"] = [f"topic{i % 3} word{i % 5} common" for i in range(22)]
    rows = [(u, f"c{c}", int(rng.integers(1, 6)), pd.Timestamp("2022-01-01") + pd.Timedelta(days=t))
            for u in range(40) for t, c in enumerate(rng.choice(20, size=8, replace=False))]
    train = pd.DataFrame(rows, columns=["user_id", "course_id", "rating", "timestamp"])
    return train, courses


def test_sparsify_keeps_every_user_and_is_reproducible(data):
    train, _ = data
    a, b = sparsify(train, 0.5, seed=1), sparsify(train, 0.5, seed=1)
    assert a.equals(b)
    assert set(a["user_id"]) == set(train["user_id"])
    assert len(a) < len(train)
    assert len(sparsify(train, 0.0, seed=1)) == len(train)


def test_rrf_hybrid_ranks_and_scores_new_courses(data):
    train, courses = data
    model = RRFHybridRecommender(ContentBasedRecommender(min_df=1), CollaborativeRecommender()).fit(train, courses)
    recs = model.recommend(0, k=5)
    assert len(recs) == 5 and recs["score"].is_monotonic_decreasing
    modes = model.components(0, ["c20", "c1"]).set_index("course_id")["mode"]
    assert modes["c20"] == "content_only"          # c20 has no interaction: CF cannot score it


def test_restricted_recommender_only_returns_candidates(data):
    train, courses = data
    model = HybridRecommender(ContentBasedRecommender(min_df=1), CollaborativeRecommender()).fit(train, courses)
    restricted = RestrictedRecommender(model, ["c20", "c21"])
    assert set(restricted.recommend(0, k=10)["course_id"]) <= {"c20", "c21"}


def test_ablation_variants_share_components(data):
    train, courses = data
    model = HybridRecommender(ContentBasedRecommender(min_df=1), CollaborativeRecommender(), alpha=0.4).fit(train, courses)
    variants = ablation_models(model)
    assert variants["hybrid_no_cb"].alpha == 0.0 and variants["hybrid_no_cf"].alpha == 1.0
    assert variants["hybrid_no_cb"].content_model is model.content_model
    assert model.alpha == 0.4                       # the original model is untouched


def test_rating_errors_and_summarize(data):
    train, courses = data
    cf = CollaborativeRecommender().fit(train, courses)
    errors = rating_errors(cf, train.head(50))
    assert errors["mae"] >= 0 and errors["rmse"] >= errors["mae"] and 0 < errors["coverage"] <= 1
    per_user = pd.DataFrame({"method": "m", "experiment": "main", "k": 10, "metric": "ndcg",
                             "value": [0.2, 0.4, 0.6]})
    out = summarize(per_user)
    assert out.loc[0, "mean"] == pytest.approx(0.4) and out.loc[0, "n_users"] == 3