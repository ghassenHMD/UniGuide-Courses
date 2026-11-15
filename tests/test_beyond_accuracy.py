import numpy as np
import pandas as pd
import pytest

from uniguide.evaluation.beyond_accuracy import (
    ItemSimilarity, catalog_coverage, evaluate_beyond_accuracy, intra_list_diversity,
    novelty, popularity_from_train,
)
from uniguide.models.popularity import PopularityRecommender


@pytest.fixture
def sim():
    # a, b identical; c orthogonal to them
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return ItemSimilarity(vectors, ["a", "b", "c"])


def test_coverage():
    catalog = {"a", "b", "c", "d"}
    assert catalog_coverage([["a", "b"], ["a"], []], catalog) == 0.5
    assert catalog_coverage([["a", "zzz"]], catalog) == 0.25     # outside the catalogue ignored
    assert catalog_coverage([], catalog) == 0.0


def test_diversity(sim):
    assert intra_list_diversity(["a", "b"], sim) == pytest.approx(0.0)       # identical
    assert intra_list_diversity(["a", "c"], sim) == pytest.approx(1.0)       # orthogonal
    assert intra_list_diversity(["a", "b", "c"], sim) == pytest.approx(1 - 2 / 6)
    assert np.isnan(intra_list_diversity(["a"], sim))
    assert np.isnan(intra_list_diversity(["a", "unknown"], sim))


def test_diversity_accepts_sparse_vectors():
    from scipy.sparse import csr_matrix
    s = ItemSimilarity(csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]])), ["x", "y"])
    assert intra_list_diversity(["x", "y"], s) == pytest.approx(1.0)


def test_novelty_and_popularity():
    train = pd.DataFrame({"user_id": [1, 2, 3, 4, 1], "course_id": ["a", "a", "a", "a", "b"]})
    pop = popularity_from_train(train, ["a", "b", "never_rated"])
    assert pop["a"] == 1.0 and pop["b"] == 0.25 and pop["never_rated"] == 0.25   # floor: 1 user
    assert novelty(["a"], pop) == pytest.approx(0.0)
    assert novelty(["b"], pop) == pytest.approx(2.0)
    assert novelty(["a", "b"], pop) == pytest.approx(1.0)
    assert np.isnan(novelty([], pop))
    assert novelty(["b", "ghost"], pop) == pytest.approx(2.0)


def test_evaluate_beyond_accuracy_format(sim):
    train = pd.DataFrame({"user_id": [1, 1, 2, 3], "course_id": ["a", "b", "a", "c"],
                          "rating": [5, 4, 5, 5]})
    courses = pd.DataFrame({"course_id": ["a", "b", "c"]})
    model = PopularityRecommender().fit(train, courses)
    pop = popularity_from_train(train, ["a", "b", "c"])
    res = evaluate_beyond_accuracy(model, [1, 2, 3], sim, pop, {"a", "b", "c"}, k_values=(1, 2))
    assert list(res.columns) == ["method", "experiment", "seed", "k", "metric", "value"]
    assert set(res["metric"]) == {"coverage", "diversity", "novelty"}
    assert (res[res.metric == "coverage"].groupby("k").size() == 1).all()
    # k=1: one item per list -> no diversity row; novelty exists
    assert not ((res.k == 1) & (res.metric == "diversity")).any()
    assert ((res.k == 1) & (res.metric == "novelty")).any()