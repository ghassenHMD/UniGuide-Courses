
import numpy as np
import pandas as pd
import pytest

from uniguide.models.collaborative import CollaborativeRecommender
from uniguide.models.hybrid import HybridRecommender


class DummyContentModel:
    """Simple Content-Based model used for isolated tests."""

    name = "dummy_content"

    def fit(self, train, courses):
        return self

    def score(self, user_id, item_ids):
        scores = {
            "A": 0.8,
            "B": 0.4,
            "C": 0.9,
            "D": 0.6,
            "E": 0.7,
        }

        return np.array(
            [scores.get(item, np.nan) for item in item_ids],
            dtype=float,
        )


@pytest.fixture
def toy_data():
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
        "course_id": ["A", "B", "A", "B", "C",
                      "A", "C", "A", "C", "D"],
        "rating": [5, 4, 5, 4, 2, 1, 5, 1, 5, 4],
    })

    courses = pd.DataFrame({
        "course_id": ["A", "B", "C", "D", "E"],
    })

    return train, courses


@pytest.fixture
def hybrid_model(toy_data):
    train, courses = toy_data

    collaborative = CollaborativeRecommender(method="knn")
    content = DummyContentModel()

    model = HybridRecommender(
        content,
        collaborative,
        alpha=0.5,
    )

    model.fit(train, courses)

    return model


def test_hybrid_fit_and_score(hybrid_model):
    scores = hybrid_model.score(1, ["A", "B", "C", "D"])

    assert len(scores) == 4
    assert np.isfinite(scores).all()


def test_hybrid_unknown_user_uses_fallback(hybrid_model):
    scores = hybrid_model.score(999, ["A", "C"])

    assert len(scores) == 2
    assert np.isfinite(scores).all()


def test_hybrid_new_course_uses_content_score(hybrid_model):
    # E has no collaborative interaction, but has a content score.
    scores = hybrid_model.score(1, ["E"])

    assert len(scores) == 1
    assert np.isfinite(scores[0])


def test_hybrid_unknown_course_returns_nan(hybrid_model):
    scores = hybrid_model.score(1, ["UNKNOWN_COURSE"])

    assert np.isnan(scores[0])


def test_hybrid_recommend_excludes_seen_courses(hybrid_model):
    recommendations = hybrid_model.recommend(
        1,
        k=5,
        exclude_seen=True,
    )

    assert not recommendations["course_id"].isin({"A", "B"}).any()
