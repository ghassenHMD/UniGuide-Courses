
import numpy as np
import pandas as pd
import pytest

from uniguide.models.collaborative import CollaborativeRecommender


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


@pytest.mark.parametrize("method", ["knn", "svd"])
def test_model_fit_and_score(method, toy_data):
    train, courses = toy_data

    model = CollaborativeRecommender(method=method)
    fitted_model = model.fit(train, courses)

    assert fitted_model is model

    scores = model.score(1, ["A", "B", "C"])

    assert len(scores) == 3
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("method", ["knn", "svd"])
def test_unknown_user_uses_fallback(method, toy_data):
    train, courses = toy_data

    model = CollaborativeRecommender(method=method)
    model.fit(train, courses)

    scores = model.score(999, ["A"])

    assert len(scores) == 1
    assert np.isfinite(scores[0])


@pytest.mark.parametrize("method", ["knn", "svd"])
def test_unknown_item_returns_nan(method, toy_data):
    train, courses = toy_data

    model = CollaborativeRecommender(method=method)
    model.fit(train, courses)

    scores = model.score(1, ["UNKNOWN_COURSE"])

    assert np.isnan(scores[0])


@pytest.mark.parametrize("method", ["knn", "svd"])
def test_recommend_excludes_seen_courses(method, toy_data):
    train, courses = toy_data

    model = CollaborativeRecommender(method=method)
    model.fit(train, courses)

    recommendations = model.recommend(1, k=5, exclude_seen=True)

    assert not recommendations["course_id"].isin({"A", "B"}).any()


@pytest.mark.parametrize("method", ["knn", "svd"])
def test_recommend_returns_expected_columns(method, toy_data):
    train, courses = toy_data

    model = CollaborativeRecommender(method=method)
    model.fit(train, courses)

    recommendations = model.recommend(1, k=3)

    assert {"course_id", "score"}.issubset(recommendations.columns)
