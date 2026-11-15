import numpy as np
import pandas as pd
import pytest

from uniguide.models.popularity import PopularityRecommender


@pytest.fixture
def data():
    # A: 4 ratings (mean 5.0), B: 2 ratings (mean 4.0), C: 1 rating (5), D: never rated
    train = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 1, 2, 3],
        "course_id": ["A", "A", "A", "A", "B", "B", "C"],
        "rating": [5, 5, 5, 5, 4, 4, 5],
    })
    courses = pd.DataFrame({"course_id": ["A", "B", "C", "D"]})
    return train, courses


def test_weighted_rating_formula(data):
    train, courses = data
    model = PopularityRecommender(min_votes=2).fit(train, courses)
    c = 33 / 7  # global mean
    assert model.global_mean == pytest.approx(c)
    # WR = v/(v+m)*R + m/(v+m)*C
    assert model.score(1, ["A"])[0] == pytest.approx(4 / 6 * 5.0 + 2 / 6 * c)
    assert model.score(1, ["B"])[0] == pytest.approx(2 / 4 * 4.0 + 2 / 4 * c)
    assert model.score(1, ["C"])[0] == pytest.approx(1 / 3 * 5.0 + 2 / 3 * c)


def test_few_votes_are_shrunk_towards_global_mean(data):
    train, courses = data
    model = PopularityRecommender(min_votes=2).fit(train, courses)
    # C has a perfect 5.0 but only one vote: it must rank below A (same mean, more votes)
    assert model.score(1, ["A"])[0] > model.score(1, ["C"])[0]


def test_unrated_and_unknown_courses_are_nan(data):
    train, courses = data
    model = PopularityRecommender().fit(train, courses)
    assert np.isnan(model.score(1, ["D"])[0])
    assert np.isnan(model.score(1, ["ZZZ"])[0])


def test_recommend_excludes_seen_and_is_sorted(data):
    train, courses = data
    model = PopularityRecommender(min_votes=2).fit(train, courses)
    rec = model.recommend(3, k=5, exclude_seen=True)  # user 3 saw A and C
    assert list(rec["course_id"]) == ["B"]
    rec_all = model.recommend(99, k=5)  # unknown user: same ranking for everybody
    assert list(rec_all.columns) == ["course_id", "score"]
    assert rec_all["score"].is_monotonic_decreasing


def test_does_not_use_courses_outside_catalog(data):
    train, courses = data
    model = PopularityRecommender().fit(train, courses[courses["course_id"] != "C"])
    assert np.isnan(model.score(1, ["C"])[0])