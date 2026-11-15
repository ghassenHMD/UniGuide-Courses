import numpy as np
import pandas as pd
import pytest

from uniguide.models.content_based import ContentBasedRecommender


@pytest.fixture
def data():
    courses = pd.DataFrame({
        "course_id": ["ml1", "ml2", "ml3", "biz1", "biz2", "new_ml"],
        "text": [
            "machine learning neural networks python",
            "deep learning neural networks training model",
            "machine learning model python data",
            "marketing management strategy finance",
            "business strategy leadership management",
            "neural networks machine learning",   # new course: no train interaction
        ],
    })
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5],
        "course_id": ["ml1", "biz1", "biz1", "biz2", "ml1", "ml2", "ml1", "ml2", "ml1"],
        "rating": [5, 2, 5, 4, 5, 5, 3, 2, 4],
    })
    return train, courses


def make(**kw):
    return ContentBasedRecommender(min_df=1, max_df=1.0, **kw)


def test_profile_prefers_similar_content(data):
    train, courses = data
    model = make().fit(train, courses)
    s = dict(zip(["ml2", "ml3", "biz2"], model.score(1, ["ml2", "ml3", "biz2"])))
    assert s["ml2"] > s["biz2"] and s["ml3"] > s["biz2"]  # user 1 liked only ml1 (biz1 rated 2)
    s2 = dict(zip(["ml3", "biz2"], model.score(2, ["ml3", "biz2"])))
    assert s2["biz2"] > s2["ml3"]


def test_scores_are_cosines_in_unit_range(data):
    train, courses = data
    model = make().fit(train, courses)
    scores = model.score(1, courses["course_id"].tolist())
    assert np.all((scores >= -1e-9) & (scores <= 1 + 1e-9))
    assert model.score(1, ["ml1"])[0] > 0.99  # the only liked course is its own profile


def test_disliked_courses_are_not_in_profile(data):
    train, courses = data
    model = make().fit(train, courses)
    # user 4 rated everything < 4: no profile -> popularity fallback, never a crash
    pop = model.score(4, ["ml1", "biz1"])
    assert np.isfinite(pop).all()


def test_unknown_user_falls_back_to_popularity(data):
    train, courses = data
    model = make().fit(train, courses)
    s = model.score(999, ["ml1", "biz1"])
    assert np.isfinite(s).all() and s.shape == (2,)


def test_unknown_item_is_nan_and_new_course_is_scored(data):
    train, courses = data
    model = make().fit(train, courses)
    assert np.isnan(model.score(1, ["nope"])[0])
    assert np.isfinite(model.score(1, ["new_ml"])[0])      # CB can score a course without ratings
    assert model.score(1, ["new_ml"])[0] > model.score(1, ["biz2"])[0]


def test_recommend_excludes_seen_sorted_and_deterministic(data):
    train, courses = data
    rec = make().fit(train, courses).recommend(1, k=3)
    assert list(rec.columns) == ["course_id", "score"]
    assert not rec["course_id"].isin({"ml1", "biz1"}).any()
    assert rec["score"].is_monotonic_decreasing
    rec2 = make().fit(train, courses).recommend(1, k=3)
    pd.testing.assert_frame_equal(rec, rec2)


def test_score_from_ratings_matches_score(data):
    train, courses = data
    model = make().fit(train, courses)
    items = courses["course_id"].tolist()
    np.testing.assert_allclose(
        model.score(1, items), model.score_from_ratings({"ml1": 5, "biz1": 2}, items)
    )


def test_rating_offset_must_be_below_threshold():
    with pytest.raises(ValueError):
        ContentBasedRecommender(rating_offset=4, liked_threshold=4)