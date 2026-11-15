import numpy as np
import pandas as pd
import pytest

from uniguide.explain import Explainer
from uniguide.models.collaborative import CollaborativeRecommender
from uniguide.models.content_based import ContentBasedRecommender
from uniguide.models.hybrid import HybridRecommender

TOPICS = ["python data model learning", "marketing finance strategy business",
          "health nutrition medicine care", "web code software java"]


@pytest.fixture
def system():
    rng = np.random.default_rng(0)
    n_items = 24
    courses = pd.DataFrame({"course_id": [f"c{i}" for i in range(n_items + 2)]})
    courses["name"] = [f"Course {i}" for i in range(len(courses))]
    courses["institution"] = "Uni"
    courses["url"] = "https://example.org"
    courses["text"] = [TOPICS[i % 4] + f" extra{i % 5}" for i in range(len(courses))]
    rows = []
    for user in range(60):
        favorite = rng.integers(0, 4)
        for c in rng.choice(n_items, size=rng.integers(6, 14), replace=False):
            rating = int(np.clip(round(rng.normal(4.6 if c % 4 == favorite else 3.5, 0.7)), 1, 5))
            rows.append((user, f"c{c}", rating))
    train = pd.DataFrame(rows, columns=["user_id", "course_id", "rating"])
    hybrid = HybridRecommender(
        ContentBasedRecommender(min_df=1), CollaborativeRecommender(method="knn"),
        alpha=0.5, alpha_schedule=[(2, 0.9)],
    ).fit(train, courses)
    return hybrid, Explainer(hybrid, courses, train)


def test_explained_ranking_matches_hybrid(system):
    hybrid, explainer = system
    explained = explainer.explain_user(3, k=5)
    expected = hybrid.recommend(3, k=5)
    assert [e.course_id for e in explained] == expected["course_id"].tolist()
    assert np.allclose([e.score for e in explained], expected["score"])


def test_contributions_add_up_to_the_score(system):
    _, explainer = system
    for e in explainer.explain_user(3, k=10):
        assert e.cb_contribution + e.cf_contribution == pytest.approx(e.score)


def test_because_only_lists_liked_courses_from_history(system):
    _, explainer = system
    ratings = explainer.history_of(3)
    for e in explainer.explain_user(3, k=10):
        for b in e.because:
            assert ratings[b["course_id"]] >= 4
            assert b["course_id"] != e.course_id


def test_switching_popularity_then_hybrid(system):
    _, explainer = system
    assert explainer.explain_ratings({}, k=3)[0].mode == "popularity"
    one = explainer.explain_ratings({"c0": 5}, k=3)[0]
    assert one.mode == "hybrid" and one.alpha == pytest.approx(0.9)


def test_new_course_is_explained_with_content_only(system):
    hybrid, explainer = system
    modes = {e.course_id: e.mode for e in explainer.explain_user(3, k=len(hybrid.catalog))}
    assert modes["c24"] == "content_only"