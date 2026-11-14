
import pandas as pd
import pytest

from uniguide.evaluation.protocol import evaluate_model


class DummyRecommender:
    name = "dummy"

    def recommend(self, user_id, k=10, exclude_seen=True):
        recommendations = {
            1: ["A", "B", "C"],
            2: ["D", "E"],
        }
        return pd.DataFrame({
            "course_id": recommendations.get(user_id, [])[:k]
        })


def test_evaluate_model_returns_expected_columns():
    test = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3],
        "course_id": ["A", "B", "C", "D", "E", "F"],
        "is_relevant": [True, False, True, True, False, False],
    })

    results = evaluate_model(
        DummyRecommender(),
        test,
        k_values=(2,),
        experiment="test",
        seed=42,
    )

    expected = {
        "method", "experiment", "seed",
        "k", "metric", "value"
    }

    assert set(results.columns) == expected
    assert len(results) == 8
    assert set(results["method"]) == {"dummy"}
    assert set(results["k"]) == {2}
    assert set(results["metric"]) == {
        "precision", "recall", "ndcg", "hit_rate"
    }


def test_evaluate_model_excludes_users_without_relevant_items():
    test = pd.DataFrame({
        "user_id": [1, 2, 3],
        "course_id": ["A", "B", "C"],
        "is_relevant": [True, False, False],
    })

    results = evaluate_model(
        DummyRecommender(),
        test,
        k_values=(5,),
    )

    assert results["method"].nunique() == 1
    assert len(results) == 4
    assert results.attrs["evaluated_users"] == 1
    assert results.attrs[
        "excluded_users_without_relevant_items"
    ] == 2


def test_evaluate_model_rejects_invalid_columns():
    invalid_test = pd.DataFrame({
        "user_id": [1],
        "course_id": ["A"],
    })

    with pytest.raises(ValueError):
        evaluate_model(DummyRecommender(), invalid_test)
