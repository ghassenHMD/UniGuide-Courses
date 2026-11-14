import numpy as np
import pandas as pd
import pytest

from uniguide.data.splits import apply_cold_start, check_no_leakage, temporal_split


@pytest.fixture
def df():
    rows = []
    for u in range(10):
        for t in range(10):
            rows.append((u, f"c{t}", 5, pd.Timestamp("2021-01-01") + pd.Timedelta(days=t)))
    return pd.DataFrame(rows, columns=["user_id", "course_id", "rating", "timestamp"])


def test_split_sizes_and_no_leakage(df):
    train, val, test = temporal_split(df, 0.2, 0.1)
    assert len(train) + len(val) + len(test) == len(df)
    assert (test.groupby("user_id").size() == 2).all()
    assert (val.groupby("user_id").size() == 1).all()
    check_no_leakage(train, val, test)


def test_test_is_most_recent(df):
    train, _, test = temporal_split(df, 0.2, 0.1)
    assert test["timestamp"].min() > train["timestamp"].max()


def test_cold_start_truncation(df):
    cold = {"0": [0], "1": [1], "3": [2]}
    out = apply_cold_start(df, cold)
    sizes = out.groupby("user_id").size()
    assert 0 not in sizes.index
    assert sizes[1] == 1 and sizes[2] == 3 and sizes[3] == 10
    # the kept interaction of user 1 is its most recent one
    assert out[out["user_id"] == 1]["course_id"].iloc[0] == "c9"