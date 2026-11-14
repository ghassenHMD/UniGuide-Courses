"""Train / validation / test splits and cold-start groups (Step 2).

Run with:  python -m uniguide.data.splits
"""
import json

import numpy as np
import pandas as pd

from uniguide.config import get_path, load_config
from uniguide.data.text import build_course_text

MIN_INTERACTIONS = 3  # a user needs at least 1 train + 1 val + 1 test interaction


def temporal_split(df: pd.DataFrame, test_ratio: float, val_ratio: float):
    """Per-user temporal split: oldest -> train, then val, most recent -> test.

    Returns (train, val, test). Ties on the (daily) timestamp are broken by course_id
    so the split is deterministic.
    """
    df = df.sort_values(["user_id", "timestamp", "course_id"]).reset_index(drop=True)
    grouped = df.groupby("user_id")["course_id"]
    n = grouped.transform("size")
    pos = grouped.cumcount()
    n_test = np.maximum(1, np.round(n * test_ratio)).astype(int)
    n_val = np.maximum(1, np.round(n * val_ratio)).astype(int)
    assert (n - n_test - n_val >= 1).all(), "every user needs at least 1 train interaction"
    is_test = pos >= n - n_test
    is_val = (pos >= n - n_test - n_val) & ~is_test
    return df[~is_test & ~is_val], df[is_val], df[is_test]


def check_no_leakage(train, val, test) -> None:
    """Raise AssertionError if the split leaks or is not chronological."""
    def pairs(d):
        return set(zip(d["user_id"], d["course_id"]))

    assert not pairs(train) & pairs(val), "train/val overlap"
    assert not pairs(train) & pairs(test), "train/test overlap"
    assert not pairs(val) & pairs(test), "val/test overlap"

    for earlier, later, label in [(train, val, "train->val"),
                                  (pd.concat([train, val]), test, "train+val->test")]:
        last = earlier.groupby("user_id")["timestamp"].max()
        first = later.groupby("user_id")["timestamp"].min()
        both = pd.concat([last, first], axis=1, keys=["last", "first"]).dropna()
        assert (both["first"] >= both["last"]).all(), f"not chronological: {label}"


def pick_cold_users(history: pd.DataFrame, sizes, group_ratio, rng) -> dict:
    """Draw disjoint groups of cold-start users, one group per history size.

    Only users with at least max(sizes) pre-test interactions are eligible, so that
    truncating to every size is meaningful.
    """
    counts = history.groupby("user_id").size()
    eligible = np.sort(counts[counts >= max(sizes)].index.to_numpy())
    rng.shuffle(eligible)
    n_group = int(round(len(eligible) * group_ratio))
    assert n_group * len(sizes) <= len(eligible), "cold_user_group_ratio is too large"
    return {
        str(size): [int(u) for u in eligible[i * n_group:(i + 1) * n_group]]
        for i, size in enumerate(sizes)
    }


def apply_cold_start(history: pd.DataFrame, cold_users: dict) -> pd.DataFrame:
    """Keep only the N most recent interactions for each cold-start user.

    `history` is the training data (e.g. train+val). Users outside the cold groups are
    untouched; users of group "0" lose all their history.
    """
    size_of = {int(u): int(size) for size, users in cold_users.items() for u in users}
    history = history.sort_values(["user_id", "timestamp", "course_id"])
    rank_from_end = history.groupby("user_id").cumcount(ascending=False)  # 0 = most recent
    limit = history["user_id"].map(size_of)  # NaN for non cold-start users
    return history[limit.isna() | (rank_from_end < limit)]


def make_splits() -> None:
    cfg = load_config()
    s, d, seed = cfg["split"], cfg["data"], cfg["seed"]
    processed = get_path("processed")
    rng = np.random.default_rng(seed)

    inter = pd.read_parquet(processed / "interactions.parquet")
    reviews_text = pd.read_parquet(processed / "reviews_text.parquet")
    courses = pd.read_parquet(processed / "courses_base.parquet")

    # 1. Cold-start courses: removed from every collaborative set.
    all_courses = np.sort(courses["course_id"].unique())
    n_cold = int(round(len(all_courses) * s["cold_course_ratio"]))
    cold_courses = rng.choice(all_courses, size=n_cold, replace=False)
    is_cold = inter["course_id"].isin(cold_courses)
    cold_inter, warm = inter[is_cold], inter[~is_cold]

    # 2. Users must keep enough interactions after removing cold courses.
    n_per_user = warm.groupby("user_id")["course_id"].transform("size")
    warm = warm[n_per_user >= MIN_INTERACTIONS]

    # 3. Temporal split, then leakage check.
    train, val, test = temporal_split(warm, s["test_ratio"], s["val_ratio"])
    check_no_leakage(train, val, test)

    # 4. Cold-course test set: only users who exist in train (they have a history).
    cold_course_test = cold_inter[cold_inter["user_id"].isin(train["user_id"].unique())]

    # 5. Cold-start user groups, drawn from the pre-test history.
    cold_users = pick_cold_users(pd.concat([train, val]), s["cold_user_sizes"],
                                 s["cold_user_group_ratio"], rng)

    # 6. Course text built from TRAIN reviews only (no test or val text leaks in).
    train_reviews = reviews_text.merge(train[["user_id", "course_id"]], on=["user_id", "course_id"])
    courses = build_course_text(courses, train_reviews, d["max_reviews_per_course_text"], seed)
    courses["n_reviews"] = courses["course_id"].map(train.groupby("course_id").size()).fillna(0).astype(int)

    # 7. Save.
    train.to_parquet(processed / "train.parquet", index=False)
    val.to_parquet(processed / "val.parquet", index=False)
    test.to_parquet(processed / "test.parquet", index=False)
    cold_course_test.to_parquet(processed / "cold_course_test.parquet", index=False)
    courses.to_parquet(processed / "courses.parquet", index=False)
    (processed / "cold_users.json").write_text(json.dumps(cold_users))
    (processed / "cold_courses.json").write_text(json.dumps([str(c) for c in cold_courses]))

    summary = pd.DataFrame(
        [{"set": name, "n_rows": len(df), "n_users": df["user_id"].nunique(),
          "n_courses": df["course_id"].nunique()}
         for name, df in [("train", train), ("val", val), ("test", test),
                          ("cold_course_test", cold_course_test)]]
    )
    out = get_path("results") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "splits_summary.csv", index=False)
    print(summary.to_string(index=False))
    print({k: len(v) for k, v in cold_users.items()}, "users per cold-start group")


if __name__ == "__main__":
    make_splits()