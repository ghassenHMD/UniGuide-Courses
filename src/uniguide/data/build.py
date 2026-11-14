"""Step 1 pipeline: raw CSVs -> cleaned interactions.

Run with:  python -m uniguide.data.build
"""
import pandas as pd

from uniguide.config import get_path, load_config
from uniguide.data.cleaning import clean_reviews
from uniguide.data.filtering import iterative_filter, sample_users
from uniguide.data.loading import load_raw
from uniguide.data.stats import snapshot


def build_interactions() -> None:
    cfg = load_config()
    d, seed = cfg["data"], cfg["seed"]
    processed = get_path("processed")
    tables = get_path("results") / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    reviews, courses = load_raw()
    df, log = clean_reviews(reviews, courses, d)

    # Pseudo-identifier: one integer per distinct cleaned display name (deterministic).
    df["user_id"], _ = pd.factorize(df["user_name"], sort=True)

    df = iterative_filter(df, d["min_user_reviews"], d["min_course_reviews"])
    log.append(snapshot(df, f"filter users>={d['min_user_reviews']}, courses>={d['min_course_reviews']}"))

    df = sample_users(df, d["n_users_sample"], seed)
    log.append(snapshot(df, f"sample {d['n_users_sample']} users"))

    df = iterative_filter(df, d["min_user_reviews"], d["min_course_reviews"])
    log.append(snapshot(df, "re-filter after sampling"))

    df = df.copy()
    df["is_relevant"] = df["rating"] >= d["relevance_threshold"]

    interactions = df[["user_id", "course_id", "rating", "timestamp", "is_relevant"]]
    reviews_text = df[["user_id", "course_id", "review"]]
    users = df[["user_id", "user_name"]].drop_duplicates()
    kept = courses[courses["course_id"].isin(df["course_id"].unique())].reset_index(drop=True)

    interactions.to_parquet(processed / "interactions.parquet", index=False)
    reviews_text.to_parquet(processed / "reviews_text.parquet", index=False)
    users.to_parquet(processed / "users.parquet", index=False)
    kept.to_parquet(processed / "courses_base.parquet", index=False)

    log_df = pd.DataFrame(log)
    log_df.to_csv(tables / "filtering_log.csv", index=False)
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    build_interactions()