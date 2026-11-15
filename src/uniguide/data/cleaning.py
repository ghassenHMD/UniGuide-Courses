"""Cleaning of the raw reviews (Step 1)."""
import pandas as pd

from uniguide.data.stats import snapshot


def clean_reviews(reviews: pd.DataFrame, courses: pd.DataFrame, data_cfg: dict):
    """Clean raw reviews. Returns (clean_df, log).

    Output columns: user_name, course_id, rating, timestamp, review.
    `log` is a list of dicts recording the data size after each step.
    """
    log = []
    df = reviews.rename(
        columns={"reviewers": "user_name", "date_reviews": "date", "reviews": "review"}
    )
    log.append(snapshot(df, "raw", "user_name"))

    # 1. Drop incomplete rows and invalid ratings.
    df = df.dropna(subset=["review", "user_name", "date", "rating", "course_id"])
    df = df[df["rating"].between(1, 5)]
    log.append(snapshot(df, "drop incomplete rows", "user_name"))

    # 2. Join on course_id: keep only reviews of courses present in the courses file.
    df = df[df["course_id"].isin(courses["course_id"])].copy()
    log.append(snapshot(df, "keep known courses", "user_name"))

    # 3. Normalize reviewer names ("By Robert S" -> "Robert S") and drop generic accounts.
    df["user_name"] = df["user_name"].str.replace(r"^By\s+", "", regex=True).str.strip()
    lower = df["user_name"].str.lower()
    generic = {n.lower() for n in data_cfg["generic_user_names"]}
    is_generic = lower.isin(generic) | lower.str.startswith("deleted") | (lower == "")
    df = df[~is_generic].copy()
    log.append(snapshot(df, "drop generic users", "user_name"))

    # 4. Parse dates ("Feb 12, 2020"); unparsable dates become NaT and are dropped.
    df["timestamp"] = pd.to_datetime(df["date"], format="%b %d, %Y", errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop(columns="date")
    log.append(snapshot(df, "parse dates", "user_name"))

    # 5. Duplicates: same (user, course) -> keep the most recent review.
    df = df.sort_values("timestamp").drop_duplicates(["user_name", "course_id"], keep="last")
    log.append(snapshot(df, "drop duplicate (user, course)", "user_name"))

    # 6. Pseudo-users with too many reviews are most likely several people sharing a name.
    n_reviews = df["user_name"].map(df["user_name"].value_counts())
    df = df[n_reviews <= data_cfg["max_user_reviews"]]
    log.append(snapshot(df, "drop suspected homonyms", "user_name"))

    return df.reset_index(drop=True), log