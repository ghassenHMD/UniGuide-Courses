"""Filtering and sampling (Step 1)."""
import numpy as np
import pandas as pd


def iterative_filter(df, min_user_reviews, min_course_reviews, max_iter=20):
    """Keep users with >= min_user_reviews and courses with >= min_course_reviews.

    Removing courses lowers user counts (and vice versa), so we repeat until stable.
    """
    for _ in range(max_iter):
        n_before = len(df)
        user_n = df.groupby("user_id")["course_id"].transform("size")
        df = df[user_n >= min_user_reviews]
        course_n = df.groupby("course_id")["user_id"].transform("size")
        df = df[course_n >= min_course_reviews]
        if len(df) == n_before:
            break
    return df.reset_index(drop=True)


def sample_users(df, n_users, seed):
    """Randomly keep n_users users (fixed seed). No-op if there are fewer users."""
    users = np.sort(df["user_id"].unique())
    if len(users) <= n_users:
        return df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=n_users, replace=False)
    return df[df["user_id"].isin(chosen)].reset_index(drop=True)