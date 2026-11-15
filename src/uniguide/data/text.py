"""Course text construction (used in Step 2, on training reviews only)."""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_NON_LETTERS = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits and stop words, drop very short tokens."""
    text = _NON_LETTERS.sub(" ", str(text).lower())
    tokens = [t for t in text.split() if len(t) > 2 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def build_course_text(courses: pd.DataFrame, reviews: pd.DataFrame,
                      max_reviews: int, seed: int) -> pd.DataFrame:
    """Add a `text` column = name + institution + at most `max_reviews` reviews per course.

    `reviews` must have columns course_id, review. Courses without reviews (cold-start
    courses) get name + institution only.
    """
    sampled = reviews.sample(frac=1, random_state=seed).groupby("course_id").head(max_reviews)
    joined = sampled.groupby("course_id")["review"].apply(" ".join)
    out = courses.copy()
    reviews_text = out["course_id"].map(joined).fillna("")
    head = out["name"].fillna("") + " " + out["institution"].fillna("")
    out["text"] = (head + " " + reviews_text).map(clean_text)
    return out