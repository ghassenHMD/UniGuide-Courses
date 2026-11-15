"""Loading of the raw Kaggle CSV files."""
import pandas as pd

from uniguide.config import get_path


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reviews, courses) exactly as stored in data/raw/.

    The courses column `course_url` is renamed to `url` (data contract).
    """
    raw = get_path("raw")
    reviews = pd.read_csv(raw / "Coursera_reviews.csv")
    courses = pd.read_csv(raw / "Coursera_courses.csv")
    courses = courses.rename(columns={"course_url": "url"}).drop_duplicates("course_id")
    return reviews, courses