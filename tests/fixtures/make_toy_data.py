"""Generate a tiny synthetic dataset following the data contract (docs/contracts.md, section 4).

It lets Person B develop metrics and CF before the real data is ready.
Output: tests/fixtures/toy_{interactions,courses,train,test}.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uniguide.config import PROJECT_ROOT, load_config  # noqa: E402

N_USERS, N_COURSES = 50, 20
TOPICS = {
    "machine learning": "neural networks python data model training",
    "business": "management marketing strategy finance leadership",
    "health": "medicine nutrition public health patients care",
    "programming": "python code software algorithms web development",
}


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    threshold = cfg["data"]["relevance_threshold"]
    topics = list(TOPICS)

    courses = pd.DataFrame(
        {
            "course_id": [f"course-{i:02d}" for i in range(N_COURSES)],
            "name": [f"{topics[i % 4].title()} {i}" for i in range(N_COURSES)],
            "institution": [f"University {i % 5}" for i in range(N_COURSES)],
            "url": [f"https://example.org/course-{i:02d}" for i in range(N_COURSES)],
        }
    )
    courses["text"] = [
        f"{topics[i % 4]} {TOPICS[topics[i % 4]]} university {i % 5}" for i in range(N_COURSES)
    ]

    rows = []
    for u in range(N_USERS):
        favorite = rng.integers(0, 4)
        n = rng.integers(6, 12)
        chosen = rng.choice(N_COURSES, size=n, replace=False)
        for t, c in enumerate(chosen):
            base = 4.6 if c % 4 == favorite else 3.8
            rating = int(np.clip(np.round(rng.normal(base, 0.7)), 1, 5))
            rows.append((u, courses.course_id[c], rating, pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(t * 20 + rng.integers(0, 10)))))
    inter = pd.DataFrame(rows, columns=["user_id", "course_id", "rating", "timestamp"])
    inter["is_relevant"] = inter["rating"] >= threshold
    courses["n_reviews"] = courses["course_id"].map(inter.groupby("course_id").size()).fillna(0).astype(int)

    # Simple per-user temporal split (toy only: the real split is built in Step 2).
    inter = inter.sort_values(["user_id", "timestamp"])
    rank = inter.groupby("user_id").cumcount()
    size = inter.groupby("user_id")["course_id"].transform("size")
    is_test = rank >= (size * (1 - cfg["split"]["test_ratio"])).round().astype(int)

    out = PROJECT_ROOT / "tests" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(out / "toy_interactions.parquet", index=False)
    courses.to_parquet(out / "toy_courses.parquet", index=False)
    inter[~is_test].to_parquet(out / "toy_train.parquet", index=False)
    inter[is_test].to_parquet(out / "toy_test.parquet", index=False)
    print(f"Toy data written to {out}: {len(inter)} interactions, "
          f"{(~is_test).sum()} train / {is_test.sum()} test")


if __name__ == "__main__":
    main()
