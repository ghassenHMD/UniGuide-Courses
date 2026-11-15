"""Dataset statistics and plots (Step 1)."""
import matplotlib.pyplot as plt
import pandas as pd

from uniguide.config import get_path


def snapshot(df: pd.DataFrame, step: str, user_col: str = "user_id") -> dict:
    """Record the size of the data after a processing step (for the filtering log)."""
    return {
        "step": step,
        "n_rows": len(df),
        "n_users": df[user_col].nunique(),
        "n_courses": df["course_id"].nunique(),
    }


def basic_stats(inter: pd.DataFrame) -> dict:
    """Users, courses, interactions, sparsity, long-tail share, share of relevant ratings."""
    n_users = inter["user_id"].nunique()
    n_courses = inter["course_id"].nunique()
    n_inter = len(inter)
    counts = inter.groupby("course_id").size().sort_values(ascending=False)
    top_n = max(1, int(0.1 * len(counts)))
    return {
        "n_users": n_users,
        "n_courses": n_courses,
        "n_interactions": n_inter,
        "sparsity": 1 - n_inter / (n_users * n_courses),
        "avg_interactions_per_user": n_inter / n_users,
        "avg_interactions_per_course": n_inter / n_courses,
        "top10pct_courses_share": counts.head(top_n).sum() / counts.sum(),
        "share_relevant": float(inter["is_relevant"].mean()),
    }


def plot_popularity(inter: pd.DataFrame):
    """Histogram of course popularity and long-tail curve (log-log)."""
    counts = inter.groupby("course_id").size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(counts, bins=40)
    axes[0].set(title="Course popularity", xlabel="interactions per course", ylabel="courses")
    axes[1].plot(range(1, len(counts) + 1), counts.values)
    axes[1].set(title="Long tail", xlabel="course rank", ylabel="interactions",
                xscale="log", yscale="log")
    return fig


def plot_rating_distribution(inter: pd.DataFrame):
    """Bar plot of ratings 1 to 5."""
    fig, ax = plt.subplots(figsize=(5, 4))
    inter["rating"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set(title="Rating distribution", xlabel="rating", ylabel="interactions")
    return fig


def plot_reviews_per_user(inter: pd.DataFrame):
    """Histogram of the number of interactions per user."""
    counts = inter.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=40)
    ax.set(title="Interactions per user", xlabel="interactions", ylabel="users", yscale="log")
    return fig


def plot_time(inter: pd.DataFrame):
    """Number of interactions per month."""
    monthly = inter["timestamp"].dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(ax=ax)
    ax.set(title="Interactions over time", xlabel="month", ylabel="interactions")
    return fig


def save_fig(fig, name: str) -> None:
    """Save a figure to results/figures/<name>.png."""
    out = get_path("results") / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")