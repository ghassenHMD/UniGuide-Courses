"""UniGuide Courses: Streamlit demo (Step 7).

Run from the repository root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make `uniguide` importable even when the package is not pip-installed (e.g. Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uniguide.config import get_path, load_config
from uniguide.pipeline import System, build_system

st.set_page_config(page_title="UniGuide Courses", page_icon="🎓", layout="wide")
CFG = load_config()


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading data and fitting the models (first run only)...")
def get_system() -> System:
    return build_system()


@st.cache_data(show_spinner=False)
def get_relevant_test(_system: System) -> dict:
    """Held-out relevant courses per user (only used for the optional sanity check)."""
    relevant = _system.test[_system.test["is_relevant"]]
    return relevant.groupby("user_id")["course_id"].apply(set).to_dict()


@st.cache_data(show_spinner=False)
def get_suggested_users(_system: System, n: int = 30) -> list:
    """Users with a medium history and at least one relevant held-out course (demo candidates)."""
    sizes = _system.history.groupby("user_id").size()
    has_test = set(get_relevant_test(_system))
    pool = sizes[(sizes >= 8) & (sizes <= 30) & sizes.index.isin(has_test)].index.to_numpy()
    pool = np.sort(pool)
    rng = np.random.default_rng(CFG["seed"])
    chosen = rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else []
    return sorted(int(u) for u in chosen)


try:
    system = get_system()
except FileNotFoundError as err:
    st.error(
        f"Missing data file: {err.filename}\n\n"
        "Build the data first:\n\n`python -m uniguide.data.build` then `python -m uniguide.data.splits`"
    )
    st.stop()

names = system.courses.set_index("course_id")["name"].to_dict()
labels = {
    row.course_id: f"{row.name} ({row.institution})"
    for row in system.courses.itertuples(index=False)
}
relevant_test = get_relevant_test(system)


# ----------------------------------------------------------------------------- helpers
def with_names(recs: pd.DataFrame, held_out: set | None = None) -> pd.DataFrame:
    """Add rank, course name and (optionally) the held-out hit marker to a recommendation list."""
    out = recs.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out["course"] = out["course_id"].map(lambda c: labels.get(c, c))
    out["score"] = out["score"].round(3)
    cols = ["rank", "course", "score"]
    if held_out is not None:
        out["held-out hit"] = out["course_id"].map(lambda c: "✅" if c in held_out else "")
        cols.append("held-out hit")
    return out[cols]


def render_explanation(e, held_out: set | None = None) -> None:
    """One recommendation card: score, component contributions, reasons and keywords."""
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{e.rank}. {e.name}**  \n{e.institution}")
            if e.url:
                st.markdown(f"[Open the course page]({e.url})")
            if held_out is not None and e.course_id in held_out:
                st.success("Rated 4+ by this user in the held-out test set")
        with right:
            st.metric("Score", f"{e.score:.2f}")
            if e.mode == "popularity":
                st.progress(min(max(e.pop_contribution, 0.0), 1.0), text=f"Popularity {e.pop_contribution:.2f}")
            else:
                st.progress(min(max(e.cb_contribution, 0.0), 1.0), text=f"Content {e.cb_contribution:.2f}")
                st.progress(min(max(e.cf_contribution, 0.0), 1.0), text=f"Collaborative {e.cf_contribution:.2f}")
        st.write(e.text)
        if e.because:
            st.markdown("**Because you liked:**")
            for b in e.because:
                origin = "rated alike by other learners" if b["source"] == "collaborative" else "similar content"
                st.markdown(f"- {b['name']} (your rating: {b['rating']:.0f}/5, {origin}, similarity {b['similarity']:.2f})")
        if e.keywords:
            st.caption("Top TF-IDF keywords: " + ", ".join(e.keywords))


# ----------------------------------------------------------------------------- sidebar
st.title("🎓 UniGuide Courses")
st.caption("Hybrid recommender (content-based + collaborative filtering) on Coursera course reviews")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of recommendations (K)", 3, 20, int(CFG["app"]["default_k"]))

    suggested = get_suggested_users(system)
    demo_user = CFG["app"]["demo_user_id"]
    if demo_user is not None and int(demo_user) not in suggested:
        suggested = [int(demo_user)] + suggested
    default_index = suggested.index(int(demo_user)) if demo_user is not None else 0
    user_id = st.selectbox("Demo user (pseudo-identifier)", suggested, index=default_index if suggested else None)
    typed = st.number_input("...or type any user id (0 = keep the selection above)", min_value=0, value=0, step=1)
    if typed:
        user_id = int(typed)
    show_check = st.toggle("Show held-out check", value=True,
                           help="Marks recommended courses this user really rated 4+ AFTER the training period.")

    st.divider()
    st.subheader("Model")
    mcfg = CFG["model"]
    st.markdown(
        f"- Content: `{mcfg['content']}`\n- Collaborative: `{mcfg['cf_method']}`\n"
        f"- alpha (long history): `{mcfg['alpha']}`\n- switching: `{mcfg['alpha_schedule']}`"
    )
    st.caption(
        f"{system.history['user_id'].nunique():,} users · {len(system.courses):,} courses · "
        f"{len(system.history):,} training interactions"
    )

if user_id is None:
    st.warning("No suggested user found. Type a user id in the sidebar.")
    st.stop()

user_id = int(user_id)
history = system.explainer.history_of(user_id)
held_out = relevant_test.get(user_id, set()) if show_check else None

tab_rec, tab_cmp, tab_new, tab_res = st.tabs(
    ["Recommendations", "Compare methods", "New user (cold-start)", "Results"]
)

# ----------------------------------------------------------------------------- tab 1
with tab_rec:
    st.subheader(f"User {user_id}")
    n_hist = len(history)
    alpha = system.hybrid.alpha_for(n_hist)
    if n_hist == 0:
        st.info("No history for this user: the hybrid falls back to popularity.")
    else:
        st.caption(f"{n_hist} training interactions → content weight alpha = {alpha:.2f}, "
                   f"collaborative weight = {1 - alpha:.2f}")
        with st.expander("Courses already rated by this user"):
            hist_df = pd.DataFrame(
                {"course": [labels.get(c, c) for c in history], "rating": list(history.values())}
            ).sort_values("rating", ascending=False)
            st.dataframe(hist_df, hide_index=True)

    for explanation in system.explainer.explain_user(user_id, k=k):
        render_explanation(explanation, held_out)

# ----------------------------------------------------------------------------- tab 2
with tab_cmp:
    st.subheader(f"Top-{k} of each method for user {user_id}")
    methods = {
        "Popularity": system.popularity,
        "Content-based": system.content_model,
        "Collaborative": system.collaborative_model,
        "Hybrid": system.hybrid,
    }
    columns = st.columns(len(methods))
    for column, (label, model) in zip(columns, methods.items()):
        with column:
            st.markdown(f"**{label}**")
            recs = model.recommend(user_id, k=k, exclude_seen=True)
            st.dataframe(with_names(recs, held_out), hide_index=True)

# ----------------------------------------------------------------------------- tab 3
with tab_new:
    st.subheader("Simulate a new user")
    st.write(
        "Pick 0, 1, 3 or 5 courses you liked and watch the hybrid switch strategy: popularity with "
        "no history, mostly content with a short history, more collaborative with a longer one."
    )
    chosen = st.multiselect(
        "Courses you liked", options=list(labels), format_func=lambda c: labels[c], max_selections=10
    )
    ratings = {}
    if chosen:
        cols = st.columns(min(len(chosen), 5))
        for i, course_id in enumerate(chosen):
            with cols[i % len(cols)]:
                ratings[course_id] = st.slider(names.get(course_id, course_id)[:40], 1, 5, 5, key=f"r_{course_id}")

    guest_alpha = system.hybrid.alpha_for(len(ratings))
    if not ratings:
        st.info("0 interactions → popularity fallback.")
    else:
        st.caption(f"{len(ratings)} interactions → content weight alpha = {guest_alpha:.2f}, "
                   f"collaborative weight = {1 - guest_alpha:.2f}")
    for explanation in system.explainer.explain_ratings(ratings, k=k):
        render_explanation(explanation)

# ----------------------------------------------------------------------------- tab 4
with tab_res:
    st.subheader("Evaluation results")
    tables, figures = get_path("results") / "tables", get_path("results") / "figures"
    shown = False
    path = tables / "comparison_k10.csv"
    if path.exists():
        st.markdown("**Main comparison (K = 10)**")
        st.dataframe(pd.read_csv(path), hide_index=True)
        shown = True
    for path in sorted(figures.glob("*cold*.png")) + sorted(figures.glob("accuracy_vs_novelty.png")):
        st.image(str(path), caption=path.stem.replace("_", " "))
        shown = True
    if not shown:
        st.info("No precomputed results found in results/. Run notebooks 07 and 08 first.")