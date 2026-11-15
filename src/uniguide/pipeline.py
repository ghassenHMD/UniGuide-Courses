"""Assemble the final system (data + fitted models + explainer), shared by notebooks and app."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from uniguide.config import get_path, load_config
from uniguide.explain import Explainer
from uniguide.models.collaborative import CollaborativeRecommender
from uniguide.models.content_based import ContentBasedRecommender
from uniguide.models.embeddings import ContentFusionRecommender, EmbeddingContentRecommender
from uniguide.models.hybrid import HybridRecommender
from uniguide.models.popularity import PopularityRecommender


@dataclass
class System:
    """Everything the demo needs, fitted on train + validation (never on test)."""

    hybrid: HybridRecommender
    popularity: PopularityRecommender
    explainer: Explainer
    courses: pd.DataFrame          # warm catalogue (course_id, name, institution, url, text...)
    history: pd.DataFrame          # train + validation interactions
    test: pd.DataFrame             # held-out interactions (used only to display a sanity check)

    @property
    def content_model(self):
        return self.hybrid.content_model

    @property
    def collaborative_model(self):
        return self.hybrid.collaborative_model


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train + val, test, warm courses) from data/processed/."""
    processed = get_path("processed")
    train, val, test = (pd.read_parquet(processed / f"{n}.parquet") for n in ("train", "val", "test"))
    courses = pd.read_parquet(processed / "courses.parquet")
    cold = set(json.loads((processed / "cold_courses.json").read_text()))
    warm = courses[~courses["course_id"].isin(cold)].reset_index(drop=True)
    return pd.concat([train, val], ignore_index=True), test, warm


def build_content_model(model_cfg: dict):
    """Content model selected in config.yaml (tfidf, embeddings or fusion)."""
    tfidf = dict(model_cfg["tfidf"])
    tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    kind = model_cfg["content"]
    if kind == "tfidf":
        return ContentBasedRecommender(**tfidf)
    embeddings = EmbeddingContentRecommender(name_weight=model_cfg["name_weight"])
    if kind == "embeddings":
        return embeddings
    if kind == "fusion":
        return ContentFusionRecommender([ContentBasedRecommender(**tfidf), embeddings])
    raise ValueError(f"unknown content model: {kind}")


def _leaf_models(model):
    subs = getattr(model, "models", None)
    if not subs:
        return [model]
    return [leaf for sub in subs for leaf in _leaf_models(sub)]


def build_system() -> System:
    """Fit popularity, content, collaborative and hybrid models, then create the explainer."""
    cfg = load_config()
    model_cfg = cfg["model"]
    history, test, courses = load_data()

    hybrid = HybridRecommender(
        build_content_model(model_cfg),
        CollaborativeRecommender(
            method=model_cfg["cf_method"],
            n_neighbors=model_cfg["cf_n_neighbors"],
            n_components=model_cfg["cf_n_components"],
            random_state=cfg["seed"],
        ),
        alpha=model_cfg["alpha"],
        alpha_schedule=[tuple(rule) for rule in model_cfg["alpha_schedule"]],
    ).fit(history, courses)

    popularity = PopularityRecommender(m_quantile=model_cfg["popularity_m_quantile"]).fit(history, courses)

    tfidf = None
    if not any(getattr(m, "vectorizer", None) is not None for m in _leaf_models(hybrid.content_model)):
        # Embedding-only hybrid: fit a TF-IDF model just for keywords and "because you liked".
        tfidf_cfg = dict(model_cfg["tfidf"])
        tfidf_cfg["ngram_range"] = tuple(tfidf_cfg["ngram_range"])
        tfidf = ContentBasedRecommender(**tfidf_cfg).fit(history, courses)
    explainer = Explainer(hybrid, courses, history, tfidf_model=tfidf)

    return System(hybrid, popularity, explainer, courses, history, test)