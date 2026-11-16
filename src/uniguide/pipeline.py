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
from uniguide.models.hybrid import HybridRecommender, RRFHybridRecommender
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


def _model_cfg(overrides: dict | None = None) -> dict:
    """``model`` section of config.yaml, optionally overridden (used by the tuning notebooks)."""
    cfg = dict(load_config()["model"])
    cfg.update(overrides or {})
    return cfg


def _tfidf_params(model_cfg: dict) -> dict:
    params = dict(model_cfg["tfidf"])
    params["ngram_range"] = tuple(params["ngram_range"])
    return params


def build_popularity(**overrides) -> PopularityRecommender:
    return PopularityRecommender(m_quantile=_model_cfg(overrides)["popularity_m_quantile"])


def build_content_model(model_cfg: dict | None = None):
    """Content model selected in config.yaml (tfidf, embeddings or fusion)."""
    model_cfg = _model_cfg() if model_cfg is None else model_cfg
    kind = model_cfg["content"]
    if kind == "tfidf":
        return ContentBasedRecommender(**_tfidf_params(model_cfg))
    embeddings = EmbeddingContentRecommender(name_weight=model_cfg["name_weight"])
    if kind == "embeddings":
        return embeddings
    if kind == "fusion":
        return ContentFusionRecommender([ContentBasedRecommender(**_tfidf_params(model_cfg)), embeddings])
    raise ValueError(f"unknown content model: {kind}")


def build_cf(**overrides) -> CollaborativeRecommender:
    """Collaborative model from config.yaml (overrides: cf_method, cf_n_neighbors, cf_n_components)."""
    model_cfg = _model_cfg(overrides)
    return CollaborativeRecommender(
        method=model_cfg["cf_method"],
        n_neighbors=model_cfg["cf_n_neighbors"],
        n_components=model_cfg["cf_n_components"],
        random_state=load_config()["seed"],
    )


def build_hybrid(rrf: bool = False, **overrides) -> HybridRecommender:
    """Unfitted hybrid built from config.yaml (overrides: alpha, alpha_schedule, cf_*...)."""
    model_cfg = _model_cfg(overrides)
    if rrf:
        return RRFHybridRecommender(build_content_model(model_cfg), build_cf(**overrides))
    return HybridRecommender(
        build_content_model(model_cfg),
        build_cf(**overrides),
        alpha=model_cfg["alpha"],
        alpha_schedule=[tuple(rule) for rule in model_cfg["alpha_schedule"]],
    )


def model_factories(**overrides) -> dict:
    """{name: callable returning a new unfitted model} for the experiments that refit models."""
    return {
        "popularity": lambda: build_popularity(**overrides),
        "content_based": lambda: build_content_model(_model_cfg(overrides)),
        "collaborative": lambda: build_cf(**overrides),
        "hybrid": lambda: build_hybrid(**overrides),
    }


def _leaf_models(model):
    subs = getattr(model, "models", None)
    if not subs:
        return [model]
    return [leaf for sub in subs for leaf in _leaf_models(sub)]


def build_system() -> System:
    """Fit popularity, content, collaborative and hybrid models, then create the explainer."""
    model_cfg = _model_cfg()
    history, test, courses = load_data()

    hybrid = build_hybrid().fit(history, courses)
    popularity = build_popularity().fit(history, courses)

    tfidf = None
    if not any(getattr(m, "vectorizer", None) is not None for m in _leaf_models(hybrid.content_model)):
        # Embedding-only hybrid: fit a TF-IDF model just for keywords and "because you liked".
        tfidf = ContentBasedRecommender(**_tfidf_params(model_cfg)).fit(history, courses)
    explainer = Explainer(hybrid, courses, history, tfidf_model=tfidf)

    return System(hybrid, popularity, explainer, courses, history, test)