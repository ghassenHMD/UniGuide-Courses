"""Content-based recommenders using Sentence Transformers embeddings (Step 5)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from uniguide.config import get_path
from uniguide.models.base import BaseRecommender
from uniguide.models.content_based import ContentBasedRecommender


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class EmbeddingContentRecommender(ContentBasedRecommender):
    """Same idea as the TF-IDF content-based model, but course vectors are sentence embeddings.

    Course vector = normalise( name_weight * embedding(name, institution)
                               + (1 - name_weight) * mean(embedding(text chunks)) )

    * The text is cut into chunks of ``chunk_words`` words because transformer models only read
      a limited number of tokens (about 256 for MiniLM); the chunk embeddings are averaged.
    * The course name is embedded separately because it carries most of the topic information,
      whereas reviews are often generic ("great course, clear explanations").
    * Embeddings only depend on the course texts, not on the ratings, so they are cached on disk
      and computed once, however many times the model is refitted (tuning, cold-start runs...).

    User profiles, cosine scoring and the popularity fallback are inherited unchanged.
    """

    name = "content_embeddings"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        encoder=None,
        chunk_words: int = 100,
        max_chunks: int = 15,
        name_weight: float = 0.5,
        liked_threshold: float | None = None,
        rating_offset: float = 0.0,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(liked_threshold=liked_threshold, rating_offset=rating_offset)
        if not 0.0 <= name_weight <= 1.0:
            raise ValueError("name_weight must be between 0 and 1")
        if chunk_words < 1 or max_chunks < 1:
            raise ValueError("chunk_words and max_chunks must be >= 1")
        self.vectorizer = None  # no TF-IDF here
        self.model_name = model_name
        self.encoder = encoder  # any object with .encode(list[str]); loaded lazily if None
        self.chunk_words = chunk_words
        self.max_chunks = max_chunks
        self.name_weight = name_weight
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    # ------------------------------------------------------------- encoding
    def _encode(self, sentences: list[str]) -> np.ndarray:
        """Encode sentences into unit-norm vectors with the (lazily loaded) model."""
        encoder = self.encoder
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError("pip install sentence-transformers") from err
            encoder = SentenceTransformer(self.model_name)
        out = encoder.encode(
            sentences, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return _normalize_rows(np.asarray(out, dtype=np.float32))

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        parts = [
            " ".join(words[i:i + self.chunk_words])
            for i in range(0, len(words), self.chunk_words)
        ][: self.max_chunks]
        return parts or [""]

    def _cache_path(self, texts: list[str], heads: list[str] | None) -> Path:
        key = hashlib.sha1()
        key.update(f"{self.model_name}|{self.chunk_words}|{self.max_chunks}".encode())
        for t in texts + (heads or []):
            key.update(t.encode("utf-8", errors="ignore"))
            key.update(b"\x00")
        folder = self.cache_dir if self.cache_dir is not None else get_path("interim")
        return folder / f"embeddings_{key.hexdigest()[:16]}.npz"

    def _encode_courses(self, texts: list[str], heads: list[str] | None):
        """Unit-norm (body, head) embeddings per course; `head` is None without a name column.

        body = mean embedding of the text chunks; head = embedding of "name, institution".
        This is the slow part, so it is cached (it does not depend on name_weight).
        """
        path = self._cache_path(texts, heads)
        if self.use_cache and path.exists():
            stored = np.load(path)
            return stored["body"], (stored["head"] if heads is not None else None)

        chunks, owner = [], []
        for i, text in enumerate(texts):
            parts = self._chunks(text)
            chunks.extend(parts)
            owner.extend([i] * len(parts))
        chunk_vectors = self._encode(chunks)
        body = np.zeros((len(texts), chunk_vectors.shape[1]), dtype=np.float32)
        np.add.at(body, np.asarray(owner), chunk_vectors)   # sum the chunks of each course
        body = _normalize_rows(body)
        head = self._encode(heads) if heads is not None else None

        if self.use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, body=body, head=head if head is not None else np.zeros(0))
        return body, head

    def _build_item_vectors(self, courses: pd.DataFrame) -> np.ndarray:
        """Dense (n_items, dim) matrix of unit-norm course embeddings, same order as `courses`."""
        texts = courses["text"].fillna("").astype(str).tolist()
        heads = None
        if "name" in courses.columns:
            head = courses["name"].fillna("").astype(str)
            if "institution" in courses.columns:
                head = head + ", " + courses["institution"].fillna("").astype(str)
            heads = head.tolist()

        body, head = self._encode_courses(texts, heads)
        if head is None:
            return body
        return _normalize_rows(self.name_weight * head + (1 - self.name_weight) * body)

    # -------------------------------------------------------------- scoring
    def _cosine_scores(self, idx: np.ndarray, weights: np.ndarray):
        """Dense version of the parent's method: cosine between profile and every course."""
        weights = weights / weights.sum()
        profile = weights @ self.item_vectors[idx]
        norm = np.linalg.norm(profile)
        if norm == 0:
            return None
        return (self.item_vectors @ profile) / norm


class ContentFusionRecommender(BaseRecommender):
    """Combine several content models (e.g. TF-IDF + embeddings) with a weighted sum.

    Each model's scores are min-max normalised over the candidate courses (like the hybrid),
    then averaged with the given weights. A model that cannot score a course (NaN) is ignored
    for that course and the weights are renormalised. Can be used as the `content_model`
    of the hybrid recommender.
    """

    name = "content_fusion"

    def __init__(self, models: list[BaseRecommender], weights: list[float] | None = None,
                 name: str | None = None) -> None:
        super().__init__()
        if not models:
            raise ValueError("models cannot be empty")
        weights = [1.0] * len(models) if weights is None else list(weights)
        if len(weights) != len(models) or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("weights must be non-negative, sum > 0, one per model")
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        if name is not None:
            self.name = name

    def fit(self, train: pd.DataFrame, courses: pd.DataFrame) -> "ContentFusionRecommender":
        self._register_data(train, courses)
        for model in self.models:
            model.fit(train, courses)
        return self

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        out = np.full(scores.shape, np.nan)
        valid = np.isfinite(scores)
        if valid.any():
            low, high = scores[valid].min(), scores[valid].max()
            out[valid] = 0.0 if high == low else (scores[valid] - low) / (high - low)
        return out

    def score(self, user_id: int, item_ids: list[str]) -> np.ndarray:
        item_ids = list(item_ids)
        if not item_ids:
            return np.array([], dtype=float)
        parts = np.vstack(
            [self._minmax(np.asarray(m.score(user_id, item_ids), dtype=float)) for m in self.models]
        )
        available = np.isfinite(parts)
        w = self.weights[:, None] * available
        total = w.sum(axis=0)
        mixed = (np.where(available, parts, 0.0) * w).sum(axis=0) / np.where(total > 0, total, 1.0)
        return np.where(total > 0, mixed, np.nan)