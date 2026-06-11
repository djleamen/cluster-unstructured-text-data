"""Vectorize text using TF-IDF or sentence-transformer embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

VectorizerKind = Literal["tfidf", "embedding"]


@dataclass
class VectorizationResult:
    matrix: np.ndarray
    kind: VectorizerKind
    feature_names: list[str] | None = None  # only meaningful for tfidf
    vectorizer: object | None = None         # underlying object (for reuse)


def vectorize(
    texts: Sequence[str],
    kind: VectorizerKind = "tfidf",
    *,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> VectorizationResult:
    """Convert texts into a 2D numeric matrix.

    - ``tfidf``: sparse TF-IDF (returned as dense for downstream simplicity
      since corpora here are typically modest in size).
    - ``embedding``: dense sentence-transformer embeddings.
    """
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Clamp min_df for tiny corpora so sklearn does not reject the input
        # (e.g. min_df=2 with a single document).
        if isinstance(min_df, int):
            min_df = max(1, min(min_df, len(texts)))

        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            lowercase=True,
        )
        try:
            X = vec.fit_transform(texts)
        except ValueError as exc:
            if min_df == 1 or "no terms remain" not in str(exc):
                raise
            # min_df pruned every term (no term repeats); retry keeping all.
            vec = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=1,
                stop_words="english",
                lowercase=True,
            )
            X = vec.fit_transform(texts)
        return VectorizationResult(
            matrix=X.toarray(),
            kind="tfidf",
            feature_names=list(vec.get_feature_names_out()),
            vectorizer=vec,
        )

    if kind == "embedding":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model)
        emb = model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return VectorizationResult(
            matrix=np.asarray(emb),
            kind="embedding",
            feature_names=None,
            vectorizer=model,
        )

    raise ValueError(f"Unknown vectorizer kind: {kind!r}")
