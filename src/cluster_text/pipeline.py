"""End-to-end pipeline: load -> vectorize -> cluster -> summarize."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .clustering import (
    ClusterResult,
    run_dbscan,
    run_kmeans,
    suggest_kmeans_k,
)
from .loader import TextDocument
from .summarize import ClusterSummary, summarize_clusters
from .vectorizer import VectorizationResult, VectorizerKind, vectorize


@dataclass
class PipelineConfig:
    """Configuration for the clustering pipeline."""
    vectorizer: VectorizerKind = "tfidf"
    algorithm: str = "kmeans"          # "kmeans" or "dbscan"
    n_clusters: int | None = None      # None => auto-suggest for kmeans
    auto_k_range: tuple[int, int] = (2, 10)
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    dbscan_metric: str = "cosine"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_keywords: int = 10
    n_exemplars: int = 3


@dataclass
class PipelineResult:
    """Result of the clustering pipeline."""
    documents: list[TextDocument]
    vectorization: VectorizationResult
    clustering: ClusterResult
    summaries: list[ClusterSummary]
    config: PipelineConfig = field(default_factory=PipelineConfig)


def run_pipeline(
    documents: Sequence[TextDocument],
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full clustering pipeline on the given documents."""
    if not documents:
        raise ValueError("No documents to cluster.")
    cfg = config or PipelineConfig()
    texts = [d.text for d in documents]
    ids = [d.id for d in documents]

    vec = vectorize(
        texts,
        kind=cfg.vectorizer,
        embedding_model=cfg.embedding_model,
    )
    X: np.ndarray = vec.matrix

    if cfg.algorithm == "kmeans":
        if cfg.n_clusters is not None and cfg.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        k = cfg.n_clusters if cfg.n_clusters is not None else suggest_kmeans_k(
            X, *cfg.auto_k_range)
        # K-Means cannot form more clusters than there are samples.
        k = min(k, X.shape[0])
        clustering = run_kmeans(X, n_clusters=k)
    elif cfg.algorithm == "dbscan":
        clustering = run_dbscan(
            X,
            eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples,
            metric=cfg.dbscan_metric,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm!r}")

    summaries = summarize_clusters(
        texts, ids, vec, clustering,
        top_keywords=cfg.top_keywords,
        n_exemplars=cfg.n_exemplars,
    )
    return PipelineResult(
        documents=list(documents),
        vectorization=vec,
        clustering=clustering,
        summaries=summaries,
        config=cfg,
    )
