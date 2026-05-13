"""K-Means and DBSCAN clustering with quality metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Algorithm = Literal["kmeans", "dbscan"]


@dataclass
class ClusterResult:
    """Result of clustering, including labels and quality metrics."""
    algorithm: Algorithm
    labels: np.ndarray                  # shape (n_samples,); -1 = noise (DBSCAN)
    n_clusters: int
    n_noise: int = 0
    silhouette: float | None = None
    params: dict = field(default_factory=dict)
    model: object | None = None


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score, returning None when it is not defined."""
    from sklearn.metrics import silhouette_score

    mask = labels != -1
    if mask.sum() < 2:
        return None
    unique = np.unique(labels[mask])
    if len(unique) < 2 or len(unique) >= mask.sum():
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 8,
    *,
    random_state: int = 42,
    n_init: int | str = "auto",
) -> ClusterResult:
    """Run K-Means clustering on the given data."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)
    return ClusterResult(
        algorithm="kmeans",
        labels=labels,
        n_clusters=int(len(set(labels))),
        n_noise=0,
        silhouette=_silhouette_safe(X, labels),
        params={"n_clusters": n_clusters, "random_state": random_state},
        model=model,
    )


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    *,
    metric: str = "cosine",
) -> ClusterResult:
    """Run DBSCAN clustering on the given data."""
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    return ClusterResult(
        algorithm="dbscan",
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        silhouette=_silhouette_safe(X, labels),
        params={"eps": eps, "min_samples": min_samples, "metric": metric},
        model=model,
    )


def suggest_kmeans_k(
    X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> int:
    """Pick the K with the highest silhouette score in the given range."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_score = k_min, -1.0
    n = X.shape[0]
    k_max = min(k_max, max(k_min, n - 1))
    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k
