"""Summarize each cluster: keywords, exemplar texts, basic stats."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .clustering import ClusterResult
from .vectorizer import VectorizationResult


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    keywords: list[str] = field(default_factory=list)
    exemplars: list[str] = field(default_factory=list)  # representative texts
    member_ids: list[str] = field(default_factory=list)


def _top_tfidf_keywords(
    X: np.ndarray, feature_names: list[str], member_idx: np.ndarray, top_k: int
) -> list[str]:
    mean_vec = X[member_idx].mean(axis=0)
    if mean_vec.ndim > 1:
        mean_vec = np.asarray(mean_vec).ravel()
    top = np.argsort(mean_vec)[::-1][:top_k]
    return [feature_names[i] for i in top if mean_vec[i] > 0]


def _embedding_keywords(
    texts: Sequence[str], member_idx: np.ndarray, top_k: int
) -> list[str]:
    """Fallback: simple stopword-filtered token frequency."""
    import re

    stop = _BASIC_STOPWORDS
    counter: Counter[str] = Counter()
    token_re = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
    for i in member_idx:
        for tok in token_re.findall(texts[i].lower()):
            if tok not in stop:
                counter[tok] += 1
    return [w for w, _ in counter.most_common(top_k)]


def _pick_exemplars(
    X: np.ndarray, member_idx: np.ndarray, texts: Sequence[str], n: int
) -> tuple[list[int], list[str]]:
    """Pick the n points closest to the cluster centroid."""
    if len(member_idx) == 0:
        return [], []
    centroid = X[member_idx].mean(axis=0)
    diffs = X[member_idx] - centroid
    dists = np.linalg.norm(diffs, axis=1)
    order = np.argsort(dists)[:n]
    picked = member_idx[order]
    return picked.tolist(), [texts[i] for i in picked]


def summarize_clusters(
    texts: Sequence[str],
    doc_ids: Sequence[str],
    vec: VectorizationResult,
    result: ClusterResult,
    *,
    top_keywords: int = 10,
    n_exemplars: int = 3,
) -> list[ClusterSummary]:
    labels = result.labels
    summaries: list[ClusterSummary] = []
    unique = sorted(set(labels.tolist()))
    for cid in unique:
        idx = np.where(labels == cid)[0]
        if vec.kind == "tfidf" and vec.feature_names is not None:
            kws = _top_tfidf_keywords(vec.matrix, vec.feature_names, idx, top_keywords)
        else:
            kws = _embedding_keywords(texts, idx, top_keywords)
        _, ex_texts = _pick_exemplars(vec.matrix, idx, texts, n_exemplars)
        summaries.append(
            ClusterSummary(
                cluster_id=int(cid),
                size=int(len(idx)),
                keywords=kws,
                exemplars=ex_texts,
                member_ids=[doc_ids[i] for i in idx],
            )
        )
    return summaries


# A tiny built-in stopword list used only when sklearn is not driving the
# vectorization (embedding mode). Kept intentionally small.
_BASIC_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "have", "has", "had", "are",
    "was", "were", "but", "not", "you", "your", "from", "they", "them", "their",
    "our", "ours", "his", "her", "him", "she", "its", "into", "than", "then",
    "there", "here", "what", "which", "when", "where", "while", "would", "could",
    "should", "about", "also", "been", "being", "very", "just", "like", "over",
    "some", "such", "only", "more", "most", "other", "any", "all", "can", "will",
    "one", "two", "three", "get", "got", "out", "off", "use", "used", "using",
    "via", "per", "etc",
}
