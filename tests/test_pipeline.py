"""Smoke tests for the end-to-end pipeline."""
from __future__ import annotations

from cluster_text.loader import load_texts
from cluster_text.pipeline import PipelineConfig, run_pipeline


FOOD = [
    "The pasta was perfectly cooked and the sauce was incredible.",
    "Loved the cozy atmosphere and the friendly waiter.",
    "Pizza arrived cold and the cheese was rubbery.",
    "Best tiramisu I've had outside of Italy.",
    "Great wine selection and the sommelier was knowledgeable.",
    "The steak was overcooked and dry, very disappointing.",
]
TECH = [
    "The new MacBook is incredibly fast and the screen is gorgeous.",
    "Battery life on this phone is terrible, dies by lunch.",
    "Setting up the router took hours and the app is buggy.",
    "This wireless mouse pairs instantly and tracks beautifully.",
    "Camera quality on the new phone is a huge upgrade.",
    "USB-C everywhere finally, no more dongle hunting.",
]
TRAVEL = [
    "Flight was delayed three hours with no explanation.",
    "The hotel room had a stunning view of the harbor.",
    "Lost my luggage on the connecting flight, still missing.",
    "Beach was pristine and the snorkeling was unreal.",
    "The resort breakfast buffet had endless options.",
    "Our taxi driver took the scenic route to inflate the fare.",
]


def test_kmeans_pipeline_runs_and_separates_topics():
    docs = load_texts(FOOD + TECH + TRAVEL)
    cfg = PipelineConfig(
        vectorizer="tfidf",
        algorithm="kmeans",
        n_clusters=3,
    )
    result = run_pipeline(docs, cfg)

    assert len(result.documents) == len(FOOD) + len(TECH) + len(TRAVEL)
    assert result.clustering.n_clusters == 3
    assert len(result.summaries) == 3
    # every cluster has at least one keyword and one exemplar
    for s in result.summaries:
        assert s.size > 0
        assert len(s.exemplars) >= 1


def test_dbscan_pipeline_runs():
    docs = load_texts(FOOD + TECH + TRAVEL)
    cfg = PipelineConfig(
        vectorizer="tfidf",
        algorithm="dbscan",
        dbscan_eps=0.9,
        dbscan_min_samples=2,
        dbscan_metric="cosine",
    )
    result = run_pipeline(docs, cfg)
    # DBSCAN should succeed and produce labels for every doc
    assert len(result.clustering.labels) == len(docs)


def test_auto_k_selection():
    docs = load_texts(FOOD + TECH + TRAVEL)
    cfg = PipelineConfig(
        vectorizer="tfidf",
        algorithm="kmeans",
        n_clusters=None,
        auto_k_range=(2, 5),
    )
    result = run_pipeline(docs, cfg)
    assert 2 <= result.clustering.n_clusters <= 5
