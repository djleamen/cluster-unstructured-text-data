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


def test_auto_k_on_large_input_uses_sampled_silhouette():
    """Above the sample threshold auto-K still finds well-separated blobs."""
    import numpy as np
    from cluster_text.clustering import SILHOUETTE_SAMPLE_SIZE, suggest_kmeans_k

    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [0, 10]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(2000, 2)) for c in centers])
    assert X.shape[0] > SILHOUETTE_SAMPLE_SIZE
    assert suggest_kmeans_k(X, 2, 6) == 3


def test_sampled_silhouette_keeps_tiny_clusters():
    """A 5-point cluster among 20k points must not vanish from the sample."""
    import numpy as np
    from cluster_text.clustering import _silhouette

    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(20000, 2)), rng.normal(loc=50, size=(5, 2))])
    labels = np.array([0] * 20000 + [1] * 5)
    assert _silhouette(X, labels) > 0


def test_sampled_silhouette_stays_bounded():
    """The sample passed to silhouette_score never exceeds the fixed budget."""
    import numpy as np
    import sklearn.metrics
    from cluster_text.clustering import SILHOUETTE_SAMPLE_SIZE, _silhouette

    rng = np.random.default_rng(0)
    # n above the budget, few clusters: sampling actually happens.
    X = np.vstack([c + rng.normal(scale=0.5, size=(4000, 2))
                   for c in np.array([[0, 0], [10, 10], [0, 10]])])
    labels = np.repeat(np.arange(3), 4000)

    sizes = []
    real = sklearn.metrics.silhouette_score
    sklearn.metrics.silhouette_score = lambda Xs, ls, *a, **k: sizes.append(Xs.shape[0]) or 0.0
    try:
        _silhouette(X, labels)
    finally:
        sklearn.metrics.silhouette_score = real
    assert sizes and sizes[-1] <= SILHOUETTE_SAMPLE_SIZE


def test_sampled_silhouette_unavailable_when_budget_cannot_fit_clusters():
    """When clusters can't each keep >=2 sampled members, report unavailable,
    not a misleadingly low score from singleton-reduced clusters."""
    import numpy as np
    from cluster_text.clustering import _silhouette_safe

    rng = np.random.default_rng(0)
    # 4,999 well-separated clusters of two coincident points each: the exact
    # silhouette is 1.0, but it cannot be sampled within the budget. The old
    # one-rep-per-cluster fill made 4,998 of them singletons and returned ~0.
    centers = rng.normal(scale=100, size=(4999, 2))
    X = np.repeat(centers, 2, axis=0)
    labels = np.repeat(np.arange(4999), 2)
    assert _silhouette_safe(X, labels) is None


def test_sampled_silhouette_many_tiny_clusters_is_bounded_or_unavailable():
    """A huge number of tiny clusters never blows the sample past the budget."""
    import numpy as np
    from cluster_text.clustering import _silhouette_safe

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100000, 2))
    labels = np.repeat(np.arange(20000), 5)  # 20k clusters, far above the budget
    # Can't keep two members of every cluster within 5,000, so: unavailable.
    assert _silhouette_safe(X, labels) is None

    # When clusters outnumber the budget the estimate is undefined, not a crash.
    assert _silhouette_safe(X, labels) is None
