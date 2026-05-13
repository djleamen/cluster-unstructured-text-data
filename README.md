# Cluster Unstructured Text Data

A small platform that **groups patterns in unstructured `.txt` data** using classic
clustering (**K-Means** or **DBSCAN**) and surfaces each cluster's top keywords and
representative example documents.

```
   .txt files ──▶ TF-IDF / embeddings ──▶ K-Means or DBSCAN
                                                 │
                                                 ▼
                              cluster keywords + exemplars
```

## Features

- **Flexible input** — load a single `.txt`, a directory of files, pasted text, or in-memory strings. Split by line, paragraph, or whole document.
- **Two vectorizers** — TF-IDF (fast, no downloads) or sentence-transformer embeddings (semantic).
- **Two clustering algorithms** — K-Means (with optional auto-K via silhouette) and DBSCAN (great for noisy data + outlier detection).
- **Cluster summaries** — each cluster gets its top TF-IDF keywords and the documents nearest to its centroid as exemplars.
- **Three ways to use it** — Python API, CLI, or a Streamlit web UI with an interactive 2D cluster map.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI

```bash
# install the package so the `cluster-text` command is on PATH
pip install -e .

cluster-text samples/reviews.txt --algorithm kmeans -k 3
cluster-text samples/reviews.txt --algorithm dbscan --eps 0.9 --min-samples 2
cluster-text samples/ --vectorizer embedding --output outputs/report.json
```

### Streamlit UI

```bash
streamlit run app.py
```

Upload one or more `.txt` files (or paste text), pick an algorithm in the sidebar, and click **Run pipeline**. You'll get:

- summary metrics (number of clusters, silhouette, noise)
- an interactive PCA scatter plot colored by cluster
- per-cluster cards with keywords, exemplars, and full member list
- a CSV download of every document with its cluster id

### Python API

```python
from cluster_text.loader import load_texts
from cluster_text.pipeline import PipelineConfig, run_pipeline

docs = load_texts([
    "The pasta was perfectly cooked.",
    "Battery life on this phone is terrible.",
    "Flight was delayed three hours.",
    # ...
])

result = run_pipeline(
    docs,
    PipelineConfig(vectorizer="tfidf", algorithm="kmeans", n_clusters=3),
)

for s in result.summaries:
    print(s.cluster_id, s.size, s.keywords[:5])
    for ex in s.exemplars:
        print("  •", ex)
```

## Configuration knobs

| Option | Where | Notes |
| --- | --- | --- |
| `vectorizer` | `tfidf` / `embedding` | Embeddings need `sentence-transformers` (in `requirements.txt`). |
| `algorithm` | `kmeans` / `dbscan` | DBSCAN labels outliers as `-1`. |
| `n_clusters` | int or `None` | `None` → silhouette-based auto-K (range configurable). |
| `dbscan_eps`, `dbscan_min_samples`, `dbscan_metric` | DBSCAN params | Default metric is `cosine`. |

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The tests run the full pipeline (TF-IDF + K-Means + DBSCAN) on a tiny built-in corpus — no network access required.

## Project layout

```
src/cluster_text/
  loader.py           # .txt loading + splitting
  vectorizer.py       # TF-IDF + sentence-transformer embeddings
  clustering.py       # K-Means, DBSCAN, auto-K via silhouette
  summarize.py        # keywords + exemplars per cluster
  pipeline.py         # end-to-end orchestration
  cli.py              # `cluster-text` command
app.py                # Streamlit UI
samples/reviews.txt   # tiny sample corpus
tests/                # pytest suite
```
