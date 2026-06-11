"""Streamlit UI for the text clustering platform.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Make the local src/ importable when running `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cluster_text.loader import TextDocument, load_texts  # noqa: E402
from cluster_text.pipeline import PipelineConfig, run_pipeline  # noqa: E402


st.set_page_config(page_title="Text Clustering", layout="wide")
st.title("📚 Unstructured Text Clustering")
st.caption(
    "Group patterns in raw text with K-Means or DBSCAN, then explore each "
    "cluster's keywords and representative examples."
)

# --------------------------------------------------------------------------- #
# Sidebar configuration                                                       #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")
    vectorizer = st.selectbox("Vectorizer", ["tfidf", "embedding"], index=0)
    algorithm = st.selectbox("Clustering algorithm", ["kmeans", "dbscan"], index=0)

    if algorithm == "kmeans":
        auto_k = st.checkbox("Auto-select K (silhouette)", value=True)
        k_value = st.slider("K", 2, 20, 6, disabled=auto_k)
    else:
        eps = st.slider("DBSCAN eps", 0.05, 2.0, 0.5, 0.05)
        min_samples = st.slider("DBSCAN min_samples", 2, 50, 5)

    embedding_model = "all-MiniLM-L6-v2"
    if vectorizer == "embedding":
        embedding_model = st.text_input("Embedding model", "all-MiniLM-L6-v2")

    split_mode = st.selectbox("How to split files", ["line", "paragraph", "whole"], index=0)

# --------------------------------------------------------------------------- #
# Input                                                                       #
# --------------------------------------------------------------------------- #
st.subheader("1. Provide your text")
tab_upload, tab_paste = st.tabs(["📁 Upload .txt files", "📝 Paste text"])

documents: list[TextDocument] = []

with tab_upload:
    uploads = st.file_uploader(
        "Upload one or more .txt files",
        type=["txt"],
        accept_multiple_files=True,
    )
    if uploads:
        from cluster_text.loader import _split_into_records  # noqa: WPS450

        for up in uploads:
            raw = up.read().decode("utf-8", errors="replace")
            for i, rec in enumerate(_split_into_records(raw, split_mode)):
                documents.append(TextDocument(id=f"{up.name}:{i}", text=rec, source=up.name))

with tab_paste:
    pasted = st.text_area(
        "Paste raw text. Splitting follows the sidebar setting.",
        height=200,
        placeholder="One record per line, or paragraphs separated by blank lines…",
    )
    if pasted.strip():
        from cluster_text.loader import _split_into_records  # noqa: WPS450

        for i, rec in enumerate(_split_into_records(pasted, split_mode)):
            documents.append(TextDocument(id=f"pasted:{i}", text=rec, source="pasted"))

if documents:
    st.success(f"Loaded **{len(documents)}** text records.")
else:
    st.info("Upload files or paste text above to get started.")

# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
st.subheader("2. Run clustering")
run_clicked = st.button("🚀 Run pipeline", type="primary", disabled=not documents)

if run_clicked and documents:
    cfg = PipelineConfig(
        vectorizer=vectorizer,
        algorithm=algorithm,
        n_clusters=(None if algorithm == "kmeans" and auto_k else (k_value if algorithm == "kmeans" else None)),
        dbscan_eps=eps if algorithm == "dbscan" else 0.5,
        dbscan_min_samples=min_samples if algorithm == "dbscan" else 5,
        embedding_model=embedding_model,
    )
    with st.spinner("Vectorizing and clustering…"):
        result = run_pipeline(documents, cfg)
    st.session_state["result"] = result

# --------------------------------------------------------------------------- #
# Results                                                                     #
# --------------------------------------------------------------------------- #
result = st.session_state.get("result")
if result:
    c = result.clustering
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(result.documents))
    col2.metric("Clusters", c.n_clusters)
    col3.metric("Noise points", c.n_noise)
    col4.metric("Silhouette", f"{c.silhouette:.3f}" if c.silhouette is not None else "n/a")

    # ----- 2D projection plot ------------------------------------------------
    st.subheader("🗺️ Cluster map (2D projection)")
    X = result.vectorization.matrix
    try:
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA

            coords = PCA(n_components=2, random_state=42).fit_transform(X)
        else:
            coords = X
        labels = result.clustering.labels
        df_plot = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "cluster": [str(l) for l in labels],
                "text": [d.text[:120] + ("…" if len(d.text) > 120 else "") for d in result.documents],
            }
        )
        fig = px.scatter(
            df_plot, x="x", y="y", color="cluster", hover_data=["text"], height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Could not render projection: {exc}")

    # ----- Per-cluster cards -------------------------------------------------
    st.subheader("📂 Clusters")
    for s in sorted(result.summaries, key=lambda x: (-x.size, x.cluster_id)):
        with st.expander(f"[{s.cluster_id}] Cluster {s.cluster_id} — {s.size} docs", expanded=False):
            if s.keywords:
                st.markdown("**Top keywords:** " + ", ".join(f"`{k}`" for k in s.keywords))
            if s.exemplars:
                st.markdown("**Representative examples:**")
                for ex in s.exemplars:
                    st.markdown(f"> {ex}")
            # Note: expanders cannot be nested, so list members directly.
            st.markdown("**All member documents:**")
            members = pd.DataFrame(
                [
                    {"id": d.id, "text": d.text}
                    for d, lbl in zip(result.documents, result.clustering.labels)
                    if int(lbl) == s.cluster_id
                ]
            )
            st.dataframe(members, use_container_width=True, height=240)

    # ----- Download ---------------------------------------------------------
    st.subheader("⬇️ Export")
    rows = []
    for d, lbl in zip(result.documents, result.clustering.labels):
        rows.append(
            {
                "id": d.id,
                "source": d.source,
                "cluster": int(lbl),
                "text": d.text,
            }
        )
    df_out = pd.DataFrame(rows)
    buf = io.StringIO()
    df_out.to_csv(buf, index=False)
    st.download_button(
        "Download clustered data (CSV)",
        buf.getvalue(),
        file_name="clusters.csv",
        mime="text/csv",
    )
