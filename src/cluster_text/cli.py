"""Command-line interface for the clustering platform."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .loader import load_text_files
from .pipeline import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    p = argparse.ArgumentParser(
        prog="cluster-text",
        description="Cluster unstructured text into themed groups.",
    )
    p.add_argument("paths", nargs="+", help="Input .txt files or directories.")
    p.add_argument(
        "--split",
        choices=["line", "paragraph", "whole"],
        default="line",
        help="How to split files into records (default: line).",
    )
    p.add_argument(
        "--vectorizer",
        choices=["tfidf", "embedding"],
        default="tfidf",
    )
    p.add_argument(
        "--algorithm",
        choices=["kmeans", "dbscan"],
        default="kmeans",
    )
    p.add_argument("-k", "--n-clusters", type=int, default=None,
                   help="Number of clusters for K-Means (default: auto).")
    p.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps.")
    p.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples.")
    p.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (used when --vectorizer=embedding).",
    )
    p.add_argument("--output", type=Path, default=None,
                   help="Write a JSON report to this path.")
    return p


def _report_dict(result) -> dict:
    """Convert the pipeline result into a dictionary for JSON reporting."""
    return {
        "config": asdict(result.config),
        "n_documents": len(result.documents),
        "clustering": {
            "algorithm": result.clustering.algorithm,
            "n_clusters": result.clustering.n_clusters,
            "n_noise": result.clustering.n_noise,
            "silhouette": result.clustering.silhouette,
            "params": result.clustering.params,
        },
        "clusters": [
            {
                "cluster_id": s.cluster_id,
                "size": s.size,
                "keywords": s.keywords,
                "exemplars": s.exemplars,
                "member_ids": s.member_ids,
            }
            for s in result.summaries
        ],
    }


def _print_human(result) -> None:
    """Print a human-readable summary of the clustering results."""
    c = result.clustering
    print(f"\nLoaded {len(result.documents)} documents.")
    print(
        f"Algorithm={c.algorithm}  clusters={c.n_clusters}  noise={c.n_noise}  "
        f"silhouette={c.silhouette}"
    )
    print("\n=== Clusters ===")
    for s in sorted(result.summaries, key=lambda x: (-x.size, x.cluster_id)):
        print(f"\n[{s.cluster_id}] Cluster {s.cluster_id}  (size={s.size})")
        if s.keywords:
            print(f"  keywords: {', '.join(s.keywords[:10])}")
        for ex in s.exemplars[:2]:
            snippet = ex.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "…"
            print(f"   • {snippet}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the command-line interface."""
    args = _build_parser().parse_args(argv)
    docs = load_text_files(args.paths, split_mode=args.split)
    if not docs:
        print("No text records found.", file=sys.stderr)
        return 2

    cfg = PipelineConfig(
        vectorizer=args.vectorizer,
        algorithm=args.algorithm,
        n_clusters=args.n_clusters,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        embedding_model=args.embedding_model,
    )
    result = run_pipeline(docs, cfg)
    _print_human(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_report_dict(result), indent=2))
        print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
