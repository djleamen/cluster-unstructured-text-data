"""Load unstructured text data from files or directories."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class TextDocument:
    """A single text record fed into the clustering pipeline."""

    id: str
    text: str
    source: str | None = None


def _split_into_records(raw: str, mode: str) -> list[str]:
    """Split a single text blob into records.

    mode:
      - "line":      one record per non-empty line
      - "paragraph": one record per blank-line-separated paragraph
      - "whole":     entire file is one record
    """
    if mode == "whole":
        text = raw.strip()
        return [text] if text else []
    if mode == "paragraph":
        parts = re.split(r"\n\s*\n", raw)
    elif mode == "line":
        parts = raw.splitlines()
    else:
        raise ValueError(f"Unknown split mode: {mode!r}")
    return [p.strip() for p in parts if p and p.strip()]


def load_text_files(
    paths: Iterable[str | Path],
    split_mode: str = "line",
    encoding: str = "utf-8",
) -> list[TextDocument]:
    """Load text records from one or more files / directories.

    Directories are scanned recursively for ``*.txt`` files.
    """
    docs: list[TextDocument] = []
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)

    for f in files:
        raw = f.read_text(encoding=encoding, errors="replace")
        for i, record in enumerate(_split_into_records(raw, split_mode)):
            docs.append(
                TextDocument(
                    id=f"{f.name}:{i}",
                    text=record,
                    source=str(f),
                )
            )
    return docs


def load_texts(texts: Iterable[str]) -> list[TextDocument]:
    """Wrap an in-memory iterable of strings as TextDocuments."""
    return [
        TextDocument(id=f"doc:{i}", text=t.strip())
        for i, t in enumerate(texts)
        if t and t.strip()
    ]
