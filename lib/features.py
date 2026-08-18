"""Shared feature layer: PCA reduction and persona/question embedding loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# PCA reducer (numpy SVD; no sklearn dependency)
# ---------------------------------------------------------------------------


class PCAReducer:
    """Center-and-project dimensionality reduction fit on an embedding matrix."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # (k, d)

    def fit(self, X: np.ndarray) -> "PCAReducer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Right singular vectors are the principal directions.
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAReducer must be fit before transform")
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Embedding loaders
# ---------------------------------------------------------------------------


def load_persona_embeddings(
    path: str | Path = "data/subj/persona_embeddings.json",
) -> Tuple[np.ndarray, List[str]]:
    """Return ``(embeddings (P, d), texts)`` ordered by persona ``idx``."""
    payload = json.loads(Path(path).read_text())
    items = sorted(payload["items"], key=lambda it: it["idx"])
    emb = np.array([it["embedding"] for it in items], dtype=np.float64)
    texts = [it["text"] for it in items]
    return emb, texts


def canonical_persona_index(
    path: str | Path = "data/subj/persona_embeddings.json",
) -> Dict[str, int]:
    """Map persona text -> canonical index (the ``idx`` order of the embeddings file)."""
    _, texts = load_persona_embeddings(path)
    return {t.strip(): i for i, t in enumerate(texts)}


def load_question_embeddings(
    path: str | Path = "data/subj/question_embeddings.json",
) -> Dict[str, dict]:
    """Map question text -> ``{qid, question, split, embedding (d,)}``."""
    payload = json.loads(Path(path).read_text())
    out: Dict[str, dict] = {}
    for it in payload["items"]:
        out[it["question"]] = {
            "qid": it["qid"],
            "question": it["question"],
            "split": it["split"],
            "embedding": np.asarray(it["embedding"], dtype=np.float64),
        }
    return out


def load_pca_question_features(
    files: Sequence[str | Path] = ("data/subj/train.jsonl", "data/subj/test.jsonl"),
    key: str = "embedding_pca10",
) -> Dict[str, dict]:
    """Return ``{question_text: {qid, question, split, feat}}`` from the precomputed
    reduced features; ``split`` comes from the file (first -> "train")."""
    out: Dict[str, dict] = {}
    for fi, path in enumerate(files):
        split = "train" if fi == 0 else "test"
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if key not in row:
                raise KeyError(
                    f"{path}: record {row.get('qid')!r} lacks '{key}'. "
                    f"Run `python pca_embeddings.py` first."
                )
            out[row["question"]] = {
                "qid": row["qid"],
                "question": row["question"],
                "split": split,
                "feat": np.asarray(row[key], dtype=np.float64),
            }
    return out
