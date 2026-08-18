"""Interaction-matrix (J) constructors: random symmetric graphs and lattices."""

from __future__ import annotations

import numpy as np


def sample_J_num_edges_symmetric(
    n: int, num_edges: int, rng: np.random.Generator
) -> np.ndarray:
    """Symmetric J in {-1, 0, +1} with exactly ``num_edges`` (even) non-zero
    off-diagonal entries: sign of the largest-|U| pairs of U_ij ~ Uniform(-1, 1)."""
    max_edges = n * (n - 1)
    if not 0 <= num_edges <= max_edges:
        raise ValueError(
            f"num_edges must be in [0, n*(n-1)] = [0, {max_edges}], got {num_edges}"
        )
    if num_edges % 2 != 0:
        raise ValueError(f"num_edges must be even for a symmetric J, got {num_edges}")
    if num_edges == 0:
        return np.zeros((n, n), dtype=np.int8)

    pairs_to_keep = num_edges // 2
    iu, ju = np.triu_indices(n, k=1)
    u = rng.uniform(-1.0, 1.0, size=iu.shape[0])
    top_idx = np.argpartition(np.abs(u), -pairs_to_keep)[-pairs_to_keep:]
    kept_i = iu[top_idx]
    kept_j = ju[top_idx]
    kept_signs = np.sign(u[top_idx]).astype(np.int8)

    J = np.zeros((n, n), dtype=np.int8)
    J[kept_i, kept_j] = kept_signs
    J[kept_j, kept_i] = kept_signs
    return J


def make_lattice_J(rows: int, cols: int, kind: str = "square",
                   weight: int = 1) -> np.ndarray:
    """Symmetric ``rows x cols`` lattice: "square" = 4-neighbor grid,
    "triangular" adds the down-left diagonal (6-neighbor)."""
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got {rows}x{cols}")
    if weight not in (-1, 1):
        raise ValueError(f"weight must be -1 or +1, got {weight}")
    if kind not in ("square", "triangular"):
        raise ValueError(f"kind must be 'square' or 'triangular', got {kind!r}")

    n = rows * cols
    J = np.zeros((n, n), dtype=np.int8)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            # Set right / down (/ down-left) once; mirroring covers the rest.
            neighbors = [(r, c + 1), (r + 1, c)]
            if kind == "triangular":
                neighbors.append((r + 1, c - 1))
            for rj, cj in neighbors:
                if 0 <= rj < rows and 0 <= cj < cols:
                    j = idx(rj, cj)
                    J[i, j] = weight
                    J[j, i] = weight
    return J
