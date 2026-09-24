# Low-rank graph episodes

The 240 GPT-4o-mini episodes cover six graphs, ten test questions per regime, and two replicas.
Each graph has 56 edges: 22 leaves share one hub in a 10-node core containing 34 edges; no node is isolated. All matrices have algebraic rank 11 and effective rank 5.92–6.88.

Generation procedure:

1. Connect 22 outer nodes to one hub in a 10-node core, then randomly place 34 edges within the core.
2. On each fixed topology, generate candidate signings using random edge flips among high-degree nodes; select the closest frustration target (0, 0.1, or 0.2).

| Graph | Effective rank | Minimum unsatisfied edges | Frustration fraction |
| --- | ---: | ---: | ---: |
| `rankLOW_frust00_s0` | 5.925 | 0/56 | 0.000000 |
| `rankLOW_frust00_s1` | 6.022 | 0/56 | 0.000000 |
| `rankLOW_frust10_s0` | 6.641 | 5/56 | 0.089286 |
| `rankLOW_frust10_s1` | 6.884 | 6/56 | 0.107143 |
| `rankLOW_frust20_s0` | 6.431 | 11/56 | 0.196429 |
| `rankLOW_frust20_s1` | 6.408 | 11/56 | 0.196429 |

### Effective rank (singular-value participation ratio)

```python
import numpy as np


def effective_rank(J):
    """Return (sum(singular values))**2 / sum(singular values**2).

    This is the singular-value participation ratio, not entropy-based rank.
    Return 0 for a zero matrix.
    """
    singular_values = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    denominator = np.sum(singular_values ** 2)
    if denominator == 0:
        return 0.0
    return float(singular_values.sum() ** 2 / denominator)
```

### Minimum unsatisfied edges (frustration index)

```python
def minimum_unsatisfied_edges(J):
    """Return the exact minimum number of unsatisfied undirected edges.

    J must be symmetric, have a zero diagonal, and contain only -1, 0, +1.
    An edge is unsatisfied when J[i, j] * s[i] * s[j] < 0.

    Remove leaves and isolated nodes, which do not contribute to the minimum,
    then enumerate the remaining binary assignments with one spin fixed.
    Cost is exponential in the remaining node count: these graphs leave a
    10-node core and require at most 2**9 = 512 assignments each.
    """
    J = np.asarray(J)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix")
    if not np.array_equal(J, J.T) or np.any(np.diag(J) != 0):
        raise ValueError("J must be symmetric with a zero diagonal")
    if not np.isin(J, [-1, 0, 1]).all():
        raise ValueError("J must contain only -1, 0, +1")

    core = J.copy()
    while len(core):
        keep = np.count_nonzero(core, axis=1) >= 2
        if keep.all():
            break
        core = core[np.ix_(keep, keep)]
    if len(core) == 0:
        return 0

    # Count each undirected edge once.
    i, j = np.nonzero(np.triu(core, k=1))
    signs = core[i, j]
    best = len(i)
    spins = np.ones(len(core), dtype=int)

    # Global sign reversal preserves satisfaction, so fix spins[0] = +1.
    for assignment in range(1 << (len(core) - 1)):
        spins[1:] = [
            1 if (assignment >> bit) & 1 else -1
            for bit in range(len(core) - 1)
        ]
        unsatisfied = int(np.count_nonzero(signs * spins[i] * spins[j] < 0))
        best = min(best, unsatisfied)
        if best == 0:
            break
    return best
```

### Frustration fraction

```python
def frustration_fraction(J):
    """Return minimum unsatisfied edges / total undirected edges.

    Uses minimum_unsatisfied_edges above. Return 0 for an edgeless graph.
    The denominator includes all edges, including those removed as leaves
    during the minimum-unsatisfied-edge calculation.
    """
    J = np.asarray(J)
    minimum = minimum_unsatisfied_edges(J)
    edge_count = int(np.count_nonzero(np.triu(J, k=1)))
    return minimum / edge_count if edge_count else 0.0
```
