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