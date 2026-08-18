"""Sample the firing schedule for the fully asynchronous run: per replica, each
step splits into ``--cells`` Bernoulli(``p_step / cells``) cells per agent, giving
ordered ``[cell, agent]`` events (collisions shuffled; time = (step-1) + cell/cells).
    python -m lib.datagen_async.make_schedule  # offline -> data/schedules/<mode>_firing_schedule.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..datagen.collect_energy import build_graph_bank, load_questions


def _replica_rng(schedule_seed: int, name: str) -> np.random.Generator:
    """Independent, platform-stable RNG per replica (sha256-derived seed)."""
    digest = hashlib.sha256(f"sched|{schedule_seed}|{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def sample_replica_schedule(
    rng: np.random.Generator,
    n: int,
    num_steps: int,
    cells: int,
    p_cell: float,
) -> List[List[List[int]]]:
    """Per step, the ordered ``[cell, agent]`` firing events of one replica."""
    steps: List[List[List[int]]] = []
    for _ in range(num_steps):
        fires = rng.random((cells, n)) < p_cell
        events: List[List[int]] = []
        for c in np.nonzero(fires.any(axis=1))[0]:
            agents = np.nonzero(fires[c])[0]
            if len(agents) > 1:
                agents = agents[rng.permutation(len(agents))]
            events.extend([int(c), int(a)] for a in agents)
        steps.append(events)
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-agents", type=int, default=32)
    ap.add_argument("--num-edges", type=int, default=112,
                    help="Non-zero entries per symmetric J (even); default ~deg 3.5.")
    ap.add_argument("--num-train-graphs", type=int, default=8)
    ap.add_argument("--num-seen-graphs", type=int, default=4)
    ap.add_argument("--num-fresh-graphs", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--cells", type=int, default=1000,
                    help="Fine cells per step; each is 1/cells of a timestep.")
    ap.add_argument("--p-step", type=float, default=0.5,
                    help="Expected firings per agent per step; the per-cell "
                         "probability is p_step / cells.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Graph-bank seed; 0 reproduces the synchronous run's bank.")
    ap.add_argument("--schedule-seed", type=int, default=0,
                    help="Seed of the Bernoulli cell sampling.")
    ap.add_argument("--mode", choices=("subjective", "objective"),
                    default="objective")
    ap.add_argument("--train-file", default=None,
                    help="Defaults to data/<subj|obj>/train.jsonl by --mode.")
    ap.add_argument("--test-file", default=None,
                    help="Defaults to data/<subj|obj>/test.jsonl by --mode.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N questions per split (testing).")
    ap.add_argument("--output", default=None,
                    help="Defaults to data/schedules/<mode>_firing_schedule.json.")
    args = ap.parse_args()

    if args.num_edges % 2 != 0:
        raise ValueError(f"--num-edges must be even, got {args.num_edges}")
    if not 0.0 < args.p_step / args.cells < 1.0:
        raise ValueError("p_step / cells must be a probability in (0, 1)")

    data_dir = "data/subj" if args.mode == "subjective" else "data/obj"
    if args.train_file is None:
        args.train_file = f"{data_dir}/train.jsonl"
    if args.test_file is None:
        args.test_file = f"{data_dir}/test.jsonl"
    if args.output is None:
        args.output = f"data/schedules/{args.mode}_firing_schedule.json"

    train_qs = load_questions(args.train_file, args.limit)
    test_qs = load_questions(args.test_file, args.limit)

    # Lattices are excluded by design for this experiment; the random Js are
    # drawn before the lattices in build_graph_bank, so the bank is identical
    # to the synchronous run's for the same seed.
    train_graphs, test_graphs, seen_ids = build_graph_bank(
        n=args.num_agents,
        num_train_graphs=args.num_train_graphs,
        num_seen_graphs=args.num_seen_graphs,
        num_fresh_graphs=args.num_fresh_graphs,
        num_edges=args.num_edges,
        seed=args.seed,
        include_lattices=False,
    )

    p_cell = args.p_step / args.cells
    replicas: Dict[str, Any] = {}
    total_events = 0
    collision_cells = 0
    plan = [("train", train_qs, train_graphs), ("test", test_qs, test_graphs)]
    for split, questions, graphs in plan:
        for q in questions:
            for gid, _J in graphs:
                name = f"{q['qid']}__{gid}__rep00"
                rng = _replica_rng(args.schedule_seed, name)
                steps = sample_replica_schedule(
                    rng, args.num_agents, args.num_steps, args.cells, p_cell,
                )
                n_events = sum(len(s) for s in steps)
                total_events += n_events
                for s in steps:
                    cells_used = [c for c, _a in s]
                    collision_cells += len(cells_used) - len(set(cells_used))
                replicas[name] = {
                    "qid": q["qid"],
                    "split": split,
                    "graph_id": gid,
                    "graph_seen": gid in seen_ids,
                    "num_events": n_events,
                    "steps": steps,
                }

    payload = {
        "meta": {
            "experiment": f"{args.mode}_firing_schedule",
            "mode": args.mode,
            "num_agents": args.num_agents,
            "num_steps": args.num_steps,
            "cells_per_step": args.cells,
            "p_step": args.p_step,
            "p_cell": p_cell,
            "schedule_seed": args.schedule_seed,
            "collision_policy": "uniform_random_order_within_cell",
            "seed": args.seed,
            "num_edges": args.num_edges,
            "num_train_graphs": args.num_train_graphs,
            "num_seen_graphs": args.num_seen_graphs,
            "num_fresh_graphs": args.num_fresh_graphs,
            "lattice_ids": [],
            "train_file": args.train_file,
            "test_file": args.test_file,
            "limit": args.limit,
            "seen_graph_ids": sorted(seen_ids),
        },
        "replicas": replicas,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, ensure_ascii=False)

    n_rep = len(replicas)
    expected = args.num_agents * args.p_step * args.num_steps
    print(f"[make_schedule] {n_rep} replicas -> {out}")
    print(f"[make_schedule] events: total {total_events:,}, "
          f"mean/replica {total_events / max(1, n_rep):.1f} "
          f"(expected {expected:.1f}), "
          f"cells with >1 firing: {collision_cells}")


if __name__ == "__main__":
    main()
