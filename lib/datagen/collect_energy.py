"""Collect the train + test trajectory dataset
    python -m lib.datagen.collect_energy                    # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .dynamics import Replica, run_forward_dynamics
from .graph import make_lattice_J, sample_J_num_edges_symmetric
from .utils import make_mock_pi, make_openai_pi, make_together_pi

# Known models -> API backend. Unknown models fall back to name-based inference
# (org-prefixed "org/name" -> together, bare names -> openai).
BACKENDS: Dict[str, str] = {
    "gpt-4o-mini": "openai",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": "together",
    "Qwen/Qwen2.5-7B-Instruct-Turbo": "together",
    "google/gemma-3n-E4B-it": "together",
}


def model_slug(model: str) -> str:
    """Folder-safe model name: org prefix dropped, lowercased."""
    return model.split("/")[-1].lower()


def resolve_backend(model: str, backend: str | None) -> str:
    """Resolve the API backend: explicit flag > registry > infer from name."""
    if backend is not None:
        return backend
    return BACKENDS.get(model) or ("together" if "/" in model else "openai")


def load_canonical_personas(
    path: str | Path, num_agents: int | None = None
) -> List[str]:
    """Personas in fixed canonical order (agent i is persona i; tiled if needed)."""
    with open(path) as f:
        data = json.load(f)
    personas = [d["persona"] if isinstance(d, dict) else d for d in data]
    if num_agents is None:
        return personas
    return [personas[i % len(personas)] for i in range(num_agents)]


def load_questions(path: str | Path, limit: int | None) -> List[Dict[str, Any]]:
    """Read ``{qid, question[, choices]}`` rows from a train/test .jsonl file."""
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        row: Dict[str, Any] = {"qid": r["qid"], "question": r["question"]}
        if "choices" in r:
            row["choices"] = r["choices"]
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_graph_bank(
    n: int,
    num_train_graphs: int,
    num_seen_graphs: int,
    num_fresh_graphs: int,
    num_edges: int,
    seed: int,
    lattice_rows: int = 4,
    lattice_cols: int = 8,
    lattice_weight: int = 1,
    include_lattices: bool = True,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]], set]:
    """Return (train_graphs, test_graphs, seen_graph_ids): random Js ``J0..`` for
    train, first ``num_seen_graphs`` of them + fresh ``Jf0..`` for test, and the
    two lattices appended to both splits ("seen")."""
    if num_seen_graphs > num_train_graphs:
        raise ValueError("num_seen_graphs cannot exceed num_train_graphs")
    if include_lattices and lattice_rows * lattice_cols != n:
        raise ValueError(
            f"lattice_rows*lattice_cols ({lattice_rows}x{lattice_cols}="
            f"{lattice_rows * lattice_cols}) must equal num_agents ({n})"
        )
    rng = np.random.default_rng(seed)
    train_random = [
        (f"J{g}", sample_J_num_edges_symmetric(n, num_edges, rng))
        for g in range(num_train_graphs)
    ]
    fresh_graphs = [
        (f"Jf{g}", sample_J_num_edges_symmetric(n, num_edges, rng))
        for g in range(num_fresh_graphs)
    ]
    lattices = (
        [(kind, make_lattice_J(lattice_rows, lattice_cols, kind, lattice_weight))
         for kind in ("square", "triangular")]
        if include_lattices else []
    )

    train_graphs = train_random + lattices
    test_graphs = train_random[:num_seen_graphs] + fresh_graphs + lattices
    seen_ids = {gid for gid, _ in train_graphs}
    return train_graphs, test_graphs, seen_ids


def build_replicas(
    personas: List[str],
    train_qs: List[Dict[str, Any]],
    test_qs: List[Dict[str, Any]],
    train_graphs: List[Tuple[str, np.ndarray]],
    test_graphs: List[Tuple[str, np.ndarray]],
    seen_ids: set,
    trajectories: int,
    mode: str = "subjective",
) -> Tuple[List[Replica], Dict[str, Any]]:
    """One replica per (question, graph, trajectory); also build the manifest."""
    replicas: List[Replica] = []
    manifest: Dict[str, Any] = {}
    plan = [("train", train_qs, train_graphs), ("test", test_qs, test_graphs)]
    for split, questions, graphs in plan:
        for q in questions:
            for gid, J in graphs:
                for rep in range(trajectories):
                    name = f"{q['qid']}__{gid}__rep{rep:02d}"
                    replicas.append(
                        Replica(
                            name=name,
                            personas=list(personas),
                            statement=q["question"],
                            J=J,
                            mode=mode,
                            choices=q.get("choices"),
                        )
                    )
                    manifest[name] = {
                        "qid": q["qid"],
                        "split": split,
                        "question": q["question"],
                        "graph_id": gid,
                        "graph_seen": gid in seen_ids,
                        "repeat": rep,
                        "num_edges": int((J != 0).sum()),
                    }
    return replicas, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-agents", type=int, default=32)
    ap.add_argument("--num-edges", type=int, default=112,
                    help="Non-zero entries per symmetric J (even); default ~deg 3.5.")
    ap.add_argument("--num-train-graphs", type=int, default=8)
    ap.add_argument("--num-seen-graphs", type=int, default=4,
                    help="How many train Js are reused at test time.")
    ap.add_argument("--num-fresh-graphs", type=int, default=4,
                    help="How many brand-new Js are sampled for test.")
    ap.add_argument("--lattice-rows", type=int, default=4,
                    help="Rows of the square/triangular lattices "
                         "(rows*cols must equal --num-agents).")
    ap.add_argument("--lattice-cols", type=int, default=8,
                    help="Cols of the square/triangular lattices.")
    ap.add_argument("--lattice-weight", type=int, default=1, choices=(-1, 1),
                    help="Bond sign for every lattice edge (+1 ferromagnetic).")
    ap.add_argument("--no-lattices", action="store_true",
                    help="Random-J bank only: skip the lattices in both splits.")
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=256)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="Model name; known models: " + ", ".join(BACKENDS))
    ap.add_argument("--backend", choices=("openai", "together"), default=None,
                    help="API backend; inferred from the model name by default.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--mode", choices=("subjective", "objective"),
                    default="subjective",
                    help="subjective -> AGREE/DISAGREE; objective -> answer A/B.")
    ap.add_argument("--train-file", default=None,
                    help="Defaults to data/<subj|obj>/train.jsonl by --mode.")
    ap.add_argument("--test-file", default=None,
                    help="Defaults to data/<subj|obj>/test.jsonl by --mode.")
    ap.add_argument("--personas", default=None,
                    help="Defaults to data/<subj|obj>/personas.json by --mode.")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to data/<model-slug>/<mode>_energy.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N questions per split (testing).")
    ap.add_argument("--mock", action="store_true", help="Offline deterministic LM.")
    args = ap.parse_args()

    if args.num_edges % 2 != 0:
        raise ValueError(f"--num-edges must be even, got {args.num_edges}")

    backend = resolve_backend(args.model, args.backend)

    # Mode-specific defaults (data/<subj|obj>/...).
    data_dir = "data/subj" if args.mode == "subjective" else "data/obj"
    if args.train_file is None:
        args.train_file = f"{data_dir}/train.jsonl"
    if args.test_file is None:
        args.test_file = f"{data_dir}/test.jsonl"
    if args.personas is None:
        args.personas = f"{data_dir}/personas.json"
    if args.output_dir is None:
        args.output_dir = f"data/{model_slug(args.model)}/{args.mode}_energy"

    train_qs = load_questions(args.train_file, args.limit)
    test_qs = load_questions(args.test_file, args.limit)
    if args.mode == "objective":
        missing = [q["qid"] for q in train_qs + test_qs if "choices" not in q]
        if missing:
            raise ValueError(
                f"objective mode needs A/B 'choices' per question; missing for "
                f"{len(missing)} (e.g. {missing[:3]})"
            )

    train_graphs, test_graphs, seen_ids = build_graph_bank(
        n=args.num_agents,
        num_train_graphs=args.num_train_graphs,
        num_seen_graphs=args.num_seen_graphs,
        num_fresh_graphs=args.num_fresh_graphs,
        num_edges=args.num_edges,
        seed=args.seed,
        lattice_rows=args.lattice_rows,
        lattice_cols=args.lattice_cols,
        lattice_weight=args.lattice_weight,
        include_lattices=not args.no_lattices,
    )

    personas = load_canonical_personas(args.personas, args.num_agents)
    replicas, manifest = build_replicas(
        personas, train_qs, test_qs, train_graphs, test_graphs,
        seen_ids, args.trajectories, mode=args.mode,
    )

    config_meta = {
        "experiment": f"{args.mode}_energy",
        "mode": args.mode,
        "model": "mock" if args.mock else args.model,
        "backend": "mock" if args.mock else backend,
        "temperature": args.temperature,
        "num_agents": args.num_agents,
        "num_edges": args.num_edges,
        "num_train_graphs": args.num_train_graphs,
        "num_seen_graphs": args.num_seen_graphs,
        "num_fresh_graphs": args.num_fresh_graphs,
        "lattice_ids": [] if args.no_lattices else ["square", "triangular"],
        "lattice_rows": args.lattice_rows,
        "lattice_cols": args.lattice_cols,
        "lattice_weight": args.lattice_weight,
        "trajectories": args.trajectories,
        "num_steps": args.num_steps,
        "k": args.k,
        "seed": args.seed,
        "persona_file": args.personas,
        "persona_order": "canonical_fixed",
        "seen_graph_ids": sorted(seen_ids),
    }

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "manifest.json", "w") as f:
        json.dump({"meta": config_meta, "replicas": manifest}, f,
                  ensure_ascii=False, indent=2)

    if args.mock:
        pi = make_mock_pi(seed=args.seed)
    elif backend == "together":
        pi = make_together_pi(model=args.model, temperature=args.temperature)
    else:
        pi = make_openai_pi(model=args.model, temperature=args.temperature)

    print(f"[collect] replicas={len(replicas)} train_q={len(train_qs)} "
          f"test_q={len(test_qs)} train_J={len(train_graphs)} test_J={len(test_graphs)} "
          f"-> {args.output_dir}", flush=True)

    t0 = time.time()
    run_forward_dynamics(
        replicas=replicas,
        pi=pi,
        num_steps=args.num_steps,
        k=args.k,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        config_meta=config_meta,
    )
    print(f"[collect] done in {time.time() - t0:.1f}s; outputs at {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
