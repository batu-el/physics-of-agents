"""Collect trajectories by replaying a pre-sampled firing schedule (the schedule
JSON is the design's source of truth; this CLI adds only model/sampling knobs).
    python -m lib.datagen_async.make_schedule           # stage 1: offline
    python -m lib.datagen_async.collect_energy_seq      # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..datagen.collect_energy import (
    BACKENDS,
    build_graph_bank,
    load_canonical_personas,
    load_questions,
    model_slug,
    resolve_backend,
)
from ..datagen.utils import make_mock_pi, make_openai_pi, make_together_pi
from .dynamics_seq import SeqReplica, run_seq_forward_dynamics


def build_replicas_from_schedule(
    schedule: Dict[str, Any],
    personas: List[str],
    split: str = "both",
) -> Tuple[List[SeqReplica], Dict[str, Any]]:
    """One replica per schedule entry (filtered by ``split``) + the manifest;
    questions and the graph bank are rebuilt from the schedule's ``meta``, so
    an entry whose qid/graph cannot be resolved is an error."""
    meta = schedule["meta"]
    train_qs = load_questions(meta["train_file"], meta.get("limit"))
    test_qs = load_questions(meta["test_file"], meta.get("limit"))
    if meta["mode"] == "objective":
        missing = [q["qid"] for q in train_qs + test_qs if "choices" not in q]
        if missing:
            raise ValueError(
                f"objective mode needs A/B 'choices' per question; missing for "
                f"{len(missing)} (e.g. {missing[:3]})"
            )
    qmap = {q["qid"]: q for q in train_qs + test_qs}

    train_graphs, test_graphs, _seen = build_graph_bank(
        n=meta["num_agents"],
        num_train_graphs=meta["num_train_graphs"],
        num_seen_graphs=meta["num_seen_graphs"],
        num_fresh_graphs=meta["num_fresh_graphs"],
        num_edges=meta["num_edges"],
        seed=meta["seed"],
        include_lattices=False,
    )
    jmap = {gid: J for gid, J in train_graphs + test_graphs}

    replicas: List[SeqReplica] = []
    manifest: Dict[str, Any] = {}
    for name in sorted(schedule["replicas"]):
        entry = schedule["replicas"][name]
        if split != "both" and entry["split"] != split:
            continue
        q = qmap.get(entry["qid"])
        if q is None:
            raise ValueError(f"schedule replica {name}: qid {entry['qid']} "
                             f"not found in {meta['train_file']}/{meta['test_file']}")
        J = jmap.get(entry["graph_id"])
        if J is None:
            raise ValueError(f"schedule replica {name}: unknown graph "
                             f"{entry['graph_id']}")
        replicas.append(
            SeqReplica(
                name=name,
                personas=list(personas),
                statement=q["question"],
                J=J,
                mode=meta["mode"],
                choices=q.get("choices"),
                firing_schedule=[
                    [[int(c), int(a)] for c, a in step]
                    for step in entry["steps"]
                ],
            )
        )
        manifest[name] = {
            "qid": entry["qid"],
            "split": entry["split"],
            "question": q["question"],
            "graph_id": entry["graph_id"],
            "graph_seen": entry["graph_seen"],
            "repeat": 0,
            "num_edges": int((J != 0).sum()),
            "num_events": entry["num_events"],
        }
    return replicas, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule",
                    default="data/schedules/objective_firing_schedule.json",
                    help="Firing-schedule JSON from make_schedule.py; the "
                         "source of truth for the experiment design.")
    ap.add_argument("--split", choices=("both", "train", "test"), default="both",
                    help="Which schedule replicas to run (test = eval-only, "
                         "~half the cost).")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-workers", type=int, default=256)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="Model name; known models: " + ", ".join(BACKENDS))
    ap.add_argument("--backend", choices=("openai", "together"), default=None,
                    help="API backend; inferred from the model name by default.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--personas", default=None,
                    help="Defaults to data/<subj|obj>/personas.json by the "
                         "schedule's mode.")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to data/<model-slug>/<mode>_energy_seq.")
    ap.add_argument("--mock", action="store_true", help="Offline deterministic LM.")
    args = ap.parse_args()

    with open(args.schedule) as f:
        schedule = json.load(f)
    smeta = schedule["meta"]
    mode = smeta["mode"]

    backend = resolve_backend(args.model, args.backend)

    data_dir = "data/subj" if mode == "subjective" else "data/obj"
    if args.personas is None:
        args.personas = f"{data_dir}/personas.json"
    if args.output_dir is None:
        args.output_dir = f"data/{model_slug(args.model)}/{mode}_energy_seq"

    personas = load_canonical_personas(args.personas, smeta["num_agents"])
    replicas, manifest = build_replicas_from_schedule(
        schedule, personas, split=args.split,
    )

    config_meta = {
        "experiment": f"{mode}_energy_seq",
        "update_rule": "scheduled_independent_clocks",
        "schedule_file": args.schedule,
        "schedule_meta": smeta,
        "split_filter": args.split,
        "mode": mode,
        "model": "mock" if args.mock else args.model,
        "backend": "mock" if args.mock else backend,
        "temperature": args.temperature,
        "num_agents": smeta["num_agents"],
        "num_steps": smeta["num_steps"],
        "k": args.k,
        "persona_file": args.personas,
        "persona_order": "canonical_fixed",
        "seen_graph_ids": smeta["seen_graph_ids"],
    }

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "manifest.json", "w") as f:
        json.dump({"meta": config_meta, "replicas": manifest}, f,
                  ensure_ascii=False, indent=2)

    if args.mock:
        pi = make_mock_pi(seed=smeta["seed"])
    elif backend == "together":
        pi = make_together_pi(model=args.model, temperature=args.temperature)
    else:
        pi = make_openai_pi(model=args.model, temperature=args.temperature)

    print(f"[collect_seq] replicas={len(replicas)} "
          f"schedule={args.schedule} -> {args.output_dir}", flush=True)

    t0 = time.time()
    run_seq_forward_dynamics(
        replicas=replicas,
        pi=pi,
        k=args.k,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        config_meta=config_meta,
    )
    print(f"[collect_seq] done in {time.time() - t0:.1f}s; "
          f"outputs at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
