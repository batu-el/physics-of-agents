"""Collect the frontier mixed-model trajectory dataset via OpenRouter.

One fixed random graph J0, n = 64 agents split across two frontier model
families (32 + 32), t = 8 timesteps, 1 episode per question, run over every
objective (40) and subjective (20) question:

    python -m lib.datagen_frontier.collect_frontier        # needs OPENROUTER_API_KEY

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..datagen.collect_energy import load_canonical_personas, load_questions, model_slug
from ..datagen.graph import sample_J_num_edges_symmetric
from ..datagen.utils import make_mock_pi
from .dynamics_mixed import MixedReplica, run_forward_dynamics_mixed
from .openrouter import REASONING_CHOICES, make_openrouter_pis

DEFAULT_MODEL_A = "openai/gpt-5.6-sol"
DEFAULT_MODEL_B = "deepseek/deepseek-v4-flash-0731"


def assign_models(model_a: str, model_b: str, n: int, scheme: str) -> List[str]:
    """Per-agent model list: "block" = first half A / second half B (with the
    32-persona bank tiled over 64 agents, agents i and i+n/2 share a persona
    across the two families), "interleave" = A, B, A, B, ..."""
    if n % 2 != 0:
        raise ValueError(f"num_agents must be even for a 50/50 split, got {n}")
    if scheme == "block":
        return [model_a] * (n // 2) + [model_b] * (n // 2)
    if scheme == "interleave":
        return [model_a if i % 2 == 0 else model_b for i in range(n)]
    raise ValueError(f"scheme must be 'block' or 'interleave', got {scheme!r}")


def load_all_questions(data_dir: str, limit: int | None) -> List[Dict[str, Any]]:
    """All questions for a mode (train + test pooled; the split label stays in
    the manifest)."""
    rows: List[Dict[str, Any]] = []
    for split in ("train", "test"):
        for q in load_questions(f"{data_dir}/{split}.jsonl", None):
            rows.append({**q, "split": split})
    return rows[:limit] if limit is not None else rows


def build_replicas(
    personas: List[str],
    agent_models: List[str],
    questions: List[Dict[str, Any]],
    graph_id: str,
    J: np.ndarray,
    episodes: int,
    mode: str,
) -> Tuple[List[MixedReplica], Dict[str, Any]]:
    """One replica per (question, episode) on the single fixed graph."""
    replicas: List[MixedReplica] = []
    manifest: Dict[str, Any] = {}
    for q in questions:
        for rep in range(episodes):
            name = f"{q['qid']}__{graph_id}__rep{rep:02d}"
            replicas.append(
                MixedReplica(
                    name=name,
                    personas=list(personas),
                    statement=q["question"],
                    J=J,
                    mode=mode,
                    choices=q.get("choices"),
                    agent_models=list(agent_models),
                )
            )
            manifest[name] = {
                "qid": q["qid"],
                "split": q["split"],
                "question": q["question"],
                "graph_id": graph_id,
                "repeat": rep,
                "num_edges": int((J != 0).sum()),
            }
    return replicas, manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-agents", type=int, default=64)
    ap.add_argument("--num-edges", type=int, default=224,
                    help="Non-zero entries of the symmetric J0 (even); "
                         "default keeps avg degree 3.5 as in the n=32 runs.")
    ap.add_argument("--episodes", type=int, default=1,
                    help="Episodes (replicas) per question; default 1.")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--k", type=int, default=1,
                    help="Spin samples per agent per step (majority vote); "
                         "default 1 = a single sample per agent.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for J0; both modes reuse the same graph.")
    ap.add_argument("--max-workers", type=int, default=256)
    ap.add_argument("--model-a", default=DEFAULT_MODEL_A,
                    help="OpenRouter model for the first 32 agents.")
    ap.add_argument("--model-b", default=DEFAULT_MODEL_B,
                    help="OpenRouter model for the last 32 agents.")
    ap.add_argument("--assignment", choices=("block", "interleave"),
                    default="block",
                    help="How agents are split between the two models.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--reasoning", choices=REASONING_CHOICES, default="none",
                    help="OpenRouter reasoning control; 'none' disables "
                         "reasoning tokens (the protocol demands snap "
                         "judgements), 'default' omits the parameter.")
    ap.add_argument("--mode", choices=("subjective", "objective", "both"),
                    default="both",
                    help="Which dataset(s) to run; 'both' runs objective "
                         "then subjective on the same J0.")
    ap.add_argument("--output-root", default="data/frontier",
                    help="Outputs go to <output-root>/<mode>_energy.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N questions per mode (testing).")
    ap.add_argument("--mock", action="store_true", help="Offline deterministic LMs.")
    args = ap.parse_args()

    if args.num_edges % 2 != 0:
        raise ValueError(f"--num-edges must be even, got {args.num_edges}")

    # One fixed random graph, shared by both modes.
    rng = np.random.default_rng(args.seed)
    graph_id = "J0"
    J0 = sample_J_num_edges_symmetric(args.num_agents, args.num_edges, rng)

    agent_models = assign_models(
        args.model_a, args.model_b, args.num_agents, args.assignment,
    )

    if args.mock:
        # Distinct mock seeds so the two "families" behave differently offline.
        pis = {
            m: make_mock_pi(seed=args.seed + idx)
            for idx, m in enumerate(dict.fromkeys(agent_models))
        }
    else:
        pis = make_openrouter_pis(
            models=agent_models,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning=args.reasoning,
        )

    modes = ("objective", "subjective") if args.mode == "both" else (args.mode,)
    for mode in modes:
        data_dir = "data/subj" if mode == "subjective" else "data/obj"
        questions = load_all_questions(data_dir, args.limit)
        if mode == "objective":
            missing = [q["qid"] for q in questions if "choices" not in q]
            if missing:
                raise ValueError(
                    f"objective mode needs A/B 'choices' per question; missing "
                    f"for {len(missing)} (e.g. {missing[:3]})"
                )

        persona_file = f"{data_dir}/personas.json"
        personas = load_canonical_personas(persona_file, args.num_agents)

        replicas, manifest = build_replicas(
            personas, agent_models, questions, graph_id, J0,
            args.episodes, mode,
        )

        config_meta = {
            "experiment": f"frontier_{mode}_energy",
            "mode": mode,
            "backend": "mock" if args.mock else "openrouter",
            "model_a": "mock" if args.mock else args.model_a,
            "model_b": "mock" if args.mock else args.model_b,
            "assignment": args.assignment,
            "agent_models": agent_models,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "reasoning": args.reasoning,
            "num_agents": args.num_agents,
            "num_edges": args.num_edges,
            "graph_ids": [graph_id],
            "episodes": args.episodes,
            "num_steps": args.num_steps,
            "k": args.k,
            "seed": args.seed,
            "persona_file": persona_file,
            "persona_order": "canonical_fixed_tiled",
        }

        output_dir = (
            f"{args.output_root}/{model_slug(args.model_a)}"
            f"__{model_slug(args.model_b)}/{mode}_energy"
        )
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "manifest.json", "w") as f:
            json.dump({"meta": config_meta, "replicas": manifest}, f,
                      ensure_ascii=False, indent=2)

        print(f"[collect_frontier] mode={mode} replicas={len(replicas)} "
              f"questions={len(questions)} n={args.num_agents} steps={args.num_steps} "
              f"-> {output_dir}", flush=True)

        t0 = time.time()
        run_forward_dynamics_mixed(
            replicas=replicas,
            pis=pis,
            num_steps=args.num_steps,
            k=args.k,
            max_workers=args.max_workers,
            output_dir=output_dir,
            config_meta=config_meta,
        )
        print(f"[collect_frontier] mode={mode} done in {time.time() - t0:.1f}s; "
              f"outputs at {output_dir}", flush=True)


if __name__ == "__main__":
    main()
