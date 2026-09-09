"""Forward dynamics with per-agent models (mixed model families): identical
protocol to `lib.datagen.dynamics`, but every LM call is routed to the pi of
the acting agent's model."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..datagen.dynamics import (
    Replica,
    _aggregate_spin,
    _build_inboxes,
    _spin_call_with_retry,
)
from ..datagen.samplers import Pi, message_sampler


@dataclass
class MixedReplica(Replica):
    """A `Replica` whose agent i runs on model ``agent_models[i]``."""

    agent_models: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.agent_models) != self.n:
            raise ValueError(
                f"agent_models has length {len(self.agent_models)}, "
                f"expected n={self.n}"
            )


def _pi_for(pis: Dict[str, Pi], model: str) -> Pi:
    try:
        return pis[model]
    except KeyError:
        raise KeyError(f"No pi registered for model {model!r}; have {sorted(pis)}")


def _run_message_phase(
    replicas: List[MixedReplica],
    pis: Dict[str, Pi],
    executor: ThreadPoolExecutor,
    step_idx: int,
) -> None:
    """One LM call per sender (the message doesn't depend on the recipient),
    broadcast along every edge with J[i, j] != 0; the sender's own model writes."""
    fut_meta: Dict[Future, Tuple[int, int]] = {}
    # Per replica: sender i -> list of recipients j (J[i, j] != 0).
    recips_per_replica: List[Dict[int, List[int]]] = []
    for r_idx, rep in enumerate(replicas):
        # spins_history[-1] is s(t-1): this runs after the previous spin phase.
        current_spins = rep.spins_history[-1]
        nz_i, nz_j = np.nonzero(rep.J)
        recips: Dict[int, List[int]] = {}
        for i, j in zip(nz_i.tolist(), nz_j.tolist()):
            recips.setdefault(i, []).append(j)
        recips_per_replica.append(recips)
        for i in recips:
            fut = executor.submit(
                message_sampler,
                persona=rep.personas[i],
                question=rep.statement,
                messages_agree=rep.inbox_agree[i],
                messages_disagree=rep.inbox_disagree[i],
                current_spin=int(current_spins[i]),
                pi=_pi_for(pis, rep.agent_models[i]),
                mode=rep.mode,
                choices=rep.choices,
            )
            fut_meta[fut] = (r_idx, i)

    agent_msgs: List[Dict[int, str]] = [{} for _ in replicas]
    for fut in tqdm(
        as_completed(fut_meta),
        total=len(fut_meta),
        desc=f"step {step_idx}: messages",
        leave=False,
    ):
        r_idx, i = fut_meta[fut]
        try:
            agent_msgs[r_idx][i] = fut.result()
        except Exception as exc:  # noqa: BLE001
            agent_msgs[r_idx][i] = f"[error: {exc}]"

    for r_idx, rep in enumerate(replicas):
        step_msgs: Dict[Tuple[int, int], str] = {}
        for i, js in recips_per_replica[r_idx].items():
            msg = agent_msgs[r_idx][i]
            for j in js:
                step_msgs[(i, j)] = msg
        step_msgs = dict(sorted(step_msgs.items()))
        rep.inbox_agree, rep.inbox_disagree = _build_inboxes(
            rep, step_msgs, step_idx,
        )
        rep.messages_history.append(step_msgs)


def _run_spin_phase(
    replicas: List[MixedReplica],
    pis: Dict[str, Pi],
    k: int,
    executor: ThreadPoolExecutor,
    step_idx: int,
    use_inbox: bool,
) -> None:
    """Sample k spins per (replica, agent) with the agent's own model and
    aggregate to S(t) by majority."""
    fut_meta: Dict[Future, Tuple[int, int, int]] = {}
    for r_idx, rep in enumerate(replicas):
        for a_idx in range(rep.n):
            agree = rep.inbox_agree[a_idx] if use_inbox else []
            disagree = rep.inbox_disagree[a_idx] if use_inbox else []
            for s_idx in range(k):
                fut = executor.submit(
                    _spin_call_with_retry,
                    rep.personas[a_idx],
                    rep.statement,
                    agree,
                    disagree,
                    _pi_for(pis, rep.agent_models[a_idx]),
                    rep.mode,
                    rep.choices,
                )
                fut_meta[fut] = (r_idx, a_idx, s_idx)

    raw: List[np.ndarray] = [
        np.zeros((rep.n, k), dtype=np.int8) for rep in replicas
    ]
    for fut in tqdm(
        as_completed(fut_meta),
        total=len(fut_meta),
        desc=f"step {step_idx}: spins",
        leave=False,
    ):
        r_idx, a_idx, s_idx = fut_meta[fut]
        try:
            raw[r_idx][a_idx, s_idx] = fut.result()
        except Exception:
            raw[r_idx][a_idx, s_idx] = 0

    for r_idx, rep in enumerate(replicas):
        spins = np.array(
            [_aggregate_spin(raw[r_idx][a].tolist()) for a in range(rep.n)],
            dtype=np.int8,
        )
        rep.spins_raw_history.append(raw[r_idx])
        rep.spins_history.append(spins)


def _save_replica(rep: MixedReplica, out_dir: Path, meta: Dict[str, Any]) -> None:
    """Write ``rep``'s full state (including the per-agent model assignment)
    to ``out_dir/<name>.json``."""
    payload = {
        "meta": {**meta, "name": rep.name, "n": rep.n},
        "personas": rep.personas,
        "agent_models": rep.agent_models,
        "statement": rep.statement,
        "mode": rep.mode,
        "choices": rep.choices,
        "J": rep.J.astype(int).tolist(),
        "spins_history": [s.astype(int).tolist() for s in rep.spins_history],
        "spins_raw_history": [
            s.astype(int).tolist() for s in rep.spins_raw_history
        ],
        "messages_history": [
            {f"{i},{j}": msg for (i, j), msg in step_msgs.items()}
            for step_msgs in rep.messages_history
        ],
    }
    with open(out_dir / f"{rep.name}.json", "w") as f:
        json.dump(payload, f, ensure_ascii=False)


def run_forward_dynamics_mixed(
    replicas: List[MixedReplica],
    pis: Dict[str, Pi],
    num_steps: int = 8,
    k: int = 5,
    max_workers: int = 64,
    output_dir: Optional[str] = None,
    config_meta: Optional[Dict[str, Any]] = None,
) -> List[MixedReplica]:
    """Advance every replica through ``num_steps`` rounds (step 0: spins with
    empty inboxes; then messages -> inboxes -> spins), then save one JSON per
    replica to ``output_dir``."""
    config_meta = dict(config_meta or {})

    print(
        f"[run_forward_dynamics_mixed] {len(replicas)} replicas, "
        f"num_steps={num_steps}, k={k}, max_workers={max_workers}, "
        f"models={sorted(pis)}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _run_spin_phase(replicas, pis, k, executor, step_idx=0, use_inbox=False)
        for t in tqdm(range(1, num_steps + 1), desc="steps", position=0):
            _run_message_phase(replicas, pis, executor, step_idx=t)
            _run_spin_phase(replicas, pis, k, executor, step_idx=t, use_inbox=True)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for rep in replicas:
            _save_replica(rep, out_path, config_meta)

    return replicas
