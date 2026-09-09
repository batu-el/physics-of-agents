"""Forward dynamics: sequential steps, parallel LM calls within each step,
one output JSON per replica."""

from __future__ import annotations

import json
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from .samplers import Pi, message_sampler, spin_sampler


@dataclass
class Replica:
    """One (statement, J) trial: personas, inboxes, and spin/message history."""

    name: str
    personas: List[str]
    statement: str
    J: np.ndarray  # (n, n) int in {-1, 0, +1}
    mode: str = "subjective"  # "subjective" (AGREE/DISAGREE) or "objective" (A/B)
    choices: Optional[Dict[str, str]] = None  # A/B options in objective mode

    inbox_agree: List[List[str]] = field(default_factory=list)
    inbox_disagree: List[List[str]] = field(default_factory=list)

    # spins_history[t]: (n,) aggregated spins, len = num_steps + 1.
    # spins_raw_history[t]: (n, k) raw per-sample spins.
    # messages_history[t]: (i, j) -> message string for step t+1.
    spins_history: List[np.ndarray] = field(default_factory=list)
    spins_raw_history: List[np.ndarray] = field(default_factory=list)
    messages_history: List[Dict[Tuple[int, int], str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.personas)
        if not self.inbox_agree:
            self.inbox_agree = [[] for _ in range(n)]
        if not self.inbox_disagree:
            self.inbox_disagree = [[] for _ in range(n)]
        if self.J.shape != (n, n):
            raise ValueError(f"J has shape {self.J.shape}, expected {(n, n)}")

    @property
    def n(self) -> int:
        return len(self.personas)


def _aggregate_spin(samples: List[int]) -> int:
    """Majority vote over k spin samples; tie-break with the first sample."""
    s = sum(samples)
    if s > 0:
        return +1
    if s < 0:
        return -1
    return samples[0] if samples else 0


def _spin_call_with_retry(
    persona: str,
    statement: str,
    agree: List[str],
    disagree: List[str],
    pi: Pi,
    mode: str = "subjective",
    choices: Optional[Dict[str, str]] = None,
    parse_retries: int = 2,
) -> int:
    """spin_sampler with parse retries; 0 on persistent failure."""
    for _ in range(parse_retries + 1):
        try:
            return spin_sampler(persona, statement, agree, disagree, pi,
                                mode=mode, choices=choices)
        except ValueError:
            continue
    return 0


def _build_inboxes(
    rep: Replica,
    step_msgs: Dict[Tuple[int, int], str],
    step_idx: int,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Split one step's messages into per-receiver (agree, disagree) inboxes,
    shuffled deterministically per (replica, step, receiver)."""
    n = rep.n
    agree: List[List[str]] = [[] for _ in range(n)]
    disagree: List[List[str]] = [[] for _ in range(n)]
    incoming: Dict[int, List[Tuple[int, str]]] = {}
    for (i, j), msg in sorted(step_msgs.items()):
        incoming.setdefault(j, []).append((i, msg))
    for j, items in incoming.items():
        random.Random(f"{rep.name}|{step_idx}|{j}").shuffle(items)
        for i, msg in items:
            sign = int(rep.J[i, j])
            if sign == +1:
                agree[j].append(msg)
            elif sign == -1:
                disagree[j].append(msg)
    return agree, disagree

# Legacy GPT-4o-mini and Gemma-3n-E4B runs sampled one message per recipient without recipient conditioning; Qwen, Llama, and async runs use one shared message per sender per round.
def _run_message_phase(
    replicas: List[Replica],
    pi: Pi,
    executor: ThreadPoolExecutor,
    step_idx: int,
) -> None:
    """One LM call per sender (the message doesn't depend on the recipient),
    broadcast along every edge with J[i, j] != 0."""
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
                pi=pi,
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
    replicas: List[Replica],
    pi: Pi,
    k: int,
    executor: ThreadPoolExecutor,
    step_idx: int,
    use_inbox: bool,
) -> None:
    """Sample k spins per (replica, agent) and aggregate to S(t) by majority."""
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
                    pi,
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


def _save_replica(rep: Replica, out_dir: Path, meta: Dict[str, Any]) -> None:
    """Write ``rep``'s full state to ``out_dir/<name>.json``."""
    payload = {
        "meta": {**meta, "name": rep.name, "n": rep.n},
        "personas": rep.personas,
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


def run_forward_dynamics(
    replicas: List[Replica],
    pi: Pi,
    num_steps: int = 8,
    k: int = 5,
    max_workers: int = 64,
    output_dir: Optional[str] = None,
    config_meta: Optional[Dict[str, Any]] = None,
) -> List[Replica]:
    """Advance every replica through ``num_steps`` rounds (step 0: spins with
    empty inboxes; then messages -> inboxes -> spins), then save one JSON per
    replica to ``output_dir``."""
    config_meta = dict(config_meta or {})

    print(
        f"[run_forward_dynamics] {len(replicas)} replicas, "
        f"num_steps={num_steps}, k={k}, max_workers={max_workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _run_spin_phase(replicas, pi, k, executor, step_idx=0, use_inbox=False)
        for t in tqdm(range(1, num_steps + 1), desc="steps", position=0):
            _run_message_phase(replicas, pi, executor, step_idx=t)
            _run_spin_phase(replicas, pi, k, executor, step_idx=t, use_inbox=True)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for rep in replicas:
            _save_replica(rep, out_path, config_meta)

    return replicas
