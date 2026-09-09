"""Schedule-driven fully asynchronous forward dynamics: events run strictly one
at a time (read board -> post message -> k-vote resample); silent agents keep their
opinion and last broadcast. Synchronous schema + event-level record, one JSON each."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from ..datagen.dynamics import (
    Replica,
    _aggregate_spin,
    _build_inboxes,
    _run_spin_phase,
    _spin_call_with_retry,
)
from ..datagen.samplers import Pi, message_sampler


@dataclass
class SeqReplica(Replica):
    """A replica driven by a pre-sampled firing schedule."""

    # firing_schedule[t-1]: ordered [cell, agent] events of step t >= 1.
    firing_schedule: List[List[List[int]]] = field(default_factory=list)
    # event_spins_history[t-1][e]: the spin recorded at event e of step t.
    event_spins_history: List[List[int]] = field(default_factory=list)
    # event_raw_history[t-1][e]: the k raw spin samples of event e.
    event_raw_history: List[List[List[int]]] = field(default_factory=list)


def _advance_replica_chain(
    rep: SeqReplica,
    pi: Pi,
    k: int,
    lm_pool: ThreadPoolExecutor,
    bar: Optional[tqdm],
) -> None:
    """Advance one replica event by event in its own chain thread; only LM
    calls go through the shared ``lm_pool``. The board (latest message per
    edge) is snapshotted to ``messages_history`` after each step."""
    board: Dict[Tuple[int, int], str] = {}
    nz_i, nz_j = np.nonzero(rep.J)
    recips: Dict[int, List[int]] = {}
    for i, j in zip(nz_i.tolist(), nz_j.tolist()):
        recips.setdefault(i, []).append(j)

    for t in range(1, len(rep.firing_schedule) + 1):
        events = rep.firing_schedule[t - 1]
        cur = rep.spins_history[-1].copy()
        # Raw rows default to the carried opinion; a firing agent's row is
        # overwritten by its LAST event's samples (all events are kept in
        # event_raw_history).
        raw = np.repeat(cur[:, None], k, axis=1).astype(np.int8)
        ev_spins: List[int] = []
        ev_raw: List[List[int]] = []
        for _cell, i in events:
            # Speak: post a message consistent with this agent's most recent
            # spin (cur[i] tracks within-step updates too).
            if i in recips:
                agree, disagree = _build_inboxes(rep, board, t)
                try:
                    msg = lm_pool.submit(
                        message_sampler,
                        persona=rep.personas[i],
                        question=rep.statement,
                        messages_agree=agree[i],
                        messages_disagree=disagree[i],
                        current_spin=int(cur[i]),
                        pi=pi,
                        mode=rep.mode,
                        choices=rep.choices,
                    ).result()
                except Exception as exc:  # noqa: BLE001
                    msg = f"[error: {exc}]"
                for j in recips[i]:
                    board[(i, j)] = msg
            agree, disagree = _build_inboxes(rep, board, t)
            futs = [
                lm_pool.submit(
                    _spin_call_with_retry,
                    rep.personas[i],
                    rep.statement,
                    agree[i],
                    disagree[i],
                    pi,
                    rep.mode,
                    rep.choices,
                )
                for _ in range(k)
            ]
            samples: List[int] = []
            for f in futs:
                try:
                    samples.append(f.result())
                except Exception:  # noqa: BLE001
                    samples.append(0)
            raw[i, :] = samples
            s = _aggregate_spin(samples)
            cur[i] = s
            ev_spins.append(int(s))
            ev_raw.append([int(x) for x in samples])
            if bar is not None:
                bar.update(1)

        rep.event_spins_history.append(ev_spins)
        rep.event_raw_history.append(ev_raw)
        rep.spins_raw_history.append(raw)
        rep.spins_history.append(cur.astype(np.int8))
        rep.messages_history.append(dict(sorted(board.items())))


def _save_replica(rep: SeqReplica, out_dir: Path, meta: Dict[str, Any]) -> None:
    """Write ``rep``'s full state (synchronous schema + schedule and
    event-level fields) to ``out_dir/<name>.json``."""
    payload = {
        "meta": {**meta, "name": rep.name, "n": rep.n},
        "personas": rep.personas,
        "statement": rep.statement,
        "mode": rep.mode,
        "choices": rep.choices,
        "J": rep.J.astype(int).tolist(),
        "firing_schedule": [
            [[int(c), int(a)] for c, a in step] for step in rep.firing_schedule
        ],
        "event_spins_history": [list(s) for s in rep.event_spins_history],
        "event_raw_history": [
            [list(e) for e in step] for step in rep.event_raw_history
        ],
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


def run_seq_forward_dynamics(
    replicas: List[SeqReplica],
    pi: Pi,
    k: int = 5,
    max_workers: int = 256,
    output_dir: Optional[str] = None,
    config_meta: Optional[Dict[str, Any]] = None,
    max_chains: int = 1024,
) -> List[SeqReplica]:
    """Advance every replica through its firing schedule (step 0: all agents,
    empty boards; then one chain thread per replica sharing one LM pool) and
    save one JSON per replica to ``output_dir``. Failed chains are not saved."""
    config_meta = dict(config_meta or {})

    total_events = sum(
        sum(len(s) for s in rep.firing_schedule) for rep in replicas
    )
    print(
        f"[run_seq_forward_dynamics] {len(replicas)} replicas, "
        f"{total_events:,} scheduled events, k={k}, "
        f"max_workers={max_workers} (schedule-driven sequential events)",
        flush=True,
    )

    failed: Dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as lm_pool:
        _run_spin_phase(replicas, pi, k, lm_pool, step_idx=0, use_inbox=False)

        with tqdm(total=total_events, desc="events (sequential)") as bar, \
                ThreadPoolExecutor(
                    max_workers=max(1, min(len(replicas), max_chains)),
                ) as chain_pool:
            futs = {
                chain_pool.submit(
                    _advance_replica_chain, rep, pi, k, lm_pool, bar,
                ): rep
                for rep in replicas
            }
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc is not None:
                    failed[futs[fut].name] = exc

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for rep in replicas:
            if rep.name not in failed:
                _save_replica(rep, out_path, config_meta)

    if failed:
        for name, exc in list(failed.items())[:10]:
            print(f"[run_seq_forward_dynamics] chain FAILED: {name}: {exc}",
                  flush=True)
        raise RuntimeError(
            f"{len(failed)}/{len(replicas)} replica chains failed (not saved)"
        )
    return replicas
