"""Shared helpers for the res/ notebooks: data loading, features, archetype
classifiers, the Adam logistic fit, and the discrete-model rollout."""

import json
from pathlib import Path

import numpy as np

MODELS = [("gpt-4o-mini", "GPT-4o-mini", "gpt"),
          ("google/gemma-3n-E4B-it", "Gemma-3n-E4B", "gma"),
          ("Qwen/Qwen3.5-9B", "Qwen3.5-9B", "qwn"),
          ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3-8B", "lma")]
REGIMES = ["subjective", "objective"]
N_AGENTS = 32
TAU = 0.2                        # split-band half-width for group archetypes

IND_CLASSES = ["Frozen", "Switcher", "Intermittent", "Oscillating"]
GRP_CLASSES = ["Persistent Split", "Convergence", "Divergence",
               "Majority Switch", "Persistent Majority"]
IND_ABBR = ["F", "S", "I", "O"]
GRP_ABBR = ["PS", "C", "D", "MS", "PM"]

# x-tensor column layout: [s_prev | bias p q p⊗q | drive_pos drive_neg]
S_PREV = 0
FIELD = slice(1, 17)
D_POS, D_NEG = 17, 18

BANK_DIR = {"objective": "obj", "subjective": "subj"}


def load_data(data_dir=None):
    """Everything the notebooks share: runs, question banks, persona features
    and the structural graph classification."""
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    runs = json.load(open(data_dir / "clean_runs.json"))
    banks, P = {}, {}
    for regime, sub in BANK_DIR.items():
        bank = {}
        for split in ("train", "test"):
            for line in (data_dir / sub / f"{split}.jsonl").read_text().splitlines():
                q = json.loads(line)
                bank[q["question"]] = {
                    "split": split, "qid": q["qid"],
                    "q": np.array(q["embedding_pca10"][:3]),
                    "truth": {"A": 1, "B": -1}.get(q.get("answer"), 0)}
        banks[regime] = bank
        items = sorted(json.load(open(data_dir / sub / "persona_embeddings.json"))
                       ["items"], key=lambda it: it["idx"])
        P[regime] = pca3(np.array([it["embedding"] for it in items], dtype=float))
    return runs, banks, P, graph_classes(runs, banks)


def opinions(run):
    """(9, 32) per-agent opinion path: the mean of the k = 5 thermal resamples."""
    return np.asarray(run["spins_raw_history"], dtype=float).mean(axis=2)


def pca3(X):
    """Top-3 PCA scores of X (rows = items): center, SVD, project."""
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def graph_classes(runs, banks):
    """J.tobytes() -> seen / train_only / fresh / lattice, identified from the
    graph's structure and which question splits it appears under."""
    appears, mats = {}, {}
    for r in runs:
        J = np.array(r["J"], dtype=np.int8)
        mats[J.tobytes()] = J
        appears.setdefault(J.tobytes(), set()).add(
            banks[r["mode"]][r["statement"]]["split"])

    def cls(key):
        if (mats[key] >= 0).all():
            return "lattice"
        if appears[key] == {"train"}:
            return "train_only"
        if appears[key] == {"test"}:
            return "fresh"
        return "seen"
    return {key: cls(key) for key in mats}


def static_field(P_reg, q):
    """(32, 16) static field block per agent: [bias | p | q | p ⊗ q]."""
    pq = (P_reg[:, :, None] * q[None, None, :]).reshape(N_AGENTS, -1)
    return np.concatenate([np.ones((N_AGENTS, 1)), P_reg,
                           np.tile(q, (N_AGENTS, 1)), pq], axis=1)


def build_episodes(model, regime, runs, banks, P, gclass):
    """All episodes of one model x regime, as aligned arrays: static fields phi
    (E, 32, 16), graphs J (E, 32, 32), single-vote spins (E, 9, 32), question
    split, graph class, a stable graph id, truth sign tau, qid and replica."""
    keys = sorted(gclass)
    gid = {key: i for i, key in enumerate(keys)}
    out = {k: [] for k in ("phi", "J", "spins", "split", "gcls", "gid",
                           "tau", "qid", "rep")}
    for r in runs:
        if r["model"] != model or r["mode"] != regime:
            continue
        info = banks[regime][r["statement"]]
        key = np.array(r["J"], dtype=np.int8).tobytes()
        out["phi"].append(static_field(P[regime], info["q"]))
        out["J"].append(np.array(r["J"], dtype=float))
        out["spins"].append(np.array(r["spins_history"], dtype=float))
        out["split"].append(info["split"])
        out["gcls"].append(gclass[key])
        out["gid"].append(gid[key])
        out["tau"].append(float(info["truth"]))
        out["qid"].append(info["qid"])
        out["rep"].append(r["replica"])
    return {k: np.array(v) for k, v in out.items()}


def build_xy(model, regime, runs, banks, P, gclass, T=8):
    """The transition tensors of one model x regime: x (E, 8, 32, 19) =
    [s_prev | bias p q p⊗q | drive_pos drive_neg] per (transition, agent),
    y (E, 8, 32) = the next spin s(t+1) (0 = unparsed), plus the episodes."""
    ep = build_episodes(model, regime, runs, banks, P, gclass)
    S, phi = ep["spins"], ep["phi"]
    s_in = S[:, :-1]
    pos = np.einsum("eij,etj->eti", np.maximum(ep["J"], 0), s_in)
    neg = np.einsum("eij,etj->eti", np.minimum(ep["J"], 0), s_in)
    x = np.concatenate([s_in[..., None],
                        np.broadcast_to(phi[:, None], (len(S), T) + phi.shape[1:]),
                        pos[..., None], neg[..., None]], axis=-1)
    return x, S[:, 1:].astype(int), ep


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def adam(grad_fn, x0, iters=1500, lr=0.05, tol=1e-9):
    """Full-batch Adam (b1 = 0.9, b2 = 0.999) with a plateau stop checked
    every 100 steps; grad_fn maps parameters to (loss, gradient)."""
    x, m, v = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
    b1, b2, e, prev = 0.9, 0.999, 1e-8, np.inf
    for t in range(1, iters + 1):
        loss, g = grad_fn(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + e)
        if t % 100 == 0:                             # plateau stop
            if prev - loss < tol * 100:
                break
            prev = loss
    return x


class Logistic:
    """Logistic regression fit by full-batch Adam on standardized columns:
    mean cross-entropy plus a small l2 (1e-3) on the persona-question field
    weights only (columns 1..15 of the [field | couplings] designs); the bias
    (column 0) and any coupling columns (16+) are unpenalized. Couplings are
    initialized at 0.1, everything else at zero. The objective is convex, so
    Adam reaches the unique optimum."""

    def __init__(self, l2=1e-3, iters=1500):
        self.l2, self.iters, self.w = l2, iters, None

    def fit(self, X, y01, offset=None):
        """offset: fixed per-row logit added to the fit (for two-stage fits
        of collinear designs); not part of the returned w."""
        off = 0.0 if offset is None else offset
        mu, sd = X.mean(axis=0), X.std(axis=0)
        mu[0], sd[0] = 0.0, 1.0
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (X - mu) / sd
        f = np.arange(1, min(FIELD.stop - 1, Z.shape[1]))

        def grad_fn(b):
            p = sigmoid(Z @ b + off)
            nll = (-(y01 * np.log(np.clip(p, 1e-12, None))
                     + (1 - y01) * np.log(np.clip(1 - p, 1e-12, None))).mean()
                   + 0.5 * self.l2 * float(b[f] @ b[f]))
            g = Z.T @ (p - y01) / len(Z)
            g[f] += self.l2 * b[f]
            return nll, g

        b0 = np.zeros(Z.shape[1])
        b0[FIELD.stop - 1:] = 0.1                    # coupling init
        b = adam(grad_fn, b0, iters=self.iters)
        # undo the standardization so w applies to raw design rows
        self.w = b / sd
        self.w[0] = b[0] - np.sum(b[1:] * mu[1:] / sd[1:])
        return self

    def predict_spin(self, X):
        return np.where(X @ self.w > 0.0, 1, -1)


def two_stage_fit(field, signed, unsigned, parsed, y01):
    """Two-stage fit for the collinear coupling designs ((|J| s) is a linear
    combination of the signed drives): beta_0 on the unsigned neighborhood
    first, then the signed betas with beta_0's drive as a fixed offset.
    Returns the full weight vector [field(16) | signed betas | beta_0]."""
    s1 = Logistic().fit(np.concatenate([field, unsigned[:, None]],
                                       axis=-1)[parsed], y01)
    beta_0 = s1.w[16]
    s2 = Logistic().fit(np.concatenate([field, signed], axis=-1)[parsed], y01,
                        offset=(beta_0 * unsigned)[parsed])
    return np.concatenate([s2.w, [beta_0]])


def drives(J_e, s, k):
    """Peer-drive columns (E, n, k) from a state s (E, n):
    k=1 [(J s)], k=3 [(J+ s) | (J- s) | (|J| s)]."""
    pos = np.einsum("eij,ej->ei", np.maximum(J_e, 0), s)
    neg = np.einsum("eij,ej->ei", np.minimum(J_e, 0), s)
    cols = [pos + neg] if k == 1 else [pos, neg, pos - neg]
    return np.stack(cols, axis=-1)


def discrete_rollout(phi, J_e, s0, w, k, T=8, mode="deterministic", rng=None):
    """Free-run the fitted discrete update for T steps from s0, feeding its own
    spins back in; deterministic (sign) or stochastic (Bernoulli). (E, T, n)."""
    h = phi @ w[:16]
    s = s0.astype(float)
    out = np.empty((len(s0), T, s0.shape[1]), dtype=int)
    for t in range(T):
        u = h + drives(J_e, s, k) @ w[16:]
        if mode == "deterministic":
            s = np.where(u > 0, 1.0, -1.0)
        else:
            p = 1.0 / (1.0 + np.exp(-np.clip(u, -30.0, 30.0)))
            s = np.where(rng.random(p.shape) < p, 1.0, -1.0)
        out[:, t] = s
    return out


def fcba(pred, y_te, s0, ep_mask):
    """Flip-and-class balanced accuracy (%), the paper's Appendix C.5 metric:
    the unweighted mean of accuracy over the four transition groups
    (flip vs stay) x (next spin +1 vs -1). A rule that never flips anyone, or
    always predicts one label, scores exactly 50."""
    s_prev = np.concatenate([s0[:, None].astype(int), y_te[:, :-1]], axis=1)
    defined = ep_mask[:, None, None] & (y_te != 0) & (s_prev != 0)
    correct = pred == y_te
    groups = [defined & (y_te != s_prev) & (y_te == v) for v in (1, -1)]
    groups += [defined & (y_te == s_prev) & (y_te == v) for v in (1, -1)]
    return 100 * float(np.mean([correct[g].mean() for g in groups if g.any()]))


def raw_acc(pred, y_te, ep_mask):
    """Raw accuracy (%): correct share over every scoreable transition
    (y != 0), with no flip or class reweighting."""
    m = ep_mask[:, None, None] & (y_te != 0)
    return 100 * float((pred == y_te)[m].mean())


def flip_counts(traj):
    """Sign switches per agent from opinion paths (E, S, n); zeros carry the
    last sign. -> (E, n) counts."""
    E, S, n = traj.shape
    prev = np.zeros((E, n))
    flips = np.zeros((E, n), dtype=int)
    for t in range(S):
        s = np.sign(traj[:, t])
        flips += (s != 0) & (prev != 0) & (s != prev)
        prev = np.where(s != 0, s, prev)
    return flips


def individual_archetypes(traj):
    """Flip-count class per agent, index into IND_CLASSES: F=0 frozen,
    1 switcher, 2 intermittent, >=3 oscillating."""
    return np.minimum(flip_counts(traj), 3)


def group_archetypes(n0, nT, closed=False):
    """Split-band class from the initial vs final net opinion n(t), index into
    GRP_CLASSES; closed=True counts |n| = TAU as split (single-vote spins
    never land on the edge, so both conventions agree there)."""
    scalar = np.isscalar(n0) or np.ndim(n0) == 0
    n0, nT = np.atleast_1d(n0).astype(float), np.atleast_1d(nT).astype(float)
    inside = (lambda n: np.abs(n) <= TAU) if closed else (lambda n: np.abs(n) < TAU)
    in0, inT = inside(n0), inside(nT)
    cls = np.full(n0.shape, 4)                       # persistent majority
    cls[in0 & inT] = 0                               # persistent split
    cls[in0 & ~inT] = 1                              # convergence
    cls[~in0 & inT] = 2                              # divergence
    cls[~in0 & ~inT & (n0 * nT < 0)] = 3             # majority switch
    return int(cls[0]) if scalar else cls
