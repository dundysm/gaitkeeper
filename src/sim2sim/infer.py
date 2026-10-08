"""Observation layout from a trace (plan section 7.4).

From ``(obs, action)`` alone: columns that equal an earlier action exactly give
the last action term and the history length; columns that hold another
column's previous value exactly give the history layout (term major or frame
major) and order; the reset rows give the history init rule. With raw state
(``qpos``, ``qvel``, ``command``) each newest column is matched against raw
signals by an affine fit, which labels the term and gives its scale, sign,
offset and joint order. Without raw state a few weak labels remain (a unit
norm 3 vector, a piecewise constant 3 vector, a unit circle pair).

The answer is withheld, not guessed: while any column is ambiguous (it is
constant, or copies or matches more than one candidate) no layout or term
boundaries are emitted. A constant command, little yaw or a joint that never
moves makes columns ambiguous, so an under-excited trace abstains (AT4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .tables import TABLES
from .terms import StateLayout, quat_to_mat
from .trace import Trace

EXACT_REL = 1e-6  # history copies and the last action are exact float32 copies
FIT_DEFICIT = 1e-6  # 1 - r^2 below which a column is an affine image of a raw signal
MAX_LAG = 8

SIGNAL_TERM = {
    "projected_gravity": "projected_gravity",
    "base_ang_vel": "base_ang_vel",
    "base_ang_vel_world": "base_ang_vel (world frame)",
    "command": "velocity_commands",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
}


@dataclass
class Block:
    start: int  # first column of the term (all history slots)
    width: int  # columns per slot
    label: str
    source: str  # equality | raw | obs | none
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inference:
    status: str = "abstained"  # inferred | partial (some spans unlabeled) | abstained
    n_cols: int = 0
    history_length: int | None = None
    layout: str | None = None  # term_major | time_major | none
    order: str | None = None  # oldest_first | newest_first
    frame_width: int | None = None
    init: str | None = None
    blocks: list[Block] = field(default_factory=list)
    last_action_cols: int = 0
    ambiguous_cols: list[int] = field(default_factory=list)
    unlabeled_cols: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    raw_state: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def lines(self) -> list[str]:
        out = [f"infer: {self.status} | {self.n_cols} observation columns"]
        if self.history_length is not None:
            out.append(
                f"  history {self.history_length}"
                + (f", {self.layout}" if self.layout else "")
                + (f", {self.order}" if self.order and self.history_length > 1 else "")
                + (f", frame width {self.frame_width}" if self.frame_width else "")
                + (f", init {self.init}" if self.init else "")
            )
        out.append(f"  last action columns (exact copies): {self.last_action_cols}")
        if self.ambiguous_cols:
            out.append(f"  ambiguous columns: {_ranges(self.ambiguous_cols)}")
        for b in self.blocks:
            h = self.history_length or 1
            end = b.start + b.width * (h if self.layout == "term_major" else 1) - 1
            d = ", ".join(f"{k} {v}" for k, v in b.detail.items() if k not in ("columns",))
            out.append(
                f"  cols {b.start:4d}..{end:<4d} width {b.width:<3d} {b.label}"
                + (f" ({b.source}{'; ' + d if d else ''})" if b.source != "none" else "")
            )
        if self.unlabeled_cols:
            out.append(f"  unlabeled columns: {_ranges(self.unlabeled_cols)}")
        for r in self.reasons:
            out.append(f"  {'abstain' if self.status == 'abstained' else 'note'}: {r}")
        return out


def _ranges(cols: list[int]) -> str:
    cols = sorted(cols)
    parts, s = [], None
    for i, c in enumerate(cols):
        if s is None:
            s = c
        if i + 1 == len(cols) or cols[i + 1] != c + 1:
            parts.append(f"{s}" if s == c else f"{s}..{c}")
            s = None
    return ", ".join(parts)


def _episode_step(trace: Trace) -> np.ndarray:
    ep = trace.get("episode_step")
    if ep is not None:
        return np.asarray(ep, int)
    reset = np.asarray(trace["reset"], bool)
    out = np.zeros(len(reset), int)
    for k in range(len(reset)):
        out[k] = 0 if (reset[k] or k == 0) else out[k - 1] + 1
    return out


def _close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per column: every row of a equals b to float32 copy precision."""
    return np.all(np.abs(a - b) <= EXACT_REL * np.maximum(1.0, np.abs(b)), axis=0)


def _last_action(obs, act, steady, res: Inference) -> dict[int, tuple[int, int]]:
    """Columns that copy an earlier action: col -> (action index, lag)."""
    la: dict[int, tuple[int, int]] = {}
    amb: set[int] = set()
    for lag in range(1, MAX_LAG + 1):
        rows = steady
        cur = obs[rows]
        past = act[rows - lag]
        for j in range(act.shape[1]):
            for c in np.where(_close(cur, past[:, j : j + 1]))[0]:
                c = int(c)
                if c in la and la[c] != (j, lag):
                    amb.add(c)
                la.setdefault(c, (j, lag))
    for c in amb:
        la.pop(c, None)
    res.ambiguous_cols.extend(sorted(amb))
    return la


def _successors(obs32, steady, var0) -> tuple[dict[int, int], set[int]]:
    """succ[c] = c' when obs[t, c] == obs[t - 1, c'] on every steady row, uniquely."""
    prev = obs32[steady - 1]
    cur = obs32[steady]
    keys: dict[bytes, list[int]] = {}
    for c in range(prev.shape[1]):
        if not var0[c]:
            keys.setdefault(prev[:, c].tobytes(), []).append(c)
    succ: dict[int, int] = {}
    amb: set[int] = set()
    for c in range(cur.shape[1]):
        if var0[c]:
            continue
        hit = keys.get(cur[:, c].tobytes(), [])
        if len(hit) == 1 and hit[0] != c:
            succ[c] = hit[0]
        elif len(hit) > 1:
            amb.add(c)
    return succ, amb


def _walk_term_major(succ, D, H, order) -> list[tuple[int, int]] | None:
    """Term blocks (start, width) from the shift structure, or None if it does not tile."""
    step = {c: abs(s - c) for c, s in succ.items()}
    blocks, c = [], 0
    while c < D:
        c2 = c
        while c2 < D and c2 not in step:
            c2 += 1
        if c2 >= D:
            return None
        d = step[c2]
        if (order == "oldest_first" and c2 != c) or (order == "newest_first" and c2 != c + d):
            return None
        if c + H * d > D:
            return None
        sign = 1 if order == "oldest_first" else -1
        older = range(c, c + (H - 1) * d) if sign > 0 else range(c + d, c + H * d)
        if any(succ.get(k) != k + sign * d for k in older):
            return None
        blocks.append((c, d))
        c += H * d
    return blocks


def _signals(trace: Trace) -> dict[str, tuple[np.ndarray, list[str] | None]]:
    lay = StateLayout.from_meta(trace.meta)
    qpos = np.asarray(trace["qpos"], np.float64)
    qvel = np.asarray(trace["qvel"], np.float64)
    quat = qpos[:, 3:7]
    if lay.quat_order == "xyzw":
        quat = quat[:, [3, 0, 1, 2]]
    R = quat_to_mat(quat)
    w = qvel[:, 3:6]
    if lay.ang_vel_frame == "world":
        w_body = np.einsum("tji,tj->ti", R, w)
        w_world = w
    else:
        w_body = w
        w_world = np.einsum("tij,tj->ti", R, w)
    grav = np.einsum("tji,j->ti", R, np.array([0.0, 0.0, -1.0]))
    names = list(lay.joint_names)
    return {
        "projected_gravity": (grav, None),
        "base_ang_vel": (w_body, None),
        "base_ang_vel_world": (w_world, None),
        "command": (np.asarray(trace["command"], np.float64), None),
        "joint_pos": (qpos[:, 7:], names),
        "joint_vel": (qvel[:, 6:], names),
    }


def _raw_match(obs, cols, sig, rows) -> tuple[dict[int, tuple[str, int, float, float]], set[int]]:
    """Affine match of each column to exactly one raw signal column."""
    names, mats = [], []
    for k, (x, _) in sig.items():
        for i in range(x.shape[1]):
            names.append((k, i))
        mats.append(x)
    X = np.concatenate(mats, 1)[rows]
    Y = obs[rows][:, cols]
    Xc = X - X.mean(0)
    Oc = Y - Y.mean(0)
    xs = np.linalg.norm(Xc, axis=0)
    os_ = np.linalg.norm(Oc, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Oc.T @ Xc) / np.outer(os_, xs)
    r[:, xs < 1e-9] = 0.0
    deficit = 1 - r**2
    out: dict[int, tuple[str, int, float, float]] = {}
    amb: set[int] = set()
    for a, c in enumerate(cols):
        hit = np.where(deficit[a] < FIT_DEFICIT)[0]
        if len(hit) > 1:
            amb.add(c)
        elif len(hit) == 1:
            k, i = names[hit[0]]
            x = X[:, hit[0]]
            A = np.stack([x, np.ones_like(x)], 1)
            (s, b), *_ = np.linalg.lstsq(A, Y[:, a], rcond=None)
            out[c] = (k, i, float(s), float(b))
    return out, amb


def _describe_raw(matches, sig) -> tuple[str, dict[str, Any]] | None:
    kinds = {m[0] for m in matches}
    if len(kinds) != 1:
        return None
    k = kinds.pop()
    idx = [m[1] for m in matches]
    sc = np.array([m[2] for m in matches])
    off = np.array([m[3] for m in matches])
    d: dict[str, Any] = {}
    s0 = float(np.median(sc))
    if np.allclose(sc, s0, rtol=1e-3):
        d["scale"] = float(f"{s0:.4g}")
    else:
        d["scale"] = [float(f"{v:.4g}") for v in sc]
    if np.all(np.abs(off) <= 1e-4 * np.maximum(1, np.abs(sc))):
        d["offset"] = "none"
    elif k == "joint_pos":
        d["offset"] = "per joint (default pose subtracted)"
    else:
        d["offset"] = [float(f"{v:.4g}") for v in off]
    names = sig[k][1]
    if names is not None:
        order = [names[i] for i in idx]
        if len(set(order)) != len(order):
            return None
        known = [t for t, tab in TABLES.items() if tab == order]
        d["joint_order"] = (
            known[0]
            if known
            else ("trace order" if idx == sorted(idx) and len(idx) == len(names) else order)
        )
    elif idx != list(range(len(idx))):
        d["axes"] = idx
    return SIGNAL_TERM[k], d


def _obs_label(x: np.ndarray, d: int) -> str | None:
    """Weak labels from observations alone (newest slot of one block)."""
    if d == 3:
        n = np.linalg.norm(x, axis=1)
        if n.mean() > 1e-6 and n.std() <= 1e-3 * n.mean():
            return f"unit norm 3 vector, norm {n.mean():.3g} (gravity direction)"
        changes = np.any(np.diff(x, axis=0) != 0, axis=1).mean()
        if changes < 0.05:
            return "piecewise constant 3 vector (command)"
    if d == 2:
        n = np.linalg.norm(x, axis=1)
        on = n > 1e-6
        if on.mean() >= 0.1 and n[on].std() <= 1e-3 * n[on].mean():
            if on.all():
                return "unit circle pair (phase clock)"
            return f"unit circle pair, zero on {1 - on.mean():.0%} of rows (phase clock with a stand rule)"
    return None


def infer(trace: Trace, use_raw: bool = True) -> Inference:
    obs32 = np.asarray(trace["obs"], np.float32)
    obs = obs32.astype(np.float64)
    act = np.asarray(trace["action"], np.float32).astype(np.float64)
    T, D = obs.shape
    res = Inference(n_cols=D)
    ep = _episode_step(trace)
    steady = np.where(ep > 2 * MAX_LAG)[0]
    if len(steady) < 10:
        res.reasons.append("fewer than 10 rows past the history warm up")
        return res
    var0 = np.all(obs32[steady] == obs32[steady][:1], axis=0)
    if var0.any():
        res.ambiguous_cols.extend(int(c) for c in np.where(var0)[0])
        res.reasons.append(
            f"{int(var0.sum())} columns never change (constant command, idle joint or unused slot)"
        )

    # Last action and history length.
    la = _last_action(obs, act, steady, res)
    res.last_action_cols = len(la)
    lags = sorted({v[1] for v in la.values()})
    H_eq = len(lags) if lags and lags == list(range(1, len(lags) + 1)) else None
    if lags and H_eq is None:
        res.reasons.append(f"last action lags {lags} are not 1..H")

    # Shift structure.
    succ, amb = _successors(obs32, steady, var0)
    res.ambiguous_cols.extend(sorted(amb))
    hops = 0
    for c in succ:
        n, k = 0, c
        while k in succ and n <= D:
            k, n = succ[k], n + 1
        hops = max(hops, n)
    H_sh = hops + 1
    if H_eq is not None and H_eq != H_sh:
        res.reasons.append(f"history length {H_eq} from the last action but {H_sh} from shifts")
    if H_eq is None and not succ:
        res.reasons.append(
            "no column copies an action or an earlier column exactly (noise added after"
            " history?); history length and boundaries undetermined"
        )
        res.ambiguous_cols = sorted(set(res.ambiguous_cols))
        return res
    H = H_eq if H_eq is not None else H_sh
    res.history_length = H

    fresh: list[tuple[int, int]] = []  # (start, width) of the newest slot of each block
    blocks: list[tuple[int, int]] | None = None
    if H == 1:
        res.layout, res.order = "none", None
        fresh = [(0, D)]
    else:
        steps = np.array([s - c for c, s in succ.items()])
        if (steps > 0).all():
            res.order = "oldest_first"
        elif (steps < 0).all():
            res.order = "newest_first"
        else:
            res.reasons.append("history shifts point both ways")
        widths = sorted({abs(int(s)) for s in steps})
        if res.order and len(widths) == 1 and widths[0] * H == D:
            res.layout, res.frame_width = "time_major", widths[0]
            W = widths[0]
            fresh = [(D - W, W)] if res.order == "oldest_first" else [(0, W)]
        elif res.order:
            blocks = _walk_term_major(succ, D, H, res.order)
            if blocks is None:
                res.reasons.append("history blocks do not tile the observation")
            else:
                res.layout = "term_major"
                for c, d in blocks:
                    fresh.append((c + (H - 1) * d, d) if res.order == "oldest_first" else (c, d))
        if res.layout:
            res.init = _init_rule(obs, np.asarray(trace["reset"], bool), H, res, blocks)

    # Labels on the newest slot.
    fresh_cols = [c for s, w in fresh for c in range(s, s + w)]
    raw: dict[int, tuple[str, int, float, float]] = {}
    sig = None
    if use_raw and all(k in trace.arrays for k in ("qpos", "qvel", "command")):
        try:
            sig = _signals(trace)
        except (KeyError, ValueError):
            sig = None
    if sig is not None:
        res.raw_state = True
        cand = [c for c in fresh_cols if c not in la and not var0[c]]
        rows = np.arange(T)
        raw, amb_raw = _raw_match(obs, cand, sig, rows)
        res.ambiguous_cols.extend(sorted(amb_raw))
    la_new = {c: v for c, v in la.items() if v[1] == 1}

    def label_span(s: int, w: int) -> list[Block]:
        """Split one newest slot span into labeled runs."""
        out: list[Block] = []
        c = s
        while c < s + w:
            if c in la_new:
                kind = "la"
            elif c in raw:
                kind = raw[c][0]
            else:
                kind = None
            e = c + 1
            while e < s + w:
                k2 = "la" if e in la_new else (raw[e][0] if e in raw else None)
                if k2 != kind:
                    break
                e += 1
            out.append(_block(c, e - c, kind, la_new, raw, sig, obs))
            c = e
        return out

    if res.layout == "term_major" and blocks is not None:
        for (c, d), (s, w) in zip(blocks, fresh):
            parts = label_span(s, w)
            if len(parts) != 1:
                res.reasons.append(f"block at column {c} mixes signals {[p.label for p in parts]}")
                b = Block(c, d, "mixed", "none")
            else:
                b = parts[0]
                b.start = c
            if b.source == "none":
                lab = _obs_label(obs[:, s : s + w], d)
                if lab:
                    b.label, b.source = lab, "obs"
            res.blocks.append(b)
        _rate_pairs(res.blocks, fresh, obs)
    elif fresh:
        s, w = fresh[0]
        for b in label_span(s, w):
            if b.source == "none":
                lab = _obs_label(obs[:, b.start : b.start + b.width], b.width)
                if lab:
                    b.label, b.source = lab, "obs"
            res.blocks.append(b)
    res.unlabeled_cols = [
        c
        for b in res.blocks
        if b.source == "none"
        for c in range(b.start, b.start + b.width)
        if c not in res.ambiguous_cols
    ]
    if res.unlabeled_cols and res.layout != "term_major":
        for b in res.blocks:
            if b.source == "none":
                b.label = "unlabeled span (one or more terms)"
    res.ambiguous_cols = sorted(set(res.ambiguous_cols))
    if res.ambiguous_cols:
        res.reasons.insert(0, f"{len(res.ambiguous_cols)} ambiguous columns remain")
    blocking = res.ambiguous_cols or any(
        r.startswith(("history", "last action lags", "block at")) for r in res.reasons
    )
    if blocking or not res.layout:
        res.status = "abstained"
        # Withhold boundaries and layout; keep what exact equality established.
        res.blocks = []
        res.layout = res.order = res.frame_width = res.init = None
    elif res.unlabeled_cols:
        # Structure is settled; the unlabeled spans are reported as such, with no
        # claim about terms inside them.
        res.status = "partial"
    else:
        res.status = "inferred"
    return res


def _block(c, w, kind, la_new, raw, sig, obs) -> Block:
    if kind == "la":
        idx = [la_new[k][0] for k in range(c, c + w)]
        d: dict[str, Any] = {"lag": 1}
        if idx != list(range(idx[0], idx[0] + w)):
            d["action_index"] = idx
        return Block(c, w, "last_action", "equality", d)
    if kind is not None and sig is not None:
        got = _describe_raw([raw[k] for k in range(c, c + w)], sig)
        if got is not None:
            return Block(c, w, got[0], "raw", got[1])
    return Block(c, w, "unlabeled", "none")


def _rate_pairs(blocks: list[Block], fresh: list[tuple[int, int]], obs: np.ndarray) -> None:
    """Weak label: an unlabeled block that tracks the per-step change of another.

    One scalar ratio for the whole block, relative residual below 0.6 (R10). It
    names a likely position and velocity pair, nothing more.
    """
    open_ = [(b, f) for b, f in zip(blocks, fresh) if b.source == "none"]
    for bv, (sv, wv) in open_:
        best = None
        for bp, (sp, wp) in open_:
            if bp is bv or wp != wv or bp.label.startswith("rate"):
                continue
            fd = np.diff(obs[:, sp : sp + wp], axis=0)
            v = obs[1:, sv : sv + wv]
            den = float((fd * fd).sum())
            if den <= 0:
                continue
            k = float((fd * v).sum() / den)
            resid = float(np.linalg.norm(v - k * fd) / max(np.linalg.norm(v), 1e-12))
            if resid < 0.6 and (best is None or resid < best[2]):
                best = (bp, k, resid)
        if best is not None:
            bp, k, resid = best
            bv.label = f"rate of the block at column {bp.start} (likely joint velocity)"
            bv.source, bv.detail = (
                "obs",
                {"ratio_per_step": round(k, 3), "residual": round(resid, 2)},
            )
            if bp.source == "none":
                bp.label, bp.source = "integrates to the rate block (likely joint position)", "obs"


def _init_rule(obs, reset, H, res: Inference, blocks) -> str | None:
    """History contents on reset rows: repeat_first, zeros, or undetermined."""
    seen: set[str] = set()
    for r in np.where(reset)[0]:
        if res.layout == "time_major":
            W = res.frame_width or 0
            groups = [obs[r].reshape(H, W)]
        else:
            groups = [obs[r, c : c + H * d].reshape(H, d) for c, d in blocks or []]
        for g in groups:
            if res.order == "newest_first":
                g = g[::-1]
            newest, older = g[-1], g[:-1]
            informative = np.abs(newest) > 1e-6
            if not informative.any():
                continue
            if np.allclose(older[:, informative], newest[informative], rtol=1e-6, atol=1e-7):
                seen.add("repeat_first")
            elif np.allclose(older, 0.0):
                seen.add("zeros")
            else:
                seen.add("other")
    if not seen:
        return None
    if len(seen) > 1:
        res.reasons.append(f"history init differs across terms {sorted(seen)}")
        return "conflicting"
    return seen.pop()
