"""Run an exported policy: on recorded inputs (boundary B) or in a closed loop.

Recurrent graphs take their state as extra inputs and return the next state as
extra outputs. The pairing is in-order (extra input k goes with extra output k)
unless the contract's ``policy_io.graph.recurrent`` names it. State starts at
zero and is zeroed again at every reset.

A port that runs two graphs and picks one by the command each step (a walking and a
standing policy) states it as ``policy_io.graph.switch``; ``load_policy`` builds either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def _shape(dims: list[Any]) -> tuple[int, ...]:
    return tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in dims)


class OnnxPolicy:
    def __init__(self, path: str | Path, recurrent: list[dict[str, str]] | None = None) -> None:
        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        self.input_name = ins[0].name
        self.output_name = outs[0].name
        self.n_in = int(ins[0].shape[-1])
        self.n_out = int(outs[0].shape[-1])
        if recurrent is None:
            recurrent = [{"in": i.name, "out": o.name} for i, o in zip(ins[1:], outs[1:])]
        if len(ins) - 1 != len(recurrent):
            raise NotImplementedError(
                f"{len(ins) - 1} extra inputs but {len(recurrent)} state pairs"
            )
        self.recurrent = recurrent
        shapes = {i.name: _shape(i.shape) for i in ins}
        types = {i.name: i.type for i in ins}
        self._state_spec = {
            r["in"]: (shapes[r["in"]], np.float64 if "double" in types[r["in"]] else np.float32)
            for r in recurrent
        }
        self._out_names = [self.output_name] + [r["out"] for r in recurrent]
        self.state: dict[str, np.ndarray] = {}
        self.reset()

    @property
    def is_recurrent(self) -> bool:
        return bool(self.recurrent)

    def reset(self) -> None:
        self.state = {k: np.zeros(s, dtype=t) for k, (s, t) in self._state_spec.items()}

    def step(self, obs: np.ndarray) -> np.ndarray:
        """One closed-loop inference; carries recurrent state."""
        feed = {self.input_name: np.asarray(obs, dtype=np.float32).reshape(1, -1)}
        feed.update(self.state)
        res = self.sess.run(self._out_names, feed)
        for r, v in zip(self.recurrent, res[1:]):
            self.state[r["in"]] = v
        return np.asarray(res[0][0], dtype=np.float64)

    def __call__(self, obs: np.ndarray, reset: np.ndarray | None = None) -> np.ndarray:
        """Rows in order, one at a time, as the env ran them."""
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((obs.shape[0], self.n_out), dtype=np.float32)
        self.reset()
        for k in range(obs.shape[0]):
            if reset is not None and k > 0 and reset[k]:
                self.reset()
            out[k] = self.step(obs[k])
        return out


class SwitchedPolicy:
    """Two exported graphs and a rule choosing one each step by the harness's command:
    ``above`` while the command's norm exceeds ``threshold``, else ``below`` (gr00t_wbc's
    walking and balance policies). The caller sets ``command`` before each ``step``."""

    def __init__(self, models: dict[str, OnnxPolicy], rule: dict[str, Any]) -> None:
        self.models = models
        self.rule = rule
        self.threshold = float(rule["threshold"])
        self.by = rule.get("by", "command_norm")
        first = models["above"]
        self.path = first.path
        self.n_in, self.n_out = first.n_in, first.n_out
        self.command = np.zeros(3)

    @property
    def is_recurrent(self) -> bool:
        return any(m.is_recurrent for m in self.models.values())

    def reset(self) -> None:
        for m in self.models.values():
            m.reset()

    def choose(self, command: np.ndarray) -> str:
        c = np.asarray(command, dtype=np.float64)[:3]
        v = float(np.linalg.norm(c[:2] if self.by == "command_planar_norm" else c))
        return "above" if v > self.threshold else "below"

    def step(self, obs: np.ndarray) -> np.ndarray:
        return self.models[self.choose(self.command)].step(obs)

    def __call__(
        self, obs: np.ndarray, reset: np.ndarray | None = None, command: np.ndarray | None = None
    ) -> np.ndarray:
        """Rows in order; ``command`` per row picks the graph (without it, ``above``)."""
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((obs.shape[0], self.n_out), dtype=np.float32)
        self.reset()
        for k in range(obs.shape[0]):
            if reset is not None and k > 0 and reset[k]:
                self.reset()
            self.command = (
                np.full(3, np.inf) if command is None else np.asarray(command[k], dtype=float)
            )
            out[k] = self.step(obs[k])
        return out


def load_policy(contract: Any, path: str | Path) -> OnnxPolicy | SwitchedPolicy:
    """The policy a contract runs from ``path``: one graph, or the pair its
    ``policy_io.graph.switch`` names (found next to ``path``)."""
    rec = contract.get("policy_io.graph.recurrent", None) or None
    sw = contract.get("policy_io.graph.switch", None)
    if not sw:
        return OnnxPolicy(path, rec)
    d = Path(path).parent
    files = {k: d / Path(str(sw[k])).name for k in ("above", "below")}
    missing = [str(f) for f in files.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"the contract switches between {files['above'].name} and {files['below'].name} "
            f"by the command; not found next to {path}: {missing}"
        )
    return SwitchedPolicy({k: OnnxPolicy(f, rec) for k, f in files.items()}, sw)
