"""Run an exported policy on recorded inputs (boundary B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        if len(ins) != 1:
            raise NotImplementedError("recurrent or multi-input policies need state lifting")
        self.input_name = ins[0].name
        self.n_in = int(ins[0].shape[-1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        out = np.empty((obs.shape[0], self.sess.get_outputs()[0].shape[-1]), dtype=np.float32)
        for k in range(obs.shape[0]):  # one row at a time, as the env ran it
            out[k] = self.sess.run(None, {self.input_name: obs[k : k + 1]})[0][0]
        return out
