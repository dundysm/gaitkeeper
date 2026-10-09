"""Files and settings from before the rename (sim2sim to gaitkeeper) still load."""

import json

import numpy as np
import yaml

from gaitkeeper.contract import SCHEMA, Contract
from gaitkeeper.env import env
from gaitkeeper.fixtures import data_dir
from gaitkeeper.trace import TRACE_SCHEMA, Trace


def test_trace_with_the_old_schema_and_runner_flag_loads(tmp_path, clean_log):
    p = tmp_path / "log.npz"
    clean_log.save(p)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    meta["schema"] = "sim2sim/trace/v1"
    meta.pop("written_by_gaitkeeper_runner", None)
    meta["written_by_sim2sim_runner"] = True
    old = tmp_path / "old.npz"
    np.savez_compressed(old, meta=np.array(json.dumps(meta)), **arrays)
    t = Trace.load(old)
    assert t.meta["schema"] == TRACE_SCHEMA
    assert t.self_consistent_only
    assert np.array_equal(t["obs"], clean_log["obs"])


def test_contract_with_the_old_schema_loads(tmp_path, truth):
    p = tmp_path / "contract.yaml"
    truth.save(p)
    d = yaml.safe_load(p.read_text())
    d["schema"] = "sim2sim/contract/v3"
    p.write_text(yaml.safe_dump(d))
    assert Contract.load(p).data["schema"] == SCHEMA


def test_old_env_names_and_cache_dir_are_read(tmp_path, monkeypatch):
    for k in ("GAITKEEPER_DATA", "SIM2SIM_DATA"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SIM2SIM_DEBUG", "1")
    assert env("DEBUG") == "1"
    monkeypatch.setenv("GAITKEEPER_DEBUG", "2")
    assert env("DEBUG") == "2"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert data_dir() == tmp_path / "gaitkeeper"
    (tmp_path / "sim2sim").mkdir()
    assert data_dir() == tmp_path / "sim2sim"
    (tmp_path / "gaitkeeper").mkdir()
    assert data_dir() == tmp_path / "gaitkeeper"
