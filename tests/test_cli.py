import numpy as np

from sim2sim.cli import main


def test_verify_exit_codes(tmp_path, clean_log, files, harness):
    files.save(tmp_path / "c.yaml")
    clean_log.save(tmp_path / "clean.npz")
    assert (
        main(
            [
                "verify",
                str(tmp_path / "clean.npz"),
                "--contract",
                str(tmp_path / "c.yaml"),
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    bad = clean_log.slice_steps(0, clean_log.n_steps)
    bad.arrays["target"] = bad["target"] - np.float32(0.1)
    bad.save(tmp_path / "bad.npz")
    assert main(["verify", str(tmp_path / "bad.npz"), "--contract", str(tmp_path / "c.yaml")]) == 2
    assert (tmp_path / "r.json").exists()


def test_inspect_writes_contract(tmp_path, files):
    files.save(tmp_path / "c.yaml")
    assert (
        main(["inspect", "--contract", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "o.yaml")])
        == 0
    )
    assert (tmp_path / "o.yaml").read_text().startswith("schema")
