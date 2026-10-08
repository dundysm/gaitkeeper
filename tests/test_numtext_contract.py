import numpy as np
import pytest

from sim2sim.contract import SCHEMA, Contract, compare_values
from sim2sim.readers.numtext import load_yaml_with_text, parse_csv_floats, resolution_of


@pytest.mark.parametrize(
    "text,res",
    [
        ("0.075", 0.0005),
        ("14.3", 0.05),
        ("1", 0.5),
        ("1.0", 0.05),
        ("2.5e-3", 0.00005),
        ("-0.25", 0.005),
    ],
)
def test_resolution_is_half_the_last_printed_digit(text, res):
    assert resolution_of(text) == pytest.approx(res)


def test_csv_floats_keep_their_text():
    vals = parse_csv_floats("0.548,0.35,14.251")
    assert [float(v) for v in vals] == [0.548, 0.35, 14.251]
    assert [v.text for v in vals] == ["0.548", "0.35", "14.251"]


def test_yaml_numbers_keep_their_text(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("scale: [0.07, 0.55]\nstep_dt: 0.02\n")
    y = load_yaml_with_text(p)
    assert y["scale"][0].text == "0.07" and float(y["scale"][0]) == 0.07


def test_contract_paths_provenance_and_roundtrip(tmp_path):
    c = Contract({"schema": SCHEMA})
    c.set("control.actuators.kp", {"a": 1.0}, "file", "onnx", resolution=0.0005)
    c.set("model.armature", None, "unknown", "not exported")
    assert c.get("control.actuators.kp.a") == 1.0
    assert c.prov("control.actuators.kp.a").source == "file"  # longest prefix
    assert c.resolution("control.actuators.kp") == 0.0005
    assert c.unknown_fields() == ["model.armature"]
    with pytest.raises(KeyError):
        c.set("nonsense.x", 1)
    c.save(tmp_path / "c.yaml")
    d = Contract.load(tmp_path / "c.yaml")
    assert d.get("control.actuators.kp") == {"a": 1.0}
    assert d.prov("control.actuators.kp").resolution == 0.0005


def test_compare_values_separates_rounding_from_disagreement():
    assert compare_values("x", 0.075, 0.0745, 0.0005, 0.0).kind == "rounding"
    assert compare_values("x", 0.07, 0.0745, 0.005, 0.0).kind == "rounding"
    assert compare_values("x", 0.07, 0.0745, 0.0005, 0.0).kind == "differs"
    d = compare_values("x", {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.2}, 0.0, 0.0)
    assert d.kind == "differs" and d.keys == ["b"]
    assert compare_values("x", None, 1.0, 0, 0).kind == "missing_left"
    assert np.isclose(compare_values("x", 1.0, 1.1, 0, 0).max_abs, 0.1)
