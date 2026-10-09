import numpy as np
import pytest
import synth

from gaitkeeper.compare import Candidate, _classify, verify
from gaitkeeper.inject import defects, history5, judge


def test_clean_log_passes_against_the_rounded_contract(clean_log, files, policy):
    rep = verify(clean_log, files, policy)
    assert rep.verdict == "PASS", rep.summary()
    assert {k: b.status for k, b in rep.boundaries.items()} == {
        "B": "pass",
        "A": "pass",
        "C": "pass",
    }
    classes = {t.term: t.tol_class for b in rep.boundaries.values() for t in b.terms}
    assert classes["target"] == "export_rounding" and classes["action"] == "exact"


def test_rounding_is_admitted_only_through_its_class(clean_log, files, policy):
    exact = files.copy()
    for p in (
        "control.actions.joint_pos.scale",
        "control.actions.joint_pos.offset",
        "control.default_joint_pos",
    ):
        exact.prov(p).resolution = 0.0
    rep = verify(clean_log, exact, policy)
    assert rep.boundaries["C"].status == "fail"


def test_wrong_policy_file_fails_b(clean_log, files):
    rep = verify(clean_log, files, synth.LinearPolicy(seed=4))
    assert rep.boundaries["B"].status == "fail" and rep.verdict == "CONTRACT"


@pytest.mark.parametrize("d", defects(), ids=lambda d: d.name)
def test_injected_defect_is_named_or_abstained(d, harness, files, policy):
    log = d.build(harness)
    contract = history5(files) if d.contract_variant == "history5" else files
    rep = verify(log, contract, None if d.contract_variant == "history5" else policy)
    r = judge(d, rep)
    assert r["correct"], (r, rep.summary())


def test_harness_clip_below_the_contract_limit_is_a_limit_difference(harness, files, policy):
    tr = harness.standard()
    t = tr["target"].astype(np.float64)
    eff = harness.effort(t)
    j = int(np.argmax(np.abs(eff).max(0)))
    c = 0.6 * float(np.abs(eff[:, j]).max())
    eff[:, j] = np.clip(eff[:, j], -c, c)
    tr.arrays["effort"] = eff.astype(np.float32)
    c2 = files.copy()
    lim = {n: 1e4 for n in harness.names}
    c2.set("model.effort_limit", lim, "user", "test: training limits above the harness clip")
    rep = verify(tr, c2, policy)
    assert rep.boundaries["C"].status == "pass", rep.summary()
    lim = [f for f in rep.findings if f.startswith("LIMIT_DIFFERENCE_ACTIVE")]
    assert lim and harness.names[j] in lim[0] and f"clips at {c:.4g}" in lim[0]


def test_gain_error_is_not_explained_away_as_a_clip(harness, files, policy):
    tr = harness.standard()
    eff = harness.effort(tr["target"].astype(np.float64), kp=harness.kp * 0.7)
    tr.arrays["effort"] = eff.astype(np.float32)
    rep = verify(tr, files, policy)
    assert rep.boundaries["C"].status == "fail"
    assert not any("clips at" in f for f in rep.findings)


def test_equally_simple_fits_abstain():
    o = np.ones((4, 2))
    e = np.zeros((4, 2))
    cands = [
        Candidate("explanation one", 1, lambda: np.ones((4, 2))),
        Candidate("explanation two", 1, lambda: np.ones((4, 2))),
    ]
    name, ambiguous, _ = _classify(o, e, lambda p: np.full_like(p, 1e-6), cands)
    assert name is None and ambiguous == ["explanation one", "explanation two"]


def test_simplest_fit_wins():
    o = np.ones((4, 2))
    e = np.zeros((4, 2))
    cands = [
        Candidate("simple", 1, lambda: np.ones((4, 2))),
        Candidate("complex", 5, lambda: np.ones((4, 2))),
    ]
    assert _classify(o, e, lambda p: np.full_like(p, 1e-6), cands)[0] == "simple"


def test_self_written_trace_is_labeled_and_capped(clean_log, files, policy):
    log = clean_log.slice_steps(0, clean_log.n_steps)
    log.meta["written_by_gaitkeeper_runner"] = True
    rep = verify(log, files, policy)
    assert rep.label == "SELF_CONSISTENT" and rep.evidence == "L1"


def test_controller_assumptions_are_reported(clean_log, files, policy):
    rep = verify(clean_log, files, policy)
    assert any(f.startswith("CONTROLLER_ASSUMED") for f in rep.findings)


def test_invalid_input(files):
    from gaitkeeper.trace import Trace

    rep = verify(Trace({"obs": np.zeros((3, 2))}, {}, "harness"), files)
    assert rep.verdict == "INVALID_INPUT"
