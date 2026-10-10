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


def test_a_clipping_deploy_passes_c_when_the_contract_records_the_clip(harness, files, policy):
    """Boundary C applies the contract's processed-target clip, as the runner does."""
    names = list(files.get("policy_io.joints.names"))
    tr = harness.standard()
    t = tr["target"].astype(np.float64)
    of = np.array([files.get("control.actions.joint_pos.offset")[n] for n in names])
    tn = list(tr.meta.get("target_joint_names", names))
    col = [names.index(n) for n in tn]
    # A band narrow enough that every joint clips on some rows.
    half = 0.5 * np.abs(t - of[col][None]).max(0)
    lo, hi = of[col] - half, of[col] + half
    tc = np.clip(t, lo[None], hi[None])
    tr.arrays["target"] = tc.astype(np.float32)
    tr.arrays["effort"] = harness.effort(tc).astype(np.float32)
    assert (tc != t).any(0).all()

    pairs = [None] * len(names)
    for j, i in enumerate(col):
        pairs[i] = [float(lo[j]), float(hi[j])]
    clipped = files.copy()
    clipped.set("control.actions.joint_pos.clip", pairs, "user", "test")
    clipped.set("control.actions.joint_pos.clip_stage", "processed", "user", "test")
    rep = verify(tr, clipped, policy)
    assert rep.boundaries["C"].status == "pass", rep.summary()
    # Without the clip in the contract the same trace is a mapping failure.
    assert verify(tr, files, policy).boundaries["C"].status == "fail"


def test_action_clip_shapes():
    from gaitkeeper.contract import Contract, action_clip_pairs

    c = Contract({})
    assert action_clip_pairs(c, 3) is None
    c.set("control.actions.joint_pos.clip", [-1.0, 2.0], "file", "t")
    assert action_clip_pairs(c, 3) == [(-1.0, 2.0)] * 3
    c.set("control.actions.joint_pos.clip", [[-1, 1], [-2, 2], [-3, 3]], "file", "t")
    assert action_clip_pairs(c, 3)[2] == (-3.0, 3.0)
    c.set("control.actions.joint_pos.clip", [[-1, 1]], "file", "t")
    with pytest.raises(ValueError, match="joint_pos.clip"):
        action_clip_pairs(c, 3)
    c.set("control.actions.joint_pos.clip_stage", "none", "file", "t")
    assert action_clip_pairs(c, 3) is None


def test_a_wrong_scale_under_a_binding_clip_is_still_named(harness, files, policy):
    names = list(files.get("policy_io.joints.names"))
    tr = harness.standard()
    raw = tr["action"].astype(np.float64)
    sc = np.array([files.get("control.actions.joint_pos.scale")[n] for n in names])
    of = np.array([files.get("control.actions.joint_pos.offset")[n] for n in names])
    e = raw * sc * 2.0 + of
    half = 0.8 * np.abs(e - of).max(0)
    lo, hi = of - half, of + half
    tn = list(tr.meta.get("target_joint_names", names))
    col = [names.index(n) for n in tn]
    t = np.clip(e, lo, hi)[:, col]
    tr.arrays["target"] = t.astype(np.float32)
    tr.arrays["effort"] = harness.effort(t).astype(np.float32)
    c = files.copy()
    c.set(
        "control.actions.joint_pos.clip",
        [[float(a), float(b)] for a, b in zip(lo, hi)],
        "user",
        "t",
    )
    c.set("control.actions.joint_pos.clip_stage", "processed", "user", "t")
    tc = verify(tr, c, policy).boundaries["C"].terms[0]
    assert tc.status == "fail" and "scale" in (tc.pattern or ""), tc.pattern
