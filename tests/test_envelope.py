from gaitkeeper.envelope import Envelope, _dead, _dead_text, _pick, grids


def _row(c, a, fell=None):
    return {"cmd": (c, 0.0, 0.0), "achieved": (a, 0.0, 0.0), "fell_at": fell}


ROWS = [
    _row(-0.3, -0.25),
    _row(-0.2, -0.1),
    _row(-0.15, 0.0),
    _row(-0.1, 0.0),
    _row(0.1, 0.0),
    _row(0.2, 0.0),
    _row(0.22, 0.1),
    _row(0.5, 0.48),
    _row(1.5, 0.1, fell=3.0),
]


def test_dead_zone_edges_scan_outward():
    d = _dead(ROWS, 0, [-0.5, 1.0])
    assert (d["pos_edge"], d["pos_first"], d["neg_edge"], d["neg_first"]) == (
        0.2,
        0.22,
        -0.15,
        -0.2,
    )
    assert "ignored for |cmd| <= 0.20 on the + side (first tracked +0.22)" in _dead_text(
        "vx", d, [-0.5, 1.0], None
    )
    assert d["limited"]


def test_dead_zone_below_the_stand_threshold_is_by_design():
    rows = [_row(0.05, 0.0), _row(0.1, 0.11), _row(-0.05, 0.0), _row(-0.1, -0.09)]
    d = _dead(rows, 0, [-0.5, 1.0])
    assert "by design" in _dead_text("vx", d, [-0.5, 1.0], 0.1) and not d.get("limited")


def test_scenarios_stay_inside_reference_and_limit():
    env = Envelope(
        {"vx": ROWS}, {}, "trained", {"vx": [-1.0, 2.0]}, {"vx": [-0.5, 0.4]}, {"vx": [-1.0, 2.0]}
    )
    # 0.5 tracks but is outside the limit; 0.22 tracks but is the dead zone edge itself
    assert _pick(env, "vx", 1, 1.0) is None
    env.limit = {"vx": [-0.5, 0.6]}
    assert _pick(env, "vx", 1, 1.0) == 0.5
    assert _pick(env, "vx", -1, -0.5) == -0.3  # first tracked -0.2: -0.3 is clear of it


def test_scenario_commands_stay_clear_of_the_dead_zone_edge():
    """Near the edge tracking depends on seed and start (the #145 lateral case:
    first tracked +0.28, limit 0.30): no lateral scenario rather than a coin flip."""
    rows = [_row(0.25, 0.0), _row(0.28, 0.23), _row(0.3, 0.25)]
    env = Envelope({"vx": rows}, {}, "trained", {}, {"vx": [-0.3, 0.3]}, None)
    assert _pick(env, "vx", 1, 0.3) is None
    env.limit = {"vx": [-0.5, 0.5]}
    env.rows["vx"].append(_row(0.4, 0.35))
    assert _pick(env, "vx", 1, 0.3) == 0.4
    g = grids({"vx": [-0.5, 1.0], "vy": [-0.3, 0.3], "wz": [-0.2, 0.2]})
    assert 0.2 in g["vx"] and 0.22 in g["vx"] and 1.0 in g["vx"] and -1.0 in g["vx"]
    assert 0.25 in g["vy"] and 0.28 in g["vy"] and 0.2 in g["wz"]
