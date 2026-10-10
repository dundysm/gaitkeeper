"""Command line: read a contract, verify a trace, run, check, map the envelope, report deploy deviations.

Exit codes follow plan section 4 wherever a verdict is given: PASS 0, CONTRACT 1,
INVALID_INPUT 2, PHYSICS 3, POLICY_UNDER_TASK 4, UNDETERMINED 5, UNSUPPORTED 6.
L1 findings without a verdict exit 0 when the gate passes and 5 when it fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import Contract
from .env import env
from .trace import Trace


def _contract(args: argparse.Namespace) -> Contract:
    if args.contract:
        c = Contract.load(args.contract)
    elif getattr(args, "deploy", None):
        from .readers.unitree_deploy import read_unitree_deploy

        c, findings = read_unitree_deploy(args.deploy, args.onnx, robot=args.robot)
        for f in findings:
            print(f"reader {f.kind}: {f.path}: {f.message}", file=sys.stderr)
    elif args.onnx:
        from .readers.mjlab_export import read_mjlab_export

        c, findings = read_mjlab_export(args.onnx, args.yaml)
        for f in findings:
            print(f"reader {f.kind}: {f.path}: {f.message}", file=sys.stderr)
    else:
        sys.exit(
            "give --contract, --deploy (Unitree deploy.yaml), or --onnx (and optionally --yaml)"
        )
    for name in getattr(args, "preset", None) or []:
        from .presets import apply_preset

        filled = apply_preset(c, name)
        print(f"preset {name}: filled {len(filled)} field(s)", file=sys.stderr)
    for item in getattr(args, "set", None) or []:
        import yaml

        path, sep, value = item.partition("=")
        if not sep or not path:
            sys.exit(f"--set {item!r}: expected PATH=VALUE, e.g. timing.policy_dt=0.02")
        try:
            c.set(path.strip(), yaml.safe_load(value), "user", "--set on the command line")
        except KeyError as e:
            sys.exit(f"--set {item!r}: {e}")
        print(f"set {path.strip()} = {value} (provenance: user)", file=sys.stderr)
    return c


def _policy(args: argparse.Namespace, contract: Contract, required: bool = True):
    path = getattr(args, "policy", None) or args.onnx
    if not path:
        if required:
            sys.exit("give --policy (or --onnx)")
        return None
    from .policy import OnnxPolicy

    rec = contract.get("policy_io.graph.recurrent", None)
    return OnnxPolicy(path, rec or None)


def _triple(text: str) -> tuple[float, float, float]:
    v = [float(x) for x in text.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected vx,vy,wz")
    return (v[0], v[1], v[2])


def cmd_run(args: argparse.Namespace) -> int:
    from .runner import External, Push, PushGenerator, RunConfig, Runner, load_schedule

    c = _contract(args)
    pol = _policy(args, c, required=args.policy_mode == "policy")
    r = Runner(c, args.mjcf, pol)
    pushes = []
    for p in args.push or []:
        f = p.split(",")
        pushes.append(
            Push(
                float(f[0]),
                "force",
                (float(f[1]), float(f[2]), float(f[3])),
                f[4] if len(f) > 4 else None,
                float(f[5]) if len(f) > 5 else 0.1,
            )
        )
    for p in args.kick or []:
        f = [float(x) for x in p.split(",")]
        pushes.append(Push(f[0], "velocity", (f[1], f[2], 0.0)))
    gen = None
    if args.push_every:
        gen = PushGenerator(
            every_s=args.push_every, force=args.push_force, velocity=args.push_velocity
        )
    ext = None
    if args.hold:
        ext = [External(joints=args.hold.split(","), drive="hold", obs=args.unowned_obs)]
    cfg = RunConfig(
        backend=args.backend,
        seconds=args.seconds,
        command=args.command,
        schedule=load_schedule(args.schedule) if args.schedule else None,
        seed=args.seed,
        policy_mode=args.policy_mode,
        pushes=pushes,
        push_generator=gen,
        external=ext,
        limit_source=args.limits,
        timestep=args.timestep,
        integrator=args.integrator,
        armature=args.armature,
        record=bool(args.record),
    )
    res = r.run(cfg)
    print("\n".join(Runner.controller_text(res.controller)))
    print(res.summary())
    worst = res.sat_frac.argmax()
    print(
        f"torque: largest share at limit {res.sat_frac[worst]:.1%} ({res.names[worst]}); "
        f"peak {res.tau_peak.max():.1f} N m ({res.names[res.tau_peak.argmax()]})"
    )
    for p in res.pushes:
        print(f"push at {p['t']:.2f} s: {p}")
    if args.record:
        r.to_trace(res, {"command": list(args.command)}).save(args.record)
        print(f"wrote {args.record} (self-consistent only: written by this runner)")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "fell_at": res.fell_at,
                    "dist": res.dist,
                    "tail": [res.vx_tail, res.vy_tail, res.wz_tail],
                    "controller": res.controller,
                    "findings": res.findings,
                    "pushes": res.pushes,
                    "tau_peak": dict(zip(res.names, res.tau_peak.tolist())),
                    "sat_frac": dict(zip(res.names, res.sat_frac.tolist())),
                },
                indent=1,
                default=str,
            )
        )
    return 0 if res.survived else 5


def cmd_check(args: argparse.Namespace) -> int:
    from .checks import run_checks
    from .runner import Runner

    c = _contract(args)
    pol = _policy(args, c, required=False)
    r = Runner(c, args.mjcf, pol)
    which = (
        tuple(args.only.split(",")) if args.only else ("S16", "S17a", "S17b", "S18", "S19", "S21")
    )
    print("\n".join(Runner.controller_text(r.controller(r.choose_backend(args.backend)))))
    res = run_checks(r, args.backend, args.scenario, which=which)
    for x in res:
        print(x.text())
    if args.json:
        Path(args.json).write_text(
            json.dumps([{"id": x.id, "status": x.status, "lines": x.lines} for x in res], indent=1)
        )
    return 4 if any(x.status == "FAIL" for x in res) else 0


def cmd_envelope(args: argparse.Namespace) -> int:
    from .behavior import contract_header, probe
    from .envelope import sweep

    c = _contract(args)
    path = args.policy or args.onnx
    if not path:
        sys.exit("give --policy (or --onnx)")
    env = sweep(c, args.mjcf, path, backend=args.backend, workers=args.workers)
    pr = None
    if not args.no_probes or args.physics:
        pr = probe(
            env,
            c,
            args.mjcf,
            path,
            backend=args.backend,
            workers=args.workers,
            pushes=not args.no_probes,
            push_seeds=args.push_seeds,
            physics=args.physics,
        )
        if args.no_probes:
            pr.standstill, pr.kicks, pr.yaw_walk = [], [], None
    lines = pr.lines() if pr else []
    frag = []
    if pr and pr.fragility:
        i = next(k for k, ln in enumerate(lines) if ln.startswith("FRAGILITY"))
        lines, frag = lines[:i], lines[i:]
    print(contract_header(c))
    print("\n".join(env.lines(middle=lines, tail=frag)))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "command": "envelope",
                    "evidence": "L1",
                    "contract": contract_header(c),
                    "controller": env.controller,
                    "reference_range": env.ref_name,
                    "rows": env.rows,
                    "dead": env.dead,
                    "scenarios": env.scenarios,
                    "probes": pr.to_json() if pr else None,
                    "findings": env.findings,
                },
                indent=1,
                default=str,
            )
        )
    return 0


def cmd_deviation(args: argparse.Namespace) -> int:
    from .deviation import compare

    c = _contract(args)
    if args.reference:
        ref = Contract.load(args.reference)
        name = f"reference {Path(args.reference).name}"
    elif args.onnx:
        from .readers.mjlab_export import read_mjlab_export

        ref, _ = read_mjlab_export(args.onnx, None)
        name = "ONNX metadata"
    else:
        sys.exit("give --reference contract.yaml or --onnx with training metadata")
    actions = names = None
    if args.trace:
        tr = Trace.load(args.trace)
        actions = tr["action"]
        names = list(ref.get("policy_io.joints.names"))
    dev = compare(c, ref, actions, names, name)
    print("\n".join(dev.lines()))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    c = _contract(args)
    if args.out:
        c.save(args.out)
        print(f"wrote {args.out}")
    else:
        print(c.dump())
    unknown = c.unknown_fields()
    if unknown:
        print(f"unknown fields: {unknown}", file=sys.stderr)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .diagnose import diagnose_trace
    from .task import TaskSpec

    try:
        trace = Trace.load(args.trace)
    except (OSError, ValueError) as e:
        print(f"INVALID_INPUT: {e}")
        return 2
    contract = _contract(args)
    policy = None
    pol_path = args.policy or args.onnx
    if pol_path:
        from .policy import OnnxPolicy

        policy = OnnxPolicy(pol_path)
    task = None
    if args.task_schedule:
        from .runner import load_schedule

        sched = load_schedule(args.task_schedule)
        task = TaskSpec(
            Path(args.task_schedule).stem, args.task_seconds or sched[-1][0] + 5.0, sched
        )
    dg = diagnose_trace(
        trace,
        contract,
        policy,
        onnx=pol_path,
        target=args.mjcf,
        seeds=tuple(range(1, args.seeds + 1)),
        run_counterfactual=not args.no_counterfactual,
        task=task,
        workers=args.workers,
    )
    print("\n".join(dg.lines()))
    if args.json:
        rep = dg.report
        out = dg.to_json()
        out.update(
            {
                "verdict": dg.decision.verdict,
                "evidence": dg.decision.evidence,
                "label": rep.label if rep else None,
                "findings": rep.findings if rep else [],
                "excitation": rep.excitation if rep else [],
                "boundaries": {
                    k: {
                        "status": b.status,
                        "patterns": b.patterns,
                        "notes": b.notes,
                        "terms": [t.__dict__ for t in b.terms],
                    }
                    for k, b in (rep.boundaries.items() if rep else [])
                },
            }
        )
        Path(args.json).write_text(json.dumps(out, indent=1, default=str))
    return dg.decision.exit_code


def cmd_residual(args: argparse.Namespace) -> int:
    from .residual import dynamics_residual

    trace = Trace.load(args.trace)
    contract = _contract(args)
    d = dynamics_residual(trace, contract, args.mjcf, fits=args.fits)
    print("\n".join(d.lines()))
    if args.json:
        Path(args.json).write_text(json.dumps(d.to_json(), indent=1, default=str))
    return 0


def cmd_tour(args: argparse.Namespace) -> int:
    import math

    import yaml

    from .behavior import contract_header
    from .tour import TourOptions, run_tour

    c = _contract(args)
    path = args.policy or args.onnx
    if not path:
        sys.exit("give --policy (or --onnx)")
    wps = None
    if args.waypoints:
        d = yaml.safe_load(Path(args.waypoints).read_text())
        rows = d["waypoints"] if isinstance(d, dict) else d
        wps = [(float(r[0]), float(r[1]), float(r[2]) if len(r) > 2 else 0.0) for r in rows]
    opts = TourOptions(
        waypoints=wps,
        point_s=args.point_s,
        arms=args.arms,
        arms_obs=args.arms_obs,
        hold_gains=args.hold_gains,
        punches=args.punches,
        backend=args.backend,
    )
    print(contract_header(c))
    src = "the benchmark's draws per seed" if wps is None else f"{len(wps)} from {args.waypoints}"
    print(
        f"Tour: waypoints {src}, {args.point_s:g} s each; arms {args.arms}"
        + (f" (gains {args.hold_gains}, observed {args.arms_obs})" if args.arms != "policy" else "")
        + f"; punches {args.punches}"
    )
    out = run_tour(c, args.mjcf, path, list(range(args.seeds)), opts, args.workers)
    for r in out["runs"]:
        surv = "complete" if r["outcome"] == "complete" else f"fell at {r['survival_s']:.2f} s"
        pos = "-" if math.isnan(r["pos_err_cm"]) else f"{r['pos_err_cm']:.0f} cm"
        yaw = "-" if math.isnan(r["yaw_err_deg"]) else f"{r['yaw_err_deg']:.0f} deg"
        print(
            f"  seed {r['seed']:<3d} {surv:22s} {r['targets']:2d} waypoints scored, "
            f"{r['reached']:2d} reached; error {pos}, {yaw}"
        )
    print(
        f"Mean survival {out['mean_survival_s']:.1f} of {out['seconds']:.0f} s; "
        f"complete {out['complete']} of {len(out['runs'])}"
    )
    print("Evidence L1: this runner, this model, these assumptions; not an attribution.")
    if args.json:
        Path(args.json).write_text(json.dumps({"command": "tour", **out}, indent=1, default=str))
    return 0 if out["complete"] == len(out["runs"]) else 5


def _read_adapter(path: str, mjcf: str, cls: str = "Policy"):
    from .readers.twb_probe import ProbeError, read_twb_adapter

    try:
        return read_twb_adapter(path, mjcf, cls)
    except ProbeError as e:
        print(f"gaitkeeper: {path}: unsupported: {e}", file=sys.stderr)
        raise SystemExit(6) from None


def cmd_adapter(args: argparse.Namespace) -> int:
    r, cs, v = _read_adapter(args.adapter, args.mjcf, args.cls)
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Adapter {r.name}: obs {r.obs_dim}, actions {r.act_dim}, owned() {r.owned}")
    terms = ", ".join(
        f"{t['source_name']}[{t['dim']}]" + ("*" if "index" in t else "") for t in r.terms
    )
    h = r.history
    print(f"  observation: {terms}" + (" (* reordered or partial)" if "*" in terms else ""))
    print(
        f"  history {h['length']}"
        + (f" {h['layout']}, {h['order']}, {h['init']}" if h["length"] > 1 else "")
    )
    if r.clock:
        print(f"  clock {r.clock}")
    held = [k for k, m in enumerate(r.p2m) if m >= r.owned]
    print(
        f"  harness holds {len(held)} policy joint(s); observed as {sorted(set(r.joint_obs[k] for k in held)) or '-'}"
    )
    for f in r.findings:
        print(f"Finding   {f}")
    for variant, c in cs.items():
        path = out / f"{r.name}.{variant}.yaml"
        c.save(path)
        print(f"wrote {path}")
    (out / f"{r.name}.probe.json").write_text(json.dumps(r.to_json(), indent=1, default=str))
    if v["ok"]:
        print(
            f"Verified: gaitkeeper builds the adapter's observation to {v['max_abs']:.1g} on random inputs"
        )
        return 0
    bad = {k: round(x, 4) for k, x in v["per_term"].items() if x >= 1e-4}
    print(f"UNVERIFIED: the observation differs from the adapter's: {bad}")
    return 5


def cmd_bench(args: argparse.Namespace) -> int:
    import yaml

    from .behavior import contract_header
    from .bench import STAGE_TEXT, run_bench

    if args.adapter:
        _, cs, v = _read_adapter(args.adapter, args.mjcf)
        if not v["ok"]:
            print(
                "warning: the adapter's observation is not reproduced (see gaitkeeper adapter)",
                file=sys.stderr,
            )
        c, port = cs["trained"], cs["port"]
    else:
        c = _contract(args)
        port = Contract.load(args.port) if args.port else None
    path = args.policy or args.onnx
    if not path:
        sys.exit("give --policy (or --onnx)")
    upstream = None
    if args.upstream:
        upstream = Contract.load(args.upstream)
    elif args.upstream_deploy:
        from .readers.unitree_deploy import read_unitree_deploy

        upstream, _ = read_unitree_deploy(args.upstream_deploy, None, robot=args.robot)
    wps = None
    if args.waypoints:
        d = yaml.safe_load(Path(args.waypoints).read_text())
        rows = d["waypoints"] if isinstance(d, dict) else d
        wps = [(float(r[0]), float(r[1]), float(r[2]) if len(r) > 2 else 0.0) for r in rows]
    name = args.name or (
        Path(args.adapter).parent.name
        if args.adapter
        else Path(args.contract or args.deploy or path).stem
    )
    print(contract_header(c))
    rep = run_bench(
        c,
        args.mjcf,
        path,
        list(range(args.seeds)),
        port=port,
        upstream=upstream,
        stages=args.stages.split(",") if args.stages else None,
        waypoints=wps,
        point_s=args.point_s,
        workers=args.workers,
        backend=args.backend,
        name=name,
        progress=lambda k: print(f"  running: {STAGE_TEXT.get(k, k)}", file=sys.stderr),
    )
    if args.adapter:
        rep.notes.insert(
            0,
            "both contracts are read from the adapter: 'own setup' is the adapter's values with "
            "the policy driving every joint it lists, not the upstream training config",
        )
    if args.envelope:
        from .envelope import sweep

        env = sweep(c, args.mjcf, path, backend=args.backend, workers=args.workers)
        rep.envelope = env.lines()
    print("\n".join(rep.lines()))
    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_json(), indent=1, default=str))
    if args.md:
        Path(args.md).write_text(rep.markdown())
    return 0 if rep.gate() else 5


def cmd_task(args: argparse.Namespace) -> int:
    from .diagnose import diagnose_task
    from .runner import Push, PushGenerator, load_schedule
    from .task import TaskSpec

    c = _contract(args)
    path = args.policy or args.onnx
    if not path:
        sys.exit("give --policy (or --onnx)")
    sched = load_schedule(args.schedule)
    pushes = []
    for p in args.push or []:
        f = p.split(",")
        pushes.append(
            Push(
                float(f[0]),
                "force",
                (float(f[1]), float(f[2]), float(f[3])),
                f[4] if len(f) > 4 else None,
                float(f[5]) if len(f) > 5 else 0.1,
            )
        )
    gen = None
    if args.push_every:
        gen = PushGenerator(
            every_s=args.push_every,
            first_s=args.push_first,
            force=args.push_force,
            fixed=True,
            duration=args.push_duration,
            body=args.push_body,
            direction="horizontal",
        )
    spec = TaskSpec(
        args.name or Path(args.schedule).stem,
        args.seconds or sched[-1][0] + 5.0,
        sched,
        pushes,
        gen,
        args.hold.split(",") if args.hold else None,
        args.unowned_obs,
    )
    from .behavior import contract_header

    print(contract_header(c))
    dg = diagnose_task(
        c, args.mjcf, path, spec, tuple(range(1, args.seeds + 1)), args.backend, args.workers
    )
    env = dg.envelope
    print("\n".join(env.lines()))
    print("\n".join(dg.lines()))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {"command": "task", **dg.to_json(), "scenarios": env.scenarios},
                indent=1,
                default=str,
            )
        )
    return dg.decision.exit_code


def cmd_infer(args: argparse.Namespace) -> int:
    from .infer import infer

    res = infer(Trace.load(args.trace), use_raw=not args.no_raw)
    print("\n".join(res.lines()))
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_json(), indent=1, default=str))
    return 0 if res.status == "inferred" else 3


DEMO_ARMS = [
    f"{side}_{j}"
    for side in ("left", "right")
    for j in (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    )
]


def cmd_fetch(args: argparse.Namespace) -> int:
    from . import fixtures

    sets = fixtures.manifest()
    if args.list or not args.sets:
        root = fixtures.data_dir()
        print(f"fixture sets (directory {root}; set GAITKEEPER_DATA to change it):")
        for name, s in sets.items():
            state = "present" if s.present() else "not fetched"
            group = f" [{s.group}]" if s.group else ""
            print(f"  {name:18s} {state:12s} {s.repo}@{s.commit[:7]}{group}: {s.about}")
        if not args.sets:
            print("fetch with `gaitkeeper fetch <set or group> ...` or `gaitkeeper fetch all`")
            print(
                "(`all` leaves out grouped sets such as the golden traces: `gaitkeeper fetch golden`)"
            )
        return 0
    names = fixtures.expand(args.sets)
    for n in names:
        fixtures.fetch(n, log=lambda m: print(m, file=sys.stderr))
        print(f"{n}: {fixtures.get(n).dir()}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """The issue 145 setup on unitree_mujoco's G1, fetched on first use."""
    from importlib import resources

    from . import fixtures

    need = ("g1_rl_lab", "g1_unitree_mujoco")
    for n in need:
        if not fixtures.get(n).present():
            if args.no_fetch:
                fixtures.require(n)
            fixtures.fetch(n, log=lambda m: print(m, file=sys.stderr))
    pol, scene = fixtures.get("g1_rl_lab"), fixtures.get("g1_unitree_mujoco")
    tour = resources.files("gaitkeeper").joinpath("data/issue145_tour.yaml")
    print(
        "Demo: unitree_rl_lab issue 145, where the official G1 policy is reported to score 0% on a\n"
        "waypoint tour in a third-party benchmark, and the question is whether the harness is wrong.\n"
        "  policy   unitree_rl_lab G1 29 dof velocity policy as shipped (deploy.yaml, policy.onnx)\n"
        "  target   unitree_mujoco's G1 29 dof scene, simulated on the CPU in MuJoCo\n"
        "  task     small commands and in-place turns (data/issue145_tour.yaml), arms held at\n"
        "           the default pose, then 600 N punches on the torso every 3 s from 27 s\n"
        f"  seeds    {args.seeds}; first the command response map, then the task (under a minute)\n",
        flush=True,
    )
    argv = [
        "task",
        "--deploy",
        str(pol.path("deploy.yaml")),
        "--onnx",
        str(pol.path("policy.onnx")),
        "--preset",
        "unitree_rl_lab_g1_29dof_velocity@4960b84",
        "--mjcf",
        str(scene.path(scene.scene)),
        "--schedule",
        str(tour),
        "--seconds",
        "34",
        "--name",
        "issue 145 tour",
        "--hold",
        ",".join(DEMO_ARMS),
        "--push-every",
        "3",
        "--push-first",
        "27",
        "--push-force",
        "600",
        "--push-duration",
        "0.1",
        "--push-body",
        "torso_link",
        "--seeds",
        str(args.seeds),
    ]
    if args.workers:
        argv += ["--workers", str(args.workers)]
    if args.json:
        argv += ["--json", args.json]
    code = main(argv)
    print(
        "\nHow to read this: in this runner the policy stands still for small commands (the DEAD\n"
        "ZONE lines above) and does not turn in place, which is most of what the tour asks for,\n"
        "so the tour fails with a harness that follows the contract; the punches then knock it\n"
        f"over. This is an L1 finding (this runner, these assumptions): `gaitkeeper task` exits {code}\n"
        "here. It does not show that the benchmark's harness is correct, and it attributes\n"
        "nothing; that needs a golden trace from the training simulator\n"
        "(README: Exit codes, Evidence levels)."
    )
    return 0


# Flags whose value must be an existing path.
_PATH_FLAGS = (
    "contract",
    "onnx",
    "yaml",
    "deploy",
    "mjcf",
    "policy",
    "schedule",
    "reference",
    "port",
    "waypoints",
    "adapter",
    "upstream",
    "upstream_deploy",
)


def _check_inputs(args: argparse.Namespace) -> str | None:
    for flag in _PATH_FLAGS:
        v = getattr(args, flag, None)
        if v and not Path(v).exists():
            return f"--{flag.replace('_', '-')} {v}: no such file"
    t = getattr(args, "trace", None)
    if (
        isinstance(t, str)
        and t
        and args.cmd in ("verify", "residual", "infer", "deviation")
        and not Path(t).exists()
    ):
        return f"trace {t}: no such file or directory"
    return None


def _floor_warning(mjcf: str) -> str | None:
    """A model with nothing on the world body has no floor: the robot can only fall."""
    if not mjcf.endswith(".xml"):
        return None
    try:
        from .models import load_model

        m = load_model(mjcf).model
    except Exception:
        return None  # the command itself reports why the model does not load
    if int(m.body_geomnum[0]) == 0:
        return (
            f"warning: {mjcf} has no geom on the world body (no floor), so the robot can only "
            "fall; give the scene file (for example scene_29dof.xml) rather than the robot file"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    ap = argparse.ArgumentParser(prog="gaitkeeper")
    ap.add_argument("--version", action="version", version=f"gaitkeeper {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def contract_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--contract", help="contract.yaml")
        p.add_argument("--onnx", help="exported policy; its metadata is read as the contract")
        p.add_argument("--yaml", help="deploy.yaml exported next to the policy")
        p.add_argument("--deploy", help="Unitree deploy.yaml, read as what the robot runs")
        p.add_argument("--robot", default="unitree_g1_29dof", help="SDK table for --deploy")
        p.add_argument("--preset", action="append", help="named training preset for unknown fields")
        p.add_argument(
            "--set",
            action="append",
            metavar="PATH=VALUE",
            help="set one contract field (YAML value), e.g. timing.policy_dt=0.02; marked as user-stated",
        )

    def sim_args(p: argparse.ArgumentParser) -> None:
        contract_args(p)
        p.add_argument("--mjcf", required=True, help="target MJCF scene")
        p.add_argument("--policy", help="policy file (defaults to --onnx)")
        p.add_argument(
            "--backend",
            choices=["native_implicit", "explicit_zoh", "python_pd", "standin_implicit"],
        )
        p.add_argument("--json")

    p = sub.add_parser(
        "demo", help="the issue 145 setup end to end (fetches its files on first use)"
    )
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int)
    p.add_argument("--json")
    p.add_argument("--no-fetch", action="store_true", help="fail instead of downloading")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("fetch", help="download pinned third-party models and policies")
    p.add_argument("sets", nargs="*", help="fixture set names, or all; none lists them")
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("inspect", help="read a contract from exported files")
    contract_args(p)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser(
        "verify", help="boundaries B, A, C of a trace, then D and the closed loop with --mjcf"
    )
    p.add_argument("trace", help="golden trace directory or harness log .npz")
    contract_args(p)
    p.add_argument("--policy", help="policy file for boundary B (defaults to --onnx)")
    p.add_argument("--mjcf", help="target model: adds D, the closed loop and the counterfactual")
    p.add_argument("--seeds", type=int, default=12, help="seeded starts for the closed loop")
    p.add_argument("--no-counterfactual", action="store_true")
    p.add_argument("--task-schedule", help="a requested task: command schedule run on the target")
    p.add_argument("--task-seconds", type=float)
    p.add_argument("--workers", type=int)
    p.add_argument("--json")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("residual", help="boundary D only: the dynamics residual of a trace")
    p.add_argument("trace", help="golden trace directory with physics-rate channels")
    contract_args(p)
    p.add_argument("--mjcf", required=True, help="analysis target (MJCF, .mjb or trace directory)")
    p.add_argument("--fits", action="store_true", help="parameter fits (research)")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_residual)

    p = sub.add_parser("task", help="a requested task with no reference: L1 findings")
    sim_args(p)
    p.add_argument("--schedule", required=True, help="command schedule, YAML or CSV t,vx,vy,wz")
    p.add_argument("--name")
    p.add_argument("--seconds", type=float)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hold", help="comma separated joints the policy does not own")
    p.add_argument("--unowned-obs", choices=["real", "echo_action", "default"], default="real")
    p.add_argument("--push", action="append", help="t,fx,fy,fz[,body[,duration]]")
    p.add_argument("--push-every", type=float, help="horizontal pushes of fixed size every N s")
    p.add_argument("--push-first", type=float, default=2.0)
    p.add_argument("--push-force", type=float, default=0.0)
    p.add_argument("--push-duration", type=float, default=0.1)
    p.add_argument("--push-body", default="torso_link")
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser(
        "tour", help="closed-loop waypoint tour (the teleop-walking-benchmark tour by default)"
    )
    sim_args(p)
    p.add_argument("--waypoints", help="YAML list of [x, y, yaw] relative to the start pose")
    p.add_argument("--point-s", type=float, default=5.0, help="seconds per waypoint")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument(
        "--arms",
        choices=["policy", "hold", "walk"],
        default="policy",
        help="who drives the arm joints: the policy, a hold at the benchmark's stance, "
        "or the benchmark's random walk",
    )
    p.add_argument(
        "--arms-obs", choices=["real", "echo_action", "default", "contract"], default="real"
    )
    p.add_argument("--hold-gains", choices=["armature", "policy"], default="armature")
    p.add_argument("--punches", choices=["none", "benchmark"], default="none")
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_tour)

    p = sub.add_parser(
        "bench",
        help="the tour at each step from the policy's own setup to the full benchmark, "
        "with a port contract compared and run alongside",
    )
    sim_args(p)
    p.add_argument("--port", help="contract of what a benchmark port runs (ownership, gains, obs)")
    p.add_argument(
        "--adapter",
        help="teleop-walking-benchmark policy.cpp: read both contracts from it (needs a C++ compiler)",
    )
    p.add_argument(
        "--upstream", help="contract of the policy as its authors trained or deployed it"
    )
    p.add_argument("--upstream-deploy", help="the same, from the authors' Unitree deploy.yaml")
    p.add_argument("--stages", help="comma separated subset of own,arms_hold,arms_walk,punches")
    p.add_argument("--waypoints", help="YAML list of [x, y, yaw]; default the benchmark's draws")
    p.add_argument("--point-s", type=float, default=5.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--envelope", action="store_true", help="add the command envelope")
    p.add_argument("--name")
    p.add_argument("--md", help="write the report as markdown")
    p.add_argument("--workers", type=int)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser(
        "adapter",
        help="read a teleop-walking-benchmark policy adapter (policy.cpp) into trained and port contracts",
    )
    p.add_argument("adapter", help="policies/<name>/policy.cpp")
    p.add_argument("--mjcf", required=True, help="the benchmark's G1 MJCF (for armature gains)")
    p.add_argument("--out", help="directory for <name>.trained.yaml, .port.yaml, .probe.json")
    p.add_argument("--cls", default="Policy", help="policy class in the adapter's namespace")
    p.set_defaults(fn=cmd_adapter)

    p = sub.add_parser("infer", help="observation layout from a trace, abstaining when ambiguous")
    p.add_argument("trace", nargs="?", help="golden trace directory or harness log .npz")
    p.add_argument("--trace", dest="trace_opt", help="same as the positional argument")
    p.add_argument("--no-raw", action="store_true", help="use (obs, action) only")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_infer)

    p = sub.add_parser("run", help="closed-loop run of a policy in a target MJCF")
    sim_args(p)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--command", type=_triple, default=(0.5, 0.0, 0.0), help="vx,vy,wz")
    p.add_argument("--schedule", help="command schedule, YAML or CSV rows t,vx,vy,wz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-mode", choices=["policy", "zero", "random"], default="policy")
    p.add_argument("--push", action="append", help="t,fx,fy,fz[,body[,duration]]")
    p.add_argument("--kick", action="append", help="t,vx,vy base velocity kick")
    p.add_argument("--push-every", type=float, help="periodic random pushes every N s")
    p.add_argument("--push-force", type=float, default=0.0)
    p.add_argument("--push-velocity", type=float, default=0.0)
    p.add_argument(
        "--hold", help="comma separated joints the policy does not own, held at the default"
    )
    p.add_argument("--unowned-obs", choices=["real", "echo_action", "default"], default="real")
    p.add_argument("--limits", choices=["model", "contract"], default="model")
    p.add_argument("--timestep", type=float, help="python_pd only")
    p.add_argument(
        "--integrator", choices=["euler", "implicitfast", "implicit", "rk4"], help="python_pd only"
    )
    p.add_argument("--armature", type=float, help="override armature on the policy joints")
    p.add_argument("--record", help="write the run as a harness trace (.npz)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("check", help="static and linearized checks S16, S17a, S17b, S18, S19, S21")
    sim_args(p)
    p.add_argument("--scenario", type=_triple, action="append", help="vx,vy,wz scenario command")
    p.add_argument("--only", help="comma separated check ids")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("envelope", help="command response map with dead zones and scenarios")
    sim_args(p)
    p.add_argument("--workers", type=int)
    p.add_argument("--no-probes", action="store_true", help="skip standstill, kick and push probes")
    p.add_argument("--push-seeds", type=int, default=10, help="seeds per push level (30 s runs)")
    p.add_argument(
        "--physics",
        action="store_true",
        help="fragility sweep: friction, contact softness, armature, kp scale, step size",
    )
    p.set_defaults(fn=cmd_envelope)

    p = sub.add_parser("deviation", help="deploy values against training values, per joint")
    contract_args(p)
    p.add_argument("--reference", help="reference contract (for example contract.live.yaml)")
    p.add_argument(
        "--trace", help="trace whose actions turn scale differences into target differences"
    )
    p.set_defaults(fn=cmd_deviation)

    args = ap.parse_args(argv)
    if args.cmd == "infer":
        args.trace = args.trace or args.trace_opt
        if not args.trace:
            ap.error("infer needs a trace")
    problem = _check_inputs(args)
    if problem:
        print(f"gaitkeeper {args.cmd}: {problem}", file=sys.stderr)
        return 2
    if getattr(args, "mjcf", None):
        w = _floor_warning(args.mjcf)
        if w:
            print(w, file=sys.stderr)
    if env("DEBUG"):
        return args.fn(args)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except ModuleNotFoundError as e:
        extra = {"mujoco": " (pip install -e .[sim])"}.get(str(e.name), "")
        print(f"gaitkeeper {args.cmd}: needs the Python package {e.name!r}{extra}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(
            f"gaitkeeper {args.cmd}: {msg}\n  (set GAITKEEPER_DEBUG=1 for the full traceback)",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
