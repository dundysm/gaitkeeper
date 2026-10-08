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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim2sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def contract_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--contract", help="contract.yaml")
        p.add_argument("--onnx", help="exported policy; its metadata is read as the contract")
        p.add_argument("--yaml", help="deploy.yaml exported next to the policy")
        p.add_argument("--deploy", help="Unitree deploy.yaml, read as what the robot runs")
        p.add_argument("--robot", default="unitree_g1_29dof", help="SDK table for --deploy")
        p.add_argument("--preset", action="append", help="named training preset for unknown fields")

    def sim_args(p: argparse.ArgumentParser) -> None:
        contract_args(p)
        p.add_argument("--mjcf", required=True, help="target MJCF scene")
        p.add_argument("--policy", help="policy file (defaults to --onnx)")
        p.add_argument(
            "--backend",
            choices=["native_implicit", "explicit_zoh", "python_pd", "standin_implicit"],
        )
        p.add_argument("--json")

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
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
