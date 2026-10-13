"""Build the G1 port audit page (docs/results/index.html) from docs/results/data.json.

    python tools/results_page.py                 # docs/results/index.html, a full document
    python tools/results_page.py --fragment X    # the same page without the document shell

data.json is written by running `gaitkeeper adapter` and `gaitkeeper bench` over the
teleop-walking-benchmark checkout; see the page's "Reproduce" section.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "data.json"

STAGES = [
    ("upstream", "Authors' config"),
    ("own", "Port's values"),
    ("arms_hold", "Harness holds arms"),
    ("arms_walk", "Random arm walk"),
    ("punches", "Plus punches"),
    ("port", "Port, full benchmark"),
]
CAUSES = {
    "policy": ("Policy", "Falls under its authors' own config"),
    "port": ("Port values", "The port's values break it"),
    "arms": ("Arm handoff", "Trained to drive its arms; the harness takes them"),
    "walk": ("Arm walk", "Stands, but the random arm motion knocks it down"),
    "punch": ("Punches", "Survives the arms, falls to the punches"),
    "unclear": ("Unclear", "Falls with the port's values; no upstream config read"),
    "none": ("Survives", "Survives every stage within 5 s of its own setup"),
}

CSS = """
/* Layout: one reading column, a summary band, then a list of policies that open in place. */
:root {
  --bg: #f4f6f8; --panel: #ffffff; --ink: #121820; --muted: #566273; --rule: #d6dce4;
  --accent: #1f5f8b; --bar: #9fb3c8; --bar-up: #1f5f8b;
  --c-policy: #5d6b7d; --c-port: #b4372b; --c-arms: #a3650a; --c-walk: #c98a12;
  --c-punch: #7a5bb0; --c-unclear: #8a94a3; --c-none: #2e7d4f;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141a; --panel: #161d25; --ink: #e6ebf1; --muted: #9aa6b5; --rule: #2a3440;
  --accent: #6fb3e0; --bar: #3e5166; --bar-up: #6fb3e0;
  --c-policy: #8f9db0; --c-port: #ef6f5f; --c-arms: #e0a03a; --c-walk: #f0c050;
  --c-punch: #ad92e0; --c-unclear: #7d8796; --c-none: #5fbf86; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f141a; --panel: #161d25; --ink: #e6ebf1; --muted: #9aa6b5; --rule: #2a3440;
  --accent: #6fb3e0; --bar: #3e5166; --bar-up: #6fb3e0;
  --c-policy: #8f9db0; --c-port: #ef6f5f; --c-arms: #e0a03a; --c-walk: #f0c050;
  --c-punch: #ad92e0; --c-unclear: #7d8796; --c-none: #5fbf86; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--f-body); margin: 0; }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; }
h1, h2 { font-family: var(--f-display); text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 800; line-height: 1.05; }
h2 { font-size: 1.35rem; font-weight: 700; margin-bottom: 12px; }
p { margin: 0; max-width: 68ch; }
a { color: var(--accent); }
code, .mono { font-family: var(--f-mono); font-size: 0.86em; }
.eyebrow { font: 600 0.72rem/1 var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
header { display: grid; gap: 14px; margin-bottom: 36px; }
.lede { font-size: 1.08rem; color: var(--ink); }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); font: 0.8rem var(--f-mono); }
section { margin-top: 44px; display: grid; gap: 14px; }
.band { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 20px; align-items: start; }
@media (max-width: 760px) { .band { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px; min-width: 0; display: grid; gap: 12px; }
.stack { display: flex; height: 26px; border-radius: 3px; overflow: hidden; background: var(--rule); }
.stack span { display: block; height: 100%; }
.legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 8px 16px; font-size: 0.86rem; }
.legend div { display: flex; gap: 8px; align-items: baseline; }
.sw { width: 10px; height: 10px; border-radius: 2px; flex: none; transform: translateY(1px); }
.num { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.big { font: 700 2rem/1 var(--f-display); }
.chip { display: inline-flex; align-items: center; gap: 6px; font: 600 0.74rem/1 var(--f-mono); letter-spacing: 0.02em; padding: 5px 8px; border-radius: 3px; border: 1px solid currentColor; white-space: nowrap; }
.list { display: grid; gap: 0; border-top: 1px solid var(--rule); }
details { border-bottom: 1px solid var(--rule); }
summary { list-style: none; cursor: pointer; display: grid; grid-template-columns: 9.5rem 7.5rem 128px minmax(0, 1fr) 4.6rem 4.6rem; gap: 14px; align-items: center; padding-block: 12px; }
summary::-webkit-details-marker { display: none; }
summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
summary .name { font: 600 0.95rem var(--f-mono); overflow-wrap: anywhere; }
summary .name small { display: block; font: 400 0.78rem var(--f-mono); color: var(--muted); }
summary .verdict { color: var(--muted); font-size: 0.9rem; min-width: 0; }
summary .s { text-align: right; }
.head { display: grid; grid-template-columns: 9.5rem 7.5rem 128px minmax(0, 1fr) 4.6rem 4.6rem; gap: 14px; padding-block: 8px; color: var(--muted); font: 600 0.7rem var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; }
.head .s { text-align: right; }
@media (max-width: 820px) {
  summary, .head { grid-template-columns: minmax(0, 1fr) auto; }
  summary .verdict, summary svg, .head span:nth-child(n+3) { grid-column: 1 / -1; }
  summary .s, .head { display: none; }
}
.detail { display: grid; gap: 14px; padding: 4px 0 20px; }
.stages { overflow-x: auto; }
table.runs { min-width: 0; }
table.runs td, table.runs th { padding: 4px 10px 4px 0; }
table.runs td.off { color: var(--accent); font-weight: 600; }
table { border-collapse: collapse; font-size: 0.86rem; min-width: 480px; }
th, td { text-align: left; padding: 6px 14px 6px 0; border-bottom: 1px solid var(--rule); }
th { font: 600 0.7rem var(--f-mono); letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
td.num, th.num { text-align: right; }
pre { margin: 0; overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 12px; font: 0.78rem/1.5 var(--f-mono); color: var(--ink); }
ul { margin: 0; padding-left: 1.1rem; display: grid; gap: 4px; max-width: 75ch; }
.notread { display: grid; grid-template-columns: 9.5rem minmax(0, 1fr); gap: 8px 14px; font-size: 0.9rem; }
.notread .name { font: 600 0.9rem var(--f-mono); }
.defs { display: grid; grid-template-columns: 11rem minmax(0, 1fr); gap: 8px 16px; font-size: 0.92rem; }
@media (max-width: 560px) { .defs, .notread { grid-template-columns: 1fr; } }
svg text { fill: var(--muted); font: 10px var(--f-mono); }
.scatter .pt { fill: var(--accent); }
.scatter .ax { stroke: var(--rule); }
.scatter .diag { stroke: var(--muted); stroke-dasharray: 3 4; }
.spark .b { fill: var(--bar); }
.spark .bu { fill: var(--bar-up); }
.spark .base { stroke: var(--rule); }
@media (prefers-reduced-motion: no-preference) { details[open] .detail { animation: in .18s ease-out; } }
@keyframes in { from { opacity: .4; transform: translateY(-3px); } to { opacity: 1; transform: none; } }
"""


REPO = "https://github.com/dundysm/gaitkeeper"
# Refused adapters with an open issue for the missing piece.
ISSUES: dict[str, int] = {}  # a refused port with an open issue for its missing piece


def e(s: object) -> str:
    return html.escape(str(s))


def spark(st: dict, total: float) -> str:
    w, h, bw, gap = 128, 34, 16, 6
    out = [
        f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="survival at each stage">'
    ]
    out.append(f'<line class="base" x1="0" x2="{w}" y1="{h - 0.5}" y2="{h - 0.5}"/>')
    for i, (k, label) in enumerate(STAGES):
        x = i * (bw + gap) + 1
        if k not in st:
            out.append(
                f'<rect x="{x}" y="{h - 2}" width="{bw}" height="1" class="b" opacity=".35"><title>{e(label)}: not run</title></rect>'
            )
            continue
        v = st[k]["mean_survival_s"]
        bh = max(1.5, (h - 2) * v / total)
        cls = "bu" if k == "upstream" else "b"
        out.append(
            f'<rect x="{x}" y="{h - 1 - bh:.1f}" width="{bw}" height="{bh:.1f}" class="{cls}"><title>{e(label)}: {v:.1f} s</title></rect>'
        )
    out.append("</svg>")
    return "".join(out)


def scatter(pols: list[dict], total: float) -> str:
    W, H, P = 300, 220, 34
    s = (W - P - 10) / total
    t = (H - P - 10) / total
    out = [
        f'<svg class="scatter" viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="gaitkeeper survival against the benchmark\'s">'
    ]
    out.append(
        f'<line class="ax" x1="{P}" y1="{H - P}" x2="{W - 10}" y2="{H - P}"/><line class="ax" x1="{P}" y1="10" x2="{P}" y2="{H - P}"/>'
    )
    out.append(
        f'<line class="diag" x1="{P}" y1="{H - P}" x2="{P + total * s}" y2="{H - P - total * t}"/>'
    )
    for v in (0, 30, 60, 90):
        out.append(f'<text x="{P + v * s}" y="{H - P + 14}" text-anchor="middle">{v}</text>')
        out.append(f'<text x="{P - 6}" y="{H - P - v * t + 3}" text-anchor="end">{v}</text>')
    out.append(
        f'<text x="{P + (W - P - 10) / 2}" y="{H - 4}" text-anchor="middle">gaitkeeper, port under the full benchmark (s)</text>'
    )
    out.append(
        f'<text transform="translate(10 {(H - P) / 2}) rotate(-90)" text-anchor="middle">benchmark, MuJoCo, same runs (s)</text>'
    )
    for p in pols:
        if p.get("stages") and same(p) is not None:
            x, y = p["stages"]["port"]["mean_survival_s"], same(p)
            out.append(
                f'<circle class="pt" cx="{P + x * s:.1f}" cy="{H - P - y * t:.1f}" r="4"><title>{e(p["name"])}: {x:.1f} s here, {y:.1f} s in the benchmark</title></circle>'
            )
    out.append("</svg>")
    return "".join(out)


def name_html(p: dict) -> str:
    """A port's name, and its variant under it (gr00t_wbc / h066_p012)."""
    port, var = p.get("port"), p.get("variant")
    if port and var and var.startswith(port + "_"):
        return f"{e(port)}<small>{e(var[len(port) + 1 :])}</small>"
    return e(p["name"])


def chip(cause: str) -> str:
    label, _ = CAUSES[cause]
    return f'<span class="chip" style="color: var(--c-{cause})">{e(label)}</span>'


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):  # ties share their mean rank
        m = x == v
        r[m] = r[m].mean()
    return r


def same(p: dict) -> float | None:
    """The benchmark's MuJoCo mean over the runs benched here (seed n is its run id n); its
    mean over every run it published when the runs are not given."""
    v = p.get("benchmark_mujoco_same")
    return p.get("benchmark_mujoco") if v is None else v


def agreement(d: dict) -> dict:
    """Agreement of port survival with the benchmark's, computed from the published values so
    anyone can reproduce it from data.json: per policy, against the benchmark's mean over the
    same runs (and over all its runs); per run, against the same run there, next to how
    often the benchmark's PhysX runs land within the same margin of its MuJoCo runs."""
    pols = [
        p
        for p in d["policies"]
        if p.get("stages") and "port" in p["stages"] and same(p) is not None
    ]
    x = np.array([p["stages"]["port"]["mean_survival_s"] for p in pols])
    y = np.array([same(p) for p in pols])
    out = {
        "n": len(pols),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_ranks(x), _ranks(y))[0, 1]),
    }
    full = np.array([p["benchmark_mujoco"] for p in pols])
    out["pearson_all"] = float(np.corrcoef(x, full)[0, 1])
    short = x < 15.0
    out["n_short"] = int(short.sum())
    out["pearson_short"] = float(np.corrcoef(x[short], y[short])[0, 1])
    px = [
        (p["stages"]["port"]["mean_survival_s"], p["benchmark_physx_same"])
        for p in pols
        if p.get("benchmark_physx_same") is not None
    ]
    if len(px) > 2:
        a_, c_ = np.array(px).T
        out["n_physx"] = len(px)
        out["pearson_physx"] = float(np.corrcoef(a_, c_)[0, 1])
    here, mj, phx = [], [], []
    for p in pols:
        r = p.get("runs") or {}
        if "mujoco" in r and "physx" in r:
            here += r["here"]
            mj += r["mujoco"]
            phx += r["physx"]
    if here:
        h, m, q = np.array(here), np.array(mj), np.array(phx)
        out["n_runs"] = len(h)
        out["runs_within_1"] = int((np.abs(h - m) <= 1.0).sum())
        out["runs_within_1_physx"] = int((np.abs(q - m) <= 1.0).sum())
        out["runs_pearson"] = float(np.corrcoef(h, m)[0, 1])
        out["runs_pearson_physx"] = float(np.corrcoef(q, m)[0, 1])
    gap = max(pols, key=lambda p: abs(p["stages"]["port"]["mean_survival_s"] - same(p)))
    out["widest"] = {
        "name": gap["name"],
        "here": gap["stages"]["port"]["mean_survival_s"],
        "mujoco": same(gap),
        "physx": gap.get("benchmark_physx_same"),
    }
    return out


def page(d: dict) -> tuple[str, str]:
    pols = d["policies"]
    ag = agreement(d)
    changes_html = ""
    if d.get("changes"):
        items = "".join(f"<li>{c}</li>" for c in d["changes"])
        changes_html = (
            "<section>\n  <h2>Since the last audit</h2>\n"
            f'  <ul style="font-size: .92rem">{items}</ul>\n</section>\n'
        )
    physx_text = (
        f" Against the benchmark&#8217;s PhysX runs it is {ag['pearson_physx']:.2f}."
        if "pearson_physx" in ag
        else ""
    )
    runs_text = (
        f" Run by run, {ag['runs_within_1']} of the {ag['n_runs']} runs here end within 1 s of the "
        f"same run in the benchmark&#8217;s MuJoCo (Pearson {ag['runs_pearson']:.2f}); its own PhysX "
        f"runs do so {ag['runs_within_1_physx']} times (Pearson {ag['runs_pearson_physx']:.2f})."
        if "n_runs" in ag
        else ""
    )
    w = ag["widest"]
    widest_text = (
        f" The widest gap is {e(w['name'])}: {w['here']:.1f} s here, {w['mujoco']:.1f} s in the "
        f"benchmark&#8217;s MuJoCo"
        + (f" and {w['physx']:.1f} s in its PhysX" if w.get("physx") is not None else "")
        + ", not yet explained."
    )
    n_refused = sum(p.get("read") == "unsupported" for p in pols)
    unbenched_read = [
        p["name"] for p in pols if p.get("read") == "verified" and not p.get("stages")
    ]
    sunny_text = (
        f", and {', '.join(unbenched_read)} {'reads' if len(unbenched_read) == 1 else 'read'} "
        "but is not benched"
        if unbenched_read
        else ""
    )
    total = float(d["tour_s"])
    benched = [p for p in pols if p.get("stages")]
    order = ["port", "arms", "walk", "punch", "policy", "unclear", "none"]
    benched.sort(key=lambda p: (order.index(p["cause"]), p["name"]))
    notread = [p for p in pols if not p.get("stages")]
    counts = {c: sum(p.get("cause") == c for p in benched) for c in order}
    n_all = len(pols)

    stack = "".join(
        f'<span style="width:{100 * counts[c] / n_all:.2f}%; background: var(--c-{c})" title="{e(CAUSES[c][0])}: {counts[c]}"></span>'
        for c in order
        if counts[c]
    )
    legend = (
        "".join(
            f'<div><span class="sw" style="background: var(--c-{c})"></span><span><b class="num">{counts[c]}</b> {e(CAUSES[c][0])}: {e(CAUSES[c][1].lower())}</span></div>'
            for c in order
            if counts[c]
        )
        + f'<div><span class="sw" style="background: var(--rule)"></span><span><b class="num">{len(notread)}</b> not benched (reasons below)</span></div>'
    )

    rows = []
    for p in benched:
        st = p["stages"]
        bench = "–" if same(p) is None else f"{same(p):.1f}"
        mine = f"{st['port']['mean_survival_s']:.1f}"
        stage_rows = "".join(
            f'<tr><td>{e(label)}</td><td class="num">{st[k]["mean_survival_s"]:.1f}</td><td class="num">{st[k]["complete"]}/{st[k]["runs"]}</td></tr>'
            for k, label in STAGES + [("port_quiet", "Port, arms still, no punches")]
            if k in st
        )
        notes = "".join(f"<li>{e(n)}</li>" for n in p.get("notes", []))
        rr = p.get("runs") or {}
        runs_html = ""
        if "mujoco" in rr:
            head_ = "".join(f'<th class="num">{i}</th>' for i in rr["ids"])

            def cells(vals: list, ref: list | None = None) -> str:
                return "".join(
                    f'<td class="num{" off" if ref is not None and abs(v - r_) > 1.0 else ""}">{v:.0f}</td>'
                    for v, r_ in zip(vals, ref or vals)
                )

            body_ = f"<tr><td>Here</td>{cells(rr['here'], rr['mujoco'])}</tr>"
            body_ += f"<tr><td>Benchmark, MuJoCo</td>{cells(rr['mujoco'])}</tr>"
            if "physx" in rr:
                body_ += f"<tr><td>Benchmark, PhysX</td>{cells(rr['physx'], rr['mujoco'])}</tr>"
            runs_html = (
                '<div class="stages"><table class="runs"><thead><tr><th>Run</th>'
                f"{head_}</tr></thead><tbody>{body_}</tbody></table></div>"
            )
        af = "".join(f"<li>{e(f)}</li>" for f in p.get("adapter_findings", []))
        pr = p.get("probe") or {}
        src = p.get("port") or p["name"]
        var = f" (variant {e(p['variant'])})" if p.get("variant") else ""
        hist = (
            f"laid out as {pr.get('history')} [term, lag] chunks"
            if pr.get("layout") == "chunks"
            else f"history {pr.get('history')}"
        )
        read = (
            f"Read from <code>policies/{e(src)}/policy.cpp</code>{var}: {pr.get('obs')} observations "
            f"({e(', '.join(pr.get('terms', [])))}), {hist}, {pr.get('actions')} actions, "
            f"owned() = {pr.get('owned')}."
        )
        up = (
            f"<p>Upstream config: {e(p['upstream'])}.</p>"
            if p.get("upstream")
            else "<p>No upstream config read.</p>"
        )
        dev = [ln for ln in p.get("deviation", []) if "float32 rounding" not in ln]
        dev_html = f"<pre>{e(chr(10).join(dev))}</pre>" if dev else ""
        rows.append(f"""
<details id="{e(p["name"])}">
  <summary><span class="name">{name_html(p)}</span>{chip(p["cause"])}{spark(st, total)}<span class="verdict">{e(p["verdict"])}</span><span class="s num">{bench}</span><span class="s num">{mine}</span></summary>
  <div class="detail">
    <div class="stages"><table><thead><tr><th>Stage</th><th class="num">Mean survival (s)</th><th class="num">Completed</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
    {runs_html}
    {f"<ul>{notes}</ul>" if notes else ""}
    <p>{read}</p>
    {f'<p class="eyebrow">What reading the adapter found</p><ul>{af}</ul>' if af else ""}
    {up}
    {dev_html}
  </div>
</details>""")

    nr = "".join(
        f'<span class="name">{e(p["name"])}</span><span>{e(p.get("reason", "Not read."))}'
        + (
            f" Benchmark: {p['benchmark_mujoco']:.1f} s in MuJoCo."
            if p.get("benchmark_mujoco") is not None
            else ""
        )
        + (
            f' <a href="{REPO}/issues/{ISSUES[p["name"]]}">Open issue #{ISSUES[p["name"]]}</a>'
            if p["name"] in ISSUES
            else ""
        )
        + "</span>"
        for p in sorted(notread, key=lambda p: p["name"])
    )

    head = f"""<title>G1 Port Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>"""
    return (
        head,
        f"""<div class="wrap">
<header>
  <span class="eyebrow">gaitkeeper {e(d["gaitkeeper"])} · teleop-walking-benchmark</span>
  <h1>G1 Port Audit</h1>
  <p class="lede">Why each Unitree G1 walking policy in rhoyn's teleop-walking-benchmark scores what it does. Each port was read from its own adapter code, then the benchmark's 90 s waypoint tour was run one change at a time: the policy as its authors configured it, the port's values, the harness taking the arms, the random arm walk, the punches.</p>
  <div class="meta"><span>{e(d["benchmark"])}</span><span>runs 0&#8211;{d["seeds"] - 1}, the benchmark&#8217;s own draws</span><span>MuJoCo, the benchmark's g1_29dof.xml</span><span>evidence L1</span><span>{e(d["generated"])}</span></div>
</header>

<section>
  <div class="band">
    <div class="panel">
      <span class="eyebrow">What stops each policy</span>
      <div class="stack">{stack}</div>
      <div class="legend">{legend}</div>
    </div>
    <div class="panel">
      <span class="eyebrow">Agreement with the benchmark</span>
      <div style="display:flex; gap: 22px; align-items: baseline; flex-wrap: wrap">
        <span><span class="big num">{ag["pearson"]:.2f}</span> <span class="eyebrow">Pearson</span></span>
        <span><span class="big num">{ag["spearman"]:.2f}</span> <span class="eyebrow">Spearman</span></span>
        {f'<span><span class="big num">{ag["runs_within_1"]}/{ag["n_runs"]}</span> <span class="eyebrow">runs within 1 s</span></span>' if "n_runs" in ag else ""}
      </div>
      {scatter(pols, total)}
      <p style="font-size: .84rem; color: var(--muted)">{ag["n"]} policies. Survival of the port under the full benchmark here, against the benchmark&#8217;s own MuJoCo runs with the same run ids: seed n here draws the benchmark&#8217;s run n (its waypoints, arm targets and punches, from its own random streams). The port contracts are read from the adapters, none written by hand. Over the {ag["n_short"]} that last under 15 s here it is {ag["pearson_short"]:.2f}; against the benchmark&#8217;s mean over all its runs, {ag["pearson_all"]:.2f}.{physx_text}{runs_text}{widest_text}</p>
    </div>
  </div>
</section>

<section>
  <h2>Verdicts</h2>
  <p style="color: var(--muted); font-size: .92rem">Bars: mean survival at each stage, out of {d["tour_s"]} s (the first, darker bar is the authors' config where one was read). Open a row for the stage table, what reading the adapter found, and the values that differ.</p>
  <div class="list">
    <div class="head"><span>Policy</span><span>Stops it</span><span>Stages</span><span>Verdict</span><span class="s">Bench (s)</span><span class="s">Here (s)</span></div>
    {"".join(rows)}
  </div>
</section>

{changes_html}
<section>
  <h2>Not benched</h2>
  <p style="color: var(--muted); font-size: .92rem">The reader probes an adapter by running it on the CPU and checks that gaitkeeper rebuilds its observation exactly; it refuses rather than guesses. {n_refused} adapters are refused{sunny_text}:</p>
  <div class="notread">{nr}</div>
</section>

<section>
  <h2>How to read a verdict</h2>
  <div class="defs">
    <b>Every run</b><span>Seed n draws the benchmark&#8217;s run n: its waypoints, arm targets and punches, from its own random streams. A run starts as the benchmark&#8217;s do, 3 s on a crane at the harness&#8217;s stance and then let go, and ends when the pelvis is below 0.2 m.</span>
    <b>Authors' config</b><span>The policy as its authors trained or deployed it: their gains, poses and observation, read from their config (Isaac Lab env.yaml, unitree_rl_gym, a Unitree deploy.yaml). Run with the same drive as the port, so stages differ only in values.</span>
    <b>Port's values</b><span>The values the benchmark's adapter feeds and applies, with the policy driving every joint it lists.</span>
    <b>Harness holds arms</b><span>The harness takes the arm joints from the policy and holds them at its stance with its own gains.</span>
    <b>Random arm walk</b><span>The harness moves the arms at random, as teleoperation does.</span>
    <b>Plus punches</b><span>A punch on a random link every waypoint, rising to 500 N.</span>
    <b>Port, full benchmark</b><span>All of the above with the port's own view of the arms (some ports hide arm motion from the policy).</span>
    <b>Bench, Here</b><span>Mean survival over runs 0&#8211;{d["seeds"] - 1}: the benchmark&#8217;s MuJoCo numbers for those run ids, and the port under the full benchmark here on the same draws. A row&#8217;s run table marks a run here or in the benchmark&#8217;s PhysX that ends more than 1 s from the benchmark&#8217;s MuJoCo run.</span>
  </div>
  <p style="color: var(--muted); font-size: .92rem">Evidence L1: one runner, one model, stated assumptions. A verdict names the step that costs survival here; it is not a claim about the benchmark's own numbers, and a step that is not run cannot be blamed.</p>
</section>

<section>
  <h2>Reproduce</h2>
<pre>pip install "gaitkeeper[sim]=={e(d["gaitkeeper"])}"
git clone https://github.com/rhoyn/teleop-walking-benchmark twb   # at 106faa0

git clone https://github.com/unitreerobotics/unitree_rl_gym   # rl_gym's authors' config

# read a port from its adapter (needs g++ or clang++); a port with variants reads each
gaitkeeper adapter twb/policies/rl_gym/policy.cpp --mjcf twb/assets/g1_29dof.xml --out contracts/
gaitkeeper adapter twb/policies/gr00t_wbc/policy.cpp --mjcf twb/assets/g1_29dof.xml --out contracts/

# run the ladder, with the authors' config as the first stage when there is one
gaitkeeper bench --adapter twb/policies/rl_gym/policy.cpp --onnx twb/policies/rl_gym/model.onnx \\
    --upstream unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml \\
    --mjcf twb/assets/g1_29dof.xml --seeds 10 --md rl_gym.md

# a variant, by name (its models are found next to the one given)
gaitkeeper bench --adapter twb/policies/gr00t_wbc/policy.cpp --variant gr00t_wbc_h066_p012 \\
    --onnx twb/policies/gr00t_wbc/model_walk.onnx --mjcf twb/assets/g1_29dof.xml --seeds 10</pre>
  <p>Source, method and every command: <a href="https://github.com/dundysm/gaitkeeper">github.com/dundysm/gaitkeeper</a>.</p>
  <p>Interactive: <a href="../demo/">command dead zones of the unitree_rl_lab G1 policy</a>, drag the command and watch it stand or walk.</p>
</section>

<section>
  <h2>Run it on your own policy</h2>
  <p>The same checks on a G1 policy whose config gaitkeeper reads, in one to two and a half minutes on a laptop:</p>
<pre>gaitkeeper doctor --onnx policy.onnx --config &lt;your config&gt; --mjcf scene.xml</pre>
  <p style="color: var(--muted); font-size: .92rem">The config is read by content: an Isaac Lab env.yaml, a Unitree deploy.yaml, a unitree_rl_gym or legged_gym config, or a contract. Doctor reports the contract fields no file states, the commands the policy ignores, and survival on the tour with the arms its own, moved at random and punched.</p>
</section>

<section>
  <h2>Help</h2>
  <p>A verdict that looks wrong, a config gaitkeeper cannot read, or harness logs for a blind test are all useful. <a href="https://github.com/dundysm/gaitkeeper/contribute">Good first issues</a> · <a href="https://github.com/dundysm/gaitkeeper/blob/main/CONTRIBUTING.md">Contributing</a> · <a href="https://github.com/dundysm/gaitkeeper/discussions">Discussions</a></p>
</section>
</div>
""",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "index.html"))
    ap.add_argument("--fragment", help="also write the page without the document shell here")
    a = ap.parse_args()
    d = json.loads(Path(a.data).read_text())
    head, body = page(d)
    doc = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Why each Unitree G1 walking policy in teleop-walking-benchmark '
        'scores what it does: ports read from their adapter code and run one change at a time with gaitkeeper.">\n'
        + head
        + "\n</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )
    Path(a.out).write_text(doc)
    if a.fragment:
        Path(a.fragment).write_text(head + "\n" + body)
    print(f"wrote {a.out}")
    write_readme_svg()


# -- README figure ---------------------------------------------------------------------

SVG_CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --edge:#d1d9e0; --cell:#0969da; --pt:#0969da;
  --policy:#59636e; --port:#cf222e; --arms:#9a6700; --walk:#bf8700; --punch:#8250df; --unclear:#8c959f;
  --none:#1a7f37; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --edge:#3d444d; --cell:#4493f8; --pt:#4493f8;
    --policy:#9198a1; --port:#f85149; --arms:#d29922; --walk:#e3b341; --punch:#a371f7; --unclear:#6e7681;
    --none:#3fb950; } }
text { font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif; fill: var(--fg); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.bg { fill: var(--bg); }
.muted { fill: var(--muted); }
.h { font-size: 15px; font-weight: 700; }
.lab { font-size: 11px; fill: var(--muted); }
.name { font-size: 12px; }
.val { font-size: 10.5px; }
.cell { fill: var(--cell); }
.empty { fill: none; stroke: var(--edge); stroke-dasharray: 2 3; }
.ax { stroke: var(--edge); }
.diag { stroke: var(--muted); stroke-dasharray: 3 4; }
.pt { fill: var(--pt); }
.big { font-size: 26px; font-weight: 700; }
"""


def readme_svg(d: dict) -> str:
    total = float(d["tour_s"])
    order = ["port", "arms", "walk", "punch", "policy", "unclear", "none"]
    pols = [p for p in d["policies"] if p.get("stages")]
    pols.sort(key=lambda p: (order.index(p["cause"]), p["name"]))
    cols = [
        ("upstream", "Authors'"),
        ("own", "Port"),
        ("arms_hold", "Arms held"),
        ("arms_walk", "Arm walk"),
        ("punches", "Punches"),
        ("port", "Full"),
    ]
    x0, cw, ch, gap, top = 292, 54, 19, 3, 74
    W = 960
    H = top + len(pols) * (ch + gap) + 40
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
        '<title id="t">G1 Port Audit: what stops each policy</title>',
        f'<desc id="d">Mean survival on the teleop-walking-benchmark tour (out of {total:.0f} s) at each stage, '
        f"for {len(pols)} ported G1 policies, with the cause gaitkeeper names; and gaitkeeper survival against the benchmark.</desc>",
        f"<style>{SVG_CSS}</style>",
        f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
        '<text x="20" y="30" class="h">What stops each policy</text>',
        f'<text x="20" y="48" class="lab">Mean survival (s of {total:.0f}) on the benchmark tour, one change at a time</text>',
    ]
    for i, (_, lab) in enumerate(cols):
        o.append(
            f'<text x="{x0 + i * (cw + gap) + cw / 2}" y="{top - 8}" class="lab" text-anchor="middle">{e(lab)}</text>'
        )
    for r, p in enumerate(pols):
        y = top + r * (ch + gap)
        o.append(f'<text x="20" y="{y + 13.5}" class="name mono">{e(p["name"])}</text>')
        o.append(
            f'<text x="192" y="{y + 13.5}" class="name" style="fill: var(--{p["cause"]}); font-weight:600">{e(CAUSES[p["cause"]][0])}</text>'
        )
        for i, (k, _) in enumerate(cols):
            x = x0 + i * (cw + gap)
            st = p["stages"].get(k)
            if st is None:
                o.append(
                    f'<rect class="empty" x="{x + 0.5}" y="{y + 0.5}" width="{cw - 1}" height="{ch - 1}" rx="2"/>'
                )
                continue
            v = st["mean_survival_s"]
            a = 0.12 + 0.88 * v / total
            light = a > 0.55
            o.append(
                f'<rect class="cell" x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" opacity="{a:.2f}"><title>{e(p["name"])}, {e(_)}: {v:.1f} s</title></rect>'
            )
            col = ' style="fill:#ffffff"' if light else ""
            o.append(
                f'<text x="{x + cw / 2}" y="{y + 13.5}" class="val mono" text-anchor="middle"{col}>{v:.0f}</text>'
            )
    o.append(
        f'<text x="20" y="{H - 16}" class="lab">Dashed: stage not run (no upstream config read, or the policy lists no arms). '
        "Full results: dundysm.github.io/gaitkeeper/results</text>"
    )
    # scatter
    sx, sy, sw, sh = 700, 96, 236, 230
    s = sw / total
    t = sh / total
    o.append(f'<text x="{sx - 30}" y="30" class="h">Agreement with the benchmark</text>')
    o.append(
        f'<text x="{sx - 30}" y="68" class="big">{agreement(d)["pearson"]:.2f}</text><text x="{sx + 34}" y="68" class="lab">Pearson</text>'
    )
    o.append(
        f'<text x="{sx + 104}" y="68" class="big">{agreement(d)["spearman"]:.2f}</text><text x="{sx + 168}" y="68" class="lab">Spearman</text>'
    )
    o.append(
        f'<line class="ax" x1="{sx}" y1="{sy + sh}" x2="{sx + sw}" y2="{sy + sh}"/><line class="ax" x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy + sh}"/>'
    )
    o.append(f'<line class="diag" x1="{sx}" y1="{sy + sh}" x2="{sx + sw}" y2="{sy}"/>')
    for v in (0, 30, 60, 90):
        o.append(
            f'<text x="{sx + v * s}" y="{sy + sh + 15}" class="lab mono" text-anchor="middle">{v}</text>'
        )
        o.append(
            f'<text x="{sx - 6}" y="{sy + sh - v * t + 4}" class="lab mono" text-anchor="end">{v}</text>'
        )
    o.append(
        f'<text x="{sx + sw / 2}" y="{sy + sh + 32}" class="lab" text-anchor="middle">gaitkeeper: port, full benchmark (s)</text>'
    )
    o.append(
        f'<text transform="translate({sx - 32} {sy + sh / 2}) rotate(-90)" class="lab" text-anchor="middle">benchmark, MuJoCo, same runs (s)</text>'
    )
    for p in pols:
        if same(p) is not None:
            x, y = p["stages"]["port"]["mean_survival_s"], same(p)
            o.append(
                f'<circle class="pt" cx="{sx + x * s:.1f}" cy="{sy + sh - y * t:.1f}" r="4.5"><title>{e(p["name"])}: {x:.1f} s here, {y:.1f} s in the benchmark</title></circle>'
            )
    o.append("</svg>")
    return "\n".join(o)


def write_readme_svg() -> None:
    d = json.loads(DATA.read_text())
    out = ROOT / "docs" / "assets" / "audit.svg"
    out.write_text(readme_svg(d))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
