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
}

CSS = """
/* Layout: one reading column, a summary band, then a list of policies that open in place. */
:root {
  --bg: #f4f6f8; --panel: #ffffff; --ink: #121820; --muted: #566273; --rule: #d6dce4;
  --accent: #1f5f8b; --bar: #9fb3c8; --bar-up: #1f5f8b;
  --c-policy: #5d6b7d; --c-port: #b4372b; --c-arms: #a3650a; --c-walk: #c98a12;
  --c-punch: #7a5bb0; --c-unclear: #8a94a3;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141a; --panel: #161d25; --ink: #e6ebf1; --muted: #9aa6b5; --rule: #2a3440;
  --accent: #6fb3e0; --bar: #3e5166; --bar-up: #6fb3e0;
  --c-policy: #8f9db0; --c-port: #ef6f5f; --c-arms: #e0a03a; --c-walk: #f0c050;
  --c-punch: #ad92e0; --c-unclear: #7d8796; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f141a; --panel: #161d25; --ink: #e6ebf1; --muted: #9aa6b5; --rule: #2a3440;
  --accent: #6fb3e0; --bar: #3e5166; --bar-up: #6fb3e0;
  --c-policy: #8f9db0; --c-port: #ef6f5f; --c-arms: #e0a03a; --c-walk: #f0c050;
  --c-punch: #ad92e0; --c-unclear: #7d8796; color-scheme: dark; }
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
summary .name { font: 600 0.95rem var(--f-mono); }
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
        f'<text transform="translate(10 {(H - P) / 2}) rotate(-90)" text-anchor="middle">benchmark, MuJoCo (s)</text>'
    )
    for p in pols:
        if p.get("stages") and p.get("benchmark_mujoco") is not None:
            x, y = p["stages"]["port"]["mean_survival_s"], p["benchmark_mujoco"]
            out.append(
                f'<circle class="pt" cx="{P + x * s:.1f}" cy="{H - P - y * t:.1f}" r="4"><title>{e(p["name"])}: {x:.1f} s here, {y:.1f} s in the benchmark</title></circle>'
            )
    out.append("</svg>")
    return "".join(out)


def chip(cause: str) -> str:
    label, _ = CAUSES[cause]
    return f'<span class="chip" style="color: var(--c-{cause})">{e(label)}</span>'


def page(d: dict) -> tuple[str, str]:
    pols = d["policies"]
    total = float(d["tour_s"])
    benched = [p for p in pols if p.get("stages")]
    order = ["port", "arms", "walk", "punch", "policy", "unclear"]
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
        bench = "–" if p.get("benchmark_mujoco") is None else f"{p['benchmark_mujoco']:.1f}"
        mine = f"{st['port']['mean_survival_s']:.1f}"
        stage_rows = "".join(
            f'<tr><td>{e(label)}</td><td class="num">{st[k]["mean_survival_s"]:.1f}</td><td class="num">{st[k]["complete"]}/{st[k]["runs"]}</td></tr>'
            for k, label in STAGES + [("port_quiet", "Port, arms still, no punches")]
            if k in st
        )
        notes = "".join(f"<li>{e(n)}</li>" for n in p.get("notes", []))
        af = "".join(f"<li>{e(f)}</li>" for f in p.get("adapter_findings", []))
        pr = p.get("probe") or {}
        read = (
            f"Read from <code>policies/{e(p['name'])}/policy.cpp</code>: {pr.get('obs')} observations "
            f"({e(', '.join(pr.get('terms', [])))}), history {pr.get('history')}, {pr.get('actions')} actions, "
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
  <summary><span class="name">{e(p["name"])}</span>{chip(p["cause"])}{spark(st, total)}<span class="verdict">{e(p["verdict"])}</span><span class="s num">{bench}</span><span class="s num">{mine}</span></summary>
  <div class="detail">
    <div class="stages"><table><thead><tr><th>Stage</th><th class="num">Mean survival (s)</th><th class="num">Completed</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
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
  <div class="meta"><span>{e(d["benchmark"])}</span><span>{d["seeds"]} seeds per stage</span><span>MuJoCo, the benchmark's g1_29dof.xml</span><span>evidence L1</span><span>{e(d["generated"])}</span></div>
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
        <span><span class="big num">{d["pearson"]:.2f}</span> <span class="eyebrow">Pearson</span></span>
        <span><span class="big num">{d["spearman"]:.2f}</span> <span class="eyebrow">Spearman</span></span>
      </div>
      {scatter(pols, total)}
      <p style="font-size: .84rem; color: var(--muted)">{d["n_corr"]} policies. Survival of the port under the full benchmark here, against the benchmark's own MuJoCo runs, with no contract written by hand.</p>
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

<section>
  <h2>Not benched</h2>
  <p style="color: var(--muted); font-size: .92rem">The reader probes an adapter by running it on the CPU and checks that gaitkeeper rebuilds its observation exactly. It refuses rather than guesses:</p>
  <div class="notread">{nr}</div>
</section>

<section>
  <h2>How to read a verdict</h2>
  <div class="defs">
    <b>Authors' config</b><span>The policy as its authors trained or deployed it: their gains, poses and observation, read from their config (Isaac Lab env.yaml, unitree_rl_gym, a Unitree deploy.yaml). Run with the same drive as the port, so stages differ only in values.</span>
    <b>Port's values</b><span>The values the benchmark's adapter feeds and applies, with the policy driving every joint it lists.</span>
    <b>Harness holds arms</b><span>The harness takes the arm joints from the policy and holds them at its stance with its own gains.</span>
    <b>Random arm walk</b><span>The harness moves the arms at random, as teleoperation does.</span>
    <b>Plus punches</b><span>A punch on a random link every waypoint, rising to 500 N.</span>
    <b>Port, full benchmark</b><span>All of the above with the port's own view of the arms (some ports hide arm motion from the policy).</span>
  </div>
  <p style="color: var(--muted); font-size: .92rem">Evidence L1: one runner, one model, stated assumptions. A verdict names the step that costs survival here; it is not a claim about the benchmark's own numbers, and a step that is not run cannot be blamed.</p>
</section>

<section>
  <h2>Reproduce</h2>
<pre>pip install "gaitkeeper[sim]=={e(d["gaitkeeper"])}"
git clone https://github.com/rhoyn/teleop-walking-benchmark twb   # at 6d331a8

git clone https://github.com/unitreerobotics/unitree_rl_gym   # rl_gym's authors' config

# read a port from its adapter (needs g++ or clang++)
gaitkeeper adapter twb/policies/rl_gym/policy.cpp --mjcf twb/assets/g1_29dof.xml --out contracts/

# run the ladder, with the authors' config as the first stage when there is one
gaitkeeper bench --adapter twb/policies/rl_gym/policy.cpp --onnx twb/policies/rl_gym/model.onnx \\
    --upstream unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml \\
    --mjcf twb/assets/g1_29dof.xml --seeds 3 --md rl_gym.md</pre>
  <p>Source, method and every command: <a href="https://github.com/dundysm/gaitkeeper">github.com/dundysm/gaitkeeper</a>.</p>
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


if __name__ == "__main__":
    main()
