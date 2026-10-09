"""Render the README's hero animation from gaitkeeper's own runner.

Two closed-loop runs of the unitree_rl_lab G1 velocity policy on unitree_mujoco's G1
scene, side by side: a 0.15 m/s forward command (inside the dead zone, the robot
stands still) and a 0.50 m/s command (it walks). Each panel shows the command and the
measured forward speed.

    pip install pillow
    gaitkeeper fetch g1_rl_lab g1_unitree_mujoco
    MUJOCO_GL=osmesa python tools/render_readme_media.py --out docs/assets/dead-zone.gif

Needs an offscreen OpenGL (MUJOCO_GL=egl on a GPU machine, osmesa on CPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gaitkeeper import fixtures
from gaitkeeper.policy import OnnxPolicy
from gaitkeeper.presets import apply_preset
from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
from gaitkeeper.runner import RunConfig, Runner

PRESET = "unitree_rl_lab_g1_29dof_velocity@4960b84"
BG = (13, 17, 23)  # GitHub dark background
FG = (230, 237, 243)
MUTED = (139, 148, 158)
RED = (248, 81, 73)
GREEN = (63, 185, 80)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["Inter-SemiBold.otf", "Inter-Medium.otf"] if bold else ["Inter-Regular.otf"]
    names += ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    roots = [Path("/usr/share/fonts/opentype/inter"), Path("/usr/share/fonts/truetype/dejavu")]
    for n in names:
        for r in roots:
            if (r / n).exists():
                return ImageFont.truetype(str(r / n), size)
    return ImageFont.load_default()


def record(runner: Runner, vx: float, seconds: float) -> dict[str, np.ndarray]:
    res = runner.run(RunConfig(command=(vx, 0.0, 0.0), seconds=seconds, record=True))
    assert res.survived, f"fell at {res.fell_at}"
    return {"qpos": res.log["qpos"], "vel": res.vel}


def frames(
    model_path: str, run: dict[str, np.ndarray], stride: int, w: int, h: int
) -> list[np.ndarray]:
    m = mujoco.MjModel.from_xml_path(model_path)
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(h, 480)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    m.mat_reflectance[:] = 0.0  # reflections cost GIF bytes and add nothing
    cam = (
        mujoco.MjvCamera()
    )  # fixed side view: a walking robot crosses the frame, a standing one does not
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    x0 = float(run["qpos"][0, 0])
    cam.lookat[:] = (x0 + 1.45, float(run["qpos"][0, 1]), 0.6)
    cam.distance, cam.azimuth, cam.elevation = 3.9, 90.0, -6.0
    opt = mujoco.MjvOption()
    out = []
    for q in run["qpos"][::stride]:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam, opt)
        out.append(r.render().copy())
    return out


def panel(img: np.ndarray, title: str, cmd: float, speed: float, ok: bool) -> Image.Image:
    w, h = img.shape[1], img.shape[0]
    head = 64
    p = Image.new("RGB", (w, h + head), BG)
    p.paste(Image.fromarray(img), (0, head))
    dr = ImageDraw.Draw(p)
    dr.text((16, 10), title, font=font(17, True), fill=FG)
    dr.text((16, 36), f"command {cmd:.2f} m/s", font=font(14), fill=MUTED)
    col = GREEN if ok else RED
    txt = f"measured {abs(speed) if abs(speed) < 0.005 else speed:+.2f} m/s"
    tw = dr.textlength(txt, font=font(14, True))
    dr.text((w - 16 - tw, 36), txt, font=font(14, True), fill=col)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/dead-zone.gif")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=17)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--colors", type=int, default=64)
    args = ap.parse_args()

    pol, scene = fixtures.get("g1_rl_lab"), fixtures.get("g1_unitree_mujoco")
    for s in (pol, scene):
        fixtures.require(s.name)
    c, _ = read_unitree_deploy(str(pol.path("deploy.yaml")), str(pol.path("policy.onnx")))
    apply_preset(c, PRESET)
    model_path = str(scene.path(scene.scene))
    runner = Runner(c, model_path, OnnxPolicy(str(pol.path("policy.onnx"))))

    policy_hz = 50
    stride = max(1, round(policy_hz / args.fps))
    runs = [
        ("Inside the dead zone", 0.15, record(runner, 0.15, args.seconds)),
        ("Outside it", 0.50, record(runner, 0.50, args.seconds)),
    ]
    rendered = [frames(model_path, r, stride, args.width, args.height) for _, _, r in runs]
    n = min(len(f) for f in rendered)
    gap = 8
    out = []
    for i in range(n):
        k = min(i * stride, len(runs[0][2]["vel"]) - 1)
        lo = max(0, k - policy_hz // 2)  # half-second moving average of forward speed
        panels = []
        for (title, cmd, r), fr in zip(runs, rendered):
            v = float(np.mean(r["vel"][lo : k + 1, 0]))
            panels.append(panel(fr[i], title, cmd, v, ok=v > 0.5 * cmd))
        W = sum(p.width for p in panels) + gap * (len(panels) - 1)
        canvas = Image.new("RGB", (W, panels[0].height), BG)
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + gap
        out.append(canvas)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # One palette for every frame: the scene's colors from a sample of frames, plus exact slots
    # for the label colors so the red and green survive. Shared, so frames compress together.
    sample = out[:: max(1, len(out) // 8)]
    strip = Image.new("RGB", (out[0].width, out[0].height * len(sample)), BG)
    for i, im in enumerate(sample):
        strip.paste(im, (0, i * out[0].height))
    labels = [BG, FG, MUTED, RED, GREEN]
    scene_pal = strip.quantize(colors=args.colors - len(labels), method=Image.Quantize.MEDIANCUT)
    entries = scene_pal.getpalette()[: 3 * (args.colors - len(labels))]
    for c in labels:
        entries += list(c)
    palette = Image.new("P", (1, 1))
    palette.putpalette(entries + [0] * (768 - len(entries)))
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in out]
    pal[0].save(
        args.out,
        save_all=True,
        append_images=pal[1:],
        duration=round(1000 * stride / policy_hz),  # real time
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"{args.out}: {len(out)} frames, {Path(args.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
