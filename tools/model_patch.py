"""Write ``model_patch.json`` for a recorded trace, in the source environment.

The patch lets any MuJoCo version rebuild the simulated model from the
exported XML: solver options and every numeric field that differs, startup
randomization included. Run it with the MuJoCo the recorder used:

    python tools/model_patch.py runs/g1_golden_a
"""

import json
import sys
from pathlib import Path

import mujoco

from gaitkeeper.models import apply_randomization, model_patch


def main(path: str) -> None:
    p = Path(path)
    live = mujoco.MjModel.from_binary_path(str(p / "model.mjb"))
    dr = p / "dr.json"
    if dr.exists():
        apply_randomization(live, json.loads(dr.read_text()))
    xml = mujoco.MjModel.from_xml_path(str(p / "model_xml" / "scene.xml"))
    patch = model_patch(live, xml)
    (p / "model_patch.json").write_text(json.dumps(patch))
    print(
        f"{p / 'model_patch.json'}: options {sorted(patch['opt'])}, {len(patch['fields'])} fields"
    )


if __name__ == "__main__":
    main(sys.argv[1])
