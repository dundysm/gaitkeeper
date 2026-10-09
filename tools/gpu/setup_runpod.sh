#!/usr/bin/env bash
# Set up a GPU machine for the Isaac Lab session (docs/GPU_SESSION.md).
#
#   bash tools/gpu/setup_runpod.sh            # installs into $WORK (default /workspace)
#
# Installs, each pinned: Isaac Sim 5.1.0 and Isaac Lab 2.3.0 (the versions unitree_rl_lab
# 4960b84 pins), unitree_rl_lab at 4960b84, the G1 29 dof USD from unitree_model, and
# gaitkeeper from this checkout. Then runs `record_isaaclab.py doctor`, which builds the task
# and steps it. Safe to rerun: finished steps are skipped.
#
# Isaac Sim asks you to accept the NVIDIA Omniverse EULA on its first start. Read it; the
# script does not answer for you. To answer it up front: export OMNI_KIT_ACCEPT_EULA=YES.
set -euo pipefail

WORK="${WORK:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # the gaitkeeper checkout
VENV="$WORK/isaac-env"
ISAACLAB_REF="v2.3.0"
RL_LAB_COMMIT="4960b84"
USD_REL="G1/29dof/usd/g1_29dof_rev_1_0"
mkdir -p "$WORK"
cd "$WORK"
log() { printf '\n== %s\n' "$*"; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }

log "checks"
glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
python3 - "$glibc" <<'PY' || die "GLIBC $glibc; Isaac Sim 5.1 pip wheels need 2.35 or newer (Ubuntu 22.04+)"
import sys
major, minor = map(int, sys.argv[1].split("."))
sys.exit(0 if (major, minor) >= (2, 35) else 1)
PY
command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1')"
[ "${mem%%.*}" -ge 15000 ] || echo "warning: ${mem} MiB of GPU memory; 16 GB or more is the tested size"
df -h "$WORK" | tail -1
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  # Vulkan loader and the X libraries Kit loads even headless; tmux for long runs.
  DEBIAN_FRONTEND=noninteractive apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git tmux libvulkan1 libglu1-mesa \
    libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libsm6 libxext6 libegl1 libgl1 >/dev/null
fi
command -v git >/dev/null || die "no git"
[ -f /etc/vulkan/icd.d/nvidia_icd.json ] || [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] \
  || echo "warning: no NVIDIA Vulkan ICD; start the container with NVIDIA_DRIVER_CAPABILITIES=all"

log "python 3.11 venv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  if ! command -v uv >/dev/null; then
    python3 -m pip install --quiet uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv --python 3.11 --seed "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' \
  || die "the venv is not Python 3.11"
pip install --quiet --upgrade pip

log "torch 2.7.0 (cu128) and Isaac Sim 5.1.0"
python -c 'import torch, sys; sys.exit(torch.__version__.split("+")[0] != "2.7.0")' 2>/dev/null \
  || pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c 'import importlib.metadata as m, sys; sys.exit(m.version("isaacsim") != "5.1.0")' 2>/dev/null \
  || pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

log "Isaac Lab $ISAACLAB_REF"
[ -d IsaacLab ] || git clone --depth 1 --branch "$ISAACLAB_REF" https://github.com/isaac-sim/IsaacLab.git
(cd IsaacLab && [ "$(git describe --tags)" = "$ISAACLAB_REF" ] || die "IsaacLab is not at $ISAACLAB_REF")
installed() { pip show "$1" >/dev/null 2>&1; }   # not `import`: a same-named folder in $WORK would pass
# flatdict (an Isaac Lab dependency) does not build with an isolated, recent setuptools.
installed flatdict || { pip install --quiet "setuptools<81" wheel && pip install --quiet --no-build-isolation flatdict==4.0.1; }
installed isaaclab || (cd IsaacLab && ./isaaclab.sh --install)
installed isaaclab || die "Isaac Lab did not install; see the output above"

log "unitree_rl_lab $RL_LAB_COMMIT"
if [ ! -d unitree_rl_lab ]; then
  git clone https://github.com/unitreerobotics/unitree_rl_lab.git
fi
(cd unitree_rl_lab && git fetch --quiet origin && git checkout --quiet "$RL_LAB_COMMIT")
installed unitree_rl_lab || pip install -e unitree_rl_lab/source/unitree_rl_lab
installed unitree_rl_lab || die "unitree_rl_lab did not install"

log "G1 29 dof USD from the unitree_model dataset"
pip install --quiet "huggingface_hub>=0.24"
USD_DIR="$(python - "$WORK/unitree_model" "$USD_REL" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
dest, rel = sys.argv[1], sys.argv[2]
sha = HfApi().dataset_info("unitreerobotics/unitree_model").sha
snapshot_download(
    "unitreerobotics/unitree_model", repo_type="dataset", revision=sha,
    allow_patterns=[f"{rel}/*"], local_dir=dest,
)
open(f"{dest}/REVISION", "w").write(sha + "\n")
print(f"{dest}/{rel}")
PY
)"
USD="$(find "$USD_DIR" -maxdepth 1 -name '*.usd' | sort | awk 'NR==1')"
[ -f "$USD" ] || die "no .usd under $USD_DIR"
echo "USD $USD (unitree_model revision $(cat "$WORK/unitree_model/REVISION"))"

log "gaitkeeper from $HERE"
pip install --quiet -e "$HERE[sim,dev]" onnxruntime
gaitkeeper fetch all

POLICY="$(python -c 'from gaitkeeper.fixtures import get; print(get("g1_rl_lab").dir())')"
cat > "$WORK/gaitkeeper_gpu.env" <<ENV
# source this before the commands in docs/GPU_SESSION.md
source "$VENV/bin/activate"
export GK="$HERE"
export USD="$USD"
export ONNX="$POLICY/policy.onnx"
export DEPLOY="$POLICY/deploy.yaml"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-}"   # set to YES once you have read the EULA
export SCENE="\$(python -c 'from gaitkeeper.fixtures import get; f = get("g1_unitree_mujoco"); print(f.path(f.scene))')"
ENV
echo "wrote $WORK/gaitkeeper_gpu.env"

log "smoke test (first start of Isaac Sim takes several minutes)"
# shellcheck disable=SC1091
source "$WORK/gaitkeeper_gpu.env"
python "$HERE/tools/record_isaaclab.py" doctor --usd "$USD" --onnx "$ONNX"
log "ready: source $WORK/gaitkeeper_gpu.env, then follow docs/GPU_SESSION.md"
