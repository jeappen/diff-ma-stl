#!/usr/bin/env bash
# Smoke tests for the gcbfplus Docker image built from ./Dockerfile.
# Assumes the image is already built, e.g.:
#   GITHUB_PAT=$(gh auth token) DOCKER_BUILDKIT=1 docker build --secret id=github_pat,env=GITHUB_PAT -t gcbfplus:jax2404 .
#
# Usage:
#   bash scripts/test_docker_image.sh [IMAGE_TAG]
#
# Runs three checks:
#   1. GPU visibility inside the container
#   2. Core package imports (jax, gcbfplus.algo, jaxproxqp, ds.*, gym)
#   3. End-to-end: short test.py rollout with the pretrained DubinsCar GCBF+
#      controller (needs pretrained/ mounted from the host)

set -euo pipefail

IMAGE="${1:-gcbfplus:jax2404}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Gurobi WLS license file on the host. Required by the STLPY planner path
# (ds.stl → gurobipy). Default: .secrets/gurobi.lic in the repo (git-ignored
# and docker-ignored, never baked into the image). Override by exporting
# GUROBI_LICENSE before running.
GUROBI_LICENSE="${GUROBI_LICENSE:-$REPO_ROOT/.secrets/gurobi.lic}"
GUROBI_MOUNT=()
if [[ -f "$GUROBI_LICENSE" ]]; then
  GUROBI_MOUNT=(-v "$GUROBI_LICENSE:/opt/gurobi/gurobi.lic:ro")
  echo "Gurobi license: $GUROBI_LICENSE"
else
  echo "WARN: Gurobi license not found at $GUROBI_LICENSE — STLPY planner tests will fail." >&2
fi

cd "$REPO_ROOT"

echo "=========================================================="
echo "[1/3] GPU visible inside $IMAGE"
echo "=========================================================="
docker run --rm --gpus all "$IMAGE" \
  python -c "import jax, jaxlib; \
print('jax', jax.__version__, 'jaxlib', jaxlib.__version__); \
print('devices:', jax.devices()); \
print('default device of ones:', jax.numpy.ones(3).device)"

echo
echo "=========================================================="
echo "[2/3] Core package imports"
echo "=========================================================="
docker run --rm -i --gpus all "${GUROBI_MOUNT[@]}" "$IMAGE" python - <<'PY'
import importlib, os, sys
failures = []
# `gcbfplus.algo` (not just `gcbfplus`) forces the full import chain through
# env.wrapper -> diffusion.planner -> diffusion.util.data, which pulls `gym`.
# Importing only `gcbfplus` does not traverse deep enough to catch a missing
# runtime dep like gym. diff-spec exposes itself as `ds`, not `diff_spec`.
for mod in ("gcbfplus.algo", "jaxproxqp", "gym",
            "ds", "ds.stl", "ds.stl_jax", "ds.ma_stl_jax",
            "gurobipy", "stlpy"):
    try:
        importlib.import_module(mod)
        print(f"OK   {mod}")
    except Exception as e:
        print(f"FAIL {mod}: {e}")
        failures.append(mod)

# Confirm the Gurobi WLS license is visible + valid by instantiating a Model.
# gurobipy raises GurobiError if the license is missing or invalid.
if os.path.isfile(os.environ.get("GRB_LICENSE_FILE", "")):
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.start()
        gp.Model("smoke", env=env)
        print(f"OK   gurobi license at {os.environ['GRB_LICENSE_FILE']}")
        env.dispose()
    except Exception as e:
        print(f"FAIL gurobi license: {e}")
        failures.append("gurobi-license")
else:
    print("SKIP gurobi license check (no license file mounted)")

if failures:
    print(f"\nHARD FAIL: {failures}")
    sys.exit(1)
PY

echo
echo "=========================================================="
echo "[3/3] End-to-end: short pretrained DubinsCar rollout"
echo "=========================================================="
# Exit 77 (autotools "SKIPPED" convention) when step 3's prerequisites are
# missing, so callers / CI can distinguish "end-to-end check did not run"
# from "end-to-end check passed". Steps 1 and 2 have already succeeded by
# this point, but we do not advertise an all-passed result for a partial run.
if [[ ! -d "$REPO_ROOT/pretrained/DubinsCar/gcbf+" ]]; then
  echo "SKIP: $REPO_ROOT/pretrained/DubinsCar/gcbf+ not found on host." >&2
  echo "      Download/copy pretrained models before running this step." >&2
  echo "      (Steps 1 and 2 passed; step 3 was not run.)" >&2
  exit 77
fi

MOUNTS=(-v "$REPO_ROOT/pretrained:/workspace/pretrained:ro")
if [[ -d "$REPO_ROOT/datasets" ]]; then
  MOUNTS+=(-v "$REPO_ROOT/datasets:/workspace/datasets:ro")
fi

if [[ ${#GUROBI_MOUNT[@]} -eq 0 ]]; then
  echo "SKIP: STLPY planner requires a Gurobi license mount (GUROBI_LICENSE not found)." >&2
  echo "      Drop a WLS gurobi.lic at .secrets/gurobi.lic or export GUROBI_LICENSE." >&2
  echo "      (Steps 1 and 2 passed; step 3 was not run.)" >&2
  exit 77
fi

# STLPY uses a Gurobi MILP solver, which is not JIT-friendly — hence
# --nojit-rollout. `mseq3` with spec-len 15 is the smallest spec that actually
# exercises the stlpy planner path end-to-end. --async-planner + --ignore-on-
# finish match the paper's evaluation setting: sync
# mode gives near-zero finish rate because all agents advance goal timers in
# lockstep and the STL spec fails as soon as any agent is behind schedule.
docker run --rm --gpus all "${MOUNTS[@]}" "${GUROBI_MOUNT[@]}" "$IMAGE" \
  python test.py \
    --path pretrained/DubinsCar/gcbf+/ \
    --epi 1 \
    --area-size 4 \
    -n 8 \
    --obs 0 \
    --nojit-rollout \
    --planner stlpy \
    --spec mseq3 \
    --spec-len 15 \
    --goal-sample-interval 20 \
    --async-planner \
    --ignore-on-finish \
    --no-video

echo
echo "All smoke tests passed for $IMAGE."
