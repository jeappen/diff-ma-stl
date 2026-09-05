# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/jax:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# libGL for matplotlib/video backends; git for the HTTPS VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential pkg-config \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Freeze the base image's exact jax/jaxlib/ml_dtypes versions (e.g. 0.4.x on
# 24.04-py3) into a pip constraints file. Passing --constraint forces pip to
# keep these installed verbatim; if a dep is incompatible, resolution errors
# out loudly instead of silently replacing the container's build.
RUN python - <<'PY' > /tmp/jax-pins.txt
import importlib.metadata as m
for pkg in ("jax", "jaxlib", "ml_dtypes"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        pass
PY
RUN echo "=== jax pins ===" && cat /tmp/jax-pins.txt

# Preinstall CPU-only torch BEFORE diff-spec. diff-spec depends on torch, and
# plain `pip install torch` grabs the CUDA build from PyPI — those wheels ship
# their own CUDA/cuDNN libs and can shadow the NGC JAX image's CUDA stack,
# breaking JAX at runtime. Using PyTorch's CPU index pins us to CPU wheels, so
# diff-spec's later install sees torch already satisfied and doesn't try to
# pull a GPU build.
RUN pip3 install --constraint /tmp/jax-pins.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

# Append the installed CPU torch/torchvision versions to the pin file so any
# later install in this Dockerfile (diff-spec's --force-reinstall, etc.) can't
# silently swap them for the CUDA wheel on PyPI.
RUN python - <<'PY' >> /tmp/jax-pins.txt
import importlib.metadata as m
for pkg in ("torch", "torchvision"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        pass
PY

# Install Python deps. The VCS installs (jaxproxqp, diff-spec) are public repositories; an
# optional GitHub token (BuildKit secret `github_pat`, see scripts/rebuild_docker_image.sh)
# is only needed for private forks. When present it is configured into git via `insteadOf`
# inside this single RUN, so pip's metadata records only the plain `https://github.com/...`
# URLs and no token reaches an image layer; the git config is unset at the end of the RUN.
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN --mount=type=secret,id=github_pat,required=false \
    TOKEN="$(cat /run/secrets/github_pat 2>/dev/null || true)" \
 && if [ -n "$TOKEN" ]; then git config --global url."https://${TOKEN}:x-oauth-basic@github.com/".insteadOf "https://github.com/"; fi \
 && pip install --constraint /tmp/jax-pins.txt -r /tmp/requirements-docker.txt \
 && pip install --constraint /tmp/jax-pins.txt --no-deps \
      "git+https://github.com/oswinso/jaxproxqp.git" \
 && pip install --constraint /tmp/jax-pins.txt \
      "diff-spec @ git+https://github.com/jeappen/diff-spec.git@feature/jax" \
 && (git config --global --unset url."https://${TOKEN}:x-oauth-basic@github.com/".insteadOf || true) \
 && unset TOKEN

# Fail the build if anything slipped past the constraint and changed jax, or
# if torch accidentally came back as a CUDA build (which would shadow the NGC
# CUDA stack that JAX depends on).
RUN python - <<'PY'
import importlib.metadata as m
expected = dict(l.strip().split("==") for l in open("/tmp/jax-pins.txt") if l.strip())
for pkg, want in expected.items():
    got = m.version(pkg)
    print(f"{pkg}: want {want}, got {got}")
    assert got == want, f"{pkg} changed from {want} to {got} — a dep in requirements-docker.txt is conflicting"

import torch
print(f"torch: version={torch.__version__}, cuda_available={torch.cuda.is_available()}")
assert "+cpu" in torch.__version__ or not torch.cuda.is_available(), (
    f"torch is a CUDA build ({torch.__version__}); it must be CPU-only so it does "
    "not shadow the NGC JAX image's CUDA/cuDNN libraries"
)
PY

# Copy the repo and install in editable mode.
# If the repo itself is private, swap COPY for a PAT-authenticated HTTPS clone
# under another `--mount=type=secret,id=github_pat` RUN.
COPY . /workspace
RUN pip install -e .

# Verify JAX still sees the GPU at build time (will only show CPU during build,
# but imports at least confirm nothing got broken). Import gcbfplus.algo (not
# just gcbfplus) so the full chain — env.wrapper -> diffusion.planner ->
# diffusion.util.data (which does `from gym import spaces`) — actually runs.
# `import gcbfplus` alone does not traverse deep enough to catch missing runtime
# deps like gym or networkx; they only surface at `docker run` time.
RUN python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)" \
 && python -c "import gcbfplus.algo; print('gcbfplus.algo import OK')" \
 && python -c "import ds.stl, ds.stl_jax, ds.ma_stl_jax; print('diff-spec (ds) import OK')"

CMD ["/bin/bash"]
