#!/usr/bin/env bash
# Remove the existing gcbfplus Docker image and dangling layers, then rebuild from the current
# Dockerfile with a clean cache. Use after Dockerfile / requirements changes.
#
# Usage:
#   bash scripts/rebuild_docker_image.sh [IMAGE_TAG]          # default gcbfplus:jax2404
#   bash scripts/rebuild_docker_image.sh IMAGE_TAG --prune-buildkit   # also wipe BuildKit's cache
#                                                                    # (affects ALL docker builds on this host)
#
# GitHub token (optional): the VCS dependencies (jaxproxqp, diff-spec) are public. If you build
# from a private fork, provide a fine-grained PAT and it is passed as a BuildKit secret (never
# stored in an image layer). Resolution order: $GITHUB_PAT, .secrets/read_gcbfplus_PAT.txt,
# `gh auth token`.

set -euo pipefail

IMAGE="${1:-gcbfplus:jax2404}"
PRUNE_BUILDKIT="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PAT_FILE="$REPO_ROOT/.secrets/read_gcbfplus_PAT.txt"
if [[ -z "${GITHUB_PAT:-}" && -f "$PAT_FILE" ]]; then
  GITHUB_PAT="$(tr -d '[:space:]' < "$PAT_FILE")"
elif [[ -z "${GITHUB_PAT:-}" ]] && command -v gh >/dev/null 2>&1 && gh auth token >/dev/null 2>&1; then
  GITHUB_PAT="$(gh auth token)"
fi
SECRET_ARGS=()
if [[ -n "${GITHUB_PAT:-}" ]]; then
  export GITHUB_PAT
  SECRET_ARGS=(--secret id=github_pat,env=GITHUB_PAT)
  echo "Using a GitHub token for the VCS installs (BuildKit secret)."
else
  echo "No GitHub token found; building with anonymous VCS installs (fine for the public repos)."
fi

echo "Removing old image tag '$IMAGE' (if present)..."
docker rmi "$IMAGE" 2>/dev/null || echo "  (no existing tag)"

echo "Pruning dangling <none> images..."
docker image prune -f

if [[ "$PRUNE_BUILDKIT" == "--prune-buildkit" ]]; then
  echo "Pruning BuildKit layer cache (affects all docker builds)..."
  docker builder prune -f
fi

echo "Building '$IMAGE' from $REPO_ROOT/Dockerfile..."
cd "$REPO_ROOT"
DOCKER_BUILDKIT=1 docker build "${SECRET_ARGS[@]}" -t "$IMAGE" .

echo
echo "Build complete. Smoke test with:"
echo "  bash scripts/test_docker_image.sh $IMAGE"
