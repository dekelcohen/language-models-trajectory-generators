#!/usr/bin/env bash
# Dev entrypoint (docker-compose.dev.yml): the python env lives on a host-mapped dir, so
# `pip install ...` inside the container survives `docker compose down` and image rebuilds.
#   DEV_ENV_KIND=venv  (default) `venv --system-site-packages`: reuses the image's torch/CUDA; only
#                      your extra / overriding packages (e.g. transformers==4.57.1) live in $DEV_ENV.
#   DEV_ENV_KIND=conda full `conda create --clone base` into $DEV_ENV (~6 GB, independent of the image).
set -euo pipefail
ENV_DIR=${DEV_ENV:-/workspace/env}
mkdir -p "$(dirname "$ENV_DIR")"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  # both services start at once: seed under a lock next to (not inside) the env dir
  exec 9>"$ENV_DIR.lock"
  flock 9
  if [ ! -x "$ENV_DIR/bin/python" ]; then
    if [ "${DEV_ENV_KIND:-venv}" = "conda" ]; then
      echo "[dev] cloning the image's conda base -> $ENV_DIR (once, several GB)"
      conda create -y -q -p "$ENV_DIR" --clone base
    else
      echo "[dev] creating venv overlay $ENV_DIR (once)"
      python -m venv --system-site-packages "$ENV_DIR"
    fi
  fi
  flock -u 9
  exec 9>&-
fi
export PATH="$ENV_DIR/bin:$PATH" VIRTUAL_ENV="$ENV_DIR"
exec "$@"
