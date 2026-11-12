"""Client-side MolmoPoint pointing: one ``molmo_point.point`` call on :mod:`gpu_server`.

The model runs on a GPU box (``docker compose ... --profile molmo``, ws :8785 / tcp :8786).
Nothing is launched locally: a missing server raises ``ConnectionError`` with a hint.

    url     GPU_SERVER_MOLMO_URL   default tcp://127.0.0.1:8786
    token   GPU_SERVER_TOKEN
    codec   GPU_SERVER_CODEC       raw (default) | jpeg | png

Model names: ``gpu-molmo`` (whatever the server loaded) or ``gpu-molmo:<url>`` to pick a server.
"""

import os

import numpy as np

DEFAULT_URL = "tcp://127.0.0.1:8786"
MODEL_PREFIX = "gpu-molmo"
MODULE = "molmo_point"


def is_molmo_model(name):
    return bool(name) and str(name).startswith(MODEL_PREFIX)


def url_from_model(name):
    """``gpu-molmo:<url>`` -> ``<url>``; plain ``gpu-molmo`` -> None (env / default)."""
    _, sep, rest = str(name or "").partition(":")
    return rest if sep and rest else None


def point_with_molmo(image, description, max_points=4, url=None, token=None, timeout=180.0,
                     image_codec=None, prompt=None, return_reply=False):
    """Point at ``description`` in ``image`` (path, PIL image or HxWx3 uint8 RGB array).

    Returns ``[[u, v], ...]`` in pixels of the given image, in model order; with
    ``return_reply`` also the server reply (``raw`` text, ``latency_s``, ``model``, ``quant``)."""
    from gpu_server.client import GpuClient
    if isinstance(image, (str, os.PathLike)):
        from PIL import Image
        image = Image.open(image)
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    arr = np.ascontiguousarray(np.asarray(image)[..., :3]).astype(np.uint8)
    url = url or os.environ.get("GPU_SERVER_MOLMO_URL") or DEFAULT_URL
    codec = image_codec or os.environ.get("GPU_SERVER_CODEC") or "raw"
    args = {"description": description, "max_points": int(max_points)}
    if prompt:
        args["prompt"] = prompt
    with GpuClient(url, timeout=timeout, token=token, image_codec=codec) as c:
        reply, _ = c.call(MODULE, "point", image=arr, **args)
    points = [[float(p[0]), float(p[1])] for p in reply.get("points") or []][: int(max_points)]
    return (points, reply) if return_reply else points
