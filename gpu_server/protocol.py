"""Wire format shared by every transport (WebSocket, TCP, HTTP body).

One message = ``MAGIC | u32 header_len | header JSON (utf-8) | array blobs``.

The header is a JSON object. Arrays travel as raw bytes after it and are described in
``header["_arrays"]`` as ``{"name", "dtype", "shape", "enc", "nbytes"}`` in blob order.
``enc`` is ``"raw"`` (C-order bytes, no copy beyond ``tobytes``), ``"png"`` (lossless,
for ``uint8`` images over slow links) or ``"jpeg"`` (lossy, quality ``JPEG_QUALITY``; for
internet tunnels such as a Kaggle-hosted server). Raw is the default: a 256x256 RGB frame is 192 KB,
which costs well under a millisecond on localhost or a LAN, while PNG costs ~2-3 ms to
encode. There is no base64 anywhere.

Requests carry ``op`` (see :mod:`gpu_server.host`) and an optional ``id`` that the reply
echoes, so a late reply can be matched or dropped.
"""

import json
import struct

import numpy as np

PROTOCOL_VERSION = 1
MAGIC = b"GS1"
_HDR = struct.Struct(">I")
IMAGE_CODECS = ("raw", "png", "jpeg")
JPEG_QUALITY = 90


class ProtocolError(ValueError):
    """Malformed message."""


def jsonable(obj):
    """Recursively convert numpy scalars/arrays (e.g. in tracker ``meta``) into JSON types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and obj != obj:  # NaN is not valid JSON
        return None
    return obj


def _png_encode(arr):
    import cv2
    img = arr[..., ::-1] if arr.ndim == 3 else arr        # cv2 writes BGR
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ProtocolError("PNG encoding failed")
    return buf.tobytes()


def _jpeg_encode(arr):
    import cv2
    img = arr[..., ::-1] if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ProtocolError("JPEG encoding failed")
    return buf.tobytes()


def _img_decode(blob, shape):
    import cv2
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ProtocolError("image decoding failed")
    if img.ndim == 3:
        img = img[..., ::-1]
    return np.ascontiguousarray(img).reshape(shape)


def _is_image(arr):
    return arr.dtype == np.uint8 and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1, 3, 4)))


def pack(header, arrays=None, image_codec="raw"):
    """Serialise ``header`` (dict) and ``arrays`` ({name: ndarray}) into one ``bytes``."""
    if image_codec not in IMAGE_CODECS:
        raise ValueError(f"image_codec must be one of {IMAGE_CODECS}, got {image_codec!r}")
    header = dict(header)
    descs, blobs = [], []
    for name, arr in (arrays or {}).items():
        arr = np.asarray(arr)
        # JPEG has no alpha / single-channel-3D variant: those images go lossless instead.
        enc = image_codec if image_codec != "raw" and _is_image(arr) else "raw"
        if enc == "jpeg" and not (arr.ndim == 2 or arr.shape[2] == 3):
            enc = "png"
        if enc == "png":
            blob = _png_encode(arr)
        elif enc == "jpeg":
            blob = _jpeg_encode(arr)
        else:
            blob = np.ascontiguousarray(arr).tobytes()
        descs.append({"name": str(name), "dtype": arr.dtype.str, "shape": list(arr.shape),
                      "enc": enc, "nbytes": len(blob)})
        blobs.append(blob)
    header["_arrays"] = descs
    head = json.dumps(jsonable(header), separators=(",", ":")).encode("utf-8")
    return b"".join([MAGIC, _HDR.pack(len(head)), head, *blobs])


def unpack(buf):
    """Inverse of :func:`pack`. Returns ``(header, arrays)`` with writable arrays (torch warns
    on read-only buffers; the copy costs ~20 us for a 192 KB frame)."""
    mv = memoryview(buf)
    if len(mv) < 7 or bytes(mv[:3]) != MAGIC:
        raise ProtocolError("bad magic (not a gpu_server message)")
    (hlen,) = _HDR.unpack(mv[3:7])
    off = 7 + hlen
    try:
        header = json.loads(bytes(mv[7:off]).decode("utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"bad header JSON: {exc}") from exc
    arrays = {}
    for d in header.pop("_arrays", []):
        end = off + int(d["nbytes"])
        if end > len(mv):
            raise ProtocolError(f"truncated message (array {d['name']!r})")
        blob = mv[off:end]
        shape = tuple(d["shape"])
        if d.get("enc", "raw") in ("png", "jpeg"):
            arrays[d["name"]] = _img_decode(bytes(blob), shape)
        else:
            arrays[d["name"]] = np.frombuffer(blob, dtype=np.dtype(d["dtype"])).reshape(shape).copy()
        off = end
    return header, arrays
