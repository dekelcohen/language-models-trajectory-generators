"""RoMa v2 dense matcher as a gpu_server module (stateless re-detection).

Method ``match`` - find query points of image A in image B::

    inputs : img_a (H,W,3) uint8, img_b (H',W',3) uint8, points_a (N,2) float [optional]
    args   : box [x0,y0,x1,y1]  -> "crop-only" mode: match only this crop of A (the stored
                                  appearance of the target); points stay in full-A pixels
             local_radius_px (8), overlap_thr (0.3), dense (false)
    outputs: points_b (N,2)        direct warp at each query      <- the re-detection
             overlap (N,)          RoMa co-visibility at each query (low = not visible in B)
             points_b_local (N,2)  median displacement of confident pixels within the
                                   radius (NaN when < 3 such pixels), n_local (N,)
             [dense] warp_b (h,w,2) B pixel of every low-res A(/crop) pixel, overlap_a (h,w)

Without ``points_a`` the query is the box centre (or the image centre).

Config: ``root`` (RoMaV2 checkout, default ``$ROMAV2_ROOT``), ``torch_home`` (weights cache,
default ``$TORCH_HOME``), ``setting`` (turbo | fast |
base | precise), ``amp`` (auto | bf16 | fp16 | fp32). RoMa v2 hard-codes ``torch.bfloat16``;
bf16 is unsupported on a T4 and ~100x slow on CPU, so for ``fp16``/``fp32`` (``auto``: fp32
on CPU, fp16 below sm80, bf16 otherwise) the checkout's sources are patched in place from
``*.py.orig`` backups - the same patch as ``tests/tracking_tools/romav2_redetect_colab.ipynb``.
The device is RoMa's own choice (CUDA when available); weights come from torch.hub
(``$TORCH_HOME``). Needs torch >= 2.4 (DINOv3 hub code), so it usually runs in a different
env than CoTracker - one more reason it lives behind the server.
"""

import glob
import logging
import os
import shutil
import sys
import time

import numpy as np

from gpu_server.modules.base import GpuModule, ModuleError, require

logger = logging.getLogger("gpu_server.romav2")

SETTINGS = ("turbo", "fast", "base", "precise")


def resolve_amp(amp, torch):
    if amp and amp != "auto":
        return amp
    if not torch.cuda.is_available():
        return "fp32"
    return "fp16" if torch.cuda.get_device_capability(0)[0] < 8 else "bf16"


def patch_amp(root, dtype):
    """Rewrite ``torch.bfloat16`` in the RoMaV2 sources (from ``.orig`` backups). Returns
    the number of files whose text differs from the original."""
    to = {"fp16": "torch.float16", "fp32": "torch.float32"}.get(dtype)
    n = 0
    for py in glob.glob(os.path.join(root, "src", "romav2", "**", "*.py"), recursive=True):
        orig = py + ".orig"
        if not os.path.exists(orig):
            shutil.copy(py, orig)
        with open(orig, encoding="utf-8") as f:
            src = f.read()
        new = src.replace("torch.bfloat16", to) if to else src
        n += new != src
        with open(py, encoding="utf-8") as f:
            cur = f.read()
        if cur != new:
            with open(py, "w", encoding="utf-8") as f:
                f.write(new)
    return n


class RoMaV2Module(GpuModule):
    name = "romav2"
    description = "RoMa v2 dense matching; 'match' re-detects query points of A in B"
    stateful = False

    def __init__(self, **config):
        super().__init__(**config)
        self.model = None
        self.amp = None
        self.setting = self.config.get("setting", "turbo")

    def load(self):
        if self.config.get("torch_home"):
            os.environ["TORCH_HOME"] = str(self.config["torch_home"])
        import torch
        root = self.config.get("root") or os.environ.get("ROMAV2_ROOT")
        if not root or not os.path.isdir(os.path.join(root, "src", "romav2")):
            raise ModuleError("romav2: set config root=<RoMaV2 checkout> or $ROMAV2_ROOT "
                              "(git clone https://github.com/Parskatt/RoMaV2)")
        self.amp = resolve_amp(self.config.get("amp", "auto"), torch)
        if "romav2" not in sys.modules:
            n = patch_amp(root, self.amp)
            logger.info("[romav2] amp=%s (%d source files differ from the bf16 original)", self.amp, n)
            sys.path.insert(0, os.path.join(root, "src"))
        from romav2 import RoMaV2
        self.model = RoMaV2(RoMaV2.Cfg(setting=self.setting))
        self.model.apply_setting(self.setting)
        self._torch = torch

    def unload(self):
        self.model = None
        if self._torch_cuda():
            self._torch.cuda.empty_cache()

    def _torch_cuda(self):
        return getattr(self, "_torch", None) is not None and self._torch.cuda.is_available()

    def info(self):
        if self.model is None:
            return {}
        dev = next(self.model.parameters()).device
        out = {"device": str(dev), "setting": self.setting, "amp": self.amp,
               "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6)}
        if self._torch_cuda():
            out["vram_mb"] = round(self._torch.cuda.memory_allocated() / 2**20)
        return out

    def m_configure(self, args, arrays):
        setting = args.get("setting", self.setting)
        if setting not in SETTINGS:
            raise ModuleError(f"romav2: setting must be one of {SETTINGS}")
        self.model.apply_setting(setting)
        self.setting = setting
        return self.info(), {}

    # -- readouts ----------------------------------------------------------
    def _sample(self, preds, pts, size_a, size_b):
        """Bilinear lookup of the dense warp at A pixels -> (B pixels (N,2), overlap (N,))."""
        torch = self._torch
        import torch.nn.functional as F
        (wa, ha), (wb, hb) = size_a, size_b
        dev = preds["warp_AB"].device
        p = torch.as_tensor(np.asarray(pts, np.float32).reshape(-1, 2), device=dev)
        g = torch.stack((p[:, 0] / wa * 2 - 1, p[:, 1] / ha * 2 - 1), -1)[None, None]
        w = F.grid_sample(preds["warp_AB"].permute(0, 3, 1, 2).float(), g, align_corners=False)[0, :, 0].T
        ov = F.grid_sample(preds["overlap_AB"].permute(0, 3, 1, 2).float(), g, align_corners=False)[0, 0, 0]
        xb = torch.stack(((w[:, 0] + 1) / 2 * wb, (w[:, 1] + 1) / 2 * hb), -1)
        return xb.cpu().numpy().astype(np.float64), ov.cpu().numpy().astype(np.float64)

    def m_match(self, args, arrays):
        if self.model is None:
            raise ModuleError("romav2: module not loaded")
        img_a, img_b = require(arrays, "img_a", "img_b")
        img_a, img_b = np.ascontiguousarray(img_a[..., :3]), np.ascontiguousarray(img_b[..., :3])
        ha, wa = img_a.shape[:2]
        hb, wb = img_b.shape[:2]
        box = args.get("box")
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(wa, x1), min(ha, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise ModuleError(f"romav2: box {box} is empty or too small inside {wa}x{ha}")
            src = np.ascontiguousarray(img_a[y0:y1, x0:x1])
            origin = np.array([x0, y0], float)
        else:
            src, origin = img_a, np.zeros(2)
        hs, ws = src.shape[:2]
        pts = arrays.get("points_a")
        if pts is None:
            pts = origin[None] + np.array([[ws / 2.0, hs / 2.0]])
        pts = np.asarray(pts, float).reshape(-1, 2)
        q = pts - origin

        cuda = self._torch_cuda()
        if cuda:
            self._torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = self.model.match(src, img_b)
        if cuda:
            self._torch.cuda.synchronize()
        match_s = time.perf_counter() - t0

        pb, ov = self._sample(preds, q, (ws, hs), (wb, hb))
        thr = float(args.get("overlap_thr", 0.3))
        r = float(args.get("local_radius_px", 8.0))
        d = np.arange(-np.floor(r), np.floor(r) + 1)
        off = np.stack(np.meshgrid(d, d), -1).reshape(-1, 2)
        off = off[np.hypot(off[:, 0], off[:, 1]) <= r]
        local = np.full_like(pb, np.nan)
        n_local = np.zeros(len(q), np.int32)
        for i, qa in enumerate(q):
            nb = qa + off
            nb = nb[(nb[:, 0] >= 0) & (nb[:, 0] < ws) & (nb[:, 1] >= 0) & (nb[:, 1] < hs)]
            nbb, nov = self._sample(preds, nb, (ws, hs), (wb, hb))
            keep = nov >= thr
            n_local[i] = int(keep.sum())
            if n_local[i] >= 3:
                local[i] = pts[i] + np.median(nbb[keep] - (nb[keep] + origin), axis=0)

        out = {"points_b": pb, "overlap": ov, "points_b_local": local, "n_local": n_local}
        reply = {"match_s": round(match_s, 4), "setting": self.setting, "amp": self.amp,
                 "src_size": [ws, hs], "origin": origin.tolist()}
        if args.get("dense"):
            w = preds["warp_AB"][0].float()
            out["warp_b"] = np.stack((((w[..., 0] + 1) / 2 * wb).cpu().numpy(),
                                      ((w[..., 1] + 1) / 2 * hb).cpu().numpy()), -1).astype(np.float32)
            out["overlap_a"] = preds["overlap_AB"][0, ..., 0].float().cpu().numpy().astype(np.float32)
        logger.debug("[romav2] match %dx%d -> %dx%d box=%s n=%d %.3fs", ws, hs, wb, hb, box, len(q), match_s)
        return reply, out
