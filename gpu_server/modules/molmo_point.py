"""MolmoPoint (Ai2) pointing VLM as a gpu_server module (stateless).

Method ``point`` - point at a described object in one image::

    inputs : image (H,W,3) uint8 RGB
    args   : description ("the USB port") -> prompt "Point to the USB port."
             or prompt (used verbatim), max_points (optional cap), max_new_tokens (optional)
    outputs: {"points": [[u, v], ...] pixels of the input image, "raw": generated text,
              "latency_s", "model", "quant"}

Method ``point_video`` - point in a clip (``MolmoPoint-Vid-4B``; minimal, unverified)::

    inputs : frames (T,H,W,3) uint8 RGB
    args   : description / prompt, max_points, max_new_tokens
    outputs: {"points": [[t, u, v], ...], "raw", "latency_s"}

MolmoPoint does not write coordinates as text: it emits grounding tokens that
``model.extract_image_points`` / ``extract_video_points`` decode (with the processor's
``return_pointing_metadata`` output) into ``[object_id, image_idx|time, x_px, y_px]`` rows in
source-image pixels (model cards of ``allenai/MolmoPoint-8B`` / ``-Vid-4B``).
:func:`parse_point_text` is the fallback for text-coordinate Molmo formats (Molmo 1
``<point x= y=>`` in percent, Molmo 2 ``<points coords="...">`` scaled by 1000).

Config (``--set molmo_point.<key>=...``):
  ``model``  HF repo or path; default ``$MOLMO_MODEL``, else auto by total VRAM (:func:`select_mode`):
             < 12 GB -> ``allenai/MolmoPoint-Vid-4B``, else ``allenai/MolmoPoint-8B``.
             Pre-quantized checkpoints (e.g. ``erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16``) load
             with their own quantization_config when ``quant=none``.
  ``quant``  bf16 | 8bit | 4bit | none; default ``$MOLMO_QUANT``, else auto: 8B -> ``bf16`` at
             >= 20 GB, ``4bit`` below; 4B -> ``bf16`` at >= 16 GB, ``4bit`` below. 8/4-bit use
             bitsandbytes on the language model only; the vision tower, connector and pointing
             head stay bf16 (quantizing them hurts pointing).
  ``max_new_tokens`` (default 200).
"""

import logging
import os
import re
import time

import numpy as np

from gpu_server.modules.base import GpuModule, ModuleError, require

logger = logging.getLogger("gpu_server.molmo_point")

MODEL_8B = "allenai/MolmoPoint-8B"
MODEL_4B = "allenai/MolmoPoint-Vid-4B"
QUANTS = ("bf16", "8bit", "4bit", "none")
# Same split as erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16 (lm_head is never quantized by transformers).
SKIP_QUANT_MODULES = ["model.vit", "model.connector", "model.build_vit_embedding", "model.point_predictor"]
# 8B nf4 peaks at 8.7 GiB (erdi28 card) -> needs > 8 GB; 8B bf16 peaks at 18.3 GiB.
SMALL_GPU_GB = 12.0
BF16_8B_GB = 20.0
BF16_4B_GB = 16.0

# Molmo 2 text pointing: <points coords="1 523 412 2 600 300"> (image) or
# <points coords="8.5 1 183 216;9.0 2 ..."> (video: time/frame first); coords scaled by 1000.
_COORDS_RE = re.compile(r"<(?:points|tracks)\b[^>]*?\scoords=\"([0-9\t:;, .]+)\"[^>]*>")
_FRAME_RE = re.compile(r"(?:^|\t|:|,|;)\s*([0-9.]+) ([0-9. ]+)")
_IDXY_RE = re.compile(r"([0-9]+) ([0-9]{1,4}(?:\.[0-9]+)?) ([0-9]{1,4}(?:\.[0-9]+)?)")
# Molmo 1: <point x="61.5" y="40.6" alt="..">; <points x1=".." y1=".." x2=".." y2=".." ...>; percent.
_TAG_RE = re.compile(r"<points?\b([^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"\b([xy])(\d*)\s*=\s*\"?\s*(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_point_text(text, width, height):
    """Text-coordinate Molmo pointing output -> ``[[u, v], ...]`` pixels (in order, in-image only).

    Covers Molmo 2 ``coords="..."`` (scale 1000; the frame/time column is dropped) and Molmo 1
    ``<point x y>`` / ``<points x1 y1 x2 y2 ...>`` (percent 0-100)."""
    text = text or ""
    pts = []

    def keep(u, v):
        if 0 <= u <= width and 0 <= v <= height:
            pts.append([float(u), float(v)])

    coords = list(_COORDS_RE.finditer(text))
    for m in coords:
        body = m.group(1)
        groups = [g.group(2) for g in _FRAME_RE.finditer(body)] if re.search(r"[;\t:,]", body) else []
        if not groups:
            # Single image: "idx x y idx x y ..." (or "img idx x y" with a leading image index).
            nums = body.split()
            groups = [" ".join(nums[1:])] if len(nums) % 3 == 1 else [body]
        for g in groups:
            for _, x, y in _IDXY_RE.findall(g):
                keep(float(x) / 1000.0 * width, float(y) / 1000.0 * height)
    if coords:
        return pts
    for tag in _TAG_RE.finditer(text):
        xs, ys = {}, {}
        for axis, idx, val in _ATTR_RE.findall(tag.group(1)):
            (xs if axis.lower() == "x" else ys)[idx] = float(val)
        for idx in sorted(set(xs) & set(ys), key=lambda s: int(s) if s else 0):
            keep(xs[idx] / 100.0 * width, ys[idx] / 100.0 * height)
    return pts


def extracted_to_pixels(rows, max_points=None):
    """``model.extract_*_points`` rows ``[object_id, image_idx|time, x, y]`` -> ``[[x, y], ...]``."""
    out = []
    for r in rows or []:
        r = list(np.asarray(r, dtype=float).ravel())
        if len(r) >= 2:
            out.append([float(r[-2]), float(r[-1])])
    return out[: int(max_points)] if max_points else out


def build_prompt(args):
    prompt = args.get("prompt")
    if prompt:
        return str(prompt)
    desc = str(args.get("description") or "").strip().rstrip(".")
    if not desc:
        raise ModuleError("molmo_point: pass 'description' (or a full 'prompt')")
    return f"Point to {desc}."


def select_mode(total_gb, model=None, quant=None):
    """Pick ``(model, quant, reason)`` from total VRAM (GiB; None = no CUDA). Explicit values win."""
    gb = total_gb if total_gb is not None else 0.0
    auto_model = not model
    if auto_model:
        model = MODEL_4B if gb < SMALL_GPU_GB else MODEL_8B
    if not quant:
        small = re.search(r"[-_/]4b\b", str(model).lower()) is not None
        quant = "bf16" if gb >= (BF16_4B_GB if small else BF16_8B_GB) else "4bit"
        if total_gb is None:
            quant = "bf16"  # CPU: bitsandbytes needs CUDA
    vram = "no CUDA" if total_gb is None else f"{total_gb:.1f} GiB VRAM"
    reason = f"{vram} -> {'auto' if auto_model else 'set'} model {model}, quant {quant}"
    return model, quant, reason


class MolmoPointModule(GpuModule):
    name = "molmo_point"
    description = "MolmoPoint VLM pointing; 'point' returns pixel points for a description"
    stateful = False

    def __init__(self, **config):
        super().__init__(**config)
        self.model = None
        self.processor = None
        self.model_id = self.config.get("model") or os.environ.get("MOLMO_MODEL") or None
        self.quant = self.config.get("quant") or os.environ.get("MOLMO_QUANT") or None
        self.max_new_tokens = int(self.config.get("max_new_tokens", 200))
        self.mode_reason = ""
        self._torch = None

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._torch = torch
        total_gb = (torch.cuda.get_device_properties(0).total_memory / 2**30
                    if torch.cuda.is_available() else None)
        self.model_id, self.quant, self.mode_reason = select_mode(total_gb, self.model_id, self.quant)
        if self.quant not in QUANTS:
            raise ModuleError(f"molmo_point: quant must be one of {QUANTS}, got {self.quant!r}")
        kw = {"trust_remote_code": True, "device_map": "cuda" if torch.cuda.is_available() else "cpu"}
        if self.quant == "none":
            kw["dtype"] = "auto"
        else:
            kw["dtype"] = torch.bfloat16
        if self.quant in ("8bit", "4bit"):
            from transformers import BitsAndBytesConfig
            if self.quant == "4bit":
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT_MODULES)
            else:
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_skip_modules=SKIP_QUANT_MODULES)
        logger.info("[molmo_point] mode: %s (vision tower bf16)", self.mode_reason)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, padding_side="left")
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw)
        self.model.eval()

    def unload(self):
        self.model = None
        self.processor = None
        if self._cuda():
            self._torch.cuda.empty_cache()

    def _cuda(self):
        return self._torch is not None and self._torch.cuda.is_available()

    def info(self):
        if self.model is None:
            return {}
        out = {"model": self.model_id, "quant": self.quant, "mode": self.mode_reason}
        if self._cuda():
            out["vram_mb"] = round(self._torch.cuda.memory_allocated() / 2**20)
        return out

    # -- generation --------------------------------------------------------
    def _generate(self, content, args):
        torch = self._torch
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, return_pointing_metadata=True)
        metadata = inputs.pop("metadata")
        dev = "cuda" if self._cuda() else "cpu"
        inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        max_new = int(args.get("max_new_tokens") or self.max_new_tokens)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            output = self.model.generate(
                **inputs, logits_processor=self.model.build_logit_processor_from_inputs(inputs),
                do_sample=False, max_new_tokens=max_new)
        if self._cuda():
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        gen = output[:, inputs["input_ids"].size(1):]
        text = self.processor.post_process_image_text_to_text(
            gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        return text, metadata, latency

    def m_point(self, args, arrays):
        if self.model is None:
            raise ModuleError("molmo_point: module not loaded")
        (img,) = require(arrays, "image")
        from PIL import Image
        img = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8)
        h, w = img.shape[:2]
        prompt = build_prompt(args)
        max_points = args.get("max_points")
        content = [{"type": "text", "text": prompt}, {"type": "image", "image": Image.fromarray(img)}]
        text, meta, latency = self._generate(content, args)
        try:
            rows = self.model.extract_image_points(
                text, meta["token_pooling"], meta["subpatch_mapping"], meta["image_sizes"])
            points = extracted_to_pixels(rows, max_points)
        except Exception as exc:  # noqa: BLE001 - fall back to text coordinates
            logger.warning("[molmo_point] extract_image_points failed (%s); parsing text", exc)
            points = []
        if not points:
            points = parse_point_text(text, w, h)
            if max_points:
                points = points[: int(max_points)]
        logger.debug("[molmo_point] %r -> %s (%.2fs)", prompt, points, latency)
        return {"points": points, "raw": text, "latency_s": round(latency, 3), "prompt": prompt,
                "image_size": [w, h], "model": self.model_id, "quant": self.quant}, {}

    def m_point_video(self, args, arrays):
        if self.model is None:
            raise ModuleError("molmo_point: module not loaded")
        (frames,) = require(arrays, "frames")
        from PIL import Image
        frames = np.asarray(frames)[..., :3].astype(np.uint8)
        if frames.ndim != 4:
            raise ModuleError(f"molmo_point: frames must be (T,H,W,3), got {frames.shape}")
        prompt = build_prompt(args)
        video = [Image.fromarray(f) for f in frames]
        text, meta, latency = self._generate([{"type": "text", "text": prompt}, {"type": "video", "video": video}], args)
        rows = self.model.extract_video_points(
            text, meta["token_pooling"], meta["subpatch_mapping"], meta["timestamps"], meta["video_size"])
        points = [[float(r[1]), float(r[-2]), float(r[-1])] for r in (np.asarray(x, float).ravel() for x in rows or [])]
        max_points = args.get("max_points")
        if max_points:
            points = points[: int(max_points)]
        return {"points": points, "raw": text, "latency_s": round(latency, 3), "prompt": prompt,
                "model": self.model_id, "quant": self.quant}, {}
