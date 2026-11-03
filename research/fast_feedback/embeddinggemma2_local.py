"""EmbeddingGemma 2 sim-frame / clip vs. text similarity on a local GPU (e.g. RTX A1000 4 GB).

Same image cases as the Kaggle notebook (CASES in common.py, read from local files) plus the
video case on a review-ablation head.mp4 clip. Uses bfloat16 when the GPU supports it (Ampere+),
else float32 (never float16: the model card says it NaNs).

Env (`embgemma2`; this laptop's NVIDIA driver supports only CUDA 11.7, so cu12x torch wheels see no GPU):
    conda create -n embgemma2 python=3.12 -y
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
    pip install -U sentence-transformers transformers python-dotenv requests "av<14"
  (no torchcodec: its DLLs fail on Windows here; av>=14 breaks torchvision's video reader)
Run (PYTHONPATH cleared so vlm_traj repos don't shadow packages):
    cd research/fast_feedback; $env:PYTHONPATH=""; conda run -n embgemma2 python embeddinggemma2_local.py [--no-video]
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from common import CASES, REPO, VIDEO_CASES, report, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="force float32 even if bf16 is supported")
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev != "cpu" and torch.cuda.is_bf16_supported() and not args.fp32 else torch.float32
    if dev != "cpu":
        p = torch.cuda.get_device_properties(0)
        print(f"[gpu] {p.name} {p.total_memory / 2**30:.1f} GB, dtype={dtype}")

    t0 = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2", device=dev,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},  # text + image/video only
    )
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.max_memory_allocated(0) / 2**30 if dev != "cpu" else 0
    print(f"[load] {time.time() - t0:.1f}s, {n / 1e6:.0f}M params, peak {mem:.2f} GB, prompts={list(model.prompts)}")

    def enc(x, **kw):
        with torch.no_grad():
            e = model.encode(x, normalize_embeddings=True, convert_to_tensor=True, **kw)
        assert not torch.isnan(e).any(), "NaN embeddings"
        return e

    results = {}
    for rel, name, texts, expected in CASES:
        t = time.time()
        e_img = enc({"image": Image.open(REPO / rel).convert("RGB")})
        e_txt = enc(texts, prompt_name="SearchQuery")
        sims = model.similarity(e_txt, e_img.reshape(1, -1)).squeeze(1).tolist()
        results[name] = report(f"{name}  [image, {time.time() - t:.2f}s]", texts, expected, sims)
    if not args.no_video:
        for rel, texts, expected in VIDEO_CASES:
            t = time.time()
            try:
                e_vid = enc({"video": str(REPO / rel)})
            except Exception as e:  # video decoding backends vary by platform
                print(f"\n[video] {rel} FAILED: {type(e).__name__}: {e}")
                results[Path(rel).parent.name + "/video"] = False
                continue
            e_txt = enc(texts, prompt_name="SearchQuery")
            sims = model.similarity(e_txt, e_vid.reshape(1, -1)).squeeze(1).tolist()
            results[Path(rel).parent.name + "/video"] = report(
                f"{rel}  [video, {time.time() - t:.2f}s]", texts, expected, sims)

    if dev != "cpu":
        print(f"\n[gpu] peak memory {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GB")
    summarize(results)


if __name__ == "__main__":
    main()
