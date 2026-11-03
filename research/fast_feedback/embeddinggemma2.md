# EmbeddingGemma 2 as a fast success/state checker: negative so far

Question: can `google/embeddinggemma-2` (multimodal embedding model, 440M params with text and vision encoders)
replace a VLM reviewer for quick checks? The idea is to embed a sim camera frame or clip and pick the closest of a
few text statements (door open/closed, handle grasped or not, handle hidden or visible).

**Result: no.** On our sim frames the correct statement wins 1 of 5 graded cases. Every margin is within
±0.02 cosine, while the absolute similarities are all 0.60 to 0.77. The model sees "door + robot" but does
not separate the states that matter.

## Results (local RTX A1000, bf16, 2026-11-04)

Text = query (`SearchQuery` prompt), image/video = document, L2-normalized cosine. Cases are in `common.py`.

| Case (image) | Texts (score) | Expected best | Result |
|---|---|---|---|
| `wrist_2_jaws_from_2_sides_of_lever_almost_grasp` | robot jaws grasp door handle (0.696) | n/a (single text) | n/a |
| `dark_gray_pillar_blocks_door` | handle visible 0.607 / **hidden** 0.602 | hidden | FAIL (-0.006) |
| `latched_slip_door_closed` | open 0.716 / **closed** 0.701 | closed | FAIL (-0.014) |
| `door_slight_open` | **slightly open** 0.677 / **a little open** 0.669 / fully open 0.659 / closed 0.658 | slightly / a little | PASS (+0.019) |
| `slip_mid_arc_not_grasping` | grasps handle 0.730 / **does not grasp** 0.719 | does not grasp | FAIL (-0.011) |
| `latched_slip/head.mp4` (video) | opens the door 0.766 / **fails to open, stays closed** 0.756 | fails to open | FAIL (-0.010) |

Summary: 1/5 PASS. The one pass has a margin as small as the failures, so it is likely chance.
Speed: about 0.3 s per image and 1.1 s per clip; peak GPU memory is 1.22 GB.

Takeaways:
- Negations ("does not grasp", "fails to open") and fine state differences are below the model's resolution.
  This is typical of contrastive embedding models.
- Not yet tried: other prompt types (`Classification` / `SentenceSimilarity` on both sides), MRL dims, a
  higher vision token budget, longer descriptive texts, comparing against reference images (image to image)
  instead of text, or a small probe trained on embeddings.

## Files

- `common.py` is shared by all probes:
  - `CASES` / `VIDEO_CASES`: images, texts and expected answers.
  - imgbb upload with a URL cache (`imgbb_urls.json`, `IMGBB_API_KEY` in `.env`).
  - Notebook cell helpers, plus `report` / `summarize`.
  - VQA prompt and answer helpers for generative models.
- `embeddinggemma2_local.py` is the local GPU runner (images and video). Env setup is in its docstring. In short:
  this laptop's driver supports only CUDA 11.7, so use torch 2.7.1+cu118 and `av<14`, without torchcodec.
- `make_notebook.py` generates `embeddinggemma2_kaggle.ipynb` (Kaggle 2x T4, fp32 because T4 has no bf16
  and fp16 gives NaN). Images come from imgbb via `wget`. This notebook has not been run on Kaggle yet.
- `make_molmopoint_notebook.py` generates `molmopoint4b_vid_kaggle.ipynb`. It asks the same cases as VQA to
  `allenai/MolmoPoint-Vid-4B`, plus a door-handle pointing plot. Not run yet.

Re-run locally:

```powershell
cd research\fast_feedback; $env:PYTHONPATH=""
conda run -n embgemma2 python embeddinggemma2_local.py
```
