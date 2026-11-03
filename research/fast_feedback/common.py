"""Shared cases, texts and helpers for the fast-feedback model probes (EmbeddingGemma 2, MolmoPoint, ...).

Each probe scores a sim camera frame (or clip) against short text statements; one or more of them
are the expected best match. Notebook builders use resolve_images() + the *_cell() helpers;
report() and the VQA helpers (vqa_question, parse_vqa_answer, label_token_ids, option_probs,
vqa_scores) are plain Python that gets inlined into notebooks (inline_source) and imported locally.
"""
import base64
import hashlib
import inspect
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
URL_CACHE = HERE / "imgbb_urls.json"

# (image path rel. to REPO, upload/file name, candidate texts, texts accepted as the best match)
CASES = [
    ("outputs/feedback/wrist_2_jaws_from_2_sides_of_lever_almost_grasp.png",
     "wrist_2_jaws_from_2_sides_of_lever_almost_grasp",
     ["robot jaws grasp door handle"], []),
    ("outputs/longhorizon/dark_gray_pillar_blocks_door.png", "dark_gray_pillar_blocks_door",
     ["door handle is hidden", "door handle is visible"], ["door handle is hidden"]),
    # imgbb hosts images only, so the latched_slip head.mp4 clip is represented by its final frame
    ("outputs/review_ablation/clips/latched_slip/final_head.png", "latched_slip_door_closed",
     ["door is closed", "door is open"], ["door is closed"]),
    ("outputs/review_ablation/clips/slight_open/final_head.png", "door_slight_open",
     ["door is closed", "door is slightly open", "door is a little open", "door is fully open"],
     ["door is slightly open", "door is a little open"]),
    ("outputs/review_ablation/clips/slip_mid_arc/final_head.png", "slip_mid_arc_not_grasping",
     ["the robot arm grasps the door handle", "the robot arm does not grasp the door handle"],
     ["the robot arm does not grasp the door handle"]),
]

# Local-only video cases (imgbb cannot host video): (clip rel. to REPO, candidate texts, expected)
VIDEO_CASES = [
    ("outputs/review_ablation/clips/latched_slip/head.mp4",
     ["the robot arm opens the door", "the robot arm fails to open the door, the door stays closed"],
     ["the robot arm fails to open the door, the door stays closed"]),
]

KAGGLE_T4_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
}


def report(name, texts, expected, scores, fmt="{:+.4f}"):
    """Print texts ranked by score; return PASS bool if `expected` is non-empty, else None."""
    print(f"\n{name}")
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        print(f"  {fmt.format(s)}  {t}")
    if not expected:
        return None
    good = max(s for t, s in zip(texts, scores) if t in expected)
    bad = max(s for t, s in zip(texts, scores) if t not in expected)
    ok = good > bad
    print(f"  expected best in {expected} -> {'PASS' if ok else 'FAIL'} (margin {good - bad:+.4f})")
    return ok


def summarize(results):
    scored = {k: v for k, v in results.items() if v is not None}
    print(f"\n[summary] {sum(scored.values())}/{len(scored)} PASS: "
          + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in scored.items()))


def vqa_question(texts):
    """VQA prompt for one case: lettered multiple choice over `texts`, or yes/no for a single text.

    Returns (prompt, labels); labels are the answer tokens ('A', 'B', ... or 'yes', 'no').
    """
    if len(texts) == 1:
        return (f"Is this statement true for the image: '{texts[0]}'? Answer yes or no.", ["yes", "no"])
    labels = [chr(ord("A") + i) for i in range(len(texts))]
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return ("Which statement best describes the image?\n" + opts
            + "\nAnswer with the letter only.", labels)


def parse_vqa_answer(answer, texts):
    """Map a free-text model answer to (label, chosen_text); (None, None) if unparseable.

    For single-text (yes/no) cases chosen_text is the text if the answer is 'yes', else None.
    """
    import re
    a = answer.strip()
    if len(texts) == 1:
        m = re.search(r"\b(yes|no)\b", a, re.I)
        if not m:
            return None, None
        label = m.group(1).lower()
        return label, (texts[0] if label == "yes" else None)
    labels = [chr(ord("A") + i) for i in range(len(texts))]
    pat = "|".join(labels)
    m = (re.match(rf"^\W*({pat})(?:\W|$)", a)
         or re.search(rf"(?:answer|option)\s*(?:is|:)?\s*\(?({pat})\b", a, re.I)
         or re.search(rf"\b({pat})[.)]", a))
    if m:
        label = m.group(1).upper()
        return label, texts[labels.index(label)]
    low = a.lower()
    hits = [t for t in sorted(texts, key=len, reverse=True) if t.lower() in low]
    if hits:
        return labels[texts.index(hits[0])], hits[0]
    return None, None


def label_token_ids(tokenizer, labels):
    """{label: [single-token ids of its variants ('A', ' A', 'a', ...)]} for logprob scoring."""
    out = {}
    for l in labels:
        ids = set()
        for v in {l, " " + l, l.lower(), " " + l.lower(), l.capitalize(), " " + l.capitalize()}:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        out[l] = sorted(ids)
    return out


def option_probs(first_logits, label_ids):
    """Softmax over the options of the first generated token's logits ({label: prob}), or None.

    Returns None when the greedy first token is not one of the option tokens (e.g. a preamble),
    since the first-step distribution then says little about the final choice.
    """
    import torch
    lp = torch.log_softmax(first_logits.float().reshape(-1), dim=-1)
    if not any(int(lp.argmax()) in ids for ids in label_ids.values()):
        return None
    per = {l: torch.logsumexp(lp[ids], 0) if ids else torch.tensor(-1e9) for l, ids in label_ids.items()}
    p = torch.softmax(torch.stack(list(per.values())), 0).tolist()
    return dict(zip(per, p))


def vqa_scores(texts, label, probs=None):
    """Per-text scores for report(): option probabilities if available, else 1/0 for the parsed choice.

    Single-text (yes/no) cases get one score: P(yes), or 1/0 for a yes/no answer.
    """
    if len(texts) == 1:
        if probs:
            return [probs["yes"]]
        return [1.0 if label == "yes" else 0.0]
    labels = [chr(ord("A") + i) for i in range(len(texts))]
    if probs:
        return [probs[l] for l in labels]
    return [1.0 if l == label else 0.0 for l in labels]


def inline_source(*funcs):
    return "\n\n".join(inspect.getsource(f) for f in funcs)


def upload_imgbb(path: Path, name: str, key: str, tries: int = 4) -> str:
    import requests
    b64 = base64.b64encode(path.read_bytes()).decode()
    for i in range(tries):
        r = requests.post("https://api.imgbb.com/1/upload",
                          data={"key": key, "image": b64, "name": name}, timeout=60)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code == 200 and body.get("success"):
            url = body["data"]["url"]
            print(f"[imgbb] uploaded {name} -> {url}")
            return url
        msg = (body.get("error") or {}).get("message") or r.text[:200].split("<title>")[-1].split("</title>")[0]
        print(f"[imgbb] upload FAILED {name} (try {i + 1}/{tries}): HTTP {r.status_code} {msg}")
        time.sleep(5)
    return ""


def resolve_images(cases=CASES, upload=True):
    """Return [{name, url, texts, expected, b64}] for notebooks: imgbb URL if available, else base64.

    URLs are cached by sha1(image bytes) in imgbb_urls.json so re-runs only upload what is missing.
    """
    from dotenv import dotenv_values
    key = dotenv_values(REPO / ".env").get("IMGBB_API_KEY", "") if upload else ""
    cache = json.loads(URL_CACHE.read_text()) if URL_CACHE.exists() else {}
    images = []
    for rel, name, texts, expected in cases:
        p = REPO / rel
        data = p.read_bytes()
        h = hashlib.sha1(data).hexdigest()
        url = cache.get(h) or (upload_imgbb(p, name, key) if key else "")
        if url:
            cache[h] = url
        images.append({"name": name + p.suffix, "url": url, "texts": texts, "expected": expected,
                       "b64": "" if url else base64.b64encode(data).decode()})
    URL_CACHE.write_text(json.dumps(cache, indent=1))
    return images


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def hf_login_cell():
    return code("try:\n"
                "    from kaggle_secrets import UserSecretsClient\n"
                "    from huggingface_hub import login\n"
                "    login(UserSecretsClient().get_secret('HF_TOKEN'))\n"
                "    print('HF login ok')\n"
                "except Exception as e:\n"
                "    print('No HF_TOKEN secret (fine for public models):', type(e).__name__)")


def gpu_info_cell(extra_imports=""):
    return code(f"import torch{extra_imports}\n"
                "print('torch', torch.__version__)\n"
                "for i in range(torch.cuda.device_count()):\n"
                "    p = torch.cuda.get_device_properties(i)\n"
                "    print(f'cuda:{i}', p.name, f'{p.total_memory/2**30:.1f} GB', 'bf16:', torch.cuda.is_bf16_supported())")


def images_cells(images):
    """Cells that define IMAGES (with local 'path'), fetch them (wget or base64), and show them."""
    return [
        md("## Images\nDownloaded from imgbb with `wget`; images with no URL are decoded from the embedded base64."),
        code("import base64, os, subprocess\n"
             f"IMAGES = {json.dumps(images, indent=1)}\n\n"
             "os.makedirs('imgs', exist_ok=True)\n"
             "for im in IMAGES:\n"
             "    im['path'] = os.path.join('imgs', im['name'])\n"
             "    if im['url']:\n"
             "        subprocess.run(['wget', '-q', '-O', im['path'], im['url']], check=True)\n"
             "        src = 'wget ' + im['url']\n"
             "    else:\n"
             "        open(im['path'], 'wb').write(base64.b64decode(im['b64']))\n"
             "        src = 'embedded base64'\n"
             "    print(f\"{im['name']}: {os.path.getsize(im['path'])} bytes via {src}\")"),
        code("from PIL import Image\n"
             "import matplotlib.pyplot as plt\n"
             "fig, axs = plt.subplots(1, len(IMAGES), figsize=(4 * len(IMAGES), 4))\n"
             "for ax, im in zip(axs, IMAGES):\n"
             "    ax.imshow(Image.open(im['path']).convert('RGB')); ax.set_title(im['name'], fontsize=7); ax.axis('off')\n"
             "plt.show()"),
        md("## Shared scoring helpers"),
        code(inline_source(report, summarize)),
    ]


def write_notebook(out: Path, cells, metadata=KAGGLE_T4_METADATA):
    nb = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")
