"""Write the MolmoPoint results back to a sheet and render one figure per row.

Each row of the sheet is an (image, prompt) pair, so figures are named by row
index: the same image appears under several different prompts and would
otherwise overwrite itself.

Rows whose original phrasing produced prose were re-asked with an explicit
"Point to ..." instruction. Both the prompt that was actually used and the
provenance are written to the sheet so a fallback result is never mistaken for
an answer to the user's exact wording.
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")


def load_results():
    primary = json.load(open(os.path.join(HERE, "results.json"),
                             encoding="utf-8"))
    retry_path = os.path.join(HERE, "results_retry.json")
    retry = (json.load(open(retry_path, encoding="utf-8"))
             if os.path.exists(retry_path) else {})
    return primary, retry


def resolve(key, primary, retry):
    """Prefer the answer to the user's own wording; fall back only if empty."""
    first = primary[key]
    if first["points"]:
        return first["points"], first["prompt"], "original prompt", first["raw"]
    second = retry.get(key)
    if second and second["points"]:
        return (second["points"], second["prompt"],
                "explicit pointing fallback", second["raw"])
    return [], first["prompt"], "no points returned", first["raw"]


def draw(image_path, points, title, note, out_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Keep every figure a consistent physical size regardless of source
    # resolution, so markers stay legible on a 256px crop and a 4K frame alike.
    scale = 8.0 / max(width, height)
    fig, ax = plt.subplots(figsize=(width * scale, height * scale + 1.1))
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    marker = max(6.0, min(width, height) * 0.018)
    for index, (x, y) in enumerate(points, start=1):
        ax.plot(x, y, marker="o", markersize=marker, markerfacecolor="none",
                markeredgecolor="#00ff88", markeredgewidth=2.2)
        ax.plot(x, y, marker="+", markersize=marker * 0.9, color="#ff2d55",
                markeredgewidth=1.8)
        if len(points) > 1:
            ax.annotate(str(index), (x, y),
                        textcoords="offset points", xytext=(marker * 0.8,
                                                            -marker * 0.8),
                        color="#00ff88", fontsize=9, fontweight="bold",
                        path_effects=None)

    heading = "\n".join(textwrap.wrap(title, 68))
    ax.set_title(f"{heading}\n{note}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    primary, retry = load_results()
    df = pd.read_excel(os.path.join(HERE, "pointing_prompts.xlsx"))

    points_col, count_col, used_col, source_col, fig_col = [], [], [], [], []

    for index, row in df.iterrows():
        key = str(index)
        points, prompt_used, source, _ = resolve(key, primary, retry)
        rounded = [[round(float(x), 1), round(float(y), 1)] for x, y in points]

        image_name = row["image_file_name"]
        stem = os.path.splitext(image_name)[0]
        fig_name = f"row{index:02d}_{stem}.png"
        note = (f"MolmoPoint-8B - {len(points)} point(s) - {source}"
                if points else
                f"MolmoPoint-8B returned no points - {source}")
        draw(os.path.join(HERE, image_name), points,
             f"[row {index}] {prompt_used}", note,
             os.path.join(OUTDIR, fig_name))

        points_col.append(json.dumps(rounded))
        count_col.append(len(rounded))
        used_col.append(prompt_used)
        source_col.append(source)
        fig_col.append(f"outputs/{fig_name}")
        print(f"row {index:>2} {image_name:<24} {len(rounded):>2} pts  "
              f"{source}")

    df["molmo_points_xy"] = points_col
    df["molmo_num_points"] = count_col
    df["molmo_prompt_used"] = used_col
    df["molmo_result_source"] = source_col
    df["figure"] = fig_col

    out_xlsx = os.path.join(HERE, "pointing_prompts_with_molmo.xlsx")
    df.to_excel(out_xlsx, index=False)
    df.to_csv(os.path.join(HERE, "pointing_prompts_with_molmo.csv"),
              index=False, encoding="utf-8")
    print(f"\nwrote {out_xlsx}")
    print(f"wrote {len(fig_col)} figures to {OUTDIR}")


if __name__ == "__main__":
    main()
