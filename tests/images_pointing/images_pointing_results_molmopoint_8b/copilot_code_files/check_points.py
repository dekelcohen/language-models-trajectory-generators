"""Sanity-check the decoded points against each image's real dimensions.

MolmoPoint returns coordinates in the source image's pixel space, and these
images have very different sizes, so an out-of-bounds count is the quickest
proof that the decode used the right frame of reference.
"""
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    data = json.load(open(os.path.join(HERE, "results.json"),
                          encoding="utf-8"))
    for key in sorted(data, key=int):
        rec = data[key]
        width, height = Image.open(os.path.join(HERE, rec["image"])).size
        pts = rec["points"]
        oob = [p for p in pts
               if not (0 <= p[0] <= width and 0 <= p[1] <= height)]
        preview = [[round(x), round(y)] for x, y in pts[:4]]
        print(f"row {key:>2} {rec['image']:<24} {width}x{height} "
              f"n={len(pts):>2} out-of-bounds={len(oob)} {preview}")


if __name__ == "__main__":
    main()
