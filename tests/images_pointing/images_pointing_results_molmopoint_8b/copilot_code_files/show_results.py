"""Fetch the MolmoPoint results and show what the model actually said.

7 of 11 rows returned no points, including plainly phrased ones like "point at
the tank". MolmoPoint does not emit coordinates as text - they are special
tokens decoded by extract_image_points - so an empty list can mean either the
model declined to point or the decode produced nothing. The raw reply
distinguishes those cases.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Users\etzionh\Desktop\playground\PointingZoo\vlm")
import a100  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    local = os.path.join(HERE, "results.json")
    a100.get("/home/user/etzion_vlm/dekel/results.json", local)
    data = json.load(open(local, encoding="utf-8"))

    for key in sorted(data, key=int):
        rec = data[key]
        print(f"row {key:>2} | {rec['image']:<24} | {rec['prompt'][:50]}")
        print(f"        points {len(rec['points'])}  "
              f"raw: {rec['raw']!r}"[:400])
        print()


if __name__ == "__main__":
    main()
