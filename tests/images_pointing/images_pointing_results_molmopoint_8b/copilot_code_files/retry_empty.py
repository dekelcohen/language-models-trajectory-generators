"""Retry the rows that returned prose instead of points.

Every row whose prompt contained "point at/to" produced coordinates; every row
phrased as a bare noun phrase ("far pot handle", "drone in the sky") made
MolmoPoint describe the object instead. That is a prompt-format effect, not an
inability to locate the object, so those rows are re-asked with an explicit
pointing instruction.

The original prompt and its result are kept either way, and the sheet records
which phrasing produced the final coordinates, so a fallback is never passed
off as the model answering the user's exact wording.
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, r"C:\Users\etzionh\Desktop\playground\PointingZoo\vlm")
import a100  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/home/user/etzion_vlm/dekel"
CONTAINER = "/work/dekel"


def pointify(prompt):
    """Turn any phrasing into an explicit pointing instruction."""
    cleaned = re.sub(r"^\s*point\s+(at|to)\s+", "", prompt,
                     flags=re.IGNORECASE).strip()
    return f"Point to {cleaned}"


def main():
    results = json.load(open(os.path.join(HERE, "results.json"),
                             encoding="utf-8"))
    df = pd.read_excel(os.path.join(HERE, "pointing_prompts.xlsx"))

    retry = []
    for key, rec in results.items():
        if rec["points"]:
            continue
        original = str(df.loc[int(key), "prompt"]).strip()
        retry.append({"id": int(key),
                      "path": f"{CONTAINER}/images/{rec['image']}",
                      "prompt": pointify(original)})

    if not retry:
        print("nothing to retry")
        return

    print(f"retrying {len(retry)} rows:")
    for job in retry:
        print(f"  row {job['id']:>2}: {job['prompt']}")

    local = os.path.join(HERE, "jobs_retry.json")
    with open(local, "w", encoding="utf-8") as handle:
        json.dump(retry, handle, indent=1)
    a100.put(local, f"{REMOTE}/jobs_retry.json")

    code, out = a100.run(
        "docker exec etzion_vid bash -lc "
        f"'cd {CONTAINER} && rm -f results_retry.json && PYTHONPATH=/work "
        "/work/venv_molmo/bin/python dekel_run.py --jobs jobs_retry.json "
        "--out results_retry.json 2>&1 | grep -avE \"Loading checkpoint|"
        "it/s.$|SyntaxWarning|slow image processor\"'", timeout=2400)
    print(out)
    a100.get(f"{REMOTE}/results_retry.json",
             os.path.join(HERE, "results_retry.json"))
    print("fetched results_retry.json")


if __name__ == "__main__":
    main()
