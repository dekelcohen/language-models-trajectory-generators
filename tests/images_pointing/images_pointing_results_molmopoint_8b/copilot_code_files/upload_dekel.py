"""Upload the pointing images and build the per-row job list on the A100."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, r"C:\Users\etzionh\Desktop\playground\PointingZoo\vlm")
import a100  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# Host path for uploading, container path for the job list: /home/user/etzion_vlm
# is mounted as /work inside the container, and the model runs in there.
REMOTE = "/home/user/etzion_vlm/dekel"
REMOTE_IN_CONTAINER = "/work/dekel"


def main():
    df = pd.read_excel(os.path.join(HERE, "pointing_prompts.xlsx"))
    print(a100.run(f"mkdir -p {REMOTE}/images")[1])

    for name in sorted(set(df["image_file_name"])):
        local = os.path.join(HERE, name)
        if not os.path.exists(local):
            raise SystemExit(f"missing image referenced by the sheet: {name}")
        a100.put_binary(local, f"{REMOTE}/images/{name}")
        print(f"uploaded {name}")

    jobs = [{"id": int(i),
             "path": f"{REMOTE_IN_CONTAINER}/images/{row['image_file_name']}",
             # Trailing spaces in the sheet would otherwise reach the model.
             "prompt": str(row["prompt"]).strip()}
            for i, row in df.iterrows()]

    local_jobs = os.path.join(HERE, "jobs.json")
    with open(local_jobs, "w", encoding="utf-8") as handle:
        json.dump(jobs, handle, indent=1)
    a100.put(local_jobs, f"{REMOTE}/jobs.json")

    a100.put(os.path.join(HERE, "dekel_run.py"), f"{REMOTE}/dekel_run.py")

    # Verify the paths the model will actually use resolve inside the
    # container. Uploading to the host path is not proof of that: a wrong
    # mapping only surfaces after the checkpoint has loaded, minutes later.
    missing = a100.run(
        "docker exec etzion_vid bash -lc '"
        f"for f in {REMOTE_IN_CONTAINER}/images/*; do :; done; "
        f"ls {REMOTE_IN_CONTAINER}/images | wc -l'")[1].strip()
    print(f"images visible inside container: {missing}")
    probe = jobs[0]["path"]
    code, out = a100.run(f"docker exec etzion_vid test -f {probe} && echo OK")
    if "OK" not in out:
        raise SystemExit(f"container cannot see {probe} - check the mount")
    print(f"probe OK: {probe}")


if __name__ == "__main__":
    main()
