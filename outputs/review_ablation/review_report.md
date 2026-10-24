| arm | n | success | door_state | grasp_kept | pull_dir | all-4 | false-success | next-pull err median deg (<=30 deg) |
|---|---|---|---|---|---|---|---|---|
| A video only (production) | 10 | 100% | 100% | 100% | 80% | 80% | 0 | 133 (0%) |
| B video + real-tracker evidence | 10 | 100% | 80% | 100% | 100% | 80% | 0 | 20 (100%) |
| O video + oracle-tracker evidence | 10 | 100% | 100% | 100% | 100% | 100% | 0 | 6 (100%) |
| C video + MISLEADING evidence | 10 | 30% | 30% | 70% | 50% | 10% | 5 | 45 (0%) |
| D real-tracker evidence text only (no video) | 10 | 100% | 80% | 100% | 100% | 80% | 0 | 20 (100%) |

Per scenario (all-4 correct / n):

| scenario | A | B | O | C | D |
|---|---|---|---|---|---|
| wide_open | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 |
| slight_open | 2/2 | 2/2 | 2/2 | 0/2 | 2/2 |
| slip_mid_arc | 2/2 | 0/2 | 2/2 | 0/2 | 0/2 |
| latched_slip | 0/2 | 2/2 | 2/2 | 0/2 | 2/2 |
| wrong_dir_lost | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 |
