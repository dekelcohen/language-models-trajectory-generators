| arm | n | success | door_state | grasp_kept | pull_dir | all-4 | false-success |
|---|---|---|---|---|---|---|---|
| A video only (production) | 10 | 100% | 80% | 100% | 70% | 50% | 0 |
| B video + real-tracker evidence | 10 | 100% | 80% | 100% | 80% | 60% | 0 |
| O video + oracle-tracker evidence | 10 | 100% | 100% | 100% | 90% | 90% | 0 |
| C video + MISLEADING evidence | 10 | 30% | 30% | 50% | 50% | 10% | 5 |
| D real-tracker evidence text only (no video) | 10 | 100% | 80% | 80% | 90% | 50% | 0 |

Per scenario (all-4 correct / n):

| scenario | A | B | O | C | D |
|---|---|---|---|---|---|
| wide_open | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 |
| slight_open | 1/2 | 2/2 | 2/2 | 0/2 | 2/2 |
| slip_mid_arc | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 |
| latched_slip | 0/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| wrong_dir_lost | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 |
