"""The head camera must actually show each Franka Kitchen task's target.

The framing in ``tasks.py`` is easy to break silently: a plausible-looking
camera tweak can leave the target hidden behind the Panda, or push it to the
edge of the frame, and every other test still passes -- the failure only shows
up as the VLM being unable to find the object it was asked to manipulate.

Occlusion is measured exactly rather than eyeballed: the scene is rendered
twice per task, once normally and once with the robot teleported out of the
world, and the target link's pixel counts are compared.
"""

import pytest

from sim_envs.pybullet.franka_kitchen.tasks import KITCHEN_TASKS
from tests.tools.tune_kitchen_head_camera import KitchenCameraProbe

# The arm may clip a corner of the target, but the bulk of it must be visible.
MAX_OCCLUDED_FRACTION = 0.15
# 0 = dead centre, 1 = at the frame edge. Segmentation degrades near the edges.
MAX_OFF_CENTRE = 0.35
# Below this the target is too few pixels for a VLM to segment reliably.
MIN_VISIBLE_PIXELS = 60

TASK_IDS = sorted(KITCHEN_TASKS)


@pytest.fixture(scope="module")
def framing():
    """Score every task's configured head-camera framing once."""
    results = {}
    for task_id in TASK_IDS:
        probe = KitchenCameraProbe(task_id)
        try:
            task = probe.task
            results[task_id] = probe.score(task.camera_distance,
                                           task.camera_pitch,
                                           task.camera_target_offset)
        finally:
            probe.close()
    return results


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_target_is_in_frame(framing, task_id):
    assert framing[task_id] is not None, (
        f"'{task_id}': the head camera does not show the target at all. "
        f"Re-tune with: python -m tests.tools.tune_kitchen_head_camera --task {task_id}"
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_target_not_occluded_by_arm(framing, task_id):
    r = framing[task_id]
    assert r is not None, f"'{task_id}': target not in frame"
    assert r["occluded"] <= MAX_OCCLUDED_FRACTION, (
        f"'{task_id}': the robot arm hides {r['occluded']:.0%} of the target "
        f"({r['visible_px']}/{r['potential_px']} px visible). "
        f"Re-tune with: python -m tests.tools.tune_kitchen_head_camera --task {task_id}"
    )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_target_is_centred_and_large_enough(framing, task_id):
    r = framing[task_id]
    assert r is not None, f"'{task_id}': target not in frame"
    assert r["off_centre"] <= MAX_OFF_CENTRE, (
        f"'{task_id}': target sits at {r['off_centre']:.2f} of the way to the frame "
        f"edge (pixel {r['target_px']}). Note a non-zero camera_target_offset "
        f"pushes the target off centre: {r['offset']}."
    )
    assert r["visible_px"] >= MIN_VISIBLE_PIXELS, (
        f"'{task_id}': only {r['visible_px']} px of the target are visible - too "
        f"small to segment. Move the camera closer (camera_distance={r['distance']})."
    )
