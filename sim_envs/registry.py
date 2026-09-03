"""Sim-env registry: resolve a ``--task`` name to a sim-env profile instance.

Task naming
-----------
New sim-env suites use a namespaced id: ``<suite>:<task>``, e.g.
``franka_kitchen:microwave``. This keeps task names unambiguous as more scene
suites are added.

Legacy un-namespaced names (``door``, ``sawyer_door_v3``, ``franka_door``, or
anything else which falls back to the grasp scene) are still accepted so
existing commands and logs keep working.
"""

from sim_envs.pybullet.door import SimEnvDoor
from sim_envs.pybullet.grasp import SimEnvGrasp

# Keywords that select the Adroit door scene when no suite prefix is given.
_LEGACY_DOOR_KEYWORDS = ["door", "open_door", "opendoor", "sawyer_door_v3", "franka_door", "door:hidden"]


def _franka_kitchen_suite():
    # Imported lazily: the kitchen assets/profile pull in more modules and are
    # only needed when a franka_kitchen:* task is requested.
    from sim_envs.pybullet.franka_kitchen import KITCHEN_TASKS, SimEnvKitchen

    return KITCHEN_TASKS, SimEnvKitchen


# suite name -> callable returning (task_ids, simenv_class)
_SUITES = {
    "franka_kitchen": _franka_kitchen_suite,
}


def list_task_ids():
    """Return every selectable task id (namespaced suites + legacy names)."""
    ids = ["grasp", "door", "door:hidden"]
    for suite, loader in _SUITES.items():
        try:
            task_ids, _cls = loader()
        except Exception:
            continue
        ids.extend(f"{suite}:{t}" for t in task_ids)
    return ids


def get_simenv(task_name, sim=None):
    """Return a sim-env profile instance for ``task_name``.

    ``sim`` is the :class:`~sim_adapter.base.SimAdapter` the profile issues its
    primitives through; profiles never import a simulator themselves.

    Raises ValueError for an unknown suite or an unknown task within a known
    suite - silently falling back to the grasp scene would hide typos.
    """
    name = (task_name or "").strip()
    lowered_full = name.lower()

    # Plain "door" hides the pole (no hiding object); "door:hidden" keeps the
    # pole-hiding behavior. Both bypass the generic suite (":") routing below.
    if lowered_full == "door":
        return SimEnvDoor(sim, hide_door_with_object=False)
    if lowered_full == "door:hidden":
        return SimEnvDoor(sim, hide_door_with_object=True)

    if ":" in name:
        suite, _, task = name.partition(":")
        suite = suite.strip().lower()
        task = task.strip().lower()
        loader = _SUITES.get(suite)
        if loader is None:
            raise ValueError(
                f"Unknown sim-env suite '{suite}' in task '{task_name}'. "
                f"Known suites: {sorted(_SUITES)}"
            )
        task_ids, simenv_cls = loader()
        if task not in task_ids:
            raise ValueError(
                f"Unknown task '{task}' for suite '{suite}'. "
                f"Valid tasks: {sorted(task_ids)}"
            )
        # Suite profiles take the task id; the adapter is injected afterwards so a
        # suite that still talks to pybullet directly needs no signature change.
        instance = simenv_cls(task)
        instance.sim = sim
        return instance

    lowered = name.lower()
    if any(k in lowered for k in _LEGACY_DOOR_KEYWORDS):
        return SimEnvDoor(sim, hide_door_with_object=True)
    return SimEnvGrasp(sim)
