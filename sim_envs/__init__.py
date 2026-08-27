"""Simulation environment (sim-env) profiles.

A sim-env encapsulates everything that is task/scene specific for a simulator
backend: asset loading, physics tuning, robot start pose, camera framing, the
3D-coordinates prompt section and ground-truth state reporting.

Use :func:`sim_envs.registry.get_simenv` to resolve a ``--task`` name to a
sim-env instance.
"""

from sim_envs.registry import get_simenv, list_task_ids

__all__ = ["get_simenv", "list_task_ids"]
