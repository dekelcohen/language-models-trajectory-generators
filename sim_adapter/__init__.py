"""Simulator abstraction layer. See :mod:`sim_adapter.base` for the contract."""

from sim_adapter.base import (
    DEPTH_LINEAR_METRIC,
    DEPTH_OPENGL,
    JOINT_FIXED,
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    CameraFrame,
    CameraParams,
    JointInfo,
    JointState,
    SimAdapter,
)
from sim_adapter.factory import SUPPORTED_SIMS, get_adapter

__all__ = [
    "SimAdapter",
    "JointInfo",
    "JointState",
    "CameraFrame",
    "CameraParams",
    "JOINT_REVOLUTE",
    "JOINT_PRISMATIC",
    "JOINT_FIXED",
    "DEPTH_OPENGL",
    "DEPTH_LINEAR_METRIC",
    "get_adapter",
    "SUPPORTED_SIMS",
]
