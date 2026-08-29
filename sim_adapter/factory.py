"""Resolve a ``--sim`` name to a :class:`~sim_adapter.base.SimAdapter` implementation.

Adapters are imported lazily: PyBullet and Genesis live in different conda envs and
importing the wrong one would fail at module load.
"""

SUPPORTED_SIMS = ("pybullet", "genesis")


def get_adapter(sim_name):
    name = (sim_name or "pybullet").strip().lower()
    if name == "pybullet":
        from sim_adapter.pybullet_adapter import PyBulletAdapter
        return PyBulletAdapter()
    if name == "genesis":
        from sim_adapter.genesis_adapter import GenesisAdapter
        return GenesisAdapter()
    raise ValueError(f"Unknown simulator '{sim_name}'. Supported: {sorted(SUPPORTED_SIMS)}")
