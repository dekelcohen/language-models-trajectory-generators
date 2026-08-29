"""Genesis-specific process bootstrap.

There is deliberately no task code here: ``sim_envs/pybullet/{grasp,door}.py`` contain
zero simulator calls after the Phase-1 refactor, so both simulators share them via
``sim_envs.registry``. This package only owns *process* concerns - starting Genesis in
its own interpreter and handing the IPC endpoint to the shared ``env.py`` app layer.
"""
