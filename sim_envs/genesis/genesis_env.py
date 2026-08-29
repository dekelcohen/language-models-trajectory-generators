"""Genesis child-process entry point.

Genesis needs its own conda env (``vlm_genesis``), so it cannot run inside the agent's
interpreter. The parent starts this script via
:func:`providers.genesis_launcher.launch_genesis_child` and talks to it over the
JSON-lines TCP transport in :mod:`providers.json_ipc`.

This file is a **bootstrap only**. Every IPC command is still handled by
``env.run_simulation_environment`` - the one sim-agnostic app layer - so an app-level
change to ``EXECUTE_TRAJECTORY`` / ``CAPTURE_IMAGES`` is made in exactly one place and
takes effect on both simulators.

Two modes:

* server (default) - accept the parent's connection, then run the shared app loop.
* ``--gui`` with no ``--port`` - open the interactive Genesis viewer and idle, the
  Genesis twin of ``env.run_sim_demo``. Useful for eyeballing a scene by hand::

      python sim_envs/genesis/genesis_env.py --task door --gui
"""

import argparse
import os
import sys
import traceback

# Running this file directly puts sim_envs/genesis/ on sys.path, not the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402


DEFAULT_HOST = "127.0.0.1"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Genesis simulation child process")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Host the parent is listening for (server mode).")
    parser.add_argument("--port", type=int, default=None,
                        help="TCP port for the IPC endpoint. Omit for standalone GUI mode.")
    parser.add_argument("--task", default="grasp", help="Sim-env profile to load.")
    parser.add_argument("--robot", default="franka", help="Robot model (franka only for now).")
    parser.add_argument("--mode", default="default", help="Env mode, forwarded to env.Environment.")
    parser.add_argument("--gui", action="store_true",
                        help="Open the interactive Genesis viewer instead of rendering headless.")
    parser.add_argument("--backend", default=None,
                        help="Genesis backend override: cpu | gpu | vulkan | metal.")
    return parser


def make_adapter(args):
    """Create the adapter and reserve render resolutions **before** the scene is built.

    ``scene.add_camera`` is ``@gs.assert_unbuilt`` in Genesis, so every resolution the
    app will ever ask for has to be declared up front. ``robot.get_camera_image`` only
    ever uses ``config.image_width`` x ``config.image_height`` for both the head and the
    wrist camera, so one reservation covers the whole pipeline.
    """
    from sim_adapter.genesis_adapter import GenesisAdapter

    sim = GenesisAdapter(backend=args.backend, dt=config.control_dt)
    sim.reserve_camera(config.image_width, config.image_height,
                       fov=config.fov, near=config.near_plane, far=config.far_plane)
    return sim


def run_server(args):
    """Accept the parent's connection and hand it to the shared app layer."""
    from providers.json_ipc import JsonIpcServer

    server = JsonIpcServer(args.host, args.port)
    print(f"[GenesisEnv] listening on {args.host}:{args.port}", flush=True)
    try:
        # The parent starts connecting immediately after Popen, but importing Genesis
        # (torch + taichi) can take tens of seconds on a cold start, so be generous.
        endpoint = server.accept(timeout=300.0)
    finally:
        server.close()
    print(f"[GenesisEnv] parent connected: task={args.task} robot={args.robot}", flush=True)

    import env as envmod

    args.sim = "genesis"
    sim = make_adapter(args)
    try:
        envmod.run_simulation_environment(args, endpoint, None, sim=sim)
    finally:
        endpoint.close()
        try:
            sim.disconnect()
        except Exception:
            traceback.print_exc()


def run_gui_demo(args):
    """Standalone interactive viewer - the Genesis twin of ``env.run_sim_demo``."""
    import env as envmod

    print(f"[GenesisEnv] GUI demo: task={args.task}", flush=True)
    envmod.run_sim_demo(task_p=args.task, gui=args.gui, sim_name="genesis")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    args.sim = "genesis"

    if args.port is None:
        # No parent to talk to: this is the hand-driven debug entry point.
        run_gui_demo(args)
        return 0

    try:
        run_server(args)
    except Exception:
        # The parent only sees our exit code and stderr, so make the failure loud.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
