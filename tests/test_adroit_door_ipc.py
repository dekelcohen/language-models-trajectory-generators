import os
import time
import shutil
import multiprocessing as mp
import traceback

import pytest

import config
from env import run_simulation_environment
from debug.dbg_utils import create_video_from_images


def _run_env(conn):
    class _Args:
        mode = "default"
        robot = "franka"

    import logging
    logger = logging.getLogger("env")
    logger.setLevel(logging.INFO)
    try:
        run_simulation_environment(_Args, conn, logger)
    except Exception:
        traceback.print_exc()


def _clean_images_folder():
    try:
        root = os.path.join("images")
        if os.path.exists(root):
            # Remove everything under images/*
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        # Recreate trajectory folder
        os.makedirs(config.trajectory_folder, exist_ok=True)
    except Exception as e:
        print("[Test] Failed to clean images folder:", e)
        traceback.print_exc()


@pytest.mark.skip(
    reason="Dead test against a dead command. It sends config.SET_DOOR_STATE (18) and "
    "config.CAPTURE_TRAJECTORY_FRAME (19); both constants still exist in config.py but their "
    "handlers were removed from run_simulation_environment, so the env never replies and the "
    "test blocks forever in parent_conn.recv(). Un-skip only after restoring both handlers."
)
def test_set_door_state_via_ipc_and_make_video():
    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_run_env, args=(child_conn,), daemon=True)
    proc.start()

    # Clean images/* and ensure trajectory folder exists
    _clean_images_folder()

    try:
        # Wait a bit for env to init
        time.sleep(1.0)

        # Open door over N steps; capture TinyRenderer frames
        steps = 60
        for i in range(steps):
            angle = (i / float(steps)) * 1.2
            parent_conn.send([config.SET_DOOR_STATE, {"door_angle": angle}])
            _ = parent_conn.recv()
            # Ask env to capture a trajectory frame at index i
            parent_conn.send([config.CAPTURE_TRAJECTORY_FRAME, i])
            _ = parent_conn.recv()

        # Assemble MP4 with dbg_utils -> write into images/
        create_video_from_images(
            folder_path=config.trajectory_folder,
            output_video_folder_path=os.path.join("images"),
            base_name=config.trajectory_image_base,
            start_idx=0,
            end_idx=steps - 1,
            ext="png",
            fps=config.trajectory_video_fps,
        )
        output_mp4 = os.path.join("images", f"{config.trajectory_image_base}_0_{steps - 1}.mp4")
        assert os.path.exists(output_mp4)
        assert os.path.getsize(output_mp4) > 1024

    except Exception as e:
        print("[Test] Exception in IPC door test:", e)
        traceback.print_exc()
        assert False, "IPC door state test failed due to exception"
    finally:
        try:
            parent_conn.send([config.TASK_COMPLETED])
        except Exception:
            pass
        time.sleep(0.5)
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            child_conn.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
