import numpy as np
import math
import openai
import torch
import os
import sys
import argparse
import json
import traceback
import multiprocessing
import logging
import functools
import models
import config
from numpy import pi
# LangSAM imported lazily only if selected as provider
from multiprocessing import Process, Pipe
from io import StringIO
from contextlib import redirect_stdout
from api import API
from env import run_simulation_environment
from prompts.main_prompt import MAIN_PROMPT
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, FAIL, ENDC

print = functools.partial(print, flush=True)


def _probe_metaworld_ws(server_url, logger, connect_timeout=2.0, ready_timeout=2.0, probe_timeout=3.0):
    """Try connecting to an already-running Metaworld WS server.
    Returns a connected WsJSONConnection if responsive; otherwise None.
    """
    try:
        from providers.ws_connection import WsJSONConnection
        conn = WsJSONConnection(server_url, connect_timeout=connect_timeout)
        ready = conn.recv(timeout=ready_timeout)
        if isinstance(ready, dict) and ready.get("status") == "ready":
            # Send a short probe to confirm responsiveness
            conn.send([config.GET_STATE])
            resp = conn.recv(timeout=probe_timeout)
            if isinstance(resp, dict) and "eef_pos" in resp:
                logger.info("Using existing Metaworld WS server: %s" % json.dumps(ready))
                return conn
        conn.close()
    except Exception:
        pass
    return None


def _setup_metaworld_ws(args, logger):
    """Connect to Metaworld WS server if already running; otherwise spawn and connect.
    Returns a tuple: (connection, server_proc_or_None).
    """
    host = os.environ.get("METAWORLD_WS_HOST", "127.0.0.1")
    port = int(os.environ.get("METAWORLD_WS_PORT", "8765"))
    default_url = f"ws://{host}:{port}"
    server_url = os.environ.get("METAWORLD_SERVER_URL", default_url)

    conn = _probe_metaworld_ws(server_url, logger)
    if conn:
        return conn, None

    # Spawn server without capturing stdio so pdb works normally
    server_path = os.path.join(os.path.dirname(__file__), "providers", "metaworld_server.py")
    py_exe = os.environ.get("METAWORLD_PYTHON", sys.executable)
    import subprocess
    cmd = [py_exe, server_path, "--env", args.task, "--ws-host", host, "--ws-port", str(port)]
    _p = subprocess.Popen(cmd, stdin=None, stdout=None, stderr=None, cwd=os.getcwd())

    # Connect to spawned server
    from providers.ws_connection import WsJSONConnection
    conn = WsJSONConnection(default_url, connect_timeout=15.0)
    try:
        _ready = conn.recv(timeout=15)
        if isinstance(_ready, dict) and _ready.get("status") == "ready":
            logger.info("Connected Metaworld WS server: %s" % json.dumps(_ready))
    except Exception:
        pass
    return conn, _p


def _safe_terminate(proc, logger, name="Metaworld WS server"):
    try:
        if proc is None:
            return
        # Try graceful terminate
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        if proc.poll() is None:
            # Force kill
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        if proc.poll() is None:
            logger.info(PROGRESS + f"Warning: {name} pid {proc.pid} still running; please close it manually." + ENDC)
    except Exception:
        # Best-effort only
        logger.info(PROGRESS + f"Warning: failed to terminate {name}." + ENDC)

if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["azure-gpt-5", "azure-gpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], default="azure-gpt-5", help="select language model")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    parser.add_argument("-s", "--sim", choices=["pybullet", "metaworld"], default="pybullet", help="select simulator backend")
    parser.add_argument("--transport", choices=["auto", "pipe", "ws"], default="auto", help="connection transport override; auto: pipe for pybullet, ws for metaworld")
    parser.add_argument("--task", type=str, default="sawyer_door_v3", help="task/environment name (metaworld only)")
    parser.add_argument("--seg-provider", choices=["langsam", "sam3"], default="langsam", help="select segmentation provider (LangSAM or RoboFlow SAM3)")
    parser.add_argument("--depth-format", choices=["norm_1m", "norm_zfar", "raw"], default="norm_1m", help="depth handling for reconstruction")
    parser.add_argument("--delete-images", action="store_true", help="delete image folders before recreating them")
    parser.add_argument("--track-provider", choices=["xmem", "none"], default="xmem", help="tracking provider for success verification; set to 'none' to disable XMem usage")
    args = parser.parse_args()

    # Logging
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    # Ensure image output directories exist (optionally clean before)
    from common_utils import ensure_image_dirs_exist
    ensure_image_dirs_exist(delete=args.delete_images)

    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. Please connect to a GPU instance if possible.")
        device = torch.device("cpu")

    torch.set_grad_enabled(False)

    # Load models
    if args.seg_provider == "langsam":
        from lang_sam import LangSAM
        langsam_model = LangSAM()
    else:
        langsam_model = None
    xmem_model = None
    # Lazily load XMem only if requested
    if args.track_provider == "xmem":
        try:
            sys.path.append("./XMem/")
            from XMem.model.network import XMem  # type: ignore
            xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
        except Exception as e:
            # Do not crash if XMem submodule/weights are missing when disabled; surface a clear log when enabled
            raise RuntimeError("Failed to initialize XMem model. Ensure the 'XMem' submodule and weights exist.") from e

    # Create connection to the chosen simulator, then build API with it
    server_proc = None
    if args.sim == "pybullet":
        main_connection, env_connection = Pipe()
        # Start process
        env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, logger])
        env_process.start()
        [env_connection_message] = main_connection.recv()
        logger.info(env_connection_message)
    else:
        # Metaworld: WebSockets transport only
        main_connection, server_proc = _setup_metaworld_ws(args, logger)

    # API set-up
    api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)

    detect_object = api.detect_object
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    task_completed = api.task_completed

    try:
        # User input
        command = input("Enter a command: ")
        api.command = command

        # Main task execution loop
        logger.info(PROGRESS + "STARTING TASK..." + ENDC)
    
        messages = []
    
        error = False
    
        new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
    
        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "system")
        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    
        while True:
    
            while not api.completed_task:
    
                new_prompt = ""
    
                if len(messages[-1]["content"].split("```python")) > 1:
    
                    code_block = messages[-1]["content"].split("```python")
    
                    block_number = 0
    
                    for block in code_block:
                        if not error:
                            if len(block.split("```")) > 1:
                                code = block.split("```")[0]
                                block_number += 1
                                try:                                
                                    f = StringIO()
                                    with redirect_stdout(f):                                    
                                        exec(code)
                                except Exception:
                                    error_message = traceback.format_exc()
                                    new_prompt += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                                    new_prompt += "\n"
                                    error = True
                                else:
                                    s = f.getvalue()
                                    error = False
                                    if s != "" and len(s) < 2000:
                                        new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                                        new_prompt += "\n"
                                        error = True
    
                if error:
    
                    api.completed_task = False
                    api.failed_task = False
    
                if not api.completed_task:
    
                    if api.failed_task:
    
                        logger.info(FAIL + "FAILED TASK! Generating summary of the task execution attempt..." + ENDC)
    
                        new_prompt += TASK_SUMMARY_PROMPT
                        new_prompt += "\n"
    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    
                        logger.info(PROGRESS + "RETRYING TASK..." + ENDC)
    
                        new_prompt = MAIN_PROMPT.replace("[INSERT EE POSITION]", str(config.ee_start_position)).replace("[INSERT TASK]", command)
                        new_prompt += "\n"
                        new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])
    
                        messages = []
    
                        error = False
    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "system")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    
                        api.failed_task = False
    
                    else:
    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    
                        error = False
    
            logger.info(OK + "FINISHED TASK!" + ENDC)
    
            new_prompt = input("Enter a command: ")
    
            logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
            logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    except KeyboardInterrupt:
        logger.info(PROGRESS + "Interrupted by user (Ctrl+C). Shutting down..." + ENDC)
    finally:
        # Close connection and terminate spawned server, if any
        try:
            if hasattr(main_connection, "close"):
                main_connection.close()
        except Exception:
            pass
        _safe_terminate(server_proc, logger)

        api.completed_task = False
