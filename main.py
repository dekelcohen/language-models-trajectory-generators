import numpy as np
import math
import openai
import torch
import os
import sys
import argparse
import json
import traceback
from debug.dbg_utils import init_loguru_logger
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
import segmentation_adapter  # override bbox/object filtering for segmentation
from prompts.main_prompt import MAIN_PROMPT, IN_CONTEXT_EXAMPLE, DETECT_OBJECT_TOOL, NO_DETECT_OBJECT_TOOL, DETECT_OBJECT_TOOL_INITIAL_PLANNING, NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT
from config import OK, PROGRESS, WARNING, FAIL, ENDC

print = functools.partial(print, flush=True)

# --- Prompt helper ------------------------------------------------------
def prepend_to_initial_command(command, args, logger):
    """Return command optionally prepended with file contents for first MAIN_PROMPT.
    On read error, logs a warning and returns the original command.
    """
    first_command = command
    if args.prepend_prompt:
        try:
            with open(args.prepend_prompt, "r", encoding="utf-8") as _pf:
                _pre = _pf.read().strip()
            if _pre:
                first_command = _pre + "\n" + command
                try:
                    logger.info(PROGRESS + "Prepended prompt from --prepend-prompt file." + ENDC)
                except Exception:
                    pass
        except Exception as e:
            try:
                logger.info(PROGRESS + "Warning: failed reading --prepend-prompt file." + ENDC)
            except Exception:
                pass
    return first_command

# --- LLM context helper -------------------------------------------------
EEF_POS_SNIPPET = 'Current end-effector pos (x,y,z): {eef_pos}'
def build_llm_context_images_and_pose(trajectory_step, logger):
    """Collect current images (head/wrist and current-step frames) and EE pose.

    - Returns tuple (image_paths, pose_snippet_str).
    - Silently skips missing files; logs only lightweight warnings.
    - If main_connection provided, tries call env to get freshe gripper ee pose;
    """
    image_paths = []

    def _maybe_add(path):
        if path and os.path.exists(path):
            image_paths.append(path)
        

    # Current-step trajectory frames (head + wrist) if available
    _maybe_add(config.rgb_image_trajectory_path.format(step=trajectory_step))
    _maybe_add(config.wrist_rgb_image_trajectory_path.format(step=trajectory_step))

    # Query current EE pose
    eef_pos = None
    if main_connection is not None:
        try:
            main_connection.send([config.GET_ROBOT_STATE, {}])
            resp = main_connection.recv()
            eef_pos = list(map(float, resp["eef_pos"]))
        except Exception as e:
            logger.info(PROGRESS + f"Warning: GET_ROBOT_STATE for LLM context failed: {e}" + ENDC)
                
    
    return image_paths, eef_pos


# --- Diagnostics helper -------------------------------------------------
def query_sim_objects_state(conn, logger):
    """Query and log all objects/geoms states from the simulator (WS only).
    Prints EEF pose, door joint angle, and every object entry with pos/dims.
    """
    try:
        conn.send([config.GET_STATE, {"objects": []}])  # empty all geoms
        gt = conn.recv()
        if not isinstance(gt, dict):
            return
        eef = gt.get("eef_pos")
        door_ang = gt.get("doorjoint_angle")
        objs = gt.get("objects", {}) or {}
        logger.info(PROGRESS + f"GT eef_pos={eef} doorjoint_angle={door_ang}" + ENDC)
        logger.info(PROGRESS + f"GT objects: count={len(objs)}" + ENDC)
        for name in sorted(objs.keys()):
            o = objs[name]
            pos = o.get("pos")
            dims = o.get("dims")
            kind = o.get("kind")
            logger.info(PROGRESS + f" - {name} ({kind}): pos={pos} dims={dims}" + ENDC)
    except Exception:
        pass


def _probe_metaworld_ws(server_url, logger, connect_timeout=2.0, ready_timeout=2.0, probe_timeout=3.0):
    """Try connecting to an already-running Metaworld WS server.
    Returns a connected WsJSONConnection if responsive; otherwise None.
    """
    try:
        from providers.ws_connection import WsJSONConnection
        conn = WsJSONConnection(server_url, timeout=connect_timeout)
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
    conn = WsJSONConnection(default_url, timeout=getattr(args, 'timeout', 15.0))
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

# --- Handshake helper ---------------------------------------------------
def read_env_handshake(main_connection, logger, default_pos):
    """Read the environment handshake and return (ee_pos_for_prompt, msg, coords_section, sim_state).
    Logs all errors; re-raises on receive failure to avoid silent mismatch.
    """
    try:
        payload = main_connection.recv()
    except Exception as e:
        logger.error(FAIL + f"Failed to receive env handshake: {e}" + ENDC)
        raise

    ee_pos = list(map(float, default_pos))
    msg = None
    coords_section = None
    sim_state = {}
    try:
        if isinstance(payload, (list, tuple)):
            if len(payload) == 4:
                eef_pos, coords_section, sim_state, msg = payload
                try:
                    ee_pos = list(map(float, eef_pos))
                except Exception as e:
                    logger.error(FAIL + f"Invalid ee_pos in handshake: {eef_pos} err={e}" + ENDC)
            elif len(payload) == 3:
                # Backward compatible: (eef_pos, coords_section, msg)
                eef_pos, coords_section, msg = payload
                try:
                    ee_pos = list(map(float, eef_pos))
                except Exception as e:
                    logger.error(FAIL + f"Invalid ee_pos in handshake: {eef_pos} err={e}" + ENDC)
            elif len(payload) == 2:
                eef_pos, msg = payload
                try:
                    ee_pos = list(map(float, eef_pos))
                except Exception as e:
                    logger.error(FAIL + f"Invalid ee_pos in handshake: {eef_pos} err={e}" + ENDC)
            elif len(payload) == 1:
                msg = payload[0]
            else:
                logger.error(FAIL + f"Unexpected handshake length={len(payload)} type={type(payload)}" + ENDC)
        elif isinstance(payload, str):
            msg = payload
        else:
            logger.error(FAIL + f"Unexpected handshake type: {type(payload)}" + ENDC)
    except Exception as e:
        logger.error(FAIL + f"Handshake parsing error: {e}" + ENDC)

    if msg is not None:
        try:
            logger.info(msg)
        except Exception as e:
            logger.error(FAIL + f"Failed to log env message: {e}" + ENDC)
    # Print env-specific state immediately with the finish setup message
    try:
        _sim_state_str = json.dumps(sim_state)
        logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
    except Exception as e:
        logger.error(FAIL + f"Failed to log env state: {e}" + ENDC)
    return ee_pos, msg, coords_section, sim_state

def process_cli_viz_point_arg(args, conn, logger):
    """Parse --viz-point/--vis-point and send one or more points.

    Accepts either a single point JSON list "[x,y,z]" or a JSON list of
    points "[[x1,y1,z1],[x2,y2,z2], ...]". Points are rendered as permanent
    markers in the simulator.
    """
    if not getattr(args, "viz_point", None):
        return
    try:
        raw = json.loads(args.viz_point)
        points = []                   
        # List of points [[x,y,z], ...]
        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    points.append([float(item[0]), float(item[1]), float(item[2])])
        # Single point [x,y,z]            
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            points.append([float(raw[0]), float(raw[1]), float(raw[2])])
                     
        if points:
            # Permanent, red markers by default
            conn.send([config.ADD_TRAJECTORY_POINTS, points, "blue", True, "line"])                                    
            logger.info(PROGRESS + f"Added visualization debug points in sim: {points} permanent viz points" + ENDC)            
        else:            
            logger.info(WARNING + "--viz-point provided but contained no valid coordinates" + ENDC)
            
    except Exception as e:        
        logger.info(FAIL + f"Failed to add viz point(s): {e}" + ENDC)        

if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = None
    if not openai.api_key is None and len(openai.api_key) > 0:
        client = openai.OpenAI()

    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", default="azure-gpt-5", help="select language model (e.g. azure-gpt-5, gpt-4o, or-google/gemini-2.5-flash)")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    parser.add_argument("-s", "--sim", choices=["pybullet", "metaworld"], default="pybullet", help="select simulator backend")
    parser.add_argument("--transport", choices=["auto", "pipe", "ws"], default="auto", help="connection transport override; auto: pipe for pybullet, ws for metaworld")
    parser.add_argument("--task", type=str, default="sawyer_door_v3", help="task/environment name (metaworld only)")
    parser.add_argument("--seg-provider", choices=["langsam", "sam3", "moondream"], default="langsam", help="select segmentation provider (LangSAM, RoboFlow SAM3, or Moondream)")
    parser.add_argument("--depth-format", choices=["norm_1m", "norm_zfar", "raw"], default="norm_1m", help="depth handling for reconstruction")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout seconds; <=0 disables timeouts")
    parser.add_argument("--delete-images", action="store_true", help="delete image folders before recreating them")
    parser.add_argument("--review-provider", choices=["vlm", "xmem"], default="vlm", help="review provider for success verification; vlm uses the vision-language (gpt) reviewer; xmem preserves the legacy tracking flow")
    parser.add_argument("--ovr-bbox", type=str, default=None, help="override segmentation bbox as \"x1,y1,x2,y2\" in pixels")    
    parser.add_argument("--ovr-obj",  type=str, default=None, help=(
            "Apply --ovr-bbox only to predictions whose text label (provider 'class') matches this regex. "
            "If omitted, the override applies to all predictions (legacy behavior). "
            "Examples: door.*?(handle|knob|lever) ; (?i)^door\\s+lever$ . "
        ),
    )
    # Accept both --viz-point (original) and --vis-point (alias)
    parser.add_argument("--viz-point", "--vis-point", dest="viz_point", type=str, default=None,
                        help="Add permanent 3D world visualization point(s) as JSON: \"[x,y,z]\" or \"[[x1,y1,z1],[x2,y2,z2],…]\"")
    parser.add_argument("--prepend-prompt", type=str, default=None, help="Path to a text file whose contents are prepended to the initial command (first MAIN_PROMPT only).",
    )
    parser.add_argument("--vis-traj", action="store_true", help="visualize trajectory points in the sim environment (3d sphere markers)")
    parser.add_argument("--vis-grasp", action="store_true", help="visualize grasp pose candidates in the 3D sim environment (cylinder/sphere markers)")
    parser.add_argument("--save-grasp-inputs", action="store_true", help="save binary segmentation mask (masks[0]) as .npy and projection/view matrices as .npy under images_folder after each detect_object call")
    args = parser.parse_args()

    
    try:
        os.makedirs(config.images_folder, exist_ok=True)
    except Exception as e:
        # Fail fast to avoid silent non-file logging
        raise RuntimeError(f"Failed to ensure images folder exists at '{config.images_folder}': {e}") from e

    # Logging (Loguru): emit to console and to a file under images folder
    logger = init_loguru_logger("vlm_traj.log")
    # Also wire adapter-level logger for its internal warnings
    segmentation_adapter.logger = logger

    segmentation_adapter.set_override_object_regex(args.ovr_obj)
    # Ensure image output directories exist (optionally clean before)
    from common_utils import ensure_image_dirs_exist
    ensure_image_dirs_exist(delete=args.delete_images)

    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. using CPU for local models - if any")
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
    if args.review_provider == "xmem":
        try:
            sys.path.append("./XMem/")
            from XMem.model.network import XMem  # type: ignore
            xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
        except Exception as e:
            # Do not crash if XMem submodule/weights are missing when disabled; surface a clear log when enabled
            raise RuntimeError("Failed to initialize XMem model. Ensure the 'XMem' submodule and weights exist.") from e

    # Create connection to the chosen simulator, then build API with it
    server_proc = None
    # Default EEF position for prompt if the environment does not report one
    ee_pos_for_prompt = list(map(float, config.ee_start_position))
    coords_section = None
    if args.sim == "pybullet":
        main_connection, env_connection = Pipe()
        # Start process
        # Do not pass Loguru logger across processes (not picklable on Windows).
        # Child process will initialize its own Loguru sinks.
        env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, None])
        env_process.start()
        # Receive environment handshake and derive EE position for prompt
        ee_pos_for_prompt, _msg, coords_section, sim_state = read_env_handshake(main_connection, logger, ee_pos_for_prompt)
        process_cli_viz_point_arg(args, main_connection, logger)
    else:
        # Metaworld: WebSockets transport only
        main_connection, server_proc = _setup_metaworld_ws(args, logger)

    # Per-task 3D coordinate prompt section (PyBullet: handshake-provided; Metaworld: default)

    if coords_section is None:
        coords_section = config.three_d_coordinates_prompt_section

    # API set-up
    api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)

    # Pass env sim_state into API for diagnostics
    api.sim_state = sim_state
    api.ee_pos_for_prompt = ee_pos_for_prompt
    api.coords_section = coords_section

    # The functions below are in the inpterpreter context of exec(code generated by the VLM). They are not statically referenced in the module
    detect_object = api.detect_object
    get_grasp_poses = api.get_grasp_poses
    visualize_grasp_pose = api.visualize_grasp_pose
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    task_completed = api.task_completed
    generate_linear_trajectory = api.generate_linear_trajectory
    try:
        # Query ground-truth state (all objects/geoms) for cross-check
        if args.sim == "metaworld" and hasattr(main_connection, 'send'):
            query_sim_objects_state(main_connection, logger)

        # User input
        command = input("Enter a command: ")
        if not str(command).strip():
            logger.info(PROGRESS + "No command entered. Exiting." + ENDC)
            sys.exit(0)
        api.command = command

        # Main task execution loop
        logger.info(PROGRESS + "STARTING TASK..." + ENDC)
    
        messages = []
    
        error = False
    
        # Build initial command 
        # Images paths to include in prompt - currently only rgb head-cam
        image_paths = [config.rgb_image_head_path]
        # --prepend-prompt - optional prepend a prev run conversation or any other long prompt 
        first_command = prepend_to_initial_command(command, args, logger)
        new_prompt = MAIN_PROMPT.replace("[INSERT DETECT_OBJECT_TOOL]", DETECT_OBJECT_TOOL) \
                                .replace("[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]", DETECT_OBJECT_TOOL_INITIAL_PLANNING) \
                                .replace("[INSERT EE POSITION]", str(ee_pos_for_prompt)) \
                                .replace("[INSERT TASK]", first_command) \
                                .replace("[INSERT 3D COORDINATES PROMPT SECTION]", coords_section) \
                                .replace("[INSERT IN CONTEXT EXAMPLE]", IN_CONTEXT_EXAMPLE)
         
        # Print env-specific state alongside the finish-setup message
        try:
            _sim_state_str = json.dumps(sim_state)
            logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
        except Exception:
            pass

        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, role="system", image_paths=image_paths)
        api.conversation_messages = messages
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
                    # api.failed_task preserved if previously set
    
                if not api.completed_task:
                    if api.failed_task:
                        logger.info(FAIL + "FAILED TASK! Generating summary of the task execution attempt..." + ENDC)
    
                        new_prompt += TASK_SUMMARY_PROMPT
                        new_prompt += "\n"
    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
                        api.conversation_messages = messages
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
    
                        logger.info(PROGRESS + "RETRYING TASK..." + ENDC)
                        # Mark the start of the new attempt for downstream reviewers
                        api.start_attempt_trajectory_step = api.trajectory_step

                        # Before composing the retry prompt, refresh the current robot state
                        _, eef_pos = build_llm_context_images_and_pose(api.trajectory_step, logger)
                        # On retry, use the latest known EE position rather than static config, no in context example and 
                        # no detect_object tool, since robot arm usually occuluds the target object (use first attempt obj coords instead) 
                        new_prompt = MAIN_PROMPT.replace("[INSERT DETECT_OBJECT_TOOL]", NO_DETECT_OBJECT_TOOL) \
                                .replace("[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]", NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING) \
                                .replace("[INSERT EE POSITION]", str(eef_pos)) \
                                .replace("[INSERT TASK]", command) \
                                .replace("[INSERT 3D COORDINATES PROMPT SECTION]", coords_section) \
                                .replace("[INSERT IN CONTEXT EXAMPLE]", '')
                        try:
                            _sim_state_str = json.dumps(sim_state)
                            logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
                        except Exception:
                            pass

                        new_prompt += "\n"
                        new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", messages[-1]["content"])
                        messages = []
                        error = False
    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "system")
                        api.conversation_messages = messages
                        api.failed_task = False # After retry task --> reset api.failed_task flag to resume normal flow (retry)    
                    else:    
                        logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                        # Attach current images and pose for better next-step context
                        
                        
                        
                        _imgs_paths, eef_pos = build_llm_context_images_and_pose(api.trajectory_step, logger)
                        if config.ENABLE_EEF_POS_IMAGE and eef_pos:                        
                            new_prompt += f'\n{EEF_POS_SNIPPET}\n'.format(eef_pos=eef_pos)
                        else:
                            _imgs_paths = None
                        messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user", image_paths=_imgs_paths)
                        api.conversation_messages = messages
                        logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
                        error = False
    
            logger.info(OK + "FINISHED TASK!" + ENDC)
    
            new_prompt = input("Enter a command: ").strip()
            if not new_prompt:
                logger.info(PROGRESS + "No command entered. Exiting." + ENDC)
                break

            logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
            messages = models.get_chatgpt_output(client, args.language_model, new_prompt, messages, "user")
            api.conversation_messages = messages
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






