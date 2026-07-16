"""Agent runner: one-time init/teardown and per-(sub)task execution.

This module extracts the previously-monolithic `__main__` flow of main.py into:
- init_agent(args, logger)  -> AgentContext   (one-time setup)
- teardown_agent(ctx)                         (one-time teardown)
- execute_task(ctx, prompt, max_attempts,...) (a self-contained (sub)task run)
- run_plan(ctx, command)                      (LLM planner + subtask orchestration)

Each (sub)task behaves like a function call: it owns its own messages,
completed/failed flags, attempt counter, per-attempt summaries and VLM reviewer.
The physical robot/sim state and continuous trajectory image numbering carry over
between subtasks.
"""
import os
import sys
import json
import math
import traceback
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
from numpy import pi  # noqa: F401 (kept available for exec'd code parity)

import config
import models
import segmentation_adapter
from config import OK, PROGRESS, WARNING, FAIL, ENDC
from helpers.main_utils import get_exec_locals, execute_blocks_from_log
from prompts.main_prompt import (
    MAIN_PROMPT,
    IN_CONTEXT_EXAMPLE,
    DETECT_OBJECT_TOOL,
    NO_DETECT_OBJECT_TOOL,
    DETECT_OBJECT_TOOL_INITIAL_PLANNING,
    NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING,
    COLLISION_AVOIDANCE,
    CODE_BLOCK_CONVENTIONS,
    INITIAL_PLANNING_1,
    INITIAL_PLANNING_2,
)
from prompts.error_correction_prompt import ERROR_CORRECTION_PROMPT
from prompts.print_output_prompt import PRINT_OUTPUT_PROMPT
from prompts.task_failure_prompt import TASK_FAILURE_PROMPT
from prompts.task_summary_prompt import TASK_SUMMARY_PROMPT

EEF_POS_SNIPPET = 'Current end-effector pos (x,y,z): {eef_pos}'

# Fallback user turn when a code block executed successfully but produced no
# captured stdout and no context images are attached (keeps the conversation from
# ending on an assistant message, which some providers reject).
CONTINUE_TASK_PROMPT = (
    "The previous code block executed successfully with no output. "
    "Continue with the next step of the task, or call task_completed() if the task is now complete."
)


# --- Prompt helper ------------------------------------------------------
def prepend_to_initial_command(command, args, logger):
    """Return command optionally prepended with file contents for first MAIN_PROMPT.
    On read error, logs a warning and returns the original command.
    """
    first_command = command
    if getattr(args, "prepend_prompt", None):
        try:
            with open(args.prepend_prompt, "r", encoding="utf-8") as _pf:
                _pre = _pf.read().strip()
            if _pre:
                first_command = _pre + "\n" + command
                try:
                    logger.info(PROGRESS + "Prepended prompt from --prepend-prompt file." + ENDC)
                except Exception:
                    pass
        except Exception:
            try:
                logger.info(PROGRESS + "Warning: failed reading --prepend-prompt file." + ENDC)
            except Exception:
                pass
    return first_command


# --- LLM context helper -------------------------------------------------
def build_llm_context_images_and_pose(main_connection, trajectory_step, logger):
    """Collect current trajectory frames and EE pose.

    - Returns tuple (image_paths, eef_pos).
    - Silently skips missing files; logs only lightweight warnings.
    - Queries the current gripper EE pose via main_connection when available.
    """
    image_paths = []

    def _maybe_add(path):
        if path and os.path.exists(path):
            image_paths.append(path)

    _maybe_add(config.rgb_image_trajectory_path.format(step=trajectory_step))
    _maybe_add(config.wrist_rgb_image_trajectory_path.format(step=trajectory_step))

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
    """Query and log all objects/geoms states from the simulator (WS only)."""
    try:
        conn.send([config.GET_STATE, {"objects": []}])
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

    server_path = os.path.join(os.path.dirname(__file__), "providers", "metaworld_server.py")
    py_exe = os.environ.get("METAWORLD_PYTHON", sys.executable)
    import subprocess
    cmd = [py_exe, server_path, "--env", args.task, "--ws-host", host, "--ws-port", str(port)]
    _p = subprocess.Popen(cmd, stdin=None, stdout=None, stderr=None, cwd=os.getcwd())

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
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
        if proc.poll() is None:
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
        logger.info(PROGRESS + f"Warning: failed to terminate {name}." + ENDC)


# --- Handshake helper ---------------------------------------------------
def read_env_handshake(main_connection, logger, default_pos):
    """Read the environment handshake and return (ee_pos_for_prompt, msg, coords_section, sim_state)."""
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
    try:
        _sim_state_str = json.dumps(sim_state)
        logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
    except Exception as e:
        logger.error(FAIL + f"Failed to log env state: {e}" + ENDC)
    return ee_pos, msg, coords_section, sim_state


def process_cli_viz_point_arg(args, conn, logger):
    """Parse --viz-point/--vis-point and send one or more points as permanent markers."""
    if not getattr(args, "viz_point", None):
        return
    try:
        raw = json.loads(args.viz_point)
        points = []
        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], (list, tuple)):
            for item in raw:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    points.append([float(item[0]), float(item[1]), float(item[2])])
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            points.append([float(raw[0]), float(raw[1]), float(raw[2])])

        if points:
            conn.send([config.ADD_TRAJECTORY_POINTS, points, "blue", True, "line"])
            logger.info(PROGRESS + f"Added visualization debug points in sim: {points} permanent viz points" + ENDC)
        else:
            logger.info(WARNING + "--viz-point provided but contained no valid coordinates" + ENDC)
    except Exception as e:
        logger.info(FAIL + f"Failed to add viz point(s): {e}" + ENDC)


# --- Agent context ------------------------------------------------------
class AgentContext:
    """Bundle of long-lived handles produced by init_agent and consumed by
    execute_task / run_plan / teardown_agent."""

    def __init__(self):
        self.args = None
        self.logger = None
        self.client = None
        self.api = None
        self.main_connection = None
        self.env_process = None
        self.server_proc = None
        self.llm_cache = None
        self.exec_locals = None
        self.coords_section = None
        self.sim_state = {}
        self.ee_pos_for_prompt = None


# --- One-time init ------------------------------------------------------
def init_agent(args, logger):
    """One-time setup. Returns a populated AgentContext."""
    import openai
    import torch
    from multiprocessing import Process, Pipe
    from api import API
    from env import run_simulation_environment
    import utils

    ctx = AgentContext()
    ctx.args = args
    ctx.logger = logger

    # OpenAI client (optional)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = None
    if openai.api_key is not None and len(openai.api_key) > 0:
        client = openai.OpenAI()
    ctx.client = client

    
    # Injects logger into utils global scope in modules
    models.logger = logger    
    utils.logger = logger 
    segmentation_adapter.logger = logger
    # object-filter regex
    segmentation_adapter.set_override_object_regex(args.ovr_obj)

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

    # Load segmentation/review models
    if args.seg_provider == "langsam":
        from lang_sam import LangSAM
        langsam_model = LangSAM()
    else:
        langsam_model = None
    xmem_model = None
    if args.review_provider == "xmem":
        try:
            sys.path.append("./XMem/")
            from XMem.model.network import XMem  # type: ignore
            xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
        except Exception as e:
            raise RuntimeError("Failed to initialize XMem model. Ensure the 'XMem' submodule and weights exist.") from e

    # Simulator connection
    server_proc = None
    env_process = None
    ee_pos_for_prompt = list(map(float, config.ee_start_position))
    coords_section = None
    sim_state = {}
    if args.sim == "pybullet":
        main_connection, env_connection = Pipe()
        env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, None])
        env_process.start()
        ee_pos_for_prompt, _msg, coords_section, sim_state = read_env_handshake(main_connection, logger, ee_pos_for_prompt)
        process_cli_viz_point_arg(args, main_connection, logger)
    else:
        main_connection, server_proc = _setup_metaworld_ws(args, logger)

    if coords_section is None:
        coords_section = config.three_d_coordinates_prompt_section

    # API + cache
    api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)
    llm_cache = None
    if args.llm_cache_enabled:
        from providers.llms.llm_cache import LLMCache
        llm_cache = LLMCache(cache_dir=config.llm_cache_dir, float_tolerance=config.llm_cache_float_tolerance, logger=logger)
    api.llm_cache = llm_cache
    api.sim_state = sim_state
    api.ee_pos_for_prompt = ee_pos_for_prompt
    api.coords_section = coords_section

    exec_locals = get_exec_locals(api, logger)

    ctx.api = api
    ctx.main_connection = main_connection
    ctx.env_process = env_process
    ctx.server_proc = server_proc
    ctx.llm_cache = llm_cache
    ctx.exec_locals = exec_locals
    ctx.coords_section = coords_section
    ctx.sim_state = sim_state
    ctx.ee_pos_for_prompt = ee_pos_for_prompt
    return ctx


# --- One-time teardown --------------------------------------------------
def teardown_agent(ctx):
    """Close the sim connection and terminate spawned processes. Best-effort."""
    logger = ctx.logger
    try:
        if ctx.main_connection is not None and hasattr(ctx.main_connection, "close"):
            ctx.main_connection.close()
    except Exception:
        pass
    _safe_terminate(ctx.server_proc, logger)
    try:
        if ctx.env_process is not None and ctx.env_process.is_alive():
            ctx.env_process.terminate()
            ctx.env_process.join(timeout=3)
    except Exception:
        pass
    if ctx.api is not None:
        from task_state import TaskState
        ctx.api.task = TaskState(start_trajectory_step=ctx.api.trajectory_step)


# --- Prompt builders ----------------------------------------------------
def _build_main_prompt(detect_tool, detect_initial, ee_pos, task, coords_section, in_context_example):
    """Fill all placeholders of the subtask MAIN_PROMPT."""
    return (
        MAIN_PROMPT
        .replace("[INSERT DETECT_OBJECT_TOOL]", detect_tool)
        .replace("[INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING]", detect_initial)
        .replace("[INSERT COLLISION AVOIDANCE]", COLLISION_AVOIDANCE)
        .replace("[INSERT CODE BLOCK CONVENTIONS]", CODE_BLOCK_CONVENTIONS)
        .replace("[INSERT INITIAL PLANNING 1]", INITIAL_PLANNING_1)
        .replace("[INSERT INITIAL PLANNING 2]", INITIAL_PLANNING_2)
        .replace("[INSERT EE POSITION]", str(ee_pos))
        .replace("[INSERT TASK]", task)
        .replace("[INSERT 3D COORDINATES PROMPT SECTION]", coords_section)
        .replace("[INSERT IN CONTEXT EXAMPLE]", in_context_example)
    )


class TaskResult:
    """Outcome of a single execute_task run."""

    def __init__(self, success, attempts, summaries, messages=None,
                 reviewer_reason="", improvement_steps="", accepted_without_review=False):
        self.success = success
        self.attempts = attempts
        self.summaries = summaries
        self.messages = messages or []
        self.reviewer_reason = reviewer_reason
        self.improvement_steps = improvement_steps
        # True when the loop ended by accepting the final attempt without a passing VLM review.
        self.accepted_without_review = accepted_without_review

    def as_summary_dict(self):
        """Compact, LLM-friendly view (excludes the full message transcript)."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "accepted_without_review": self.accepted_without_review,
            "reviewer_reason": self.reviewer_reason,
            "improvement_steps": self.improvement_steps,
            "summaries": self.summaries,
        }

    def __repr__(self):
        return f"TaskResult(success={self.success}, attempts={self.attempts}, accepted_without_review={self.accepted_without_review})"


# --- Per-(sub)task execution -------------------------------------------
def execute_task(ctx, prompt, max_attempts=None, in_context_example=True):
    """Run a single (sub)task to completion.

    Inputs:
      ctx : AgentContext - long-lived handles from init_agent.
      prompt : str - task prompt (e.g. "pick up the box", "open the door").
        May include review/verification instructions for the VLM reviewer.
      max_attempts : int - max attempts for THIS task (first attempt + retries with
        VLM review between each). Defaults to ctx.args.attempts when None. This is a
        small per-task cap and is distinct from the global args.attempts default.
      in_context_example : bool - include the in-context example on the first attempt.

    Returns TaskResult.
    """
    args = ctx.args
    logger = ctx.logger
    api = ctx.api
    client = ctx.client
    main_connection = ctx.main_connection
    llm_cache = ctx.llm_cache
    coords_section = ctx.coords_section
    sim_state = ctx.sim_state
    ee_pos_for_prompt = ctx.ee_pos_for_prompt

    if max_attempts is None:
        max_attempts = args.attempts

    # --- Fresh per-task state (function-call analogy) ---
    from task_state import TaskState
    task = TaskState(command=prompt, max_attempts=max_attempts, start_trajectory_step=api.trajectory_step)
    api.task = task
    messages = []
    task.conversation_messages = messages
    attempt_summaries = []
    error = False

    logger.info(PROGRESS + f"STARTING TASK... (max_attempts={max_attempts}) prompt={prompt!r}" + ENDC)

    # First-attempt prompt: detect_object tool + optional in-context example
    image_paths = [config.rgb_image_head_path]
    first_command = prepend_to_initial_command(prompt, args, logger)
    ic = IN_CONTEXT_EXAMPLE if in_context_example else ''
    new_prompt = _build_main_prompt(
        DETECT_OBJECT_TOOL, DETECT_OBJECT_TOOL_INITIAL_PLANNING,
        ee_pos_for_prompt, first_command, coords_section, ic,
    )

    try:
        _sim_state_str = json.dumps(sim_state)
        logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
    except Exception:
        pass

    logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
    messages = models.call_llm_cached(
        main_connection, client, args.language_model, new_prompt, messages, role="system",
        image_paths=image_paths if args.lm_images else None,
        options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": llm_cache},
    )
    task.conversation_messages = messages
    logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

    while not task.completed_task:
        new_prompt = ""
        if len(messages[-1]["content"].split("```python")) > 1:
            code_block = messages[-1]["content"].split("```python")
            block_number = 0
            # One shared namespace for all blocks in THIS assistant response, so a
            # variable/import defined in an earlier block is visible to later blocks.
            # It is rebuilt each turn (new assistant response) -> reset between responses.
            exec_env = globals().copy()
            exec_env.update(ctx.exec_locals)
            for block in code_block:
                if not error:
                    if len(block.split("```")) > 1:
                        code = block.split("```")[0]
                        block_number += 1
                        try:
                            f = StringIO()
                            with redirect_stdout(f):
                                exec(code, exec_env)
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
        else:
            if not task.completed_task:
                new_prompt += "No tool call detected. You must emit a ```python block calling actions or task_completed().\n"
                

        if not task.completed_task:
            if task.failed_task:
                logger.info(FAIL + "FAILED TASK! Generating summary of the task execution attempt..." + ENDC)

                new_prompt += TASK_SUMMARY_PROMPT
                new_prompt += "\n"

                logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                messages = models.call_llm_cached(main_connection, client, args.language_model, new_prompt, messages, "user", options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": llm_cache})
                task.conversation_messages = messages
                logger.info(OK + "Finished generating ChatGPT output!" + ENDC)

                attempt_summaries.append(messages[-1]["content"])

                logger.info(PROGRESS + f"RETRYING TASK (attempt {task.attempt_number + 1}/{max_attempts})..." + ENDC)
                task.start_attempt_trajectory_step = api.trajectory_step

                _, eef_pos = build_llm_context_images_and_pose(main_connection, api.trajectory_step, logger)
                # On retry: latest EE pos, no in-context example, and no detect_object tool
                # (arm usually occludes the target; reuse first-attempt object coords instead).
                new_prompt = _build_main_prompt(
                    NO_DETECT_OBJECT_TOOL, NO_DETECT_OBJECT_TOOL_INITIAL_PLANNING,
                    eef_pos, prompt, coords_section, '',
                )
                try:
                    _sim_state_str = json.dumps(sim_state)
                    logger.info(PROGRESS + f"Env state: {_sim_state_str}" + ENDC)
                except Exception:
                    pass

                combined_summary = "\n".join(
                    f"--- Attempt {i+1} Summary ---\n{s}"
                    for i, s in enumerate(attempt_summaries)
                )
                new_prompt += "\n"
                new_prompt += TASK_FAILURE_PROMPT.replace("[INSERT TASK SUMMARY]", combined_summary)
                messages = []
                error = False

                logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                messages = models.call_llm_cached(main_connection, client, args.language_model, new_prompt, messages, "system", options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": llm_cache})
                task.conversation_messages = messages
                task.failed_task = False  # reset to resume normal flow on the retry
            else:
                logger.info(PROGRESS + "Generating ChatGPT output..." + ENDC)
                _imgs_paths, eef_pos = build_llm_context_images_and_pose(main_connection, api.trajectory_step, logger)
                if config.ENABLE_EEF_POS_IMAGE and eef_pos:
                    new_prompt += f'\n{EEF_POS_SNIPPET}\n'.format(eef_pos=eef_pos)
                else:
                    _imgs_paths = None
                if not args.lm_images:
                    _imgs_paths = None
                # Never send an empty user turn: a successful code block may print
                # nothing (only logger output, which is not captured) and, with
                # ENABLE_EEF_POS_IMAGE/images off, leave both new_prompt and
                # _imgs_paths empty. Some providers (e.g. AWS Bedrock) reject a
                # conversation that ends on the assistant message. Add a concise
                # continuation nudge so the turn always has user content.
                if not (new_prompt and new_prompt.strip()) and not _imgs_paths:
                    new_prompt = CONTINUE_TASK_PROMPT
                messages = models.call_llm_cached(main_connection, client, args.language_model, new_prompt, messages, "user", image_paths=_imgs_paths, options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": llm_cache})
                task.conversation_messages = messages
                logger.info(OK + "Finished generating ChatGPT output!" + ENDC)
                error = False

    logger.info(OK + "FINISHED TASK!" + ENDC)
    return TaskResult(
        success=bool(task.review_succeeded),
        attempts=task.attempt_number,
        summaries=attempt_summaries,
        messages=messages,
        reviewer_reason=task.review_reason,
        improvement_steps=task.review_improvement_steps,
        accepted_without_review=bool(task.accepted_without_review),
    )


# --- Planner + orchestration (single agentic loop) ----------------------
def run_scene_perception(ctx, command):
    """Run the perception VLM (--perception-vlm) on the current head image.

    Returns its free-text scene analysis, injected into the planner prompt's
    SCENE ANALYSIS section. Best-effort: on any failure returns a short fallback
    string so the planner still runs (it can fall back to detect_object)."""
    from prompts.scene_perception_prompt import SCENE_PERCEPTION_PROMPT
    args = ctx.args
    logger = ctx.logger
    prompt = SCENE_PERCEPTION_PROMPT.replace("[INSERT USER COMMAND TASK]", str(command))
    image_paths = [config.rgb_image_head_path] if args.lm_images else None
    try:
        logger.info(PROGRESS + f"Perception: analyzing scene with {args.perception_vlm}..." + ENDC)
        messages = models.call_llm_cached(
            ctx.main_connection, ctx.client, args.perception_vlm, prompt, [], role="system",
            image_paths=image_paths,
            options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": ctx.llm_cache},
        )
        text = messages[-1]["content"] if messages and isinstance(messages[-1], dict) else ""
        text = (text or "").strip()
        logger.info(OK + "Perception: scene analysis ready." + ENDC)
        return text or "(perception returned no analysis)"
    except Exception as e:
        logger.info(WARNING + f"Perception VLM failed: {e}. Proceeding without scene analysis." + ENDC)
        return "(scene analysis unavailable; inspect the scene yourself with detect_object(...))"


def _build_planner_prompt(ctx, command, scene_analysis=""):
    """Fill the merged PLANNER_PROMPT placeholders."""
    from prompts.planner_prompt import PLANNER_PROMPT, RECOVERY_FROM_FAILURE
    return (
        PLANNER_PROMPT
        .replace("[INSERT CODE BLOCK CONVENTIONS]", CODE_BLOCK_CONVENTIONS)
        .replace("[INSERT INITIAL PLANNING 1]", INITIAL_PLANNING_1)
        .replace("[INSERT INITIAL PLANNING 2]", INITIAL_PLANNING_2)
        .replace("[INSERT COLLISION AVOIDANCE]", COLLISION_AVOIDANCE)
        .replace("[INSERT RECOVERY_FROM_FAILURE]", RECOVERY_FROM_FAILURE)
        .replace("[INSERT SCENE ANALYSIS]", scene_analysis)
        .replace("[INSERT 3D COORDINATES PROMPT SECTION]", ctx.coords_section)
        .replace("[INSERT EE POSITION]", str(ctx.ee_pos_for_prompt))
        .replace("[INSERT TASK]", command)
    )


def run_plan(ctx, command, max_iterations=16):
    """Drive one continuous agentic planner conversation for the user command.

    With args.no_plan, runs the raw command as a single execute_task.
    Otherwise the planner LLM observes the scene + command, then dispatches subtasks
    ONE AT A TIME via the execute_subtasks tool. After every subtask, perception is
    re-run on the (possibly changed) world state and the planner is re-invoked, so it
    can reevaluate before the next subtask - e.g. insert a "move the arm out of the
    way" subtask once an occluder is cleared, or replan on failure. The planner always
    terminates by calling plan_completed() or plan_failed().
    """
    args = ctx.args
    logger = ctx.logger

    if getattr(args, "no_plan", False):
        logger.info(PROGRESS + "Planner disabled (--no-plan): running command as a single task." + ENDC)
        return execute_task(ctx, command, max_attempts=args.attempts)

    from planner_api import PlannerAPI, get_planner_exec_locals

    planner = PlannerAPI(ctx, execute_task, logger)
    planner_locals = get_planner_exec_locals(planner, logger)

    prompt = _build_planner_prompt(ctx, command, scene_analysis=run_scene_perception(ctx, command))
    image_paths = [config.rgb_image_head_path] if args.lm_images else None
    logger.info(PROGRESS + "Planner: generating plan / dispatch..." + ENDC)
    messages = models.call_llm_cached(
        ctx.main_connection, ctx.client, args.language_model, prompt, [], role="system",
        image_paths=image_paths,
        options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": ctx.llm_cache},
    )

    iteration = 0
    while not (planner.plan_completed_flag or planner.plan_failed_flag) and iteration < max_iterations:
        iteration += 1
        new_prompt = _run_planner_code_blocks(messages, planner_locals)

        if planner.plan_completed_flag or planner.plan_failed_flag:
            break

        # Re-run perception after every subtask: the executed subtask may have
        # changed the scene (cleared occluders, opened a door, or left the arm
        # occluding the next target). The planner reevaluates this fresh analysis
        # and may insert a new subtask (e.g. move the arm away) before continuing.
        scene_analysis = run_scene_perception(ctx, command)
        new_prompt = (
            "UPDATED SCENE ANALYSIS (perception VLM, current head-camera image):\n"
            f"{scene_analysis}\n\n" + new_prompt
        )

        logger.info(PROGRESS + f"Planner: iteration {iteration}/{max_iterations}..." + ENDC)
        messages = models.call_llm_cached(
            ctx.main_connection, ctx.client, args.language_model, new_prompt, messages, "user",
            options={"max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort, "cache": ctx.llm_cache},
        )

    if planner.plan_completed_flag:
        logger.info(OK + "Planner: overall command completed." + ENDC)
    elif planner.plan_failed_flag:
        logger.info(FAIL + "Planner: overall command marked unreachable." + ENDC)
    else:
        logger.info(WARNING + f"Planner: stopped after {max_iterations} iterations without a terminal decision." + ENDC)
    return planner


def _run_planner_code_blocks(messages, planner_locals):
    """Execute the ```python blocks in the planner's latest message; return the
    follow-up user prompt built from captured stdout / errors."""
    new_prompt = ""
    content = messages[-1]["content"] if messages and isinstance(messages[-1], dict) else ""
    code_block = content.split("```python")
    if len(code_block) > 1:
        block_number = 0
        # Shared namespace for all blocks in this planner response (reset per response).
        exec_env = globals().copy()
        exec_env.update(planner_locals)
        for block in code_block:
            if len(block.split("```")) > 1:
                code = block.split("```")[0]
                block_number += 1
                try:
                    f = StringIO()
                    with redirect_stdout(f):
                        exec(code, exec_env)
                except Exception:
                    error_message = traceback.format_exc()
                    new_prompt += ERROR_CORRECTION_PROMPT.replace("[INSERT BLOCK NUMBER]", str(block_number)).replace("[INSERT ERROR MESSAGE]", error_message)
                    new_prompt += "\n"
                else:
                    s = f.getvalue()
                    if s:
                        new_prompt += PRINT_OUTPUT_PROMPT.replace("[INSERT PRINT STATEMENT OUTPUT]", s)
                        new_prompt += "\n"
    else:
        new_prompt += ("No planner tool call detected. Emit a ```python block calling "
                       "execute_subtasks([...]), or plan_completed()/plan_failed().")
    return new_prompt

