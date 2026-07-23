import os
import sys
import argparse
import functools

import config
import segmentation_adapter
from dotenv import load_dotenv
from debug.dbg_utils import init_loguru_logger
from config import OK, PROGRESS, WARNING, FAIL, ENDC
from agent_runner import init_agent, teardown_agent, run_plan, execute_blocks_from_log, query_sim_objects_state

print = functools.partial(print, flush=True)

load_dotenv()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", default="azure-gpt-5", help="select language model (e.g. azure-gpt-5, gpt-4o, or-google/gemini-2.5-flash)")
    parser.add_argument("--lm-images", action=argparse.BooleanOptionalAction, default=True, help="pass images to LLM prompts (default: True, use --no-lm-images to disable)")
    parser.add_argument("--max-tokens", type=int, default=60000, help="max completion tokens for LLM responses")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["xhigh", "high", "medium", "low", "minimal", "none"], help="reasoning effort for reasoning models (OpenRouter, Gemini)")
    parser.add_argument("--llm-cache", dest="llm_cache_enabled", action=argparse.BooleanOptionalAction, default=True, help="cache LLM responses on disk (default: True, use --no-llm-cache to disable)")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    parser.add_argument("-s", "--sim", choices=["pybullet", "metaworld"], default="pybullet", help="select simulator backend")
    parser.add_argument("--transport", choices=["auto", "pipe", "ws"], default="auto", help="connection transport override; auto: pipe for pybullet, ws for metaworld")
    parser.add_argument("--task", type=str, default="sawyer_door_v3", help="task/environment name (metaworld only)")
    parser.add_argument("--seg-provider", choices=["langsam", "sam3", "moondream"], default="moondream", help="select segmentation provider (LangSAM, RoboFlow SAM3, or Moondream)")
    parser.add_argument("--depth-format", choices=["norm_1m", "norm_zfar", "raw"], default="norm_1m", help="depth handling for reconstruction")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout seconds; <=0 disables timeouts")
    parser.add_argument("--delete-images", action="store_true", help="delete image folders before recreating them")
    parser.add_argument("--review-provider", default="vlm", help="review provider for success verification: 'vlm' (uses main model), 'vlm:<model>' (e.g. vlm:or-openai/gpt-5.5), or 'xmem'")
    parser.add_argument("--perception-vlm", default="gemini-3.5-flash", help="VLM used for scene perception/vision analysis run before every planner LLM call; its text answer is injected into the planner prompt.")
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
    parser.add_argument("--attempts", type=int, default=2, help="total number of task attempts (default: 2 = first attempt + 1 retry after VLM review). Values > 2 allow additional retries with VLM review between each.")
    parser.add_argument("--max-planner-iter", type=int, default=4, help="max planner LLM iterations (subtask dispatch/replan turns) per user command before the loop stops without a terminal decision (default: 4).")
    parser.add_argument("--no-plan", dest="no_plan", action="store_true", default=False, help="bypass the LLM planner and run the raw command as a single task.")
    parser.add_argument("--vis-traj", action="store_true", help="visualize trajectory points in the sim environment (3d sphere markers)")
    parser.add_argument("--vis-grasp", action="store_true", help="visualize grasp pose candidates in the 3D sim environment (cylinder/sphere markers)")
    parser.add_argument("--vis-box", type=str, default=None, help="Visualize 3D bounding box in sim for objects whose label matches this regex (e.g. 'handle|knob'). Uses cylinder markers visible in camera captures.")
    parser.add_argument("--save-grasp-inputs", action="store_true", help="save binary segmentation mask (masks[0]) as .npy and projection/view matrices as .npy under images_folder after each detect_object call")
    parser.add_argument("--replay-log", type=str, default=None, help="Path to a log file of conversation with ```python blocks to execute. LLM-less execution")
    parser.add_argument("--replay-vlm-review", action="store_true", default=False, help="When replaying a log, also execute VLM-review code blocks between attempts (default: False, skip VLM review)")
    parser.add_argument("--learn-from-trajs", type=str, default=None, help="Path to a text file of past trajectories to learn from. Generates an improved in-context example via LLM and exits.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    try:
        os.makedirs(config.images_folder, exist_ok=True)
    except Exception as e:
        # Fail fast to avoid silent non-file logging
        raise RuntimeError(f"Failed to ensure images folder exists at '{config.images_folder}': {e}") from e

    # Logging (Loguru): emit to console and to a file under images folder
    logger = init_loguru_logger("vlm_traj.log")
    logger.info(PROGRESS + f"Args: {vars(args)}" + ENDC)
    segmentation_adapter.logger = logger

    # Enable bash-like arrow-up command history for the interactive prompt
    from helpers.command_utils import init_command_history, record_command, read_command
    readline = init_command_history(logger)

    # One-time setup: models, simulator connection, API, exec env
    ctx = init_agent(args, logger)

    # --learn-from-trajs: LLM-based in-context example learning; exits before agent flow
    if args.learn_from_trajs:
        from helpers.main_utils import learn_from_past_trajs
        try:
            learn_from_past_trajs(ctx.client, args, ctx.coords_section, logger)
        finally:
            teardown_agent(ctx)
        return

    try:
        # NO-LLM Debug: parse a conversation log, extract python blocks and execute them
        if args.replay_log:
            logger.info(PROGRESS + f"Replay mode activated. Parsing: {args.replay_log}" + ENDC)
            execute_blocks_from_log(args.replay_log, ctx.api, logger)
            logger.info(PROGRESS + "Replay mode finished executing code blocks" + ENDC)
            return

        # Query ground-truth state (all objects/geoms) for cross-check
        if args.sim == "metaworld" and hasattr(ctx.main_connection, "send"):
            query_sim_objects_state(ctx.main_connection, logger)

        # Interactive command loop: plan + execute each user command
        while True:
            command = read_command("Enter a command: ", readline).strip()
            if not command:
                logger.info(PROGRESS + "No command entered. Exiting." + ENDC)
                break
            record_command(readline, command)
            run_plan(ctx, command)
    except KeyboardInterrupt:
        logger.info(PROGRESS + "Interrupted by user (Ctrl+C). Shutting down..." + ENDC)
    finally:
        teardown_agent(ctx)


if __name__ == "__main__":
    main()
