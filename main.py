import numpy as np
import math
import openai
import torch
import os
import sys
import argparse
import traceback
import multiprocessing
import logging
import functools
import models
import config
from numpy import pi
from lang_sam import LangSAM
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

sys.path.append("./XMem/")
print = functools.partial(print, flush=True)

from XMem.model.network import XMem

if __name__ == "__main__":

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    # Parse args
    parser = argparse.ArgumentParser(description="Main Program.")
    parser.add_argument("-lm", "--language_model", choices=["azure-gpt-5", "azure-gpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"], default="azure-gpt-5", help="select language model")
    parser.add_argument("-r", "--robot", choices=["sawyer", "franka"], default="sawyer", help="select robot")
    parser.add_argument("-m", "--mode", choices=["default", "debug"], default="default", help="select mode to run")
    parser.add_argument("-s", "--sim", choices=["pybullet", "metaworld"], default="pybullet", help="select simulator backend")
    parser.add_argument("--task", type=str, default="sawyer_door_v3", help="task/environment name (metaworld only)")
    parser.add_argument("--depth-format", choices=["norm_1m", "norm_zfar", "raw"], default="norm_1m", help="depth handling for reconstruction")
    args = parser.parse_args()

    # Logging
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    # Device
    if torch.cuda.is_available():
        logger.info("Using GPU.")
        device = torch.device("cuda")
    else:
        logger.info("CUDA not available. Please connect to a GPU instance if possible.")
        device = torch.device("cpu")

    torch.set_grad_enabled(False)

    # Load models
    langsam_model = LangSAM()
    xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)

    # Create connection to the chosen simulator, then build API with it
    if args.sim == "pybullet":
        main_connection, env_connection = Pipe()
        # Start process
        env_process = Process(target=run_simulation_environment, name="EnvProcess", args=[args, env_connection, logger])
        env_process.start()
        [env_connection_message] = main_connection.recv()
        logger.info(env_connection_message)
    else:
        from providers.subproc_connection import SubprocessJSONConnection
        server_path = os.path.join(os.path.dirname(__file__), "providers", "metaworld_server.py")
        py_exe = os.environ.get("METAWORLD_PYTHON", sys.executable)
        # Pass env/task and depth format via CLI and env vars
        cmd = [py_exe, server_path, "--env", args.task]
        # Prepare environment for the server
        env_vars = os.environ.copy()
        env_vars["DEPTH_FORMAT"] = args.depth_format
        # Allow user to set METAWORLD_REPO externally
        env_connection = SubprocessJSONConnection(cmd)
        main_connection = env_connection
        ready_msg = main_connection.recv()
        if isinstance(ready_msg, list) and ready_msg:
            logger.info(ready_msg[0])
        else:
            logger.info("\033[92mFinished setting up environment!\033[0m")

    # API set-up
    api = API(args, main_connection, logger, client, langsam_model, xmem_model, device)

    detect_object = api.detect_object
    execute_trajectory = api.execute_trajectory
    open_gripper = api.open_gripper
    close_gripper = api.close_gripper
    task_completed = api.task_completed

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

        api.completed_task = False
