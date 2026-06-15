import traceback
import math 
import os
import re
import tempfile
import numpy as np
from config import OK, PROGRESS, WARNING, FAIL, ENDC

def get_exec_locals(api, logger):
    """Returns the local variables dictionary for the exec environment."""
    # Assuming math and numpy (as np) are imported globally in main.py
    return {
        "detect_object": api.detect_object,
        "get_grasp_poses": api.get_grasp_poses,
        "visualize_grasp_pose": api.visualize_grasp_pose,
        "execute_trajectory": api.execute_trajectory,
        "open_gripper": api.open_gripper,
        "close_gripper": api.close_gripper,
        "task_completed": api.task_completed,
        "generate_linear_trajectory": api.generate_linear_trajectory,
        "api": api,
        "math": math,
        "np": np,
        "logger" : logger,
    }

def execute_blocks_from_log(log_path, api, logger):
    """
    Reads a log file, extracts all python blocks, cleans out log prefixes, 
    and executes them sequentially.
    """
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(FAIL + f"Failed to read replay log file '{log_path}': {e}" + ENDC)
        return

    # Extract all content between ```python and ```
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    raw_blocks = pattern.findall(content)

    if not raw_blocks:
        logger.info(WARNING + "No python blocks found in the provided log file." + ENDC)
        return

    logger.info(PROGRESS + f"Found {len(raw_blocks)} python block(s) to replay." + ENDC)

    # Set up the execution environment locals exactly like the main loop
    exec_locals = get_exec_locals(api, logger)
    # Combine globals and our custom locals into a single environment
    exec_env = globals().copy()
    exec_env.update(exec_locals)
    # Optional: Log prefix pattern to clean up lines like "16/06 01:11 | INFO      | "
    log_prefix_pattern = re.compile(r'^\d{2}/\d{2} \d{2}:\d{2} \| \w+\s+\| ')

    for i, block in enumerate(raw_blocks, start=1):
        logger.info(PROGRESS + f"--- Executing Block {i}/{len(raw_blocks)} ---" + ENDC)
        
        # Clean the block: remove log prefixes from the start of any line
        cleaned_lines = []
        for line in block.split('\n'):
            clean_line = log_prefix_pattern.sub('', line)
            cleaned_lines.append(clean_line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        if not code:
            continue
            
        # 1. Write the code to a physical temporary file so pdb can read it
        fd, temp_path = tempfile.mkstemp(prefix=f"replay_block_{i}_", suffix=".py", text=True)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(code)
            
        try:
            # Combine globals and locals
            exec_env = globals().copy()
            exec_env.update(exec_locals)
            
            # 2. Compile the code and associate it with the real file path!
            compiled_code = compile(code, temp_path, 'exec')
            
            # 3. Execute the compiled code
            exec(compiled_code, exec_env)
            
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(FAIL + f"Error executing block {i}:\n{error_message}" + ENDC)
            logger.info(WARNING + "Halting replay due to execution error." + ENDC)
            break
        finally:
            # 4. Clean up the temporary file (optional: comment this out if you want to inspect them)
            try:
                os.remove(temp_path)
            except Exception:
                pass
            
    logger.info(OK + "Finished replaying log blocks!" + ENDC)