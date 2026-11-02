# INPUT: [INSERT PRINT STATEMENT OUTPUT]
PRINT_OUTPUT_PROMPT = \
"""Print statement output:
[INSERT PRINT STATEMENT OUTPUT]
"""

# Sent when the assistant response contained no executable code block.
NO_TOOL_CALL_PROMPT = \
"""No tool call detected. You must emit a ```python block calling actions or task_completed().
If the attempt cannot continue (part dropped, blocked, or nothing left to try), call task_completed() now: \
it ends the attempt and an independent reviewer judges the result - it is NOT a claim of success, and a \
failed review starts a fresh attempt.
"""

# INPUT: [INSERT EXECUTED BLOCKS], [INSERT TOTAL BLOCKS], [INSERT FIRST SKIPPED BLOCK]
# Sent when execution aborted mid-response, so the LLM never assumes skipped blocks ran.
BLOCKS_NOT_EXECUTED_PROMPT = \
"""IMPORTANT: only the first [INSERT EXECUTED BLOCKS] of [INSERT TOTAL BLOCKS] code blocks in your last response were executed. \
Blocks [INSERT FIRST SKIPPED BLOCK]-[INSERT TOTAL BLOCKS] were NOT executed - none of their actions happened in the environment. \
Re-emit the remaining actions (corrected if needed) before claiming any progress, and do NOT call task_completed() \
until every required trajectory has actually been executed.
"""
