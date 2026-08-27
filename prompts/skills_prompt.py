# INPUT: [INSERT SKILLS INDEX]
#
# Level-1 skill catalog injected into both the PLANNER_PROMPT and the subtask MAIN_PROMPT
# (each with its own scope-filtered index - see skill_registry.py). Only names and
# descriptions are shown; the full instructions of a skill are pulled in on demand with
# load_skill(...), which appends them to the conversation as if they had been part of this
# prompt from the start.
#
# TURN COST: loading a skill costs a round-trip, because the body only arrives in the NEXT
# message. The rules below therefore differ per scope:
#   subtask - load_skill(...) is batched into the SAME code block as the opening
#             detect_object(...) calls. Both are read-only information gathering, the agent
#             was going to end its turn on detect_object anyway, and execute_python_blocks
#             returns the printed detect_object output AND the skill body together => no
#             extra turn is spent.
#   planner - NO batching: its only other tool, execute_subtasks(...), moves the robot, so
#             batching would run a subtask before the skill was ever read.

# Rules 2+3 of SKILLS_SECTION, substituted per scope.
TURN_RULE_SUBTASK = """2. To load it, call load_skill("<exact-name-from-the-list>") IN THE SAME CODE BLOCK as your opening detect_object(...) calls, then END YOUR TURN. Do NOT spend a separate turn on load_skill: the next message returns BOTH the printed detect_object output AND the skill's full instructions, which then stay valid for the rest of this task.
3. That turn must contain ONLY information gathering (load_skill / detect_object). Do NOT emit any trajectory, motion, gripper or task_completed code in it - the skill instructions arrive only afterwards, so a plan written in that same turn would be written without them."""

TURN_RULE_PLANNER = """2. To load it, emit a code block with just: load_skill("<exact-name-from-the-list>") and END YOUR TURN. Its full instructions come back in the next message, and stay valid for the rest of this task.
3. Do NOT call execute_subtasks(...) in that same turn - it dispatches a real robot subtask, which would run before you have read the skill. Load first, then plan on the next turn."""

SKILLS_SECTION = """AVAILABLE SKILLS (extra instructions you can pull into this conversation on demand):
Each skill below is a package of detailed, task-specific know-how that is NOT included in this prompt. You only see its name and description; the instructions themselves are loaded only if you ask for them.
[INSERT SKILLS INDEX]
How to use skills:
1. Before writing any code for the task, check the descriptions above. If one matches what you are about to do, load it FIRST - it contains procedures and pitfalls this prompt does not.
[INSERT SKILL LOADING TURN RULE]
4. Use exactly one of the names listed above; do not invent names, and do not guess at a skill's contents without loading it.
5. A skill may point to extra bundled files; read those with read_file("<path shown in the skill>") only when its instructions tell you to.
6. If no skill matches, continue normally - skills are optional.
"""

# First AVAILABLE FUNCTIONS line, substituted per scope (see TURN COST note above).
LOAD_SKILL_TOOL_SUBTASK = """load_skill(skill_name: str) -> None: Loads the full instructions of one of the AVAILABLE SKILLS listed below into this conversation. Call it in the SAME code block as your opening detect_object(...) calls and end the turn; the instructions and the detect_object output arrive together in the next message, so it costs no extra turn. Does not return anything."""

LOAD_SKILL_TOOL_PLANNER = """load_skill(skill_name: str) -> None: Loads the full instructions of one of the AVAILABLE SKILLS listed below into this conversation. Call it in its own code block and end the turn; the instructions arrive in the next message. Does not return anything."""

READ_FILE_TOOL = """read_file(path: str) -> None: Reads a file bundled with a skill (for example "<skill-name>/references/<file>.md"), as listed inside a loaded skill. Only files inside the skills directory can be read. Does not return anything - the contents are added to the conversation."""

# Appended to [INSERT DETECT_OBJECT_TOOL_INITIAL_PLANNING] so the batching rule is repeated at
# the exact point the subtask agent is told to detect objects and stop generating.
DETECT_BATCHING_LINE = """ In that same code block, ALSO call load_skill("<name>") for any skill in AVAILABLE SKILLS matching this task - its instructions come back together with the detect_object output, so batching them costs no extra turn. Write no motion code in that turn."""


def build_skills_section(index_text, scope="subtask"):
    """Render the skills prompt section, or "" when this scope has no skills."""
    if not index_text:
        return ""
    turn_rule = TURN_RULE_PLANNER if scope == "planner" else TURN_RULE_SUBTASK
    return (
        SKILLS_SECTION
        .replace("[INSERT SKILLS INDEX]", index_text)
        .replace("[INSERT SKILL LOADING TURN RULE]", turn_rule)
    )


def build_skill_tools_section(index_text, start_number=None, scope="subtask"):
    """Render the AVAILABLE FUNCTIONS entries for the skill tools, or "" when disabled.

    `start_number` numbers the entries to continue the surrounding list (e.g. 7 -> "7." and "8.").
    """
    if not index_text:
        return ""
    load_line = LOAD_SKILL_TOOL_PLANNER if scope == "planner" else LOAD_SKILL_TOOL_SUBTASK
    lines = [load_line, READ_FILE_TOOL]
    if start_number is None:
        return "\n".join(lines)
    return "\n".join(f"{start_number + i}. {line}" for i, line in enumerate(lines))


def build_detect_batching_line(index_text):
    """Extra INITIAL-PLANNING sentence telling the subtask agent to batch load_skill into its
    detect_object block. Empty when skills are disabled or detect_object is unavailable."""
    if not index_text:
        return ""
    return DETECT_BATCHING_LINE

