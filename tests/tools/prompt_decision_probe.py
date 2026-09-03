"""Does the LLM pick the *right* gripper approach? (Tier 4 probe)

The engineering risk of the side approach is small and covered by
``tests/test_side_approach.py``. The real risk is behavioural: having been given a second
way to grasp things, the model may reach for it when plain top-down was fine, or fail to
reach for it on the one handle shape that needs it.

This probe measures exactly that, and nothing else. It is deliberately **not** a pytest:
it costs tokens. There is no simulator, no image, no trajectory execution - each scenario
is one short completion asking only for the first code block the model would write.

Scoring is a confusion matrix over ``expect_side``:

* a **false positive** (side approach on a can, a lever, a drawer) is the regression this
  feature could introduce;
* a **false negative** (top-down on a vertical bar) means the trigger rules in
  ``prompts/skills/open-close-door-cabinet-drawer/SKILL.md`` are not landing.

Usage::

    python tests/tools/prompt_decision_probe.py --dry-run           # prints a prompt, costs nothing
    python tests/tools/prompt_decision_probe.py                     # default model
    python tests/tools/prompt_decision_probe.py -lm or-openai/gpt-5-mini --repeats 3

Responses go through the ordinary on-disk LLM cache, so re-running a scenario set that has
not changed is free. Exit status is non-zero if any scenario was decided wrongly.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402
import skill_registry  # noqa: E402


SKILL_NAME = "open-close-door-cabinet-drawer"

INSTRUCTIONS = """\
You are controlling a Franka arm. You may call these helpers:

    detect_object(name)
    generate_linear_trajectory(desc, start_pose, end_pose, num_points=20)
    execute_trajectory(trajectory)
    open_gripper() / close_gripper()
    side_grasp_pose(x, y, z, rotation, approach_yaw) -> horizontal end-effector pose

A top-down pose is written directly as [x, y, z, rotation], as usual.

Reply with exactly ONE ```python block: the first block you would run for this task.
Do not explain anything outside the block.
"""

#: Each scenario is a fabricated `detect_object` printout - the real thing prints exactly
#: these fields - plus the ground truth of which approach it should provoke.
SCENARIOS = [
    dict(
        id="vertical_bar_cabinet",
        expect_side=True,
        command="open the cabinet door",
        scene="""Position of cabinet door handle: [0.35, 0.62, 0.95]
Dimensions:
Width: 0.03
Length: 0.04
Height: 0.24
Position of cabinet door: [0.10, 0.70, 0.95]
Dimensions:
Width: 0.6
Length: 0.03
Height: 0.7""",
    ),
    dict(
        id="vertical_bar_fridge",
        expect_side=True,
        command="open the fridge",
        scene="""Position of fridge handle: [0.30, 0.55, 1.10]
Dimensions:
Width: 0.04
Length: 0.04
Height: 0.35
Position of fridge door: [0.05, 0.63, 1.00]
Dimensions:
Width: 0.7
Length: 0.04
Height: 1.4""",
    ),
    dict(
        id="horizontal_d_handle",
        expect_side=False,
        command="open the drawer",
        scene="""Position of drawer handle: [0.20, 0.50, 0.75]
Dimensions:
Width: 0.16
Length: 0.03
Height: 0.03
Position of drawer front: [0.20, 0.52, 0.72]
Dimensions:
Width: 0.45
Length: 0.02
Height: 0.2""",
    ),
    dict(
        id="door_lever",
        expect_side=False,
        command="open the door",
        scene="""Position of door lever: [-0.32, -0.13, 0.64]
Dimensions:
Width: 0.12
Length: 0.03
Height: 0.02
Position of door: [-0.20, -0.05, 0.80]
Dimensions:
Width: 0.8
Length: 0.04
Height: 1.9""",
    ),
    dict(
        id="round_knob",
        expect_side=False,
        command="open the cupboard",
        scene="""Position of cupboard knob: [0.28, 0.48, 1.05]
Dimensions:
Width: 0.04
Length: 0.04
Height: 0.04
Position of cupboard door: [0.10, 0.50, 1.05]
Dimensions:
Width: 0.4
Length: 0.02
Height: 0.5""",
    ),
    dict(
        id="tabletop_can",
        expect_side=False,
        command="pick up the blue can",
        scene="""Position of blue can: [0.05, 0.55, 0.10]
Dimensions:
Width: 0.07
Length: 0.07
Height: 0.12""",
    ),
    dict(
        id="tall_thin_bottle",
        expect_side=False,
        command="pick up the bottle",
        scene="""Position of bottle: [-0.10, 0.60, 0.14]
Dimensions:
Width: 0.06
Length: 0.06
Height: 0.26""",
    ),
]


def build_prompt(scenario, skill_body):
    return (
        f"{INSTRUCTIONS}\n"
        f"--- SKILL: {SKILL_NAME} ---\n{skill_body}\n--- END SKILL ---\n\n"
        f"SCENE ANALYSIS (already printed by detect_object):\n{scenario['scene']}\n\n"
        f"USER COMMAND: {scenario['command']}\n"
    )


def load_skill_body():
    metas = skill_registry.discover(skill_registry.DEFAULT_SKILLS_DIR, "subtask")
    for meta in metas:
        if meta.name == SKILL_NAME:
            return skill_registry.read_body(meta)
    raise SystemExit(f"skill {SKILL_NAME!r} not found")


def used_side_approach(answer):
    block = re.search(r"```python(.*?)```", answer, re.S)
    code = block.group(1) if block else answer
    return "side_grasp_pose" in code


def printed_trigger(answer):
    return bool(re.search(r"print\s*\(.*trigger", answer, re.I | re.S))


def ask(models, client, model, prompt, cache, args):
    messages = models._call_llm_provider_wrapper(
        client, model, prompt, [], role="user",
        options={"max_tokens": args.max_tokens,
                 "reasoning_effort": args.reasoning_effort,
                 "cache": cache,
                 "log_msgs": False},
    )
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message["content"] or ""
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-lm", "--language_model", default="azure-gpt-5")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--repeats", type=int, default=1,
                        help="samples per scenario; >1 measures how stable the choice is")
    parser.add_argument("--only", default=None, help="run just this scenario id")
    parser.add_argument("--no-llm-cache", action="store_true",
                        help="bypass the on-disk cache (every scenario costs tokens)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print one built prompt and exit - no LLM call, no tokens")
    args = parser.parse_args()

    skill_body = load_skill_body()
    scenarios = [s for s in SCENARIOS if args.only in (None, s["id"])]
    if not scenarios:
        raise SystemExit(f"no scenario matches --only {args.only!r}")

    if args.dry_run:
        print(build_prompt(scenarios[0], skill_body))
        return 0

    import openai
    import models
    from dotenv import load_dotenv
    from debug.dbg_utils import init_loguru_logger

    load_dotenv()
    models.logger = init_loguru_logger("prompt_decision_probe.log")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI() if openai.api_key else None

    cache = None
    if not args.no_llm_cache:
        from providers.llms.llm_cache import LLMCache
        cache = LLMCache(cache_dir=config.llm_cache_dir,
                         float_tolerance=config.llm_cache_float_tolerance,
                         logger=models.logger)

    rows, false_pos, false_neg = [], 0, 0
    for scenario in scenarios:
        prompt = build_prompt(scenario, skill_body)
        for attempt in range(args.repeats):
            answer = ask(models, client, args.language_model, prompt, cache, args)
            side = used_side_approach(answer)
            ok = side == scenario["expect_side"]
            if not ok and side:
                false_pos += 1
            if not ok and not side:
                false_neg += 1
            rows.append((scenario["id"], attempt, scenario["expect_side"], side, ok,
                         printed_trigger(answer) if side else None))

    print(f"\n{'scenario':<24}{'#':>3}  {'expect':>7}{'got':>7}{'':4}{'trigger printed'}")
    for name, attempt, expect, got, ok, trigger in rows:
        mark = "ok " if ok else "FAIL"
        exp = "side" if expect else "top"
        act = "side" if got else "top"
        trig = "" if trigger is None else ("yes" if trigger else "NO")
        print(f"{name:<24}{attempt:>3}  {exp:>7}{act:>7}  {mark}  {trig}")

    total = len(rows)
    print(f"\ncorrect {total - false_pos - false_neg}/{total}   "
          f"false-positive (needless side approach) {false_pos}   "
          f"false-negative (missed vertical bar) {false_neg}")
    return 1 if (false_pos or false_neg) else 0


if __name__ == "__main__":
    raise SystemExit(main())
