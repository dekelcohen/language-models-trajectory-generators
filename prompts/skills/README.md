# Skills

Lazily-loaded packages of task-specific know-how, in the `SKILL.md` format
([agentskills.io](https://agentskills.io) — the convention used by Claude Code, Copilot,
Gemini CLI and others), adapted to this repo's exec-based tool layer.

A skill costs ~50-100 tokens per turn (its name + description) until the LLM decides it is
relevant and calls `load_skill(...)`; only then does the full body enter the conversation.
This lets us add deep procedures (door levers, occlusion recovery, ...) without growing the
always-on prompt.

## Layout

```
prompts/skills/
├── open-door-cabinet-drawer/          # directory name == frontmatter `name`
│   ├── SKILL.md                       # required: frontmatter + markdown body
│   ├── references/                    # optional: level-3 files, read on demand
│   ├── scripts/                       # optional
│   └── assets/                        # optional
└── <category>/<skill-name>/SKILL.md   # one optional category level is also scanned
```

## Frontmatter

```yaml
---
name: open-door-cabinet-drawer   # required, == directory name, [a-z0-9-], <= 64 chars
scope: subtask                   # REQUIRED: planner | subtask | both
description: >                   # required, <= 1024 chars: what it does AND when to use it
  Open a hinged door, a cabinet or a drawer... Use when the command is open / unlatch...
  Not for picking up free objects.
license: MIT                     # optional
metadata:                        # optional free-form map
  category: manipulation
  tags: [door, handle]
---
```

`scope` decides which agent ever sees the skill:

| scope | Listed to the planner LLM (`PLANNER_PROMPT`) | Listed to the low-level subtask LLM (`MAIN_PROMPT`) |
|---|---|---|
| `planner` | yes | no |
| `subtask`  | no | yes |
| `both`     | yes | yes |

Discovery greps the frontmatter head for `scope` **before** parsing anything else, so a
subtask skill's description never reaches the planner's context (and vice versa). The grep is
lenient: indentation, quotes, upper case, CRLF and `scope` appearing after other keys are all
fine — but the key itself is mandatory, and a file without it is skipped with a warning.

## Writing a good skill

- **Description = what + when.** Include the trigger words a user would actually say, and say
  what the skill does *not* cover ("not for hinged doors — use ...").
- **Body under ~500 lines.** Push long tables, derivations or code into `references/` and tell
  the reader when to `read_file("<skill-name>/references/<file>.md")`.
- **Structure that works:** When to use / When NOT to use → Pre-conditions → Workflow →
  Patterns → Failure triage → Pitfalls.
- **Be prescriptive.** The reader is another LLM mid-task: give it order of operations,
  concrete numbers and the failure modes, not background theory.
- Body text is injected verbatim, so it may contain ```python blocks, formulas and tables.

## Usage

```bash
python main.py --task franka_kitchen:hinge_cabinet     # skills on by default
python main.py --list-skills                           # show what was discovered, per scope
python main.py --no-skills                             # disable entirely
python main.py --skills-dir path/to/other/skills       # use a different tree
```

Implementation: `skill_registry.py` (discovery, index rendering, sandboxed reads,
`SkillSession`), `prompts/skills_prompt.py` (the prompt section), wiring in `agent_runner.py`.
