"""Build ablated touch-guided-insertion SKILL.md variants (research/touch/skill_ablation/<v>/SKILL.md).

Use one with: python research/touch/skill_ablation/make_variants.py -> skills trees in
outputs/_ablation/skills_<v>/ (a copy of prompts/skills with the variant SKILL.md), then
main.py --skills-dir outputs/_ablation/skills_<v>.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, "prompts", "skills")
SKILL = os.path.join(SRC, "touch-guided-insertion", "SKILL.md")

# (start, end) substrings: remove text from start up to (not including) end.
A = [  # situational rules this scene never exercises (part heading, several faces / openings)
    (" The part may lie pointing anywhere:", "\n   Small parts:"),
    (" Which face holds the port is a guess too:", "\n   Holding the part, park"),
    (" A device can show\n   several similar openings", " Points a few cm apart"),
    ("A line search that ends\n   `search_exhausted` after a full sweep", "   `no_contact` / `slipped`"),
    ("   **Wrong opening**:", "   **Adapt the forces"),
]
B = A + [  # + long defensive prose (keeps the code blocks and the core numbers)
    (" **Sanity-check the touch**:", " The arm settles a few mm LOW"),
    ("   Settle correction (here and below):", "   `close_gripper()` records the grasp;"),
    ("   **Off-centre grasp -> regrasp.** Expected off_a", "   Do NOT press the held tip down"),
    ("   **Adapt the forces per retry**", None),
]


def ablate(text, cuts):
    for start, end in cuts:
        i = text.index(start)
        j = len(text) if end is None else text.index(end, i)
        text = text[:i] + text[j:]
    return text


if __name__ == "__main__":
    base = open(SKILL, encoding="utf-8").read()
    for name, cuts in (("A_no_situational", A), ("B_lean", B)):
        out = ablate(base, cuts)
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8").write(out)
        tree = os.path.join(ROOT, "outputs", "_ablation", f"skills_{name}")
        shutil.rmtree(tree, ignore_errors=True)
        shutil.copytree(SRC, tree)
        shutil.copy(os.path.join(d, "SKILL.md"), os.path.join(tree, "touch-guided-insertion", "SKILL.md"))
        print(f"{name}: {len(base)} -> {len(out)} chars; tree {tree}")
