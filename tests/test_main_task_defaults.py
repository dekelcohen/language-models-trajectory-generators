import main


def _resolve(argv):
    a = main.resolve_task_defaults(main.build_arg_parser().parse_args(argv))
    return a.language_model, a.reasoning_effort


def test_usb_task_defaults_to_sonnet_medium_but_explicit_flags_win():
    assert _resolve(["--task", "usb"]) == ("copilot-claude-sonnet-5.5", "medium")
    assert _resolve(["--task", "usb", "--reasoning-effort", "high"]) == ("copilot-claude-sonnet-5.5", "high")
    assert _resolve(["--task", "usb", "-lm", "copilot-claude-opus-5.5"]) == ("copilot-claude-opus-5.5", None)
    assert _resolve(["--task", "door"]) == (main.DEFAULT_LM, None)


def test_perception_effort_defaults_low_and_can_inherit_the_main_effort():
    from helpers.perception_scene_analysis import perception_effort
    args = lambda argv: main.resolve_task_defaults(main.build_arg_parser().parse_args(argv))
    assert perception_effort(args(["--task", "usb"])) == "low"
    assert perception_effort(args(["--task", "usb", "--perception-reasoning-effort", "inherit"])) == "medium"
    assert perception_effort(args(["--task", "usb", "--perception-reasoning-effort", "high"])) == "high"


def test_usb_wrist_pointing_defaults_to_gemini_flash_unless_given():
    parse = lambda argv: main.resolve_task_defaults(main.build_arg_parser().parse_args(argv))
    assert parse(["--task", "usb"]).wrist_pointing_model == "or-google/gemini-3.8-flash"
    assert parse(["--task", "usb", "--wrist-pointing-model", "gpu-molmo"]).wrist_pointing_model == "gpu-molmo"
    assert parse(["--task", "door"]).wrist_pointing_model is None
