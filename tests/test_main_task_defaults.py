import main


def _resolve(argv):
    a = main.resolve_task_defaults(main.build_arg_parser().parse_args(argv))
    return a.language_model, a.reasoning_effort


def test_usb_task_defaults_to_sonnet_medium_but_explicit_flags_win():
    assert _resolve(["--task", "usb"]) == ("copilot-claude-sonnet-5.5", "medium")
    assert _resolve(["--task", "usb", "--reasoning-effort", "high"]) == ("copilot-claude-sonnet-5.5", "high")
    assert _resolve(["--task", "usb", "-lm", "copilot-claude-opus-5.5"]) == ("copilot-claude-opus-5.5", None)
    assert _resolve(["--task", "door"]) == (main.DEFAULT_LM, None)
