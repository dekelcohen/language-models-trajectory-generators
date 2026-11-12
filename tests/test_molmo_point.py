"""MolmoPoint gpu_server module + client glue, CPU only (no model download)."""

import contextlib
import types

import numpy as np
import pytest

from gpu_server.modules import molmo_point as mp
from gpu_server.modules.base import ModuleError


# -- text parsers --------------------------------------------------------------
def test_parse_molmo1_point_percent():
    text = 'Sure. <point x="25.0" y="50.0" alt="port">port</point>'
    assert mp.parse_point_text(text, 200, 100) == [[50.0, 50.0]]


def test_parse_molmo1_points_multi_in_index_order():
    text = '<points x1="10" y1="20" x2="50.5" y2="100" x10="1" y10="1" alt="ports">ports</points>'
    assert mp.parse_point_text(text, 1000, 500) == [[100.0, 100.0], [505.0, 500.0], [10.0, 5.0]]


def test_parse_molmo2_coords_image_scale_1000():
    # "<idx> <x> <y>" triples, coordinates scaled by 1000.
    text = '<points coords="1 500 250 2 100 900">USB</points>'
    assert mp.parse_point_text(text, 640, 480) == [[320.0, 120.0], [64.0, 432.0]]


def test_parse_molmo2_coords_with_leading_image_index():
    text = '<points coords="1 1 500 250">x</points>'
    assert mp.parse_point_text(text, 640, 480) == [[320.0, 120.0]]


def test_parse_molmo2_video_coords_drop_time():
    text = '<points coords="8.5 1 500 500;9.0 2 250 750">p</points>'
    assert mp.parse_point_text(text, 400, 200) == [[200.0, 100.0], [100.0, 150.0]]


def test_parse_drops_out_of_image_and_garbage():
    assert mp.parse_point_text('<point x="150" y="10">', 100, 100) == []
    assert mp.parse_point_text("no points here", 100, 100) == []
    assert mp.parse_point_text(None, 100, 100) == []


def test_extracted_rows_to_pixels_uses_last_two_columns():
    rows = [[1, 0, 336.0, 100.0], [2, 0, 627.0, 187.0], np.array([3, 0, 5.0, 6.0])]
    assert mp.extracted_to_pixels(rows) == [[336.0, 100.0], [627.0, 187.0], [5.0, 6.0]]
    assert mp.extracted_to_pixels(rows, max_points=1) == [[336.0, 100.0]]
    assert mp.extracted_to_pixels(None) == []


def test_build_prompt():
    assert mp.build_prompt({"description": "the USB port."}) == "Point to the USB port."
    assert mp.build_prompt({"prompt": "Point to all mugs", "description": "x"}) == "Point to all mugs"
    with pytest.raises(ModuleError):
        mp.build_prompt({})


def test_select_mode_auto_by_vram():
    assert mp.select_mode(7.6)[:2] == (mp.MODEL_4B, "4bit")          # RTX 3070 8 GB
    assert mp.select_mode(11.9)[:2] == (mp.MODEL_4B, "4bit")
    assert mp.select_mode(15.6)[:2] == (mp.MODEL_8B, "4bit")         # 16 GB card
    assert mp.select_mode(23.6)[:2] == (mp.MODEL_8B, "bf16")         # 24 GB card
    assert mp.select_mode(None)[:2] == (mp.MODEL_4B, "bf16")         # CPU: no bitsandbytes
    assert "7.6 GiB" in mp.select_mode(7.6)[2]


def test_select_mode_explicit_values_win():
    assert mp.select_mode(7.6, mp.MODEL_8B)[:2] == (mp.MODEL_8B, "4bit")
    assert mp.select_mode(23.6, mp.MODEL_4B)[:2] == (mp.MODEL_4B, "bf16")
    assert mp.select_mode(13.0, mp.MODEL_4B)[:2] == (mp.MODEL_4B, "4bit")
    assert mp.select_mode(7.6, None, "8bit")[:2] == (mp.MODEL_4B, "8bit")
    assert mp.select_mode(40, "erdi28/MolmoPoint-8B-bnb-nf4-vision-bf16", "none")[1] == "none"


def test_module_reads_env(monkeypatch):
    monkeypatch.setenv("MOLMO_MODEL", "x/y")
    monkeypatch.setenv("MOLMO_QUANT", "8bit")
    m = mp.MolmoPointModule()
    assert (m.model_id, m.quant) == ("x/y", "8bit")
    monkeypatch.delenv("MOLMO_MODEL")
    monkeypatch.delenv("MOLMO_QUANT")
    assert mp.MolmoPointModule().model_id is None   # auto at load()


# -- m_point with a fake model / processor ------------------------------------
class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def to(self, _dev):
        return self

    def size(self, i):
        return self.a.shape[i]


class _FakeProcessor:
    def __init__(self, text):
        self.text, self.messages = text, None

    def apply_chat_template(self, messages, **kw):
        assert kw["return_pointing_metadata"]
        self.messages = messages
        img = messages[0]["content"][1]["image"]
        return {"input_ids": _T(np.zeros((1, 7))), "metadata": {
            "token_pooling": "tp", "subpatch_mapping": "sm", "image_sizes": [img.size]}}

    def post_process_image_text_to_text(self, gen, **kw):
        assert gen.shape == (1, 3)
        return [self.text]


class _FakeModel:
    def __init__(self, rows):
        self.rows = rows

    def build_logit_processor_from_inputs(self, inputs):
        return None

    def generate(self, **kw):
        return np.zeros((1, 10))

    def extract_image_points(self, text, pooling, mapping, sizes):
        assert (pooling, mapping) == ("tp", "sm")
        if self.rows is None:
            raise RuntimeError("no grounding tokens")
        return self.rows


_FAKE_TORCH = types.SimpleNamespace(
    inference_mode=contextlib.nullcontext, autocast=lambda *a, **k: contextlib.nullcontext(), bfloat16="bf16",
    cuda=types.SimpleNamespace(is_available=lambda: False))


def _module(rows, text="<|points|>"):
    m = mp.MolmoPointModule(model="fake/molmo", quant="4bit")
    m.model, m.processor, m._torch = _FakeModel(rows), _FakeProcessor(text), _FAKE_TORCH
    return m


def test_m_point_returns_pixels_from_extract_image_points():
    m = _module([[1, 0, 10.0, 20.0], [2, 0, 30.0, 40.0], [3, 0, 1.0, 2.0]])
    img = np.zeros((48, 64, 3), np.uint8)
    reply, arrays = m.call("point", {"description": "the port", "max_points": 2}, {"image": img})
    assert arrays == {}
    assert reply["points"] == [[10.0, 20.0], [30.0, 40.0]]
    assert reply["prompt"] == "Point to the port."
    assert reply["image_size"] == [64, 48] and reply["model"] == "fake/molmo" and "latency_s" in reply
    assert m.processor.messages[0]["content"][0] == {"type": "text", "text": "Point to the port."}


def test_m_point_falls_back_to_text_parser():
    m = _module(None, text='<point x="50" y="25" alt="p">p</point>')
    reply, _ = m.call("point", {"description": "p"}, {"image": np.zeros((100, 200, 3), np.uint8)})
    assert reply["points"] == [[100.0, 25.0]]


def test_m_point_requires_image_and_loaded_model():
    with pytest.raises(ModuleError):
        _module([]).call("point", {"description": "p"}, {})
    with pytest.raises(ModuleError):
        mp.MolmoPointModule().call("point", {"description": "p"}, {"image": np.zeros((4, 4, 3), np.uint8)})


def test_registered_builtin():
    from gpu_server.host import BUILTIN_MODULES, import_spec
    assert import_spec(BUILTIN_MODULES["molmo_point"]) is mp.MolmoPointModule


# -- client glue ----------------------------------------------------------------
def test_point_with_molmo_client(monkeypatch):
    from PIL import Image
    import gpu_server.client as gc
    from providers.pointing import molmo_remote as mr

    calls = {}

    class FakeClient:
        def __init__(self, url, timeout=None, token=None, image_codec="raw"):
            calls["url"], calls["codec"] = url, image_codec

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def call(self, module, method, **kw):
            calls.update(module=module, method=method, shape=kw["image"].shape, desc=kw["description"])
            return {"points": [[1, 2], [3, 4], [5, 6]], "raw": "r"}, {}

    monkeypatch.setattr(gc, "GpuClient", FakeClient)
    monkeypatch.setenv("GPU_SERVER_MOLMO_URL", "tcp://box:8786")
    rgba = Image.new("RGBA", (32, 16))
    assert mr.point_with_molmo(rgba, "port", max_points=2) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == {"url": "tcp://box:8786", "codec": "raw", "module": "molmo_point", "method": "point",
                     "shape": (16, 32, 3), "desc": "port"}
    mr.point_with_molmo(np.zeros((8, 8, 3), np.uint8), "port", url=mr.url_from_model("gpu-molmo:ws://h:8785/v1/ws"))
    assert calls["url"] == "ws://h:8785/v1/ws"
    assert mr.url_from_model("gpu-molmo") is None and mr.is_molmo_model("gpu-molmo") and not mr.is_molmo_model("gpt")
