"""Central registry of per-model-family capabilities/conventions.

Replaces ad-hoc `"gemini" in model_name.lower()` / `"claude" in ...` string
checks scattered across the codebase. To add a new model family (e.g. a new
VLM), add one `ModelInfo` entry to `MODEL_REGISTRY` below - no other file
needs to change.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Capabilities/conventions for a model family.

    Looked up by matching `family` as a case-insensitive substring against the
    full model string (e.g. "gemini-2.5-flash", "or-google/gemini-2.5-flash",
    "aws-eu.anthropic.claude-opus:4-8", "azure-gpt-5").
    """
    supports_image: bool = True
    supports_video: bool = False
    # "xy_pixels": raw [x, y] pixel coordinates.
    # "yx_norm_1000": [y, x] coordinates normalized to 0-1000.
    pointing_coords_format: str = "xy_pixels"


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "gemini": ModelInfo(supports_image=True, supports_video=True, pointing_coords_format="yx_norm_1000"),
    "gemma":  ModelInfo(supports_image=True, supports_video=False, pointing_coords_format="yx_norm_1000"),
    "claude": ModelInfo(supports_image=True, supports_video=False, pointing_coords_format="xy_pixels"),
    "gpt":    ModelInfo(supports_image=True, supports_video=False, pointing_coords_format="xy_pixels"),
    "qwen":   ModelInfo(supports_image=True, supports_video=True, pointing_coords_format="xy_pixels"),
}

# Fallback for unrecognized models: image-only, raw pixel pointing coords.
DEFAULT_MODEL_INFO = ModelInfo()


def get_model_info(model: str) -> ModelInfo:
    """Return the ModelInfo for `model` by matching a known family name as a
    substring (case-insensitive). Falls back to DEFAULT_MODEL_INFO, logging a
    warning so unrecognized models are noticed (e.g. a typo, or a new family
    that needs a MODEL_REGISTRY entry)."""
    m = (model or "").lower()
    for family, info in MODEL_REGISTRY.items():
        if family in m:
            return info
    logger.warning(
        "get_model_info: no model family matched for %r, falling back to DEFAULT_MODEL_INFO "
        "(image-only, xy_pixels). Add an entry to MODEL_REGISTRY if this model needs different "
        "capabilities.", model
    )
    return DEFAULT_MODEL_INFO
