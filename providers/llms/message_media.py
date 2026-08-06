# message_media.py
"""Provider-agnostic multimodal message building.

All providers (azure, openrouter, gemini, bedrock, openai-compatible) share ONE
canonical OpenAI-style message format; each provider module converts it to its
native shape. Media parts:

    image: {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
    video: {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,<data>"}}

The video part shape follows OpenRouter's `video_url` convention (the only
OpenAI-compatible video convention in the wild); providers that take video in a
different shape (Gemini `inline_data`, Bedrock `video` block) convert it.
"""
import base64
import mimetypes
from pathlib import Path

# Canonical media part types used across providers.
IMAGE_PART_TYPE = "image_url"
VIDEO_PART_TYPE = "video_url"
MEDIA_PART_TYPES = (IMAGE_PART_TYPE, VIDEO_PART_TYPE)


def encode_media(media_path: str) -> tuple[str, str]:
    """Encode any media file to base64 and return (mime_type, base64_string).

    Example:
        mime, b64 = encode_media("clip.mp4")
        url = f"data:{mime};base64,{b64}"
    """
    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    mime_type, _ = mimetypes.guess_type(media_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {media_path}")

    with open(media_path, "rb") as f:
        return mime_type, base64.b64encode(f.read()).decode("utf-8")


# Backwards-compatible alias (images are just media).
encode_image = encode_media


def append_images(content: list, image_paths) -> list:
    """Append inline base64 image parts to a content-parts list (in place)."""
    for image_path in image_paths or []:
        mime_type, b64 = encode_media(image_path)
        content.append({
            "type": IMAGE_PART_TYPE,
            IMAGE_PART_TYPE: {"url": f"data:{mime_type};base64,{b64}"},
        })
    return content


def append_videos(content: list, video_paths) -> list:
    """Append inline base64 video parts to a content-parts list (in place).

    Videos must be small enough to inline (provider caps: Gemini < 100 MB total
    request, Bedrock 25 MB); the reviewer clips are short 256x256 mp4s.
    """
    for video_path in video_paths or []:
        mime_type, b64 = encode_media(video_path)
        if not mime_type.startswith("video/"):
            raise ValueError(f"Not a video file: {video_path} (mime={mime_type})")
        content.append({
            "type": VIDEO_PART_TYPE,
            VIDEO_PART_TYPE: {"url": f"data:{mime_type};base64,{b64}"},
        })
    return content


def append_to_messages(new_prompt: str,
                       attach_images: list[str] | None = None,
                       messages: list | None = None,
                       role: str = "user",
                       attach_videos: list[str] | None = None):
    """Append a new message to the conversation, with optional image/video attachments.

    Args:
        new_prompt: text prompt (may be empty when media is attached)
        attach_images: image file paths, inlined as base64 `image_url` parts
        messages: existing messages list (created when None)
        role: message role (default: user)
        attach_videos: video file paths, inlined as base64 `video_url` parts

    Returns:
        Updated messages list
    """
    if messages is None:
        messages = []

    if not attach_images and not attach_videos:
        if not new_prompt:
            return messages
        messages.append({"role": role, "content": new_prompt})
        return messages

    content = [{"type": "text", "text": new_prompt}]
    append_images(content, attach_images)
    append_videos(content, attach_videos)
    messages.append({"role": role, "content": content})
    return messages


def is_media_part(part) -> bool:
    """True if a content part carries an image or a video."""
    if not isinstance(part, dict):
        return False
    return part.get("type") in MEDIA_PART_TYPES or any(k in part for k in MEDIA_PART_TYPES)
