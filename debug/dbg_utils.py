# pip install opencv-python loguru
import os
import sys
import io
import glob
import config
import cv2
from loguru import logger as loguru_logger

from helpers.image_utils import find_available_frame

def create_video_from_images(
    folder_path: str = 'images/trajectory', 
    output_video_folder_path: str = None,
    base_name: str = 'rgb_image',   # <--- New Parameter
    start_idx: int = 0, 
    end_idx: int = float('inf'), 
    ext: str = 'png', 
    fps: int = 30,
    lookahead_max: int = 50,
    output_filename: str = None,
):
    """
    Creates an .mp4 video from images in the format: <base_name>_<idx>.<ext>
    
    Args:
        folder_path (str): Directory containing images.
        output_video_folder_path (str): Directory to write the video into - if None (default) --> config.video_folder
        base_name (str): The prefix of the files (e.g., 'rgb_image' for 'rgb_image_0.png'). 
                         If None, tries to auto-detect.
        start_idx (int): Start index.
        end_idx (int): End index.
        ext (str): File extension.
        fps (int): Frames per second.
        output_filename (str): Output .mp4 file name; defaults to
            "<base_name>_<start_idx>_<end_idx>.mp4".

    Returns:
        str | None: path of the written .mp4, or None when nothing could be written.
    """
    
    # 1. Determine Base Name
    if base_name is None:
        # Auto-detection logic
        start_pattern = os.path.join(folder_path, f"*_{start_idx}.{ext}")
        matches = glob.glob(start_pattern)
        
        # Lookahead 1 frame if start frame is missing
        if not matches:
            matches = glob.glob(os.path.join(folder_path, f"*_{start_idx + 1}.{ext}"))
            
        if not matches:
            print(f"[Error] Could not auto-detect base name in {folder_path} starting at {start_idx}")
            return
            
        # Extract basename (e.g., 'path/to/trajectory_0.png' -> 'trajectory')
        filename = os.path.basename(matches[0])
        base_name = filename.rpartition('_')[0]
        print(f"[Info] Auto-detected base name: '{base_name}'")

    # 2. Find first valid frame to set Video Dimensions (use helper)
    first_file_path, start_idx = find_available_frame(
        folder_path, base_name, start_idx, end_idx, ext, lookahead_max, include_current=True
    )
    if first_file_path is None:
        print(f"[Error] Could not find start frame ({start_idx}) or lookahead ({start_idx+1}) for base '{base_name}'")
        return

    # 3. Setup Video Writer
    img = cv2.imread(first_file_path)
    if img is None:
        print(f"[Error] Could not read setup image: {first_file_path}")
        return

    height, width, _ = img.shape
    
    end_label = "inf" if end_idx == float('inf') else end_idx
    if output_filename is None:
        output_filename = f"{base_name}_{start_idx}_{end_label}.mp4"
    
    if output_video_folder_path is None:
        output_video_folder_path = config.video_folder
    os.makedirs(output_video_folder_path, exist_ok=True)
    output_path = os.path.join(output_video_folder_path, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[Info] Writing: {output_filename} | Size: {width}x{height}")

    # 4. Processing Loop
    current_idx = start_idx
    processed_frames = 0
    skip_events = 0  # count how many times we had to jump ahead

    try:
        while current_idx <= end_idx:
            
            curr_file = f"{base_name}_{current_idx}.{ext}"
            curr_path = os.path.join(folder_path, curr_file)
            
            if not os.path.exists(curr_path):
                # --- Look ahead using the same helper ---
                next_path, next_idx = find_available_frame(
                    folder_path, base_name, current_idx, end_idx, ext, lookahead_max, include_current=False
                )
                if next_path is not None:
                    skip_events += 1
                    current_idx = next_idx
                    continue
                # No further frames within lookahead window -> end sequence
                if end_idx == float('inf'):
                    print(f"[Info] Sequence ended at {current_idx}.")
                else:
                    print(f"[Info] Sequence broken at {current_idx}. Stopping.")
                break
            
            # --- Write Frame ---
            frame = cv2.imread(curr_path)
            if frame is None:
                # Corrupt file handling
                print(f"[Warning] Failed to read {curr_file}, skipping.")
                current_idx += 1
                continue

            out.write(frame)
            processed_frames += 1
            current_idx += 1
            
    finally:
        out.release()
        if skip_events > 0:
            print(f"[Warning] Skipped {skip_events} gap(s) due to missing frames (lookahead <= {lookahead_max}).")
        print(f"[Success] Saved {output_path} ({processed_frames} frames).")

    return output_path if processed_frames > 0 else None
        
# -------------------------
# Debug reminder utilities
# -------------------------

_REMINDER_INSTALLED = False
_ORIG_BREAKPOINTHOOK = None
_ORIG_EXCEPTHOOK = None
_ORIG_THREADING_EXCEPTHOOK = None

def install_debug_reminder(tip: str | None = None) -> None:
    """Install simple hooks that print a reminder when debugging pauses.

    - Prints a tip to stderr when builtin breakpoint() triggers (via sys.breakpointhook).
    - Prints the same tip on uncaught exceptions in the main thread (sys.excepthook)
      and in threads (threading.excepthook, if present).

    Notes: intentionally simple  does not handle PYTHONBREAKPOINT overrides.
    """
    global _REMINDER_INSTALLED, _ORIG_BREAKPOINTHOOK, _ORIG_EXCEPTHOOK, _ORIG_THREADING_EXCEPTHOOK
    if _REMINDER_INSTALLED:
        return
    import os, sys, threading
    message = tip or os.getenv("MW_DEBUG_TIP", "Reminder: launch tests/main.py with --timeout=0 (or -1) to allow debugging.")

    def _print_tip():
        try:
            print(f"[DEBUG TIP] {message}", file=sys.stderr, flush=True)
        except Exception:
            pass

    _ORIG_BREAKPOINTHOOK = getattr(sys, "breakpointhook", None)
    def _bp_hook(*a, **kw):
        _print_tip()
        if _ORIG_BREAKPOINTHOOK is not None:
            return _ORIG_BREAKPOINTHOOK(*a, **kw)
    sys.breakpointhook = _bp_hook

    _ORIG_EXCEPTHOOK = sys.excepthook
    def _exc_hook(t, v, tb):
        _print_tip()
        return _ORIG_EXCEPTHOOK(t, v, tb)
    sys.excepthook = _exc_hook

    if hasattr(threading, "excepthook"):
        _ORIG_THREADING_EXCEPTHOOK = threading.excepthook
        def _thread_exc_hook(args):
            _print_tip()
            return _ORIG_THREADING_EXCEPTHOOK(args)
        threading.excepthook = _thread_exc_hook

    _REMINDER_INSTALLED = True

# -------------------------
# Log file ANSI stripping
# -------------------------
import re
from typing import Any, Dict

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
# Some environments may strip the ESC byte, leaving bracket codes like "[92m" in text.
# This regex removes a limited set of such codes (color/style + erase) to avoid false positives.
_BARE_ANSI_RE = re.compile(r"\[[0-9;]*[mK]")

def _strip_ansi_for_file(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ANSI escape codes from record["message"] for file logging.

    - Strips standard CSI sequences starting with ESC.
    - Also strips bracket-only color/erase sequences (e.g., "[92m", "[0m", "[2K")
      which can appear if ESC was lost upstream.
    """
    try:
        msg = str(record.get("message", ""))
        msg = _ANSI_RE.sub("", msg)
        msg = _BARE_ANSI_RE.sub("", msg)
        record["message"] = msg
    except Exception:
        pass
    return record


class _StdStreamTee(io.TextIOBase):
    """
    Tee stdout/stderr into the file-only path without duplicating console
    """
    def __init__(self, stream, bound_logger):
        self._stream = stream
        self._logger = bound_logger
        self._buffer = ""

    def write(self, s):
        try:
            # Always forward to the real console stream
            self._stream.write(s)
        except Exception:
            pass

        try:
            # Buffer and emit in blocks up to the last newline
            self._buffer += str(s)
            last = self._buffer.rfind("\n")
            if last != -1:
               chunk = self._buffer[:last]
               self._buffer = self._buffer[last+1:]
               chunk = chunk.replace("\r", "")
               # Emit as a single record to avoid per-line prefixes
               if chunk:
                   self._logger.info(chunk)
               else:
                   self._logger.info("")
        except Exception:
            # Never raise from logger path; console already received text
            pass
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            if self._buffer:
                self._logger.info(self._buffer.rstrip("\r\n"))
                self._buffer = ""
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False
            
def init_loguru_logger(file_basename: str = "vlm_traj.log"):
    """Initialize Loguru with console + file sinks and tee stdio to file.

    - Console: pretty colors, does NOT duplicate captured prints.
    - File: receives both logger messages and any writes to stdout/stderr.
    """    
    log_file_path = os.path.join(config.images_folder, file_basename)

    # Keep references to the real stdio before we touch anything
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    # Remove default logger so we control both sinks
    loguru_logger.remove()

    # Ensure console does NOT print the re-logged stdio lines again
    def _console_filter(record):
        try:
            src = record.get("extra", {}).get("source")
            return src not in ("stdout", "stderr")
        except Exception:
            return True

    # Console sink (keep colors, no stripping)
    loguru_logger.add(
        orig_stdout,
        level="INFO",
        colorize=True,
        filter=_console_filter,
        format="{time:DD/MM HH:mm} | {level:<8} | {message}",
    )

    
    loguru_logger.add(
        log_file_path,
        level="INFO",
        enqueue=True,
        encoding="utf-8",
        colorize=False,
        format="{time:DD/MM HH:mm} | {level:<8} | {message}\n",
    )


    # Bind sources so console sink can filter them out
    sys.stdout = _StdStreamTee(orig_stdout, loguru_logger.bind(source="stdout"))
    sys.stderr = _StdStreamTee(orig_stderr, loguru_logger.bind(source="stderr"))

    return loguru_logger

