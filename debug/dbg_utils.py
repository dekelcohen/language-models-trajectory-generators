# pip install opencv-python
import cv2
import os
import glob

def create_video_from_images(
    folder_path: str = 'images/trajectory', 
    base_name: str = 'rgb_image',   # <--- New Parameter
    start_idx: int = 0, 
    end_idx: int = float('inf'), 
    ext: str = 'png', 
    fps: int = 30
):
    """
    Creates an .mp4 video from images in the format: <base_name>_<idx>.<ext>
    
    Args:
        folder_path (str): Directory containing images.
        base_name (str): The prefix of the files (e.g., 'rgb_image' for 'rgb_image_0.png'). 
                         If None, tries to auto-detect.
        start_idx (int): Start index.
        end_idx (int): End index.
        ext (str): File extension.
        fps (int): Frames per second.
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

    # 2. Find first valid frame to set Video Dimensions
    # We check start_idx, then start_idx + 1 (to handle the lookahead case)
    first_file_path = os.path.join(folder_path, f"{base_name}_{start_idx}.{ext}")
    
    if not os.path.exists(first_file_path):
        # check next frame
        next_path = os.path.join(folder_path, f"{base_name}_{start_idx + 1}.{ext}")
        if os.path.exists(next_path):
            first_file_path = next_path
        else:
            print(f"[Error] Could not find start frame ({start_idx}) or lookahead ({start_idx+1}) for base '{base_name}'")
            return

    # 3. Setup Video Writer
    img = cv2.imread(first_file_path)
    if img is None:
        print(f"[Error] Could not read setup image: {first_file_path}")
        return

    height, width, _ = img.shape
    
    end_label = "inf" if end_idx == float('inf') else end_idx
    output_filename = f"{base_name}_{start_idx}_{end_label}.mp4"
    output_path = os.path.join(folder_path, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[Info] Writing: {output_filename} | Size: {width}x{height}")

    # 4. Processing Loop
    current_idx = start_idx
    processed_frames = 0

    try:
        while current_idx <= end_idx:
            
            curr_file = f"{base_name}_{current_idx}.{ext}"
            curr_path = os.path.join(folder_path, curr_file)
            
            if not os.path.exists(curr_path):
                # --- Look Ahead Logic ---
                next_idx = current_idx + 1
                
                if next_idx > end_idx:
                    break
                    
                next_path = os.path.join(folder_path, f"{base_name}_{next_idx}.{ext}")
                
                if os.path.exists(next_path):
                    print(f"[Warning] Frame {current_idx} missing. Skipping to {next_idx}...")
                    current_idx += 1
                    continue 
                else:
                    # Both missing -> End of sequence
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
        print(f"[Success] Saved {output_path} ({processed_frames} frames).")
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

