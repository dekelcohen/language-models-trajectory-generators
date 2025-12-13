import cv2
import os
import glob

def create_video_from_images(
    folder_path: str, 
    start_idx: int = 0, 
    end_idx: int = float('inf'), 
    ext: str = 'png', 
    fps: int = 30
):
    """
    Creates an .mp4 video from a sequence of images <basename>_<idx>.<ext>.
    
    Logic:
    - Automatically detects the basename from the file at start_idx (or start_idx+1).
    - Iterates through indexes.
    - If a file is missing, checks the immediate next index (current + 1).
      - If next exists: Skips the missing frame and continues.
      - If next is also missing: Breaks and saves the video.
    """
    
    # 1. Determine Basename
    # Try to find the start file or start+1 to determine the naming pattern
    start_pattern = os.path.join(folder_path, f"*_{start_idx}.{ext}")
    matches = glob.glob(start_pattern)
    
    # If start_idx is missing, try start_idx + 1 (Lookahead for initialization)
    if not matches:
        start_pattern_next = os.path.join(folder_path, f"*_{start_idx + 1}.{ext}")
        matches = glob.glob(start_pattern_next)
        if matches:
            print(f"[Info] Frame {start_idx} missing, detected basename from {start_idx+1}")
        else:
            print(f"[Error] Could not determine basename. Neither index {start_idx} nor {start_idx+1} found in {folder_path}")
            return

    # Extract basename (e.g., 'trajectory_0.png' -> 'trajectory')
    first_file_path = matches[0]
    filename = os.path.basename(first_file_path)
    basename = filename.rpartition('_')[0]
    
    # 2. Setup Video Writer using the first available image
    img = cv2.imread(first_file_path)
    if img is None:
        print(f"[Error] Could not read image setup frame: {first_file_path}")
        return

    height, width, _ = img.shape
    
    # Define Output Filename
    end_label = "inf" if end_idx == float('inf') else end_idx
    output_filename = f"{basename}_{start_idx}_{end_label}.mp4"
    output_path = os.path.join(folder_path, output_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[Info] Processing '{basename}' | Start: {start_idx} | Output: {output_filename}")

    # 3. Processing Loop
    current_idx = start_idx
    processed_frames = 0

    try:
        while current_idx <= end_idx:
            
            curr_file = f"{basename}_{current_idx}.{ext}"
            curr_path = os.path.join(folder_path, curr_file)
            
            if not os.path.exists(curr_path):
                # --- Look Ahead Logic ---
                next_idx = current_idx + 1
                
                # Don't look ahead if we've passed the user's requested end_idx
                if next_idx > end_idx:
                    break
                    
                next_file = f"{basename}_{next_idx}.{ext}"
                next_path = os.path.join(folder_path, next_file)
                
                if os.path.exists(next_path):
                    print(f"[Warning] Frame {current_idx} missing. Found {next_idx}. Skipping gap...")
                    current_idx += 1
                    continue # Continues the while loop, which will pick up 'next_path' in the next iteration
                else:
                    # Both current and next are missing -> Sequence ended
                    if end_idx == float('inf'):
                        print(f"[Info] Sequence ended naturally at {current_idx}.")
                    else:
                        print(f"[Info] Sequence broken at {current_idx} (next frame also missing). Stopping.")
                    break
            
            # --- Frame Exists ---
            frame = cv2.imread(curr_path)
            
            if frame is None:
                print(f"[Warning] File exists but could not be read (corrupt?): {curr_file}")
                # Treat corrupt file as missing -> trigger same logic? 
                # For safety, we skip this specific frame and move on
                current_idx += 1
                continue

            out.write(frame)
            processed_frames += 1
            current_idx += 1
            
    finally:
        out.release()
        print(f"[Success] Saved {output_path} ({processed_frames} frames).")

# --- Usage Example ---
# if __name__ == "__main__":
#     # Assuming cfg.images_traj is your path
#     # create_video_from_sequence("/path/to/images", start_idx=0)