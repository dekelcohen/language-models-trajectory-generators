REVIEW_PROMPT = (
    "You are a robotics VLM reviewer. Use the full conversation history you just saw (planning and execution). "
    "verify whether the robot executed the task as planned by inspecting the provided visual evidence.\n\n"
    "Task: [INSERT TASK]\n\n"
    "[INSERT OVERALL COMMAND SECTION]"
    "3D Coordinates Context:\n[INSERT 3D COORDINATES PROMPT SECTION]\n\n"
    "[INSERT SCENE ANALYSIS SECTION]"
    "[INSERT MEDIA SECTION]"
    "Instructions:\n\n"
    "Define the success criteria according to task prompt core goal (and not if all traj were executed as expected). Ex: 1) if the task was to clear an occluding object and it no longer stands in front of / blocks the view (moved, pushed or knocked over - no matter how) --> success=True, even if the hidden target is still not visible (its location behind the occluder was only a guess). 2) if the task was to open the door and it was opened WIDE (>= ~60 deg, wide enough for a mobile robot to pass) --> success=True; a door only ajar (~10-30 deg) --> success=False"
    "Carefully compare the plan and execution in the conversation history with the visual evidence provided (reference cam + frame number or video timestamp). focus on positions and orientation of robot arm gripper in relation to the target object(s). Only use the rendered text on the image as the plan desc, it should match actual visual state" 
    "Did all executed trajectories succeeded as planned ?"
    "If grasping: Did the gripper actually grasped the target object ?"
    "If grasping: Did the grasp pose ensured a stable post-grasp actions / trajectory (object moved?) or can it be improved ?"
    "Articulation and Post grasp traj: Did the trajectories opened/rotated/pulled/pushed/operated the obj/affordances in the desired axis/directions ?"
    "Target object final pose: Did the object ended up in desired end-pose ? Ex: door opened enough ? Ex: grapsed object lifted enough height ?. Use also START-OF-ATTEMPT SCENE to compare before vs current env state and target objects"
    "Decide if the task was achieved. If uncertain or inconsistent, set success=false."
    "if success=False (and even if its succeeded), Reason to suggest global improvement_steps: positions, poses, orientations in gen code. for each trajectory - determine success : true/false and its local improvement_steps (or suggest to delete/replace/add new trajs)"
    "If attempt > 2 - start to explore more positions, poses, orientations in gen code in failed trajs - do not repeat bad poses from prev attempts"
    "Output strictly one JSON object with exactly these keys: success (true/false), reasoning (string), improvement_steps (list of strings for trajectories - inc new/delete/update)"
)

# --- [INSERT MEDIA SECTION] variants ------------------------------------------------
# Frames mode (review model cannot take video): subsampled key frames are attached.
REVIEW_MEDIA_FRAMES_SECTION = (
    "Key frames of THIS attempt (attached after the scene image, in this order: all head-camera frames, "
    "then all wrist-camera frames; chosen where the image changes most, plus each camera's FINAL frame):\n"
    "[INSERT FRAME PATHS]\n"
    "The frames marked FINAL STATE decide success. Reference evidence as 'head cam step N' / "
    "'wrist cam step N'.\n\n"
)

# Video mode (Gemini direct / via OpenRouter): one mp4 per camera, covering ONLY the
# current attempt, is attached instead of dozens of key frames.
REVIEW_MEDIA_VIDEO_SECTION = (
    "TRAJECTORY VIDEO(S) of THIS attempt (attached after the scene image, in this order):\n"
    "[INSERT VIDEO LIST]\n"
    "Each clip covers ONLY the current attempt (from the START-OF-ATTEMPT SCENE above to the final state), "
    "recorded at [INSERT VIDEO FPS] fps. Watch them end-to-end; the LAST frames are the final state that "
    "determines success. Reference evidence as 'head cam @ MM:SS' / 'wrist cam @ MM:SS'.\n\n"
)

# Filled into [INSERT SCENE ANALYSIS SECTION] when a scene analysis exists for the attempt.
# Its image is attached FIRST, before the trajectory key frames / videos, and must not be
# mistaken for one of them.
REVIEW_SCENE_ANALYSIS_SECTION = (
    "START-OF-ATTEMPT SCENE (before any robot motion in this attempt):\n"
    "The FIRST attached image is NOT a trajectory key frame - it is the head-camera image "
    "([INSERT SCENE ANALYSIS IMAGE PATH]) that the perception VLM analyzed at the beginning of this attempt. "
    "Use it as the 'before' reference to judge what actually changed during execution. "
    "[INSERT REMAINING MEDIA SENTENCE]\n\n"
    "Scene analysis of that image (perception VLM):\n[INSERT SCENE ANALYSIS]\n\n"
)

REMAINING_MEDIA_SENTENCE_FRAMES = (
    "All the REMAINING attached images are the trajectory key frames, in the order listed below."
)

# Filled into [INSERT OVERALL COMMAND SECTION] when the Task is a planner subtask.
REVIEW_OVERALL_COMMAND_SECTION = (
    "OVERALL USER COMMAND (this Task is one planner step toward it): [INSERT OVERALL COMMAND]\n"
    "Judge the Task by its GOAL and the END STATE (e.g. 'the occluder no longer blocks the target', 'the door "
    "is open'), even if an earlier attempt already achieved it. "
    "HOW-TO details the planner wrote into the Task text (where/how to grasp, exact distances, keep it "
    "upright, avoid touching X) are suggestions, not success criteria: if the goal is met they do not make "
    "it a failure. Amounts the planner chose ('about 12 cm higher', 'lift ~50 cm', 'at least 30 deg') "
    "are approximate targets: within ~25% of them, or as far as the arm could reach, counts as met when "
    "the qualitative goal holds (held, clearly lifted, visibly open). Exception: a stated MINIMUM opening for a door (e.g. '>= 60 deg, wide enough to pass') is a hard floor - a door clearly short of it (ajar, ~10-40 deg) is a failure, however far the arm could reach. Process steps written as success conditions (e.g. 'the lever has been actuated', "
    "'the latch released', 'grasped the handle first') are implied by the goal state when it is reached "
    "(an open door means its latch was released): do not fail an achieved goal for lack of proof of how "
    "it got there. Do not add requirements the Task does not state (exact clearance margins, a "
    "verified swing path, where a cleared object came to rest): if the stated condition is visibly met "
    "(e.g. the object no longer stands in front of / touches the target), the uncertainty about such "
    "extras is NOT a reason to fail. A side effect that ADVANCES the overall command (e.g. the door swung open while the "
    "pole was cleared) is not a failure either; only fail for side effects that make the overall command "
    "harder or unsafe. The ROBOT ARM itself overlapping the target in the last frame (it is parked "
    "or retracting there) is NOT an obstruction: the arm moves away on the next step; judge whether the "
    "OBJECTS are arranged as required, ignoring the arm.\n\n"
)
REMAINING_MEDIA_SENTENCE_VIDEO = (
    "The REMAINING attachments are the trajectory VIDEOS of this attempt, in the order listed below."
)
