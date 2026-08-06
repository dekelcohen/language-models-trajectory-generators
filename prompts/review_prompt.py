REVIEW_PROMPT = (
    "You are a robotics VLM reviewer. Use the full conversation history you just saw (planning and execution). "
    "verify whether the robot executed the task as planned by inspecting the provided key frames.\n\n"
    "Task: [INSERT TASK]\n\n"
    "3D Coordinates Context:\n[INSERT 3D COORDINATES PROMPT SECTION]\n\n"
    "[INSERT SCENE ANALYSIS SECTION]"
    "Key frame file paths (chronological):\n[INSERT FRAME PATHS]\n\n"
    "Instructions:\n\n"
    "Define the success criteria according to task prompt core goal (and not if all traj were executed as expected). Ex: 1) if the target object was not visible and an occluding object is removed and not the target object is visible (no matter what happened) --> success=True. 2) if the object was to open the door and it was opened --> success=True"
    "Carefully compare the plan and execution in the conversation history with the visual evidence in the provided frames (reference cam+frame numbers). focus on positions and orientation of robot arm gripper in relation to the target object(s). Only use the rendered text on the image as the plan desc, it should match actual visual state" 
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

# Filled into [INSERT SCENE ANALYSIS SECTION] when a scene analysis exists for the attempt.
# Its image is attached FIRST, before the trajectory key frames, and must not be mistaken
# for one of them.
REVIEW_SCENE_ANALYSIS_SECTION = (
    "START-OF-ATTEMPT SCENE (before any robot motion in this attempt):\n"
    "The FIRST attached image is NOT a trajectory key frame - it is the head-camera image "
    "([INSERT SCENE ANALYSIS IMAGE PATH]) that the perception VLM analyzed at the beginning of this attempt. "
    "Use it as the 'before' reference to judge what actually changed during execution. "
    "All the REMAINING attached images are the trajectory key frames, in the order listed below.\n\n"
    "Scene analysis of that image (perception VLM):\n[INSERT SCENE ANALYSIS]\n\n"
)
