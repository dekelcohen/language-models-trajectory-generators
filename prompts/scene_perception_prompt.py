# INPUT: [INSERT USER COMMAND TASK]
#
# Perception/vision prompt run by a dedicated perception VLM (--planner-perception-vlm) on the
# head-camera image BEFORE each planner LLM call. Its free-text answer is injected into
# the planner prompt's SCENE ANALYSIS section so the planner reasons over a described
# scene instead of the raw pixels.

SCENE_PERCEPTION_PROMPT = """You are a vision/perception module for a robot arm. You are given the head-camera image of a tabletop scene and the robot's user command. Analyze ONLY what is visible; do not plan robot motions.

User command: "[INSERT USER COMMAND TASK]"

Answer the following, concisely:
1. Describe all objects in this image (identity, rough location, and notable dimensions/orientation where visible).
2. Target object affordance:
   - If the affordance needed to perform the user command (e.g. a handle, lever, lid, knob, opening, grasp point) IS clearly visible, say so explicitly and finish.
   - If it is NOT clearly visible, estimate where the target object and its affordance are likely located, describe that estimate, and list the top objects that may be occluding it (including the robot arm)
3. Collision risks: if the robot arm, on its way to the target object's affordance, may collide with objects in the scene that are not easy to bypass, state them. Otherwise say there are no significant collision risks.
[INSERT AFFORDANCE POINTING SECTION]"""

# Appended to SCENE_PERCEPTION_PROMPT only when --affordance-points is enabled (default).
# COORDINATES_FORMAT_PLACEHOLDER is replaced at runtime by config.affordance_coords_format_gemini
# or config.affordance_coords_format_pixels depending on the perception VLM family.
AFFORDANCE_POINTING_SECTION = """
After the free-text answer above, output a separate block, delimited exactly as:
AFFORDANCE_POINTS:
```json
[{"point": , "label": }, ...]
```
Get 4 points for: where a robot arm should grasp next to accomplish task. The label returned should be an identifying name for the object detected.
  - If the affordance of target object clearly visible --> point to it
  - Else if there is an occluding / obstacle object, that probably needs to be cleared --> point to it 
  - Never point at objects not clearly visible and no sense in pointing to Robot Arm either

The answer should follow the json format:
[{"point": , "label": }, ...]

COORDINATES_FORMAT_PLACEHOLDER
sort the points in descending quality: the best predicted is the first one, then the next-best ...
"""
