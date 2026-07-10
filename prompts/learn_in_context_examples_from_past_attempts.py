# Prompt for learning better in context example for a task, from past experience
LEARN_IN_CONTEXT_PROMPT = \
"""
Role: You are a senior robotics AI researcher. 
Background: The user prompts the VLM Agent to perform a task (usually manipulation of objects). The main prompt contains instructions + in- context example of grasping and the code the llm generated for this task. It now attempts several times to perform the task by using perception tools (ex: detect_object --> 3D bbox of object) and generate trajs code. It sometimes fails and sometimes succeeds.




Past trajectories: [INSERT PAST TRAJS]


3D Coordinates Context:\n[INSERT 3D COORDINATES PROMPT SECTION]

Task: analyze past trajectories to extract a better in-context example (to be inserted in main prompt) that guides the Agent to success.
Extract Robot Agent Task description (user command) from the past traj
Compare failed attempts (if any) to successful ones and focus on the key differences in grasping, post-grasp actions, locations and trajs calc 
Output: A generalized in-context example - make sure no hard coded magic numbers and helps even if small changes in env and poses 
"""
