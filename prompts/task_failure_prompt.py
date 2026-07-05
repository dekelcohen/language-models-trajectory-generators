# INPUT: [INSERT TASK SUMMARY]
TASK_FAILURE_PROMPT = \
"""SUMMARY OF PREVIOUS FAILED ATTEMPTS:
[INSERT TASK SUMMARY]

PROBLEM RESOLUTION:
Can you suggest what was wrong with the plans for the trajectories, and suggest specific changes that would be appropriate?
1) Analyze carefully the previous summary of previous attempts with the review notes and improvement suggestions
2) Incrementally Explore: In the failed trajs, change one thing at a time. Do not just repeat a set of failed trajectories - as it will fail again. 
   ) Ex: If Grasping failed - change grapsing pose towards the correct object part 
   ) Ex: If pull failed - consider changing pull direction (a little bit each attempt)
Then, replan and retry the task by continuing with INITIAL PLANNING 1.
"""
