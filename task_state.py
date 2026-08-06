"""Per-(sub)task mutable state for the robot agent.

The long-lived `API` owns hardware/model/connection handles and the continuous
trajectory image counter. Everything that must be *fresh per subtask* lives here,
so each subtask gets an independent conversation, attempt counter, review outcome
and segmentation bookkeeping while the physical robot/sim state and continuous
`trajectory_step` numbering (held on API) carry over between subtasks.
"""


class TaskState:
    def __init__(self, command=None, max_attempts=None, start_trajectory_step=1):
        # Identity / config for this task
        self.command = command
        self.max_attempts = max_attempts

        # Conversation for this task
        self.conversation_messages = []

        # Attempt bookkeeping
        self.attempt_number = 0
        # First trajectory image index of the current attempt (VLM review frame sampling)
        self.start_attempt_trajectory_step = start_trajectory_step

        # Terminal flags
        self.completed_task = False
        self.failed_task = False

        # Review outcome
        self.review_succeeded = False
        self.review_reason = ""
        self.review_improvement_steps = ""
        self.accepted_without_review = False
        # Per-attempt trajectory clips (head, wrist mp4) built by task_completed() and
        # attached to the reviewer when the review model supports video input.
        self.review_clips = []

        # Perception / trajectory bookkeeping for this task
        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0

        # Scene analysis (perception VLM) this task started from, plus the exact head
        # image it was computed from. Shown to the reviewer VLM so it knows the
        # start-of-attempt scene, separately from the trajectory frames.
        self.scene_analysis = ""
        self.scene_analysis_image_path = None

    def title(self, max_len=90):
        """Short single-line task name for log lines."""
        text = " ".join(str(self.command or "(no task)").split())
        return text if len(text) <= max_len else text[:max_len - 3] + "..."
