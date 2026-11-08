---
title: VLM Trajectory Rollouts Viewer
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - read-repos
---

A static, browser-only viewer for the private dataset `RoboTrick/vlm_traj_rollouts`.
It shows the summary, head and wrist videos, per-attempt images and the log.

- Sign in with Hugging Face and grant access to the RoboTrick org, or paste a read token.
- Static Space because Gradio / Docker Spaces need a paid HF plan.

Deploy or update from the repo root:
```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='research/hf_rollout_viewer', repo_id='RoboTrick/vlm_traj_rollouts_viewer', repo_type='space')"
```
