---
title: VLM Trajectory Rollouts Viewer
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 43200
hf_oauth_scopes:
  - read-repos
hf_oauth_authorized_org: RoboTrick
---

A static, browser-only viewer for the private dataset `RoboTrick/vlm_traj_rollouts`.
It shows the summary, head and wrist videos, per-attempt images and the log.

- No token needed. The page signs you in with your Hugging Face account (OAuth, `read-repos`) as soon as it opens. The first time, approve the app and keep the RoboTrick organization ticked. After that the redirect is silent, and the login lasts 30 days.
- Only RoboTrick members can sign in. Pasting a read token is still available as a fallback.
- A browser's huggingface.co cookie cannot be used directly: the page runs on `*.hf.space`, so it needs the OAuth access token.
- Static Space because Gradio / Docker Spaces need a paid HF plan.

Deploy or update from the repo root:
```
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='research/hf_rollout_viewer', repo_id='RoboTrick/vlm_traj_rollouts_viewer', repo_type='space')"
```
