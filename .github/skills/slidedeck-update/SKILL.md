---
name: slidedeck-update
description: Add or update slides (videos/images) in the docs slide decks docs/slidedeck_robotic_manipulation_he.html / _en.html, including converting rollout videos with ffmpeg so they play in browsers. Use ONLY when the user asks to update slides / the slide deck / presentation.
---

# Slide deck update

Decks: `docs/slidedeck_robotic_manipulation_he.html` (Hebrew, RTL) and `docs/slidedeck_robotic_manipulation_en.html`.
Videos live in `docs/assets/<topic>/` (VLM agent experiments: `docs/assets/vlm/`) and are referenced relatively: `assets/vlm/<file>.mp4`.

## 1. Pick and name the video
- Rollout videos: `outputs/<date>/<rollout>/videos/rgb_image_full.mp4` (head cam) or `wrist_image_full.mp4`.
- Name: `<task>_<what>_<sim>.mp4`, lower snake case, include the task name and the simulator (e.g. `kettle_mug_pour_genesis.mp4`).

## 2. Convert for browsers (always)
Sim videos are often MPEG-4 Part 2 (`mpeg4`), which Chrome/Firefox/Edge will not play. Re-encode to H.264 + yuv420p, even dimensions, `faststart` (metadata at the front so it streams), no audio:

```powershell
ffmpeg -y -v error -i <src.mp4> -c:v libx264 -pix_fmt yuv420p -profile:v high -crf 23 -preset slow `
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -movflags +faststart -an docs\assets\vlm\<name>.mp4
ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height:format=duration,size -of compact docs\assets\vlm\<name>.mp4
```
Expect `codec_name=h264`, `pix_fmt=yuv420p`. ffmpeg is in the `vlm_traj` conda env (`...\envs\vlm_traj\Library\bin\ffmpeg.exe`).
Optional: slow a short clip with `-vf "setpts=2*PTS,scale=..."`; trim with `-ss <s> -t <s>` before `-i`.

## 3. Add the slide
Copy the structure of an existing video slide (e.g. "ניסויים VLM Agent Long Horizon"):

```html
<!-- ════ שקף · <title> ════ -->
<div class="slide bg-12">
  <span class="sn"></span>
  <div class="wrap" style="gap:2vh;">
    <h2>💧 <span class="gt"><title></span></h2>
    <div class="vrow">
      <div class="vbox" style="max-width:60vw; margin:0 auto;">
        <div>
          <div class="tags" style="margin-bottom:.8vh;"><span class="tag tp">VLM Agent — Zero-Shot - Sim</span></div>
          <h3><Hebrew subtitle></h3>
          <p><command / one-line result></p>
        </div>
        <video controls autoplay muted loop preload="metadata">
          <source src="assets/vlm/<name>.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </div>
</div>
```
- Slides are `div.slide` in document order (JS collects `.slide`); the `.sn` number is static text: leave it empty or renumber all.
- `autoplay` needs `muted`.
- Keep text short: a title, a one-line subtitle, one result line.

## 4. Verify and commit
- Open the deck in a browser and check the video plays (the source must be H.264).
- `git add docs/assets/... docs/slidedeck_*.html` and commit.
