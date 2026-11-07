# Plan: Semantic navigation ("go to the kitchen"), indoor and outdoor, with or without a map

Status: proposal (2026-11-08). Depends on [mobile_base.md](mobile_base.md) (holonomic base + `MOVE_BASE`).
Research background:
- [../models/navigation.md](../models/navigation.md): VLN, ObjectNav, nav foundation models, agentic navigation.
- [../models/slam_mapping.md](../models/slam_mapping.md): Localizer and map layers.
- [../models/mobile_manip.md](../models/mobile_manip.md): nav→manip hand-off and simulators.

---

## 1. Goals

| # | Goal | Measured by |
|---|---|---|
| N-G1 | **Point / pose goals**: "go to (x, y)" or "drive 3 m to the door" with an online occupancy map, A* and the guarded base | sim: SR ≥ 95 % on 20 random reachable goals in a 3-room house, 0 collisions |
| N-G2 | **Object goals without a map** ("find the mug"). Explore frontiers scored by a VLM / CLIP value map plus LLM priors, detect the target, approach it and verify | our `nav:` suite: SR ≥ 60 %, SPL reported. Within ~10 SR points of VLFM-style numbers when run on Habitat ObjectNav HM3D (stretch goal) |
| N-G3 | **Room / region goals** ("go to the kitchen"). This works without a map (commonsense room priors) and with a map (a room node in the semantic map / scene graph) | `nav:` room-goal episodes: SR ≥ 70 % without a map and ≥ 90 % with a saved map |
| N-G4 | **Prior-tour mode**: a recorded tour (video or frames, optional spoken labels) becomes a topological + semantic map. "Take me where I can wash dishes" resolves to a node, and the robot drives there | Mobility-VLA-style episodes: SR ≥ 80 % on the tour scene |
| N-G5 | Fits the existing loop: the planner turns the instruction into nav and manipulation subtasks, the LLM calls the nav tools, and the **VLM reviewer confirms arrival** | e2e "go to the kitchen and pick the red cup" succeeds in Genesis |
| N-G6 | **Sim2Real-clean**: only pixels, depth, odometry, intrinsics/extrinsics and the instruction. The ground-truth pose, room labels and object poses are used only for evaluation | a test asserts the nav prompts and tools never read `get_state()` / scene YAML eval fields |
| N-G7 | **Outdoor (later)**: GPS/compass or visual waypoint goals plus landmark goals ("go to the red door of building B") with a pixel-only local policy | milestone N7 demo |
| N-G8 | Pluggable specialist policies (VLFM, NoMaD/ViNT, InternVLA-N1, StreamVLN) behind one tool, served via `gpu_server` for ablations | `--nav-policy X` switches the policy without code changes |

## 2. Non-goals

- Training our own navigation foundation model in v1. We may fine-tune later with the
  OWMM-style data synthesis described in mobile_manip.md.
- Writing a SLAM system. We integrate one: odometry in sim, RTAB-Map / ORB-SLAM3 on the real robot.
- Multi-floor navigation, stairs or elevators, and opening doors *during* navigation in v1.
  A closed door ends navigation with `reason="blocked: door"`, and the planner may call the door skill.
- Dynamic crowds, high-speed outdoor driving, and GPS-RTK.
- Benchmark-chasing in Habitat as the main target. Habitat is used for *evaluation* only.

## 3. High-level approach

This is a hybrid design that is agentic and zero-shot by default. It is the same design that
SuperNav (2026) shows beats fine-tuned navigation MLLMs: *a general VLM decides, and navigation
tools execute.* Training-free harnesses also lead the 2026 multi-goal boards: NavHarness and
STEGNav on GOAT-Bench, and Fast-Slow ReAct on zero-shot HM3D. See
[navigation.md §4.1](../models/navigation.md). The design also matches our existing
planner → LLM-code → reviewer stack.

```
instruction ─► Planner (LLM) ─► subtasks: [nav: "go to kitchen"] [manip: "pick red cup"]
                                   │
                                   ▼
                   LLM code (exec) calls nav tools          ◄── skill: semantic-navigation
                   navigate_to("kitchen") / explore("mug") / look_around() / where_am_i()
                                   │
                ┌──────────────────┴─────────────────────────────── API process ─────────┐
                │ Navigator (nav/navigator.py)                                             │
                │  ├ GoalParser (LLM)      text ─► {type: room|object|landmark|xy, cues}    │
                │  ├ Localizer             odom | rtabmap | orbslam3     ─► T_odom_base     │
                │  ├ OccupancyMap          depth ─► 2D grid (5 cm), frontiers               │
                │  ├ SemanticMap           seg_provider + CLIP ─► objects ─► rooms          │
                │  ├ TopoGraph (tour)      keyframes + embeddings + edges                    │
                │  ├ GoalSelector          map hit? ─► A* goal | else frontier scoring:       │
                │  │                        VLM/CLIP value map (VLFM) + LLM room priors (ESC) │
                │  ├ PathPlanner           A* + smoothing ─► waypoints                       │
                │  ├ NavPolicy (optional)  vlfm | nomad | internvla_n1 | streamvln (gpu_server)│
                │  └ ArrivalVerifier       VLM: "is this the kitchen?" (reviewer model)       │
                └──────────────┬──────────────────────────────────────────────────────────┘
                               │ IPC: GET_NAV_OBS, FOLLOW_PATH, MOVE_BASE (Plan A)
                               ▼
                    Sim process (env.py): BaseController (guarded) + head RGB-D camera
```

**Decisions and their rationale** (details in the research docs):
1. **Map layers are separate from the decision maker.** Occupancy and frontiers handle
   geometry. A semantic object/room map plus a topological tour graph handle meaning. The LLM
   reads *text summaries and one top-down map image*. It never reads raw grids. Sources:
   VLMaps, HOV-SG, TopV-Nav, MapGPT.
2. **No-map mode = VLFM + ESC/L3MVN + ApexNav fallback.**
   - Frontiers are scored by a cheap CLIP/SigLIP value map for every frame, plus LLM room/object
     priors at the keyframes.
   - When the semantic signal is flat, scoring falls back to geometric frontiers (ApexNav).
3. **Map mode = HOV-SG-style room/object graph + Mobility-VLA goal-frame retrieval.**
   - Text maps to candidate nodes by embedding similarity.
   - The VLM then reranks the candidates' keyframes (long-context, top-k).
   - Motion uses A* on the saved occupancy map.
4. **Visual-point interface.**
   - The VLM may answer with a *pixel* ("go there"). It is lifted with depth through the existing
     `convert_2d_point_to_3d_world` and snapped to the nearest free cell.
   - This is the same mechanism as in SuperNav, PIVOT, VLMnav, CoNVOI and the InternVLA-N1 System 2.
5. **Specialist models are optional tools, not the backbone.**
   - They run under `--nav-policy` behind the `NavPolicy` interface.
   - This keeps the zero-shot stack working on Windows with no extra GPUs.
6. **Simulator.**
   - Development and demos run in **Genesis** (Windows, existing stack): a procedural
     3–4-room house, plus HSSD / ProcTHOR GLB houses imported as static meshes.
   - Standard benchmark numbers come from a separate **Habitat runner on the Linux GPU box**
     (`nav_bench/`). It reuses `nav/` unchanged through an observation adapter.
   - PyBullet only hosts the unit-level point-goal tests.
   - The R4 comparison is in mobile_manip.md §7.

## 4. Pipeline (inputs → preprocessing → models/algorithms → outputs)

### Step 1: Inputs
| Input | Type / units / frame | Rate | Producer → consumer | Failure → fallback |
|---|---|---|---|---|
| Instruction / subtask text | str | per subtask | user → planner → LLM code → `navigate_to(text)` | ambiguous → the GoalParser returns `clarify` and the planner asks or picks the nearest |
| Head RGB | H×W×3 uint8 (640×480 default) | keyframes every `NAV_KEYFRAME_M` (0.5 m) or 30° of yaw, and at every replan | env `GET_NAV_OBS` → Navigator | blank or occluded frame → skip the map update |
| Head depth | H×W float32, metres | same | same | invalid pixels (0, NaN, > `NAV_MAX_DEPTH` = 5 m) are masked |
| Intrinsics K, extrinsics `T_odom_cam` | 3×3, 4×4 (odometry frame from Plan A) | same | env (camera config + odometry) | none |
| Odometry pose | (x, y, yaw), m / rad, odometry frame, plus a covariance | every `GET_NAV_OBS` | `BaseController.odom_pose()` → Localizer | drift → the Localizer corrects it (real: RTAB-Map) or flags `lost` |
| Robot mask | H×W bool (pixels of our own arm / base) | same | env (renders the robot-only segmentation via FK. Allowed: the robot knows its own geometry) | missing → apply the arm-stow rule |
| Prior tour (optional) | video file or frame folder + optional `labels.json` `[{t, "kitchen"}]` or a transcript | offline, once | user → `nav/tour.py` | no labels → VLM auto-captions the keyframes |
| Saved map (optional) | `map_dir/{occupancy.npy, meta.json, semantic.json, topo.json, keyframes/}` | load once | previous run (`--nav-map saved`) | version mismatch → rebuild |
| Outdoor (N7) | GPS (lat, lon), compass yaw | 1–10 Hz | real sensors / sim noise model | no fix → visual odometry only |

### Step 2: Preprocessing
1. **Localization.**
   - `Localizer.update(obs) -> Pose2D(x, y, yaw), cov, lost`.
   - In sim v1 this is `OdomLocalizer`: it passes odometry through and grows the covariance with
     distance travelled.
   - On the real robot, `RtabmapLocalizer` subscribes to `/rtabmap/localization_pose` over ROS2.
   - It is a stub until N5.
2. **Depth → points → occupancy.**
   - Back-project the valid depth (stride 4) with K into the camera frame, then transform with
     `T_odom_cam` into the odometry frame.
   - Drop robot-mask pixels.
   - Points with `z ∈ [NAV_Z_MIN = 0.05, NAV_Z_MAX = robot_height + 0.1]` are obstacles. Points
     with `z < 0.05` are floor.
   - Ray-cast free cells from the camera to each hit (Bresenham on the 2D grid).
   - Log-odds update on a `NAV_CELL_M = 0.05` grid that grows dynamically (start 20×20 m).
   - Inflate obstacles by base radius + `NAV_INFLATE_M` (0.05) for planning.
3. **Frontier extraction.**
   - Frontier cells are free cells with an unknown 4-neighbour.
   - Cluster them by connected components and drop clusters smaller than `NAV_FRONTIER_MIN_CELLS` (10).
   - Each frontier is represented by the reachable free cell nearest its centroid.
4. **Open-vocabulary perception per keyframe.**
   - Run the existing `segmentation_adapter` (`--seg-provider` = langsam, sam3 or moondream) with a
     *goal-conditioned* prompt list: the goal noun, cue objects from the GoalParser (e.g. kitchen →
     fridge, stove, sink, oven, kitchen counter), and the generic set `NAV_GENERIC_OBJECTS`.
   - Each mask gives a median depth point, then a 3D centroid, then a `SemanticMap` object:
     label, score, centroid, CLIP embedding of the crop, keyframe id and count.
   - Objects are merged by label + distance < 0.5 m + embedding cosine > 0.8 (ConceptGraphs-style).
5. **Image embedding.**
   - Compute a SigLIP/CLIP embedding of the whole keyframe.
   - It goes to the TopoGraph node and to the value map: the cosine to the text "a photo of a
     {goal}" or "a {room}" is splatted into the visible FOV cone with confidence weighting by
     angle from the optical axis (VLFM).
   - The CLIP model runs locally (CPU OK, ~50 ms) or via `gpu_server` module `clip`.
6. **Room inference (N3).**
   - Segment free space into regions: a distance-transform watershed on the occupancy grid, the way
     Hydra/HOV-SG partition rooms.
   - Each region is labelled by an LLM call over its object list (cached; recomputed only when the
     object set changes), with the VLM caption of the best keyframe as a tie-breaker.
   - Output: `rooms[{id, label, conf, cells, objects, keyframes}]`.
7. **Tour ingestion (N4, offline, `nav/tour.py`).**
   - Subsample the video to keyframes (by motion and embedding novelty, ~1 per 0.5 m).
   - Poses come from the robot odometry recorded with the tour. A phone video has no odometry, so
     it uses VGGT-SLAM / MASt3R-SLAM poses with scale from known camera height (research option).
   - Then build the TopoGraph: an edge exists if the poses are within 1.5 m and the occupancy line
     is free.
   - Run steps 2–6 on all keyframes and save `map_dir/`.

### Step 3: Models / algorithms
a. **GoalParser** (LLM, one call per `navigate_to`).
   - The input is the goal text plus a short map summary (rooms and objects seen so far).
   - The output is JSON:
   `{"type":"room|object|landmark|xy|relative","target":"kitchen","cues":["fridge","stove","sink"],"near":[...],"avoid":[...],"stop_condition":"inside room" | "within 1.0 m facing target"}`.
   - Relative goals ("2 m to my left") go straight to `move_base`.
b. **Map lookup** (map exists or partly built).
   - Query `SemanticMap.query(target, cues)` and `TopoGraph.query(text, k=5)`.
   - The query uses a text-embedding cosine plus a label match.
   - If the best score is ≥ `NAV_MAP_HIT_THRESH`, the VLM reranks the top-k keyframes (one
     multi-image call: "which image shows the kitchen? answer index or none"). This is the
     Mobility VLA approach.
   - If a node is accepted, the goal is that node's pose, or the room's centroid cell. Go to (d).
c. **Exploration** (no hit yet), repeated until found or the budget is exhausted.
   - Score each frontier f:
     `S(f) = w_v·value_map(f) + w_p·prior(f) + w_d·exp(−dist(f)/λ) − w_r·revisit(f)`.
     - `value_map`: the VLFM-style cosine map.
     - `prior`: the LLM commonsense score, computed at keyframes only. The LLM sees the frontier's
       neighbouring objects and room label as text, plus the top-down map image with numbered
       frontiers (TopV-Nav / L3MVN), and returns a 0–1 score per frontier.
     - `revisit`: penalises frontiers already chosen or blacklisted (UniGoal).
   - **ApexNav switch**: if `std(value_map over frontiers) < NAV_FLAT_THRESH`, set `w_v = 0`
     (geometric exploration).
   - **Commitment**: keep the current frontier until it is reached, disappears, or a new frontier
     beats it by `NAV_SWITCH_MARGIN` (VLFM). This prevents oscillation.
   - **Optional `--nav-policy`**:
     - `vlfm` replaces the scoring with the official VLFM value map.
     - `nomad` runs NoMaD in exploration mode (goal-masked) for local waypoints when depth is bad
       (outdoors).
     - `internvla_n1` / `streamvln` take the instruction plus RGB history and return a pixel goal
       or waypoints. These are snapped to free cells and still go through our guard.
d. **Approach-point selection.**
   - For an object goal: sample free cells on a ring of radius `NAV_STANDOFF_M` (0.7 m, or the
     reach-map score when a manipulation subtask follows; mobile_manip.md §9.3) around the
     centroid.
   - Keep cells with a line of sight to the object.
   - Choose the one with the shortest path. The yaw faces the object.
   - For a room goal: choose the free cell nearest the room's centroid that is ≥ 0.5 m from
     obstacles, facing the room's highest-scoring object.
e. **Path planning and following.**
   - A* on the inflated grid (8-connected, cost = distance + `NAV_CLEARANCE_W`/clearance).
   - Ramer-Douglas-Peucker smoothing, then waypoints every ≤ 1 m.
   - The waypoints go to the env via `FOLLOW_PATH`, which runs pure pursuit on the holonomic base:
     it drives x/y directly and turns toward the next waypoint so the head camera keeps facing
     forward.
   - The Plan A guard stays active.
   - On `stopped_by=depth|contact`: mark the cells as an obstacle, replan (max `NAV_MAX_REPLANS` = 5),
     then report `blocked`.
   - Replan every `NAV_REPLAN_M` (1.0 m) travelled with fresh keyframes.
f. **Detection while exploring.**
   - On every keyframe the SemanticMap is updated.
   - If the goal label appears with score ≥ `NAV_DETECT_THRESH` on ≥ 2 keyframes (target-centric
     fusion, ApexNav), switch from exploring to approaching (d).
   - A single detection only triggers a `look_at` re-perception.
g. **Arrival verification** (one call).
   - The reviewer VLM gets the head image (plus optionally a 4-view `look_around` strip) and the
     question "Is the robot now in/at <goal>? Answer yes/no + reason".
   - "No" blacklists the node/object, and exploration resumes (SG-Nav re-perception and UniGoal
     blacklist).
   - The budget is `NAV_MAX_VERIFY` (3) per goal.
h. **Termination.**
   - Success: the verifier says yes.
   - Failure:
     - Step budget `--nav-max-steps` (keyframes) or time budget `NAV_TIMEOUT_S`.
     - No frontiers left and not found → `not_found`.
     - The Localizer reports `lost` → `lost`.

### Step 4: Outputs
- **Tool result to the LLM:**
  ```python
  {"ok": bool, "reason": "arrived|not_found|blocked|blocked: door|lost|budget",
   "goal": {"type": "room", "target": "kitchen"}, "goal_pose_odom": [x, y, yaw_deg],
   "verified": bool, "verifier_reason": str, "path_len_m": float, "keyframes": int,
   "replans": int, "evidence_image": "<path>", "map_summary": "rooms: kitchen(0.86), hallway; objects: fridge#1 ..."}
  ```
- **Map summary text** (`get_map_summary()`), at most ~40 lines.
  - Rooms with their labels and confidence, top objects per room, and unexplored-frontier counts
    per direction.
  - Odometry coordinates rounded to 0.1 m.
  - No ground truth.
- **Map image**: `images/nav/map_<step>.png`, a top-down image showing occupancy, frontiers
  (numbered), rooms (coloured), objects, the robot pose and the path.
  - It is used by the VLM (TopV-Nav style) and by the video.
- **Persisted map**: `<rollout>/nav_map/` (occupancy.npy, meta.json, semantic.json, topo.json,
  keyframes/*.jpg). It is reusable with `--nav-map saved --nav-map-path`.
- **Eval metrics** (logs and summary.md only): SR, SPL, distance to goal, path length,
  collisions, # VLM calls, wall time, ATE (odometry vs ground truth).

## 5. Code-level changes and integration points

### 5.1 New package `nav/` (pure Python, numpy / scipy / opencv; no sim imports)
| File | Classes / functions |
|---|---|
| `nav/types.py` | `@dataclass Pose2D(x, y, yaw)`, `NavObs(rgb, depth, K, T_odom_cam, pose, cov, robot_mask, t)`, `NavGoal(type, target, cues, near, avoid, stop_condition)`, `NavResult` (the dict schema above) |
| `nav/occupancy.py` | `class OccupancyMap(cell=0.05, z_band=(0.05, 1.4))`: `update(obs)`, `inflated(radius)`, `is_free(xy)`, `nearest_free(xy)`, `to_image()`, `save(dir)` / `load(dir)` |
| `nav/frontier.py` | `extract_frontiers(occ, min_cells) -> list[Frontier(id, cells, centroid, approach_xy)]` |
| `nav/value_map.py` | `class ValueMap(occ_shape)`: `update(obs, score: float, fov_deg)` (VLFM cone splat with confidence), `at(xy)`, `flatness(frontiers)`. `class TextImageScorer(provider="clip"/"siglip"/"gpu_server")`: `score(rgb, texts) -> list[float]` |
| `nav/semantic_map.py` | `class SemanticMap`: `add_detections(obs, masks, labels, scores)`, `merge()`, `query(text, cues, k) -> list[Obj]`, `objects_near(xy, r)`, `summary()`, `save/load` |
| `nav/rooms.py` | `segment_rooms(occ) -> regions` (distance-transform watershed), `label_rooms(regions, semantic_map, llm) -> rooms` (cached) |
| `nav/topo_graph.py` | `class TopoGraph`: `add_keyframe(id, pose, emb, caption)`, `connect(occ)`, `query(text, k)`, `shortest_path(a, b)`, `localize(emb) -> node` (visual relocalization), `save/load` |
| `nav/tour.py` | `build_map_from_tour(frames_or_video, poses_or_None, labels_json, out_dir, perception)`. CLI: `python -m nav.tour <video> --out map_dir` |
| `nav/path_planner.py` | `astar(grid, start, goal, clearance_w) -> cells`, `smooth(cells) -> waypoints`, `choose_approach(occ, target_xy, standoff, los=True) -> Pose2D` |
| `nav/localizer.py` | `class Localizer(ABC)`: `update(obs) -> (Pose2D, cov, lost)`. `OdomLocalizer`; `RtabmapLocalizer` (ROS2 stub); `OrbSlam3Localizer` (stub); factory `make_localizer(name)` |
| `nav/goal_parser.py` | `parse_goal(text, map_summary, llm) -> NavGoal` (JSON-mode prompt, 1 retry, logged) |
| `nav/frontier_scorer.py` | `score_frontiers(frontiers, value_map, semantic_map, rooms, goal, llm, map_image) -> list[(id, S, terms)]`, with ApexNav flat-switch and VLFM commitment |
| `nav/verifier.py` | `verify_arrival(goal, images, vlm) -> (bool, reason)`. It reuses the reviewer model/provider from `api.run_vlm_review` |
| `nav/navigator.py` | `class Navigator(obs_fn, follow_fn, move_fn, perception, llm, vlm, localizer, policy, cfg, logger)`: `navigate_to(text) -> NavResult`, `explore(goal_text, max_steps)`, `look_around(n=4)`, `find(obj)`, `where_am_i()`, `map_summary()`, `render_map(path)`, `save_map(dir)` |
| `nav/render.py` | top-down map rendering (numbered frontiers / rooms / objects / path) for the VLM and the video |

### 5.2 Pluggable specialist policies
| File | Change |
|---|---|
| `providers/nav_policy/__init__.py`, `factory.py` | `make_nav_policy(name, args) -> NavPolicy`, the same pattern as `providers/trackers/factory.py` |
| `providers/nav_policy/base.py` | `class NavPolicy(ABC)`: `reset(goal_text)`, `act(obs_history) -> {"pixel_goal": (u, v)} or {"waypoints_base": [[x, y], ...]} or {"stop": True}` |
| `providers/nav_policy/vlm_frontier.py` | default: wraps `frontier_scorer` (no extra model) |
| `providers/nav_policy/vlfm.py` | official VLFM value map (BLIP-2 ITM) via a `gpu_server` module `vlfm` |
| `providers/nav_policy/nomad.py` | NoMaD/ViNT (visualnav-transformer, MIT) via `gpu_server` module `nomad`. Goal-masked exploration and image-goal mode (tour keyframes) |
| `providers/nav_policy/internvla_n1.py` | InternVLA-N1 (InternNav, MIT) via `gpu_server` module `internvla_n1`. Returns a pixel goal and/or waypoints |
| `providers/nav_policy/streamvln.py` | StreamVLN (check its license first) |
| `gpu_server/` | register the new modules in the existing `/v1/modules/{name}/load` registry. Same `protocol.py` message format (images as JPEG bytes) |

### 5.3 Modifications to existing files
| File | Change | Called from |
|---|---|---|
| `config.py` | IPC `GET_NAV_OBS = 33`, `FOLLOW_PATH = 34`, `NAV_EVAL = 35` (eval only, see §6). `NAV_*` constants (§6.2) | `env.py`, `api.py` |
| `env.py` | command loop (next to `MOVE_BASE`): **`GET_NAV_OBS`** → `{rgb, depth, K, T_odom_cam, odom_pose, robot_mask, t}` in one atomic message. **`FOLLOW_PATH`** → `robot.base.follow_path(waypoints, max_speed, guard=True, step_fn=...)` (pure pursuit, new in `mobile/base_controller.py`). **`NAV_EVAL`** → ground-truth distance / geodesic-to-goal for the logger only. It is refused unless `args.nav_eval` is set, and it is never routed to `get_exec_locals`. The scene loader loads house-scale scenes from `sim_envs/nav/scenes/*.yaml` | the API process |
| `mobile/base_controller.py` (Plan A) | add `follow_path(waypoints_odom, max_speed, guard, step_fn) -> dict` (pure pursuit with lookahead `NAV_LOOKAHEAD_M`, holonomic: the velocity is aimed at the lookahead point and yaw → path tangent) | `env.py` `FOLLOW_PATH` |
| `api.py` | `API.__init__`: if `args.nav`, build `self.navigator = Navigator(obs_fn=self._nav_obs, follow_fn=self._follow_path, move_fn=self.move_base, perception=self.seg_adapter, llm=self._llm_call, vlm=self._vlm_call, ...)`. New tools: `navigate_to(target: str)`, `explore(goal_text: str, max_steps=None)`, `look_around(n=4)`, `find(obj: str)`, `where_am_i()`, `get_map_summary()`, `save_map(path=None)`. Each goes through `_budget_tick("nav", ...)`, logs, and appends to `self.task.nav_calls`. `get_robot_state()` adds `nav: {room: "hallway", map_rooms: n}` | LLM code via `get_exec_locals` |
| `helpers/main_utils.py:get_exec_locals` | expose the nav tools only when `api.args.nav` (default off, so goldens are unchanged) | `main.py` exec |
| `planner_api.py` / `prompts/planner_prompt.py` | when nav is on, the planner may emit `{"type": "navigate", "goal": "kitchen"}` subtasks before manipulation subtasks. `PlannerAPI.get_planner_exec_locals` gets `get_map_summary` (read-only) | `planner_api.PlannerAPI.execute_subtasks` |
| `prompts/main_prompt.py` | when nav is on, add a ~8-line block: the tool list + "use `navigate_to` for anything beyond arm reach; after arriving, re-detect objects before manipulating" | `main.py` prompt build |
| `prompts/review_prompt.py`, `api.run_vlm_review` | nav subtasks: evidence = goal + `NavResult` + before/after head images + the map image. The success question is "is the robot in/at <goal>" | review step |
| `task_state.py` | `nav_calls: list`, `nav_map_dir: str` | `api.py` |
| `segmentation_adapter.py` | no API change. The Navigator calls `segment(rgb, labels)` with batched labels. Add a `max_labels` guard and a per-keyframe timing log | `nav/semantic_map.py` |
| `sim_envs/nav/` (new) | `scenes/house3_procedural.yaml` (rooms built from boxes: kitchen with fridge/stove/sink meshes, living room, bedroom, hallway), `scenes/hssd_<id>.yaml` (GLB import), `scene_loader.py` (Genesis / PyBullet static mesh + collision), `tasks.py` (episodes: start pose, instruction, **eval-only** goal region) | `sim_envs/registry.py` |
| `sim_envs/registry.py` | suite `"nav"` (lazy): `nav:point_goal`, `nav:find_object`, `nav:go_to_room`, `nav:tour_go_to`, `nav:go_and_pick`. `_apply_required_robot` → `franka_mobile` | `main.py` task resolution |
| `nav_bench/habitat_runner.py` (new, Linux only) | runs `Navigator` on Habitat ObjectNav (HM3D-Semantics v2) / HM3D-OVON episodes through a `HabitatObsAdapter` (Habitat obs → `NavObs`; discrete actions ← waypoints). Writes the SR/SPL json | manual eval on the GPU box |
| `helpers/rollout_report.py` | adds SR, SPL, path length, # replans, # VLM calls and ATE to `summary.md`. HF `metadata.jsonl` `metric` = `"SR=..;SPL=.."`. Copies `images/nav/*.png` and the `nav_map/` folder | packaging |
| `document.md` | new "Semantic navigation" section + flags + changelog | — |
| `requirements*.txt` | `scikit-image` (watershed) if not already present; optional `open_clip_torch` | — |

### 5.4 Example LLM code (what the skill teaches)
```python
r = navigate_to("kitchen")
if not r["ok"] and r["reason"] == "not_found":
    r = explore("kitchen", max_steps=60)
if r["ok"]:
    look_around(4)
    cup = detect_object("red cup")      # re-detect after driving (odometry frame)
    ...                                 # existing manipulation code
```

## 6. New CLI arguments and configuration

### 6.1 `main.py`: new argparse group `navigation`
| Flag | Type | Default | Help |
|---|---|---|---|
| `--nav / --no-nav` | BooleanOptional | `False` (True for `nav:*` tasks) | enable the navigation tools, prompts and the Navigator. Requires `--mobile-base` |
| `--nav-map` | `{none, tour, saved}` | `none` | `none` = explore from scratch; `tour` = build from `--nav-tour` first; `saved` = load `--nav-map-path` |
| `--nav-map-path` | path | `None` | map dir to load (saved) or to write (tour) |
| `--nav-tour` | path | `None` | tour video or frame folder (+ optional `labels.json`) |
| `--nav-policy` | `{vlm_frontier, vlfm, nomad, internvla_n1, streamvln}` | `vlm_frontier` | goal-selection / local policy backend |
| `--nav-localizer` | `{odom, rtabmap, orbslam3}` | `odom` | pose source (real-robot ones are stubs until N5) |
| `--nav-value-model` | `{clip, siglip, none}` | `siglip` | per-frame text-image scorer for the value map |
| `--nav-max-steps` | int | `200` | keyframe budget per `navigate_to` / `explore` |
| `--nav-success-dist` | float m | `1.0` | eval success radius (object goals). Room goals = inside the region |
| `--nav-standoff` | float m | `0.7` | approach distance to object goals |
| `--nav-vis-map` | flag | off | save the top-down map image at every replan + add it to the video |
| `--nav-eval / --no-nav-eval` | BooleanOptional | True for `nav:*` tasks | allow the eval-only `NAV_EVAL` IPC (ground truth for metrics; never shown to the agent) |
| `--nav-debug-log` | flag | off | temporary per-keyframe logs (frontier scores, value-map stats) |

Validation in `main.py`: `--nav` without `--mobile-base` → error. `--nav-map tour` without
`--nav-tour` → error. `--nav-policy internvla_n1|nomad|vlfm|streamvln` → check that the
gpu_server module is reachable at startup, and otherwise fall back to `vlm_frontier` with a
warning.

### 6.2 `config.py`
```
GET_NAV_OBS = 33; FOLLOW_PATH = 34; NAV_EVAL = 35
NAV_CELL_M = 0.05; NAV_Z_MIN = 0.05; NAV_Z_MAX = 1.5; NAV_MAX_DEPTH = 5.0; NAV_INFLATE_M = 0.05
NAV_KEYFRAME_M = 0.5; NAV_KEYFRAME_DEG = 30; NAV_REPLAN_M = 1.0; NAV_MAX_REPLANS = 5
NAV_FRONTIER_MIN_CELLS = 10; NAV_SWITCH_MARGIN = 0.15; NAV_FLAT_THRESH = 0.02
NAV_SCORE_W = {"value": 1.0, "prior": 1.0, "dist": 0.3, "revisit": 0.5}
NAV_DETECT_THRESH = 0.35; NAV_MAP_HIT_THRESH = 0.28; NAV_MAX_VERIFY = 3; NAV_TIMEOUT_S = 600
NAV_LOOKAHEAD_M = 0.4; NAV_CLEARANCE_W = 0.2
NAV_GENERIC_OBJECTS = ["door", "table", "chair", "sofa", "bed", "fridge", "sink", "stove", "toilet", "tv", "counter"]
```

### 6.3 Scene / episode YAML (`sim_envs/nav/scenes/*.yaml`)
```yaml
scene: house3_procedural          # or hssd_102344250
asset: assets/scenes/house3/house.glb   # optional; procedural if omitted
scale: 1.0
robot_start: {x: 0.5, y: 0.5, yaw_deg: 0}     # scene setup only
episodes:
  - id: kitchen_from_bedroom
    instruction: "go to the kitchen"
    start: {x: 6.0, y: 1.0, yaw_deg: 90}
    eval: {type: room, region: [[3,4],[6,4],[6,8],[3,8]]}      # EVAL ONLY - never read by nav/
  - id: find_mug
    instruction: "find the blue mug"
    eval: {type: object, object_id: mug_1, success_dist: 1.0}  # EVAL ONLY
```
A unit test asserts that `nav/` never imports the scene loader and that no prompt contains `eval:` values.

## 7. Prompts and skills

- **New skill `prompts/skills/semantic-navigation/SKILL.md`:**
  - when to navigate vs. move the arm; the tool signatures and result schema;
  - the frames (odometry; `where_am_i`);
  - patterns: "navigate → look_around → re-detect → manipulate", and "not_found → explore with a broader goal or ask";
  - failure handling: blocked → try `explore` with `avoid`; door → hand off to the door skill;
  - **never assume coordinates of unseen places**.
- **Planner prompt**: the subtask type `navigate`, a map-summary injection (text), and the rule
  "one navigate subtask per location change".
- **GoalParser prompt** (`nav/prompts/goal_parser.txt`): JSON schema + 6 examples (room, object,
  landmark, relative, demand "I'm thirsty" → kitchen/fridge, ambiguous).
- **Frontier-prior prompt** (`nav/prompts/frontier_prior.txt`):
  - the goal, a text block per numbered frontier (nearby objects and room label), and the map image;
  - it returns `{frontier_id: score}`;
  - it must not invent objects that are not listed.
- **Verifier prompt** (`nav/prompts/verify_arrival.txt`): yes/no + reason + a "what would make it
  certain" hint (e.g. "turn left to see the stove").
- **Reviewer prompt**: nav evidence block (see §5.3).

## 8. Sim2Real and the no-privileged-information rules

- The Navigator only consumes `NavObs`: RGB, depth, K, `T_odom_cam` from **odometry**, the robot
  mask (from FK), and the text.
- It never reads `simenv.get_state()`, scene YAML `eval:` fields, semantic labels from the sim
  renderer, room polygons, or object poses.
- Room labels come from *our* perception + LLM. Using the sim's semantic segmentation is forbidden,
  even for "debugging" inside the agent loop.
- `NAV_EVAL` (ground-truth distance / geodesic) exists only for metrics. It is refused unless
  `--nav-eval` is set, and it is not reachable from `get_exec_locals`, the planner, the reviewer or
  `nav/`. It is enforced by a test.
- The map is in the **odometry frame**. Expected real errors are wheel drift (kiwi slip) plus the
  ~0.5–1 cm hand-eye error.
  - The design tolerates this: the TopoGraph is used for relocalization, and objects are
    re-detected after arrival.
  - Sim uses `--odom-noise` to test this.
- Real deployment:
  - head RGB-D (RealSense D435/D455) + wheel odometry → RTAB-Map localization (`--nav-localizer rtabmap`);
  - the same `nav/`;
  - `FOLLOW_PATH` → the LeKiwi / ROS2 `cmd_vel` bridge from Plan A.
- Outdoors (N7): RGB-D depth is unreliable in sunlight. Use stereo / monocular depth or the NoMaD
  / CityWalker pixel-only policy, plus GPS/compass from real sensors (simulated with noise).

## 9. Logging and observability

| Stage | Log (PROGRESS unless noted) |
|---|---|
| GoalParser | `[nav] goal text="go to the kitchen" -> {type:room, target:kitchen, cues:[fridge,stove,sink]} (1.2 s)` |
| Map lookup | `[nav] map hit kitchen room#2 conf=0.81 (rerank idx=3 of 5)` or `[nav] map miss -> explore` |
| Exploration step | `[nav] kf=17 pose=(3.1,2.0,90°) frontiers=6 pick=#3 S=0.74 (v=0.41 p=0.30 d=0.12 r=0) flat=False` |
| Detection | `[nav] detect fridge score=0.52 kf=17,18 -> approach (2.3, 4.1)` |
| Path | `[nav] path len=4.2 m wps=6 replan=1 stopped_by=None` |
| Verify | `[nav] verify kitchen -> yes ("fridge and stove visible")` |
| Result | `[nav] RESULT ok=True reason=arrived path=11.3 m kfs=24 vlm_calls=9 t=142 s` |
| Eval (logger only) | `[nav-eval] dist_to_goal=0.6 m success=True SPL=0.71 ATE=0.08 m` |
| Debug (`--nav-debug-log`, temporary) | per-frontier score terms, value-map histogram, occupancy update ms |

Artifacts:
- `images/nav/map_<kf>.png` (with `--nav-vis-map`);
- `nav_map/` in the rollout;
- the video gets a picture-in-picture map overlay.

## 10. Tests and evaluation

| Test | Kind | What it checks |
|---|---|---|
| `tests/nav/test_occupancy.py` | unit | synthetic depth of a wall at 2 m → occupied cells at x = 2 ± 0.05; ray-cast free cells; robot-mask pixels ignored |
| `tests/nav/test_frontier.py` | unit | known grid → expected frontier clusters; min-size filter |
| `tests/nav/test_astar.py` | unit | path around an obstacle; inflation respected; no path → None |
| `tests/nav/test_value_map.py` | unit | cone splat + confidence fusion; flatness metric |
| `tests/nav/test_semantic_map.py` | unit | merging duplicate detections; `query("kitchen", cues)` ranking |
| `tests/nav/test_topo_graph.py` | unit | edges only with free lines; `shortest_path`; save/load round trip |
| `tests/nav/test_goal_parser.py` | unit (mock LLM) | JSON parse, retry, `relative` goal shortcut |
| `tests/nav/test_no_privileged.py` | static + runtime | `nav/` imports no `sim_envs`/`env`; `NAV_EVAL` absent from `get_exec_locals`; prompts contain no `eval:` keys |
| `tests/test_nav_ipc.py` | IPC contract | `GET_NAV_OBS`, `FOLLOW_PATH` payloads |
| `tests/test_nav_pointgoal_pybullet.py` | sim | empty room + 2 boxes: point goal SR = 100 % over 10 seeds, 0 collisions |
| `tests/test_nav_genesis_house.py` | sim (skip if no Genesis) | `nav:go_to_room` kitchen from the bedroom, scripted (no LLM: GoalParser stub + mocked VLM verifier using CLIP) |
| `tests/test_pybullet_regression.py` | regression | unchanged goldens (nav off) |
| Eval: `nav:` suite | e2e, LLM | 3 scenes × {point, object, room, tour, go_and_pick} × 5 episodes; SR, SPL, VLM calls, time; packaged per repo convention + HF upload |
| Eval: Habitat ObjectNav HM3D-v2 val (subset of 200) | benchmark (Linux) | SR / SPL vs. VLFM / SG-Nav / ApexNav reported numbers (navigation.md) |
| Ablations | eval | `--nav-policy` × `--nav-map` × `--odom-noise {0, 0.02}` × `--nav-value-model` |

## 11. Milestones, risks and open questions

| N | Deliverable | Demo |
|---|---|---|
| N1 | `nav/occupancy`, `frontier`, `path_planner`, `localizer(odom)`, `GET_NAV_OBS`, `FOLLOW_PATH`, `navigate_to("x,y")` | point goals in a PyBullet room; map image |
| N2 | `value_map` + `semantic_map` + `frontier_scorer` + `verifier`; `explore("mug")` (no map) | find an object in the procedural Genesis house |
| N3 | `rooms` + GoalParser + LLM priors; `navigate_to("kitchen")` without a map; skill + planner integration | "go to the kitchen" from the bedroom, reviewer-verified |
| N4 | `topo_graph` + `tour.py` + saved maps (`--nav-map tour/saved`) | record a tour, then "take me where I can wash dishes" |
| N5 | `providers/nav_policy/*` via gpu_server (VLFM, NoMaD, InternVLA-N1) + `nav_bench/habitat_runner.py` | ablation table on Habitat HM3D subset |
| N6 | Nav + manipulation: `nav:go_and_pick` with reach-aware standoff (OWMM-style) | "bring the red cup from the kitchen to the table" |
| N7 | Outdoor: GPS/compass waypoints + landmark goals; NoMaD/CityWalker local policy; outdoor Genesis scene | "go to the red door across the courtyard" |

**Risks and open questions**
- **VLM latency and cost**: about 5–15 LLM/VLM calls per goal.
  - Mitigation: VLM calls only at keyframes and replans; per-frame scoring uses CLIP.
  - Cache the room labels. Budget with `_budget_tick`.
- **Genesis house-scale rendering speed** (an R4 warning): benchmark an HSSD GLB in N2.
  - If it is too slow, use the procedural low-poly house, with Habitat for the large-scale numbers.
- **Head camera FOV / height**: a single pitched-down camera sees little of a room.
  - `look_around()` compensates. A pan-tilt joint may be needed (decide in N2, together with Plan A M2).
- **Room segmentation quality** in open-plan homes. Fallback: object-cue clusters without hard
  polygons (an "area near the fridge").
- **Segmentation-provider recall** on small far objects: use goal-conditioned prompts plus
  re-perception from a closer viewpoint.
- **License**: ApexNav is GPL-3.0, so ideas only, no code. Check StreamVLN's license before
  integrating it. InternNav and visualnav-transformer are MIT.
- **Open question**: should the tour be robot-driven (has odometry) or phone-recorded (needs
  VGGT-SLAM / MASt3R-SLAM poses)? The v1 assumption is robot-driven.
