# Tracking → Agent Integration: Motion Semantics, Articulation and Failure Events

## Brief

The repo already has far more machinery than it looks like from `api.py` alone: `tracking/motion.py` is a complete, tested (62 tests) screw-theory toolkit — robust Kabsch rigid-fit, screw decomposition into hinge axis + angle, a `static|prismatic|revolute|free` classifier with hysteresis, plane fit, PCA line fit, sign-aligned EMA smoothing, and a principled confidence model. `tracking/kalman.py` already runs a per-object twist (screw-motion) Kalman filter that outputs `velocity`/`angular_velocity`/`rotation_deg` every frame. `tracking/monitors.py` already has a production-grade `attached_to_gripper` monitor (grace frames *and* seconds, FK-depth cross-check, tentative/predicted-aware) that is 80% of a "grasp-slip" detector. **None of this is wired together**: `tracking/session.py` never builds a `MotionBuffer`, `tracking/report.py`'s summary has no event timeline, and `prompts/skills/open-close-door-cabinet-drawer/SKILL.md` never calls `track_objects`/monitors at all. The single biggest gap, confirmed in the plan's own "still in flight" note (`docs/plans/3d_tracking.md:877-879`), is exactly this wiring.

My recommendation is therefore **not** a new estimation theory — `estimate_motion()` already *is* a principled per-window screw/BIC-style model-selector — but a **wiring, exposure and event-layer** project: (1) surface the per-point fused 3D cloud that `pose_tracker.py` already computes up into a per-object `MotionBuffer`; (2) add a secondary line/circle-fit estimator for the single genuinely new failure mode I found (a *vertical* handle bar whose points are collinear with a *parallel* hinge axis makes a true revolute motion look exactly prismatic to the Kabsch fit — real, and triggerable from `SKILL.md §2b`); (3) add 4 new small monitors (`grasp_slip_trend`, `door_stuck`, `angle_progress`, `wrong_object_moved`) built directly on fields that already exist (`pose.rotation_deg`, `lift_meta['fk_check']`); (4) add two new API calls (`get_track_features`, `estimate_articulation`) and a structured `events` list in `get_tracking_report()`; (5) extend the existing PyBullet door-scene GT harness (`tests/test_tracking_door_pybullet.py`) rather than build a new one. Below is the detailed reasoning, the formulas, the code sketches, and a phased P0–P5 plan with file names and concrete numbers (thresholds, windows, expected noise), fully grounded in what is already built and measured in this repo.

---

## Details

### 0. What exists today (read first, changes everything downstream)

| Capability | File:lines | Status |
|---|---|---|
| Ring buffer of K corresponded 3D points + mask, scattered by seed index | `tracking/motion.py:320-532` (`MotionBuffer`) | built, unused in rollouts |
| Robust Kabsch/Umeyama rigid fit (median+MAD outlier rejection, `det(R)<0` fix) | `tracking/motion.py:590-648` (`fit_rigid_motion`) | built |
| Screw decomposition: rotation angle, axis *direction and location* | `tracking/motion.py:676-712` (`screw_decompose`) | built |
| Tangential pull direction at an arbitrary body point | `tracking/motion.py:714-735` (`pull_direction`) | built |
| Plane fit (drawer-face normal, camera-sign-disambiguated) | `tracking/motion.py:737-773` (`fit_plane`) | built |
| TLS line fit on the centroid trajectory (not frame-to-frame diff) | `tracking/motion.py:775-833` (`centroid_direction`/`LineFit`) | built |
| `static\|prismatic\|revolute\|free` classifier, with hysteresis | `tracking/motion.py:834-930` (`classify_motion`, `MotionClassifier`) | built |
| Sign-aligned EMA smoothing of direction/axis | `tracking/motion.py:932-1023` | built |
| Composed per-window estimate (13 fields + reason + `to_dict()`) | `tracking/motion.py:1097-1273` (`estimate_motion`, `MotionEstimate`) | built, **never called from `session.py`** |
| Per-object twist Kalman filter: position, rotation, **velocity, angular_velocity**, green/yellow/red/black health, coast damping | `tracking/kalman.py`, `providers/tracker3d/pose_tracker.py:19-26` | built, live in rollouts |
| Relative pose since seed time, incl. `rotation_deg`, `source` (`measured`/`predicted`/`fk`), `trusted` | `tracking/types.py:130-186` (`ObjectPose`) | built, live |
| Gripper-FK↔object relative transform, re-used for FK-anchoring and a depth-based "is it still in the hand" test | `tracking/session.py:82, 420-425, 583-650, 668-691` (`grip_rel`, `_fk_check`) | built, live, **not exposed as an event** |
| `monitor(state)->{status,reason}` contract, `ok/warn/record/abort`, source shipped over IPC | `tracking/monitor.py` | built, live |
| `attached_to_gripper`, `object_not_lost`, `stays_within`, `moved_at_least`, `combine`, `default_monitor` | `tracking/monitors.py` | built, live |
| Rollout summary + JSONL + `describe()` for the reviewer | `tracking/report.py` | built, live, **no motion/event timeline** |
| `track_objects`/`get_tracking_report`/`stop_tracking` | `api.py:526-651` | built, live |
| GT eval harness incl. a door scene with real hinge+latch joints | `tests/tracking_eval.py`, `tests/test_tracking_door_pybullet.py` | built, live |
| `SKILL.md` for door/cabinet/drawer opening | `prompts/skills/open-close-door-cabinet-drawer/SKILL.md` | built, **never calls `track_objects`** |

Explicit confirmation of the gap: `docs/plans/3d_tracking.md:877-879` lists as "still in flight": *"the monitor wiring (`moves_along`, `moves_perpendicular_to_surface`, `revolute_axis_stable`) on the existing `ok\|warn\|record\|abort` contract"* — i.e. the plan itself already earmarked this as the next step. `grep` confirms `tracking/session.py` never imports `tracking.motion` or constructs a `MotionBuffer`.

A second concrete finding worth having up front: `pose_tracker.py` already computes the **per-point fused 3D array** every frame (`_fuse_points`, around `providers/tracker3d/pose_tracker.py:265-284`) in order to run Kabsch against the seed-time template, but it is collapsed to a single `world_point` + `pose.R/t` before it reaches `TrackedObjectState`/`TrackFrameReport` (`tracking/types.py:189-226`). The per-point cloud `MotionBuffer` needs is being thrown away one layer below where monitors live — that is the concrete thing to patch, not a new tracker.

---

### 1. Is piecewise-linear/segmentation the right primary approach? (comparison + recommendation)

**Recommendation: keep the existing windowed rigid-motion + screw decomposition as the primary, real-time estimator (tier A); add a dedicated line/circle (arc) fit as a secondary, always-available cross-check and fallback (tier B); add an *offline* changepoint pass only for the reviewer timeline and demo segmentation, not for real-time control.** Do not use RDP or IMM as primary mechanisms. Formalize (but do not replace) the existing `revolute_gain` heuristic as a BIC-style model comparison.

Rationale, option by option:

- **Piecewise-linear fit to raw xyz / RDP (Ramer–Douglas–Peucker).** RDP simplifies a *single* polyline by perpendicular-distance error, ignoring (a) that we have ≥3 *corresponded* points per frame — RDP only ever sees a centroid-equivalent single curve — and (b) model semantics (it will happily draw a long straight segment through the first 10° of a 0.35 m-radius door arc, because the sagitta is tiny — see the numeric estimate below). It is a good tool for **compressing a scalar/1-D time series for the reviewer's timeline** (e.g. simplify a `hinge_angle(t)` or `handle_distance_to_axis(t)` curve to a handful of knee-points for a human/VLM-readable summary), not for estimating 3-D articulation. **Use:** apply RDP (tolerance ≈ 2× tracking noise, e.g. 2–6 cm for position curves or 2–3° for angle curves) only when rendering the event timeline text, never as the geometry estimator.
- **Changepoint detection (PELT, BOCPD).** These detect *when* a 1-D or low-D statistic's distribution shifts; they do not by themselves know what a "prismatic" or "revolute" segment looks like — you still need a per-segment model fit after detecting the boundary. Good fit for: (a) **offline** re-segmentation of a finished rollout into primitives (approach / unlatch-press / arc-pull) for the reviewer and for demonstration-learning (§6), where you can afford the batch cost and want a probabilistically justified number of segments; (b) **not** good for low-latency reactive monitors, because PELT is retrospective (needs the whole series, or at least a trailing buffer, before it can place a changepoint with confidence) and BOCPD's run-length posterior needs tens of samples to resolve a true change from noise at our ~1–4 cm/frame jitter. Concretely: use **PELT** (Killick, Fearnhead & Eckley, *JASA* 2012, [doi:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745)) with a cost function = sum of per-segment `fit_rigid_motion` RMSE (not raw position error) plus a linear penalty `β·k` on segment count (classic BIC form `β = 2·ln(n)`), run once at `stop_tracking()`/`get_tracking_report()` time over the already-buffered frame history. **BOCPD** (Adams & MacKay, 2007, [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)) is the natural *online* Bayesian analogue if you ever need true probabilistic changepoint confidence in the loop — but given the existing `MotionClassifier` already provides a cheap, causal, already-tested 2-frame-dwell hysteresis classifier, I would not add BOCPD unless evaluation (§7) shows the existing classifier's false-transition rate is too high.
- **Per-window rigid-motion screw estimation (what's built).** This is exactly the right *primary* tool: it uses all K≥3 corresponded points each window (not a single centroid trace), gives axis location *and* direction (not just a direction), and its confidence model (`tracking/motion.py:1066-1095`, `_confidence`) is derived from the angular error the data can actually resolve (`rmse/motion_scale` for direction, `rmse/spread[0]` for axis tilt) rather than a bare residual — this is a better-justified uncertainty model than a naive "residual < threshold" rule. Measured recovery on synthetic screw motions (from `docs/plans/3d_tracking.md:847-849`): clean data → axis-direction error `0.0`, axis-point error `5.5e-16 m`; at σ=1 mm noise → 1.39° / 3.4 mm; at σ=4 mm (our high end) → 5.80° / 12.3 mm axis-point error. These numbers should become the P7 evaluation targets (§7).
- **Articulation-specific learned methods (Sturm et al., ScrewNet, Ditto, PARIS, ANCSH, ArtGS, UMPNet, Where2Act, FlowBot3D, GAPartNet).** These solve a *different* problem than ours and are valuable mainly as **priors**, not as the online estimator:
  - **Sturm, Stachniss & Burgard, "A Probabilistic Framework for Learning Kinematic Models of Articulated Objects"**, *JAIR* 41 (2011), [arXiv:1405.7705](https://arxiv.org/abs/1405.7705). This is the closest conceptual ancestor of `tracking/motion.py`: it fits rigid/prismatic/revolute *and other* parametric+non-parametric edge models between two observed rigid bodies from a trajectory of relative poses, and selects between them by BIC over the whole trajectory (their eq. 12 is literally `BIC = -2 ln L + k ln n`). Our setting is a **simplification** of theirs: we track one moving part (the handle) against the static world frame rather than two moving rigid bodies in a kinematic graph, so we never need their kinematic-graph structure search — but their *model-selection-by-information-criterion* principle is worth making explicit (below) rather than hand-tuned, as a documentation/robustness upgrade to `classify_motion`.
  - **ScrewNet** (Jain, Lioutikov, Niekum et al., [arXiv:2008.10518](https://arxiv.org/abs/2008.10518)), **Ditto** ([arXiv:2202.08227](https://arxiv.org/abs/2202.08227)), **PARIS** ([arXiv:2308.07391](https://arxiv.org/abs/2308.07391)), **ArtGS** ([arXiv:2502.19459](https://arxiv.org/abs/2502.19459)), **ANCSH / "Category-Level Articulated Object Pose Estimation"** ([arXiv:1912.11913](https://arxiv.org/abs/1912.11913)) all *learn* articulation parameters from one or two depth/RGB observations (before/after interaction), without point correspondence over time — they solve the problem we'd have if we *didn't* have a point tracker. Given we do have one, their main relevance is (a) validating that screw-theory is the right output representation (ScrewNet literally predicts the same `(axis_dir, axis_point, angle, pitch)` tuple `screw_decompose` returns), and (b) as the mechanism for the **a-priori** estimate before any motion is observed (§3).
  - **UMPNet** ([arXiv:2109.05668](https://arxiv.org/abs/2109.05668)), **Where2Act** ([arXiv:2101.02692](https://arxiv.org/abs/2101.02692)), **FlowBot3D** ([arXiv:2205.04382](https://arxiv.org/abs/2205.04382)) predict a *single-step* actionable direction/flow from one image, useful as alternative sources for the a-priori direction prior, not for refinement once motion is observed.
  - **GAPartNet** ([arXiv:2211.05272](https://arxiv.org/abs/2211.05272)) is the right reference for **what kind of part-level prior to bake into `SKILL.md`**: it defines canonical part classes (`hinge_handle`, `hinge_lid`, `slider_drawer`, `hinge_door`, `slider_button`, …) with canonical joint axis conventions per class — i.e. exactly the taxonomy `SKILL.md §1` already encodes informally in English ("hinged lever", "fixed bar", "knob", sliding vs. drawer). Worth citing in a docstring, not re-implementing.
- **RANSAC over model types + BIC.** Classic articulation-estimation RANSAC (e.g. Sturm et al. §4, Martin-Martin & Brock) exists because those methods **do not know point correspondence** (they fit a joint model to a cloud of *unlabelled* relative transforms, or to raw point clouds across two frames) and must guess-and-check minimal subsets. We *do* know correspondence (seed index), so full RANSAC hypothesis generation is unneeded complexity; `fit_rigid_motion`'s iterative median/MAD trimming (`tracking/motion.py:590-648`) already gets RANSAC's main benefit (one bad correspondence can't rotate the whole fit) far more cheaply. What *is* missing, and worth adding, is an explicit model-selection criterion between the **prismatic** and **revolute** candidates rather than the current single hand-tuned ratio. Today, `classify_motion` (`tracking/motion.py:834-867`) compares `rigid_residual` (rotation+translation fit, the "revolute" candidate) against `prismatic_residual` (translation-only fit, computed as `sqrt(mean(||(delta - mean(delta))||²))` — i.e. the residual of the *best possible single-direction slide* through the same points) and accepts revolute only when `rigid_residual <= gain · prismatic_residual` (`gain=0.5` entering, `0.8` once already revolute). This is already a likelihood-ratio test in disguise; make it an explicit BIC so the threshold is principled rather than tuned-by-feel:
  $$\mathrm{BIC}_{\text{model}} = n\cdot\ln\!\left(\frac{\mathrm{RSS}_{\text{model}}}{n}\right) + k_{\text{model}}\ln n,\qquad k_{\text{prismatic}}=3,\; k_{\text{revolute}}=6$$
  (`RSS = n · residual²` over the `n` corresponded points; `k` counts free parameters — a unit direction is 2 DOF but a pure-translation model also has a magnitude, giving 3; a screw has a 2-DOF axis direction, a 2-DOF (within-plane) axis location, and an angle, giving 6 counting translation along the axis as a 7th only if pitch≠0). Prefer revolute when `BIC_revolute < BIC_prismatic`. This reduces to the current `gain` rule at a particular `n`, so it can be introduced as a config-selectable alternative (`track_motion_model_select = "ratio" | "bic"`) rather than a breaking change — a P2-size addition, not core-path risk.
- **Kalman / IMM (interacting multiple models).** We already run one Kalman filter with a *single* "twist" dynamic model (`tracking/kalman.py:1-60`), which already degrades gracefully from a constant-velocity model (`ω→0` ⇒ `V→I·dt`) to a constant-twist one — this is itself a soft compromise between a prismatic and revolute *prior*, chosen once at filter-construction time (`track_kf_motion_model="twist"`, `config.py:370`). A true IMM (Blom & Bar-Shalom, *IEEE TAC* 33(8), 1988; overview: [Interacting Multiple Model (Wikipedia)](https://en.wikipedia.org/wiki/Interacting_multiple_model_algorithm)) would run 2–3 parallel filters (static-CV, prismatic-CV, revolute-CT) and maintain a model-probability vector updated via each filter's innovation likelihood, mixing state estimates each step. This is strictly more principled for *fast* online mode-switching, but it roughly triples the filter bookkeeping (covariance per model, mixing matrix, likelihood normalisation) for a gain that `MotionClassifier`'s window+hysteresis (dwell=2 windows, hysteresis gap between `angle_enter_rad=0.05` rad and `angle_exit_rad=0.02` rad) already mostly captures at our noise level (σ≈1–4 cm per point, window ≈0.3–1 s at 10 Hz). **Recommendation: do not implement IMM now; revisit only if evaluation (§7) shows classification-flicker or switch-latency is the dominant source of event-detection error** — flag as a P5/stretch item.

**Bottom line formula set to keep as the system of record** (all already implemented):
1. Rigid fit: `R, t, rmse = fit_rigid_motion(P0, P1)` (Kabsch + robust trim).
2. Screw: `θ, n̂, c, pitch = screw_decompose(R, t, ref_point=centroid)`.
3. Pull direction at the grasped point `p`: `pull_direction(n̂, c, p) = normalize(n̂ × ((p − c) − ((p−c)·n̂)n̂))`.
4. Classification: `classify_motion(θ, Δcentroid, rmse_rigid, rmse_prismatic, …)` with hysteresis (`MotionClassifier`).
5. **New (P1, below):** `fit_circle_3d` as a tier-B cross-check, specifically for the collinear-handle case (§2).

---

### 2. Aggregating the K point tracks

**Recommendation: keep the existing weighted-Kabsch-with-robust-trimming aggregation (not raw centroid, not per-point voting) as the default; add health/confidence weighting at the `_kabsch` call site (currently unweighted); explicitly document and mitigate the collinear-handle degeneracy with a concrete, previously-unreported failure mode.**

- **SE(3) Kabsch fit (current) vs. centroid-only vs. robust median vs. per-point voting.** A centroid alone can only ever report a displacement vector — exactly the motivation documented at the top of `tracking/motion.py` ("migration from a bbox/centroid tracker to a multi-point one… three or more corresponded points give you an axis, a plane and a rotation"). Per-point voting (e.g., majority-vote a discrete `static/slide/rotate` label per point, then combine) throws away the fact that points on a rigid body are *jointly* constrained — a single Kabsch fit over all points uses that constraint to average down noise (`σ/√K` scaling), which per-point voting cannot do. Robust median-of-per-point-displacements is a reasonable *cheap fallback* when correspondence is weak (e.g., only 2 points agree this frame), but is strictly dominated by the robust Kabsch fit whenever ≥3 non-collinear points are available, which `fit_rigid_motion` already is (iterative median/MAD-based point rejection *inside* the Kabsch loop, `tracking/motion.py:590-648`).
- **Weighting by health/confidence.** `_kabsch` (`tracking/motion.py:558-589`) already accepts a `weights` argument but `fit_rigid_motion` currently always calls it with `weights=None` (uniform weighting across all surviving inliers). Every per-point health/confidence signal needed to weight it already exists one layer down in `CamTrack.confidence`/`CamTrack.health` (`tracking/types.py`, `CamTrack` dataclass) and the per-point fusion weight `confidence / z²` already used inside `pose_tracker.py._fuse_points`. **Concrete P1 change:** thread the per-point fusion confidence through `MotionBuffer.push(...)` as an optional `weights` row-vector per frame-pair (reuse the *older* frame's weight, since that's the one entering the Kabsch "A" side) and pass it to `fit_rigid_motion(weights=…)`. Expected effect: a tracked point that briefly degrades to single-camera fusion (noisier, `CamTrack.confidence` typically falls by roughly half per `pose_tracker`'s single-camera branch) is automatically down-weighted in the axis/direction fit rather than contributing full-strength noise.
- **Occluded/lost/outlier points.** Already handled at three levels, which should be kept as-is: (1) `MotionBuffer` masks a missing sample `False` and nothing reads a masked entry (`tracking/motion.py:320-344`); (2) `fit_rigid_motion`'s median/MAD rejection drops the single worst *rigid-inconsistent* point even if it reports as "visible" (catches a 2D tracker that locked onto the wrong feature but still reports high confidence); (3) `estimate_motion`'s `n_common < 3` / `collinear_points` gates refuse to emit a direction at all rather than guess (`tracking/motion.py:1150-1165`). This is the right layered design — reactive (per-point reject), geometric (collinearity gate), and semantic (confidence-zero-means-don't-trust) — and nothing here needs to change.
- **The collinear-handle degeneracy — more subtle than "few points, no plane".** `is_collinear` (`tracking/motion.py:550-556`) already refuses a rotation when the *current* point cloud's singular spectrum is nearly 1-D (σ₂/σ₁ ≤ `collinear_ratio=0.02`), which correctly blocks the obvious case (5 points crammed at one pixel-cross, the historical `--seeding affordance` default, `docs/plans/3d_tracking.md:1163` 9.1). But there is a **second, worse degeneracy that this check does not catch**, and it is directly relevant to `SKILL.md §2b` ("vertical handle bar"): if the K seed points lie (even loosely) along a line **parallel to the true rotation axis** rather than merely "in a line" in some arbitrary sense, a true large-radius revolute motion is *indistinguishable from pure translation at that sub-cloud*. Proof sketch: a rigid-body rotation by angle θ about the world-Z axis maps any point `(x,y,z)` to `(x cosθ − y sinθ, x sinθ + y cosθ, z)` — the horizontal displacement does **not depend on z**. So every point on a *vertical* handle bar (which differ from each other only in z, by construction) receives **exactly the same 3-D displacement vector**, which is the literal definition of a pure translation. A vertical-bar handle on a hinged door (`SKILL.md §2b`'s whole "side approach" branch) is precisely this geometry, whereas a *horizontal* lever (the §2 default case) is **not** degenerate, because a horizontal lever's points differ in `(x,y)`, which *does* depend on θ. This failure mode would silently mis-classify a real revolute door-opening motion as `prismatic` whenever the agent is working a vertical-bar handle, and it would not be caught by `is_collinear` because the handle's *spatial* point cloud genuinely is fine (not collinear with *itself*) — it only becomes degenerate with respect to this *particular* rotation axis, which the Kabsch fit cannot know in advance.
  - **Mitigation 1 (seeding, cheap, do first):** when seeding a handle recognised as a vertical bar (`SKILL.md §2b`'s own height/footprint trigger is a ready-made signal), always add 1–2 extra `world_points` **off the bar's own line** — e.g. on the rosette/escutcheon plate at its base, or on the door surface immediately behind it — so the tracked cloud is not collinear with the hinge even though it is collinear along the bar. This is a one-line change to whatever currently builds `targets=[{"name": "door_handle", "world_points": [...]}]` from `detect_object`.
  - **Mitigation 2 (always-available cross-check, add in motion.py):** independent of the spatial-collinearity problem, a *single* point's trajectory over time (the handle's own centroid, or the gripper's FK position, which is noise-free and always available via `frame.gripper.world_pos`) traces a **straight line** under prismatic motion and a **circular arc** under revolute motion — always, regardless of how the K points are laid out spatially. `centroid_direction`/`LineFit` (`tracking/motion.py:775-833`) already computes the straight-line residual; add a matching `fit_circle_3d(points) -> CircleFit(center, radius, axis_normal, residual, confidence)` (project onto `fit_plane`'s plane, then a Taubin/Pratt 1991 algebraic circle fit — biased **Kasa** fits are a known trap for small arcs and should not be used here, since our arcs are frequently small: 10–40° per window) and compare the two residuals the same way `classify_motion` already compares rigid-vs-prismatic residuals. Because this only needs a *single* trajectory, it works even when the spatial cloud is degenerate, and it is cheap to run on the **gripper's own FK path** as a zero-noise ground truth cross-check once grasped — if the line-fit residual is low (straight) while the gripper's FK path over the same window is visibly curved, that disagreement is itself diagnostic evidence of the vertical-bar degeneracy and should demote confidence rather than silently reporting "prismatic".
  - Numerically: sagitta (deviation of the arc from its chord) for radius `R` and chord length `d` is `s ≈ d²/(8R)`. For a typical door-width radius `R=0.35 m` and our fit RMSE floor `≈1 cm`, the arc only becomes distinguishable from a straight line *by shape alone* once `d > √(8·R·k·σ) ≈ √(8·0.35·2·0.01) ≈ 0.24 m` (`k=2` as a 2σ margin) — i.e. roughly a quarter of the door's full travel. This is why relying on curvature-of-centroid-path alone (tier B) is a *secondary*, not primary, discriminator: tier A's multi-point Kabsch fit resolves the rotation ≈16× earlier in displacement terms for a *non-degenerate* (e.g. horizontal-lever) point cloud, because it uses the differential signal across points (`ω × r_i` differs point-to-point) rather than waiting for visible path curvature.

---

### 3. Computing the pull/push direction and articulation parameters

**Already built (keep):** for revolute, `direction = pull_direction(axis_dir, axis_point, query_point)` evaluated at the *handle* (not the centroid) — `MotionEstimate.pull_direction_at(point)` lets any caller re-evaluate it anywhere else on the body (`tracking/motion.py:282-295`). For prismatic, `direction` is the TLS slide direction (`LineFit.direction`), and `perpendicularity = |direction·plane_normal|` / `perpendicularity_deg` already answers "is this drawer being pulled straight out or racked" (`tracking/motion.py:1023-1038`, `_perpendicularity`).

**A-priori estimate, before any motion is observed.** Three sources, cheapest first:
1. **Geometric heuristic already encoded in `SKILL.md §1`, made explicit and numeric.** The skill already reasons "the hinge is on the **opposite** edge" of the door from the handle, and already extracts `angle_long`/`angle_short` from `detect_object`. Formalize this into a literal prior: `axis_point_prior` = the door panel's farthest edge from the handle (from `detect_object("door")`'s bounding box, projected onto the floor-parallel plane), `axis_dir_prior` = vertical (`[0,0,1]`) for a hinged door/cabinet, or the door-plane normal for a sliding door/drawer (already partially present as `plane.normal` from `fit_plane` once there are *any* points, even static ones, to fit a plane to — the plane fit does not require motion, only ≥3 non-collinear points, so it can run **before** the pull begins). This costs nothing new to compute; it needs to be written down as a `prior` dict and threaded through `track_objects`/`estimate_articulation`.
2. **Part-affordance priors (GAPartNet-style, [arXiv:2211.05272](https://arxiv.org/abs/2211.05272)):** if/when a part-segmentation or affordance model is added, its predicted joint type + canonical axis convention per part class is a strictly better prior than (1) and should simply replace it — same `prior` dict shape, no other code changes needed.
3. **VLM proposal:** the planning VLM already reasons qualitatively about hinge side / pull-vs-push (`SKILL.md` "Pull vs push" bullet); asking it to also emit a *numeric* guess (e.g. "hinge is at approximately `[x,y,z]`, axis is vertical") costs one extra sentence in its existing output and is already compatible with (1)'s shape.
- **Small-angle revolute/prismatic ambiguity, with thresholds.** As derived in §2, path-shape alone cannot discriminate until `d ≳ 0.24 m` of travel for a 0.35 m-radius hinge; but the Kabsch differential-point-signal **can**, much earlier, *provided the point cloud is not axis-degenerate* (§2's mitigation). Concrete operating thresholds already in `MotionThresholds` (`tracking/motion.py:91-144`) and recommended as the "first classification is trustworthy" gate: `min_direction_m=0.015` m of centroid travel, `angle_enter_rad=0.05` rad (2.9°) of rotation, `revolute_gain=0.5` (rigid fit must explain the data ≥2× better than a pure slide). At σ=1–4 mm noise this resolves correctly within roughly 1.5–3 cm of handle travel (empirically, the existing axis-error numbers: 1.39° at σ=1 mm, 5.80° at σ=4 mm, both well inside a single `MotionClassifier` dwell window). **Recommendation:** report `kind="revolute_or_prismatic (undetermined)"` with `confidence` scaled down (not simply "static") specifically in the window where `angle_rad` is between `angle_enter_rad` and, say, 3×`angle_enter_rad` (≈2.9°–8.7°) **and** the point cloud's extent along the candidate axis direction exceeds `0.5×` its extent perpendicular to it (a cheap proxy for "this point cloud may be axis-degenerate") — i.e. make the existing `reason` field carry this specific diagnostic string (`"axis_degenerate_candidate"`) rather than silently reporting `prismatic`.
- **Compliance / closed-loop adaptation.** Once `axis_dir`/`axis_point` (revolute) or `direction` (prismatic) is available with `confidence>0`, the natural use is **not** to re-plan the whole trajectory but to *correct the next short segment's direction* — classic articulation-aware compliant control, as in Jain & Kemp, "Pulling Open Novel Doors and Drawers with Equilibrium Point Control" (ICRA 2010) and Arduengo et al., "Robust and Adaptive Door Operation with a Mobile Robot" ([arXiv:1902.09051](https://arxiv.org/abs/1902.09051)) — both use a compliant/impedance-controlled arm that tracks a *locally re-estimated* pull direction rather than a single pre-planned arc, exactly matching our screw-estimate-refreshed-every-window design. Concretely: `generate_linear_trajectory` already interpolates waypoint-to-waypoint; propose a new helper `next_pull_waypoint(current_pose, estimate: MotionEstimate, step_m=0.02)` that advances `step_m` along `estimate.pull_direction_at(current_handle_pos)` (revolute) or `estimate.direction` (prismatic) each time the agent calls `get_track_features`, so that the trajectory the agent generates *this step* always points along the latest belief rather than the a-priori guess from step 0. This is "compliance" in the geometric-adaptation sense (correct the commanded direction), not force/impedance compliance (which would need Franka joint-torque control, out of scope for this pass — flagged as a stretch item).

---

### 4. Event detection — concrete monitor code

All of the below respects the existing contract exactly: `monitor(state: TrackFrameReport) -> {"status": "ok"|"warn"|"record"|"abort", "reason": str, **extra}`, stateful closures, never raising (exceptions are caught and downgraded to `warn` by `MonitorRunner`, `tracking/monitor.py:95-122`). New state fields needed on `TrackedObjectState` are marked **[NEW]**.

**4a. Grasp-slip (reactive + proactive), reusing `attached_to_gripper` and adding a rate-of-separation leading indicator.**

`attached_to_gripper` (`tracking/monitors.py:30-130`) already *is* the reactive slip detector: it computes `frame.distance(name)` (gripper FK ↔ tracked world point), arms itself only after an actual grasp engagement (`engage_dist`), requires both `grace_frames` *and* `grace_s` of sustained violation (so one noisy frame can't abort), and already cross-checks against the independent depth-based `lift_meta['fk_check']` support/contra counts. This covers "distance exceeds threshold with hysteresis." What's missing is the **proactive** half (Code-as-Monitor's terminology, §5) — flag a *trend* before the hard distance threshold trips, so the agent can react earlier:

```python
def grasp_slip_trend(name, rate_warn_mps=0.03, rate_abort_mps=0.08, window_s=0.5):
    """Proactive companion to attached_to_gripper: flags a GROWING gripper<->object gap
    before the hard distance threshold (track_attach_max_dist, 0.12 m) is crossed.
    Uses only trusted (non-predicted, non-tentative) evidence, same as attached_to_gripper.
    """
    history = []  # [(sim_time, distance)]

    def monitor(frame):
        obj = frame.get(name)
        if obj is None or obj.lost or getattr(obj, "predicted", False):
            return {"status": "ok"}
        pose = getattr(obj, "pose", None)
        if pose is not None and getattr(pose, "tentative", False):
            return {"status": "ok"}
        dist = frame.distance(name)
        if dist is None:
            return {"status": "ok"}
        t = frame.sim_time if frame.sim_time is not None else frame.frame_idx * 0.1
        history.append((t, dist))
        while history and t - history[0][0] > window_s:
            history.pop(0)
        if len(history) < 3:
            return {"status": "ok", "distance": round(dist, 4)}
        ts = [h[0] for h in history]; ds = [h[1] for h in history]
        # simple linear regression slope (same technique as centroid_direction's speed fit)
        import numpy as np
        rate = float(np.polyfit(ts, ds, 1)[0])
        if rate >= rate_abort_mps:
            return {"status": "abort",
                    "reason": f"'{name}' separating from the gripper at {rate:.3f} m/s",
                    "slip_rate_mps": round(rate, 4), "distance": round(dist, 4)}
        if rate >= rate_warn_mps:
            return {"status": "record",
                    "reason": f"'{name}' drifting from the gripper at {rate:.3f} m/s (watch)",
                    "slip_rate_mps": round(rate, 4), "distance": round(dist, 4)}
        return {"status": "ok", "slip_rate_mps": round(rate, 4), "distance": round(dist, 4)}

    monitor.__name__ = f"grasp_slip_trend[{name}]"
    return monitor
```
Use as `monitor=combine(attached_to_gripper("handle"), grasp_slip_trend("handle"), object_not_lost("handle"))`. Suggested thresholds: `rate_warn_mps=0.03` m/s (≈1 cm per 3 frames at 10 Hz, well above the ~1-4cm/window noise floor's implied rate of ≈0.1-0.4 m/s only if sustained, so a *sustained* 0.03 m/s trend over `window_s=0.5s` is a real signal, not jitter — jitter is frame-to-frame, not trend-over-5-frames), `rate_abort_mps=0.08` m/s (object would cross the 12 cm `track_attach_max_dist` limit in ~1.5 s at this rate, leaving time to react before `attached_to_gripper` itself aborts).

**4b. Door/drawer stuck** — gripper moved, object didn't:

```python
def door_stuck(name, min_gripper_travel_m=0.03, max_object_travel_m=0.008, after_s=0.6):
    origin_g, origin_o, t0 = None, None, None

    def monitor(frame):
        nonlocal origin_g, origin_o, t0
        g = frame.world_point("gripper")
        o = frame.world_point(name)
        if g is None or o is None:
            return {"status": "ok"}
        t = frame.sim_time if frame.sim_time is not None else frame.frame_idx * 0.1
        if origin_g is None:
            origin_g, origin_o, t0 = g.copy(), o.copy(), t
            return {"status": "ok"}
        import numpy as np
        g_travel = float(np.linalg.norm(g - origin_g))
        o_travel = float(np.linalg.norm(o - origin_o))
        if t - t0 >= after_s and g_travel >= min_gripper_travel_m and o_travel <= max_object_travel_m:
            return {"status": "record",
                    "reason": (f"gripper moved {g_travel:.3f} m but '{name}' moved only "
                               f"{o_travel:.3f} m - the mechanism may be stuck or still grasped wrong"),
                    "gripper_travel": round(g_travel, 4), "object_travel": round(o_travel, 4)}
        return {"status": "ok", "gripper_travel": round(g_travel, 4), "object_travel": round(o_travel, 4)}

    monitor.__name__ = f"door_stuck[{name}]"
    return monitor
```
`max_object_travel_m=0.008` m sits just above the ~1-4 mm green-mode fit noise (`track_pose_green_rmse_m=0.015` is the RMSE *gate*, not the steady-state centroid noise, which is smaller); `min_gripper_travel_m=0.03` m is well above FK noise (FK is exact — the real floor here is "did the agent actually command 3 cm of motion," which any real pull attempt will). Status `record` (not `abort`) per the task's own triage guidance in `SKILL.md §5` — this maps directly onto the "miscalculation vs. bad grasp" distinction the skill file already asks the LLM to make, and should be surfaced to the reviewer, not used to kill the rollout (a real stuck/latched door is sometimes expected mid-procedure, e.g. before the unlatch press).

**4c. Door opened by θ degrees (progress events)** — nearly free, built on the existing `ObjectPose.rotation_deg`:

```python
def angle_progress(name, thresholds_deg=(10, 30, 60, 90)):
    fired = set()

    def monitor(frame):
        obj = frame.get(name)
        pose = getattr(obj, "pose", None) if obj is not None else None
        if pose is None or not pose.trusted:
            return {"status": "ok"}
        deg = pose.rotation_deg
        for th in thresholds_deg:
            if deg >= th and th not in fired:
                fired.add(th)
                return {"status": "record", "reason": f"'{name}' rotated past {th} deg (now {deg:.1f})",
                        "angle_deg": round(deg, 2), "threshold_deg": th}
        return {"status": "ok", "angle_deg": round(deg, 2)}

    monitor.__name__ = f"angle_progress[{name}]"
    return monitor
```
This reuses `pose.rotation_deg`, which is *already* computed every frame by the Kalman filter relative to seed time (`tracking/types.py:160-163`), requires no new tracking math, and directly answers "door opened by N degrees" for the reviewer. For a drawer/slider, the matching helper reads `obj.pose.t` projected onto the a-priori slide direction (or `estimate.displacement`) against distance thresholds instead of degrees.

**4d. Object dropped (z falling) / wrong object moved** — both are one-line specializations of existing patterns: "dropped" = `stays_within` with a box whose floor sits just above the table + hysteresis on downward velocity (finite-difference `world_point.z`, same closure pattern as `moved_at_least`); "wrong object moved" = run `moved_at_least(other_name, dist=0.01, status="record")` on every tracked target *other than* the intended one, so an accidental bump shows up as a flagged frame without aborting the rollout.

**Suggested default combination for the door/cabinet/drawer skill:**
```python
monitor = combine(
    attached_to_gripper("handle"),
    grasp_slip_trend("handle"),
    object_not_lost("handle"),
    door_stuck("handle"),
    angle_progress("handle", thresholds_deg=(10, 30, 60, 90)),
)
```

---

### 5. Code-as-Monitor and related failure-detection work, mapped onto this system

**Code-as-Monitor (CaM)**, Zhou et al., CVPR 2025, [arXiv:2412.04455](https://arxiv.org/abs/2412.04455), [project page](https://zhoues.github.io/Code-as-Monitor/): formulates both *reactive* (did something already go wrong) and *proactive* (is something about to go wrong) failure detection as a single family of **spatio-temporal constraint-satisfaction problems**, evaluated by **VLM-generated code** against **"constraint elements"** — compact geometric primitives (points, lines, planes abstracted from full object masks/meshes) that make the constraints cheap to re-evaluate every frame and easy for the VLM to reason about and generate code against. This maps almost one-to-one onto what is already built here:

| CaM concept | This repo's equivalent |
|---|---|
| Constraint elements (points/lines/planes abstracted from the scene) | `tracking/motion.py`'s `PlaneFit`, `LineFit`, `ScrewMotion` (axis = a line; plane normal = a plane) — already the same abstraction level |
| VLM-generated monitor code, evaluated in real time | `tracking/monitor.py`'s `compile_monitor`/`MonitorRunner`, shipped as source over IPC (`api.py:587-608`) — already the same mechanism |
| Reactive detection (failure already happened) | `attached_to_gripper` (abort on detach), `stays_within` |
| Proactive detection (failure about to happen) | **gap** — `grasp_slip_trend` (§4a) and `door_stuck` (§4b) are the proactive additions this plan proposes |
| Closed-loop correction using the monitor's structured output | **gap** — §3's `next_pull_waypoint` proposal is the closed-loop hook |

The actionable takeaway from CaM for this repo is specifically the **reactive+proactive split as a design principle**, not new infrastructure: the existing `record` status (distinct from `abort`) is already the right vehicle for "proactive, not yet fatal" signals (as used by `grasp_slip_trend`/`door_stuck` above) — it just hasn't had a proactive monitor populate it for anything beyond "object not lost" and "hasn't moved yet."

**Other related work, and what to take from each:**
- **AHA** (Duan et al., [arXiv:2410.00371](https://arxiv.org/abs/2410.00371)): a VLM fine-tuned specifically to detect *and explain* manipulation failures from video, trained on a large synthetic failure-mode dataset. Relevant as the **vision-only reviewer's complement**: our tracking monitors give exact numeric evidence (distances, angles, rates) that a VLM reviewer like AHA's approach can be *given as structured context* rather than having to re-derive visually — i.e. feed `get_tracking_report()`'s event list into the reviewer prompt alongside the images, rather than asking the VLM to infer "did it slip" from pixels alone. This is exactly what §6's `events` field is for.
- **REFLECT** (Liu et al., [arXiv:2306.15724](https://arxiv.org/abs/2306.15724)): summarizes a multisensory (audio/vision/proprioception) failure into a hierarchical summary that an LLM queries to explain *why* a task failed and propose a correction. The structural idea — a **hierarchical, queryable summary** rather than a flat log — is exactly the right model for `TrackingReporter.summary()`/`describe()` once it carries an event timeline (§6): keep per-frame detail in the JSONL (already done), but make the *summary* a small number of semantically labelled events + a top-level verdict, which is what REFLECT's summary tree does one level up.
- **DoReMi** (Guo et al., [arXiv:2307.00329](https://arxiv.org/abs/2307.00329)): detects plan-execution misalignment by having an LLM generate *constraints that the current plan step implies* and checking them against low-level sensor state in real time, triggering immediate re-planning rather than waiting for the whole trajectory to finish. This is the strongest precedent for **abort-mid-trajectory with a reason fed straight back into the retry prompt** — which `api.py`'s `_handle_abort` (around `api.py:500-523`, confirmed in the code read) already implements verbatim (raises a `RuntimeError` with the reason, caught by `agent_runner.execute_python_blocks`'s error-correction path). No new work needed here beyond making sure the *new* monitors (§4) populate `reason` with DoReMi-style actionable text ("re-detect the object and re-grasp" rather than just "distance = 0.19").
- **Inner Monologue** ([arXiv:2207.05608](https://arxiv.org/abs/2207.05608)) and **"Robots That Ask For Help" / KnowNo** ([arXiv:2307.01928](https://arxiv.org/abs/2307.01928)): both are about feeding *environment feedback* back into the LLM's reasoning loop (success detection, scene description, or calibrated uncertainty) rather than running open-loop. Relevant mainly as a sanity check that the planned integration — `get_tracking_report()`/`get_track_features()` results formatted as text and handed back to the agent mid-rollout, not only post-hoc — is the right shape; KnowNo specifically argues for *calibrated* uncertainty before escalating to a human/abort, which maps onto using `MotionEstimate.confidence` (already principled, §1) as the gate for whether a `record` event should actually be trusted enough to change the agent's behaviour.
- **ReKep** (Huang et al., [arXiv:2409.01652](https://arxiv.org/abs/2409.01652)): represents manipulation constraints as **relational keypoint constraints** — Python functions of tracked keypoint positions, generated by an LLM, re-evaluated every frame to drive a real-time optimizer. This is the most direct precedent for **letting the LLM choose which points to seed and what geometric relation to monitor**, i.e. for Q5's explicit ask ("LLM with skills decides what to track, which monitor funcs, and how to post-process tracks"). ReKep's keypoint constraints are structurally identical to what `track_objects(targets=[...], monitor=...)` already lets the LLM specify — the gap (consistent with the rest of this report) is that our monitors currently only compose *pre-built* functions (`attached_to_gripper`, etc.), and the worked examples in `tracking/monitors.py`'s own docstring ("as worked examples the model imitates when it needs something bespoke") are the intended mechanism for the LLM to write a ReKep-style bespoke constraint, which §6's skill-file addition should make explicit with a worked example using `get_track_features`.

---

### 6. Proposed API additions

**`api.get_track_features(name, window=None) -> dict`** — a thin new IPC round-trip (`config.GET_TRACK_FEATURES`) that asks the simulator process for the session's `MotionEstimate.to_dict()` for `name`, either the most recent cached one (`window=None`, essentially free) or recomputed on demand over a caller-chosen window (in ref-frames at `track_kf_ref_hz=10 Hz`, so `window=10` ≈ last 1 s). Shape (identical to `MotionEstimate.to_dict()`, `tracking/motion.py:296-319`, nothing new to invent):
```python
{
  "kind": "revolute", "direction": [0.02, -0.98, 0.01], "speed": 0.07,
  "axis_dir": [0.0, 0.0, 1.0], "axis_point": [0.31, 0.82, 0.90],
  "angle_rad": 0.41, "displacement": 0.14, "confidence": 0.86,
  "n_points": 5, "window_frames": 10, "residual": 0.004, "reason": ""
}
```

**`api.estimate_articulation(name) -> dict`** — a higher-level call combining the a-priori prior (§3) with the live `get_track_features` output, with an explicit, auditable (not silently blended) contract:
```python
{
  "source": "online" | "prior" | "prior_low_confidence",
  "kind": "revolute", "axis_dir": [...], "axis_point": [...],
  "pull_direction_at_handle": [...], "confidence": 0.86,
  "prior": {"axis_dir": [0,0,1], "axis_point": [0.30, 0.80, 0.90], "basis": "door_opposite_edge_heuristic"}
}
```
`source="online"` whenever `get_track_features(...).confidence > 0`; otherwise fall back to the prior with `source="prior"` (if one was supplied to `track_objects`) and `confidence=0`, so a caller can never mistake an untested guess for measured geometry — matching `MotionEstimate`'s own existing "confidence 0 means do not use the geometry" contract (`tracking/motion.py:245-249`).

**Structured events in `get_tracking_report()`.** Extend `TrackingReporter` (`tracking/report.py`) with an `events: list` populated by (a) `MotionClassifier` label transitions (it already logs these internally, `tracking/motion.py:908-917` — just also append a structured dict instead of only `logger.info`), (b) `angle_progress`/`door_stuck`/`grasp_slip_trend` monitor outputs whose `status != "ok"`, and (c) `lift_meta['fk_check']` support/contra crossovers. Expose as `summary()["events"]` (list of `{"t": float, "type": str, "name": str, "detail": dict}`) and render a one-line-per-event timeline in `describe()`, e.g.:
```
Tracking: 142 frames, status=record.
  - 'door_handle': kind static->revolute at t=1.3s (angle 3.1 deg, axis [0,0,1]).
  - 'door_handle': rotated past 30 deg at t=2.6s.
  - 'door_handle': drifting from the gripper at 0.04 m/s at t=3.9s (watch).
```
This is the structured, numeric timeline the VLM/LLM reviewer should be given alongside images — directly answering the AHA/REFLECT takeaway in §5.

**Skill-file snippet** to add to `prompts/skills/open-close-door-cabinet-drawer/SKILL.md` (currently has *zero* tracking calls — the single largest and cheapest-to-fix gap found in this review):
```markdown
## 6. Track the handle while you pull (do this for every arc/slide in sections 2-4)

Before executing T3 (the opening motion), seed tracking on the handle so a slip or a
stuck mechanism is caught within ~0.3-0.5s instead of only at the end of the trajectory:

    track_objects(targets=[{"name": "door_handle", "world_points": handle_points}],
                  monitor=combine(attached_to_gripper("door_handle"),
                                  door_stuck("door_handle"),
                                  angle_progress("door_handle", thresholds_deg=(10, 30, 60))))
    execute_trajectory(t3)
    report = get_track_features("door_handle")
    print(f"kind={report['kind']} confidence={report['confidence']:.2f} "
          f"angle={math.degrees(report['angle_rad']):.1f} deg")

If `report['kind']` disagrees with what section 1 classified (e.g. you expected `revolute`
but the pull motion is also true along a *vertical* bar handle, see 2b), prefer the
tracked `axis_dir`/`direction` over your initial guess for the remaining waypoints -
`get_tracking_report()`'s event list will show which.
```

**Demonstration-learning path.** The same `MotionBuffer`/`estimate_motion` pipeline applies unchanged to a *recorded* trajectory of 2D points from a human or robot video: lift 2D CoTracker tracks to 3D via depth (exactly the existing `providers/tracker3d/depth_fusion.py` path, already provider-agnostic), push them into a `MotionBuffer`, and run `estimate_motion` offline to recover the same `(kind, axis, direction, angle)` tuple as imitation targets — this is architecturally what **Track2Act** ([arXiv:2405.01527](https://arxiv.org/abs/2405.01527)), **ATM (Any-point Trajectory Modeling)** ([arXiv:2401.00025](https://arxiv.org/abs/2401.00025)), **Im2Flow2Act** ([arXiv:2407.15208](https://arxiv.org/abs/2407.15208)), **General Flow** ([arXiv:2401.11439](https://arxiv.org/abs/2401.11439)), **VidBot** ([arXiv:2503.07135](https://arxiv.org/abs/2503.07135)), **Gen2Act** ([arXiv:2409.16283](https://arxiv.org/abs/2409.16283)) and **ManipTrans** ([arXiv:2503.21860](https://arxiv.org/abs/2503.21860)) all do — predict or extract point tracks from human/web video and use them (directly, or as a conditioning signal / retargeting target) as the manipulation-relevant representation, bridging the embodiment gap because point tracks (unlike joint angles) are embodiment-agnostic. The articulation layer proposed here is the natural "what do these tracks *mean*" step on top of any of them: once a demo's tracks are lifted to 3D, `estimate_motion`'s screw/classification output is exactly the per-segment articulation label + axis/direction imitation targets these methods either predict directly (Track2Act, ATM) or need as an intermediate structure to retarget onto the robot (ManipTrans). **Recommendation:** treat this as a P5+ item — the tracking/articulation code needed is already in hand; the missing piece (a demo-ingestion pipeline: load a human video, run CoTracker + depth, build a `MotionBuffer`) is a separate, larger effort than the live-rollout integration this report is scoped to, and should be its own follow-up plan once the rollout-side wiring (P1-P3 below) is in and evaluated.

---

### 7. Evaluation plan in sim

Extend the existing harness (`tests/tracking_eval.py`, `tests/test_tracking_door_pybullet.py`, `DoorSceneDriver`) rather than build a new one — it already has hinge/latch joint ground truth (`cls.hinge_index`, `cls.latch_index`, `self.joint_angles()` returning `(hinge, latch)` in radians, `tests/test_tracking_door_pybullet.py:213-269`) and already reaches ~0.75 rad (43°) hinge swing / ~0.8 rad latch turn (`docs/plans/3d_tracking.md:1740`).

| Metric | How to compute | Target (from measured numbers already in the repo) |
|---|---|---|
| Axis-direction error | angle between estimated `axis_dir` and the URDF hinge joint's world-frame axis (available from `get_link_pose`/joint axis already used by `DoorSceneDriver`) | < 2° once travelled > 0.02 rad (clean); < 6° at σ=4 mm noise (matches `docs/plans/3d_tracking.md:847-849`'s own synthetic-noise numbers) |
| Axis-point error | distance between estimated `axis_point` and the true hinge line (perpendicular distance) | < 5 mm clean, < 15 mm at σ=4 mm |
| Pull-direction error vs. time since motion start | angle between `pull_direction_at(handle)` and the GT tangent (`ω̂ × (p_handle − p_hinge)`, computed from true hinge angle rate), plotted against `t − t_motion_start` | should fall below 10° within the same displacement budget derived in §2/§3 (~1.5-3 cm of handle travel) |
| Slip-detection latency | synthetic slip injection (below) → time from injected slip to first `abort`/`record` from `grasp_slip_trend`/`attached_to_gripper` | < 0.5 s (matches `grasp_slip_trend`'s `window_s=0.5`) |
| False-alarm rate | fraction of frames flagged `record`/`abort` during a normal, successful grasp-and-pull baseline run (no injected fault) | 0 aborts; ≤ a handful of transient `warn`s during known-noisy events (e.g. a camera hand-over) |
| Angle-threshold event correctness | compare `angle_progress` fire times against GT hinge-angle crossing times | fire within 1 ref-frame (0.1 s) of the true crossing |

**Synthetic slip injection:** reuse the sim's existing constraint/attachment primitive (`robot.py:31`'s `create_fixed_constraint` is the general mechanism already used to attach the gripper mechanism to the end-effector; the object-grasp attachment in `env.py` follows the same pattern) — for a controlled slip, either (a) programmatically offset the grasped object's pose by a few mm per frame for `K` frames while leaving the constraint intact (simulates a slow slip inside the fingers), or (b) briefly remove/loosen the grasp constraint so the object free-falls under gravity for `K` frames before re-attaching (simulates a hard drop). Run both as new `tests/test_tracking_slip_pybullet.py` scenarios parameterized by slip onset frame and rate, scoring detection latency against the injected onset time, re-using `tests/tracking_eval.py`'s existing scoring plumbing.

---

### 8. Phased implementation plan

| Phase | Size | What | Files (new/changed) | Tests |
|---|---|---|---|---|
| **P0** | S (half day) | Expose per-point fused 3D cloud from `pose_tracker.py` up to `TrackedObjectState`, gated behind a config flag, no behaviour change when off | `providers/tracker3d/pose_tracker.py` (stash `meta["fused_points"]`/`meta["point_mask"]`), `tracking/types.py` (add `TrackedObjectState.points_3d`/`points_mask`, included in `to_dict()` only if present) | extend `tests/test_pose_tracker.py` to assert the new meta fields' shape/values on a known fixture |
| **P1** | M (1-2 days) | Wire `tracking.motion.MotionBuffer`/`estimate_motion` into `session.py`: one buffer + `MotionClassifier` + `MotionSmoother` per tracked target, fed every frame from P0's data, recomputed every `track_motion_every` (default 3) ref-frames; attach `TrackedObjectState.motion = estimate.to_dict()` | `tracking/session.py` (new `_update_motion` step in the per-target per-frame loop), `config.py` (`track_motion_enable=True`, `track_motion_window=10`, `track_motion_every=3`) | new `tests/test_session_motion.py`: feed a scripted door-scene rollout, assert `TrackedObjectState.motion["kind"]` transitions static→revolute at the expected frame |
| **P2** | S (1 day) | Add `fit_circle_3d` (Taubin/Pratt) to `tracking/motion.py` as tier-B cross-check; wire into `estimate_motion` only as a diagnostic field (`estimate.arc_residual`, `estimate.line_residual`), not yet changing the primary `kind` decision | `tracking/motion.py` (+`CircleFit` dataclass, `fit_circle_3d`) | extend `tests/test_motion_unit.py` with synthetic collinear-vertical-bar fixture reproducing §2's degeneracy, assert tier-B disagreement is flagged |
| **P3** | M (1-2 days) | New monitors (`grasp_slip_trend`, `door_stuck`, `angle_progress`, `wrong_object_moved`) in `tracking/monitors.py`; structured `events` list in `tracking/report.py`; new `api.get_track_features`/`api.estimate_articulation` + `config.GET_TRACK_FEATURES` IPC message + `env.py` handler | `tracking/monitors.py`, `tracking/report.py`, `api.py`, `env.py`, `config.py` | `tests/test_monitors_unit.py` additions (one test per new monitor, scripted frame sequences); `tests/test_tracking_unit.py` for the new IPC round trip |
| **P4** | S (1 day) | `SKILL.md` snippet (§6) + a priori heuristic in whatever currently builds `track_objects` targets for door/cabinet/drawer subtasks | `prompts/skills/open-close-door-cabinet-drawer/SKILL.md`, the skill-invoking code path | manual rollout smoke test on the existing door scene; no new unit tests needed (prompt-only change) |
| **P5** | M (2-3 days) | Evaluation harness extensions (§7): axis/pull-direction-error-over-time scoring, synthetic slip injection, false-alarm-rate baseline | `tests/test_tracking_door_pybullet.py` (extend `DoorSceneDriver` with GT hinge axis export), new `tests/test_tracking_slip_pybullet.py`, `tests/tracking_eval.py` (new scoring functions) | the new test files themselves are the deliverable |
| **P6 (stretch)** | L | BIC-formalized model selection (toggle, §1); IMM filter (only if P5 shows classification flicker is the dominant error source); demonstration-learning ingestion pipeline (§6) | `tracking/motion.py` (`track_motion_model_select`), `tracking/kalman_imm.py` (new, only if needed), new `tracking/demo_ingest.py` | — |

P0–P3 are the minimum to close the "still in flight" gap the plan already flagged; P4 is the cheapest, highest-leverage change (the skill file currently never calls tracking at all); P5 is what turns all of the above from "plausible" into "measured," matching this repo's existing evidence-first documentation style (`docs/plans/3d_tracking.md`'s own decision log, §10, which never adopts a change without a measured before/after).

---

## References

**Internal (this repo):** `tracking/motion.py`, `tracking/monitors.py`, `tracking/monitor.py`, `tracking/types.py`, `tracking/kalman.py`, `tracking/session.py`, `tracking/report.py`, `providers/tracker3d/pose_tracker.py`, `api.py`, `config.py`, `prompts/skills/open-close-door-cabinet-drawer/SKILL.md`, `docs/plans/3d_tracking.md` (§0, §6, §7, §9, §10, §13.11), `tests/test_tracking_door_pybullet.py`.

**External:**
- Sturm, Stachniss, Burgard. *A Probabilistic Framework for Learning Kinematic Models of Articulated Objects.* JAIR 41 (2011). [arXiv:1405.7705](https://arxiv.org/abs/1405.7705)
- Killick, Fearnhead, Eckley. *Optimal Detection of Changepoints with a Linear Computational Cost.* JASA 107(500), 2012. [doi:10.1080/01621459.2012.737745](https://doi.org/10.1080/01621459.2012.737745) (PELT)
- Adams, MacKay. *Bayesian Online Changepoint Detection.* [arXiv:0710.3742](https://arxiv.org/abs/0710.3742) (BOCPD)
- Blom, Bar-Shalom. *The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients.* IEEE TAC 33(8), 1988. Overview: [Wikipedia](https://en.wikipedia.org/wiki/Interacting_multiple_model_algorithm)
- Jain, Lioutikov, Chuck, Niekum. *ScrewNet: Category-Independent Articulation Model Estimation from Depth Images Using Screw Theory.* [arXiv:2008.10518](https://arxiv.org/abs/2008.10518)
- Jiang, Liu, Chen, Wang. *Ditto: Building Digital Twins of Articulated Objects from Interaction.* [arXiv:2202.08227](https://arxiv.org/abs/2202.08227)
- Liu et al. *PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects.* [arXiv:2308.07391](https://arxiv.org/abs/2308.07391)
- Liu et al. *ArtGS: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting.* [arXiv:2502.19459](https://arxiv.org/abs/2502.19459)
- Li et al. *Category-Level Articulated Object Pose Estimation.* [arXiv:1912.11913](https://arxiv.org/abs/1912.11913) (ANCSH)
- Xu et al. *UMPNet: Universal Manipulation Policy Network for Articulated Objects.* [arXiv:2109.05668](https://arxiv.org/abs/2109.05668)
- Mo et al. *Where2Act: From Pixels to Actions for Articulated 3D Objects.* [arXiv:2101.02692](https://arxiv.org/abs/2101.02692)
- Eisner et al. *FlowBot3D: Learning 3D Articulation Flow to Manipulate Articulated Objects.* [arXiv:2205.04382](https://arxiv.org/abs/2205.04382)
- Geng et al. *GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts.* [arXiv:2211.05272](https://arxiv.org/abs/2211.05272)
- Jain, Kemp. *Pulling Open Novel Doors and Drawers with Equilibrium Point Control.* ICRA 2010.
- Arduengo, Torras, Sentis. *Robust and Adaptive Door Operation with a Mobile Robot.* [arXiv:1902.09051](https://arxiv.org/abs/1902.09051)
- Zhou et al. *Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection.* CVPR 2025. [arXiv:2412.04455](https://arxiv.org/abs/2412.04455), [project page](https://zhoues.github.io/Code-as-Monitor/)
- Duan et al. *AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation.* [arXiv:2410.00371](https://arxiv.org/abs/2410.00371)
- Liu et al. *REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction.* [arXiv:2306.15724](https://arxiv.org/abs/2306.15724)
- Guo et al. *DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment.* [arXiv:2307.00329](https://arxiv.org/abs/2307.00329)
- Huang et al. *Inner Monologue: Embodied Reasoning through Planning with Language Models.* [arXiv:2207.05608](https://arxiv.org/abs/2207.05608)
- Ren et al. *Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners.* [arXiv:2307.01928](https://arxiv.org/abs/2307.01928) (KnowNo)
- Huang et al. *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation.* [arXiv:2409.01652](https://arxiv.org/abs/2409.01652)
- Bharadhwaj et al. *Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation.* [arXiv:2405.01527](https://arxiv.org/abs/2405.01527)
- Wen et al. *Any-point Trajectory Modeling for Policy Learning.* [arXiv:2401.00025](https://arxiv.org/abs/2401.00025) (ATM)
- Xu et al. *Flow as the Cross-Domain Manipulation Interface.* [arXiv:2407.15208](https://arxiv.org/abs/2407.15208) (Im2Flow2Act)
- Yang et al. *General Flow as Foundation Affordance for Scalable Robot Learning.* [arXiv:2401.11439](https://arxiv.org/abs/2401.11439)
- Lu et al. *VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation.* [arXiv:2503.07135](https://arxiv.org/abs/2503.07135)
- Jain et al. *Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation.* [arXiv:2409.16283](https://arxiv.org/abs/2409.16283)
- Li et al. *ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning.* [arXiv:2503.21860](https://arxiv.org/abs/2503.21860)

**Gaps / not independently verified:** "ArtiPoint" (named in the prompt) and "Dream2Manip" could not be located via arXiv title search in the time available — likely a mis-remembered title or a non-arXiv venue; recommend dropping or re-confirming the exact title/authors before citing. The exact object-grasp constraint call in `env.py` (vs. the confirmed gripper-to-EE constraint at `robot.py:31`) was not traced line-by-line for the §7 slip-injection proposal — confirm the precise API before implementing P5.
