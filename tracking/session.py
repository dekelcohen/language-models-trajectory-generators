"""TrackingSession - the per-rollout tracker orchestrator.

Lives in the **simulator process** (the cameras, depth buffers and robot state are there),
and is driven once per recorded keyframe by ``Robot.step_env_and_record``. One frame:

    1. gather a :class:`tracking.types.CameraView` per camera (RGB + metric depth + matrices)
    2. update each object's per-camera point tracker  -> 2D points + confidence
    3. deproject those points with that camera's depth -> a world point per camera
    4. score each camera (:mod:`tracking.health`) and re-seed unhealthy cameras from the
       healthy one - this is what recovers from arm occlusion and what seeds the wrist
       camera once the object finally enters its view
    5. lift the surviving cameras to one world point via a
       :class:`providers.tracker3d.base.MultiCamTracker3D` (``depth_fusion`` by default,
       which is the weighted average this session has always used)
    6. read the gripper pose from FK, optionally cross-check it visually
    7. run the monitor; an ``abort`` latches ``self.aborted`` so ``EXECUTE_TRAJECTORY``
       can stop between waypoints

Step 5 is a provider seam so that geometry-based lifts (``triangulate``, ``lapa``) can
replace depth sampling *without* touching re-seeding, health scoring, monitors, reporting
or the abort latch - those are trust and policy concerns that apply to every lift.

Every step is defensive: any exception is caught, logged once and downgraded, because a
tracking bug must never break a rollout that would otherwise have succeeded.
"""

import time

import numpy as np

import config
from providers.tracker3d.factory import get_tracker3d
from providers.trackers.factory import get_tracker
from tracking import geometry
from tracking.health import CamHealthState, decide_reseeds, score_camera
from tracking.monitor import MonitorRunner
from tracking.realtime import LatencyModel, RealtimePipeline, policy_for
from tracking.report import TrackingReporter
from sim_adapter.transforms import rotation_matrix
from tracking.types import (
    STATUS_ABORT,
    STATUS_RECORD,
    CamTrack,
    GripperState,
    MonitorResult,
    ObjectPose,
    ReseedEvent,
    TrackedObjectState,
    TrackFrameReport,
)


class _TargetTrackers:
    """Per-object tracker + health state for every camera."""

    def __init__(self, name, seed_points_world, cameras, provider, tracker_kwargs):
        self.name = name
        self.seed_points_world = np.asarray(seed_points_world, dtype=float).reshape(-1, 3)
        self.trackers = {cam: get_tracker(provider, **tracker_kwargs) for cam in cameras}
        self.health = {cam: CamHealthState() for cam in cameras}
        self.seeded = {cam: False for cam in cameras}
        # Which seed point each camera's tracked point corresponds to. A re-seed drops seed
        # points its camera cannot image, so the arrays are *not* positionally aligned
        # across cameras - a triangulator that zips them would fuse different physical
        # points into a confident, smooth, wrong 3D estimate. ``-1`` marks a point with no
        # correspondence (the centroid fallback).
        self.point_index = {cam: None for cam in cameras}
        # Local offsets of the seed points around their centroid. Re-seeding only knows a
        # single world point (the donor's estimate), so the offsets rebuild a point *set*
        # around it and the object keeps multi-point outlier rejection after a re-seed.
        centroid = self.seed_points_world.mean(axis=0)
        self.local_offsets = self.seed_points_world - centroid
        self.last_world_point = centroid
        self.lost_frames = 0
        # Pose-fitting lifts only (``pose_tracker``): the last *measured* relative pose,
        # its mode, and the object's pose in the gripper frame while it is held.
        self.last_pose = None
        self.pose_mode = None
        self.grip_rel = None


class TrackingSession:
    """Tracks a set of world-space targets (and the gripper) through a rollout."""

    def __init__(self, robot=None, env=None, cameras=None, provider=None, monitor=None,
                 track_gripper=True, interval=None, logger=None, run_id=None,
                 write_jsonl=True, tracker_kwargs=None, output_dir=None, save_depth=False,
                 tracker3d=None, tracker3d_kwargs=None, camera_fps=None, latency=None,
                 min_period_s=None):
        self.robot = robot
        self.env = env
        self.cameras = tuple(cameras or config.tracking_cameras)
        self.provider = provider or config.tracker_provider_default
        self.tracker_kwargs = dict(tracker_kwargs or {})
        self.track_gripper = bool(track_gripper)
        self.interval = max(1, int(interval if interval is not None else config.track_interval))
        self.logger = logger
        self.save_depth = bool(save_depth)
        self.targets = {}
        self.gripper_target = None
        self.frame_idx = 0
        self._calls = 0
        self.aborted = False
        self.abort_reason = None
        self.active = True
        self.last_report = None
        self.errors = 0
        self.monitor = MonitorRunner.from_spec(monitor, logger=logger)
        self.reporter = TrackingReporter(run_id=run_id, output_dir=output_dir, logger=logger,
                                         write_jsonl=write_jsonl)
        # The 3D lift. ``depth_fusion`` reproduces the historical weighted average exactly,
        # so the default path is unchanged and ``tests/golden/**`` stay valid.
        self.tracker3d_kwargs = dict(tracker3d_kwargs or {})
        self.tracker3d_kwargs.setdefault("logger", logger)
        if isinstance(tracker3d, str) or tracker3d is None:
            self.tracker3d_name = (tracker3d
                                   or getattr(config, "tracker3d_provider_default", "depth_fusion"))
            self.tracker3d = get_tracker3d(self.tracker3d_name, **self.tracker3d_kwargs)
        else:
            self.tracker3d = tracker3d
            self.tracker3d_name = getattr(tracker3d, "name", "custom")
        self.lift_errors = 0
        # Real-time model (tracking/realtime.py). ``camera_fps`` None/<=0 keeps the legacy
        # "one on_frame call = one tracked frame, committed immediately" contract; > 0 makes
        # ``tick(sim_time)`` the entry point: a camera clock plus latency charged in sim time.
        self.pipeline = None
        if camera_fps is not None and float(camera_fps) > 0:
            self.pipeline = RealtimePipeline(camera_fps=camera_fps, latency=LatencyModel(latency),
                                             policy=policy_for(self.provider),
                                             min_period_s=min_period_s)
            self._log(f"[tracking] realtime: {self.pipeline.describe()}")
        self._timing = None
        self._warmup_done = False

    # -- registration ------------------------------------------------------
    def add_target(self, name, world_points):
        """Register an object by the 3D world points that should be followed."""
        pts = np.asarray(world_points, dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            raise ValueError(f"track target '{name}' needs at least one world point")
        self.targets[name] = _TargetTrackers(name, pts, self.cameras, self.provider,
                                             self.tracker_kwargs)
        self._log(f"[tracking] registered '{name}' with {pts.shape[0]} seed point(s)")

    def add_targets(self, targets):
        for spec in targets or []:
            self.add_target(spec["name"], spec["world_points"])

    # -- per-frame entry point --------------------------------------------
    def on_frame(self, views=None, gripper_pose=None, trajectory_step=0, rgb_paths=None):
        """Advance tracking by one frame. Returns the frame report (or ``None`` if skipped).

        ``views``: ``{cam_name: CameraView}``. When omitted the session renders them
        itself from ``self.robot``/``self.env``.
        """
        if not self.active or self.aborted:
            return None
        self._calls += 1
        if (self._calls - 1) % self.interval != 0:
            return None
        report = self._compute(views, gripper_pose, trajectory_step, rgb_paths)
        if report is not None:
            self._commit(report)
        return report

    def tick(self, sim_time, views_fn=None, gripper_pose=None, trajectory_step=0):
        """Real-time entry point: call with the sim clock, as often as the sim steps.

        Publishes results whose latency has elapsed (only then do they reach
        ``last_report``, the reporter, the monitor's abort latch), and - when the camera
        clock has a new frame and the tracker can take it - renders it (``views_fn`` or
        :meth:`capture_views`), computes, and charges the compute time in sim time.
        Returns the report computed on this call (possibly not yet published), else None.
        Without a pipeline (``camera_fps`` <= 0) it degrades to :meth:`on_frame`.
        """
        if not self.active or self.aborted:
            return None
        pipe = self.pipeline
        if pipe is None:
            if callable(gripper_pose):
                gripper_pose = gripper_pose()
            return self.on_frame(views=views_fn() if views_fn else None,
                                 gripper_pose=gripper_pose, trajectory_step=trajectory_step)
        now = float(sim_time)
        pipe.publish(now, self._commit_result)
        if self.aborted:
            return None
        frame = pipe.next_frame(now)
        if frame is None or not pipe.accepts(frame[1]):
            return None
        frame_idx, frame_t = frame
        views = views_fn() if views_fn is not None else self.capture_views()
        if callable(gripper_pose):
            gripper_pose = gripper_pose()
        self._timing = {"track_s": {}, "flushes": 0, "flush_s": 0.0}
        set_time = getattr(self.tracker3d, "set_time", None)
        if callable(set_time):
            set_time(frame_t)
        t0 = time.perf_counter()
        report = self._compute(views, gripper_pose, trajectory_step, None)
        wall = time.perf_counter() - t0
        timing, self._timing = self._timing, None
        if report is None:
            return None
        warmup = timing.pop("warmup_s", 0.0)
        wall = max(0.0, wall - warmup)
        result = pipe.submit(frame_idx, frame_t, wall, timing, payload=report, views=views)
        report.sim_time = frame_t
        report.camera_frame = frame_idx
        report.available_at = result.available_t
        report.latency_s = result.available_t - frame_t
        report.timing = {"wall_s": round(wall, 5), "cost_s": round(result.cost_s, 5),
                         "queue_wait_s": round(result.queue_wait_s, 5),
                         "warmup_s": round(warmup, 3) if warmup else 0.0,
                         "flushes": timing["flushes"], "flush_s": round(timing["flush_s"], 5),
                         "track_s": {c: round(v, 5) for c, v in timing["track_s"].items()}}
        pipe.publish(now, self._commit_result)
        return report

    def drain(self):
        """Publish every result still in flight (end of a rollout / eval run)."""
        if self.pipeline is not None and self.pipeline.pending:
            self.pipeline.publish(float("inf"), self._commit_result)

    def estimate_age(self, sim_time):
        """Sim seconds between ``sim_time`` and the frame behind ``last_report``."""
        if self.pipeline is not None:
            return self.pipeline.age(float(sim_time))
        return None

    def _compute(self, views, gripper_pose, trajectory_step, rgb_paths):
        """Track one frame and advance all state; does *not* publish the result."""
        try:
            if views is None:
                views = self.capture_views()
            if not views:
                return None
            if self.save_depth:
                self._dump_depth(views)
            report = self._process(views, gripper_pose, trajectory_step, rgb_paths or {})
        except Exception as exc:            # tracking must not break the rollout
            self.errors += 1
            if self.errors <= 3:
                self._log(f"[tracking] frame {self.frame_idx} failed: {type(exc).__name__}: {exc}")
            return None
        self.frame_idx += 1
        return report

    def _commit_result(self, result):
        self._commit(result.payload)

    def _commit(self, report):
        """Make a computed report the current one: reporter, ``last_report``, abort latch."""
        if self.aborted:
            return
        self.last_report = report
        self.reporter.add(report)
        if report.monitor.status == STATUS_ABORT:
            self.aborted = True
            self.abort_reason = report.monitor.reason
            self._log(f"[tracking] ABORT at frame {report.frame_idx}: {self.abort_reason}")
        elif report.monitor.status == STATUS_RECORD:
            self._log(f"[tracking] flagged frame {report.frame_idx}: {report.monitor.reason}")
        return report

    # -- capture -----------------------------------------------------------
    def capture_views(self):
        """Render every tracked camera into a :class:`CameraView` (no disk I/O)."""
        views = {}
        for cam in self.cameras:
            try:
                views[cam] = self.robot.capture_camera_view(cam, self.env)
            except Exception as exc:
                self._log(f"[tracking] camera '{cam}' capture failed: {type(exc).__name__}: {exc}")
        return views

    # -- core --------------------------------------------------------------
    def _dump_depth(self, views):
        """--track-save-depth: persist the metric depth arrays behind each decision."""
        import os

        folder = os.path.join(self.reporter.output_dir, "depth")
        try:
            os.makedirs(folder, exist_ok=True)
            for cam, view in views.items():
                np.save(os.path.join(folder, f"{cam}_{self.frame_idx:05d}.npy"),
                        np.asarray(view.depth, dtype=np.float32))
        except OSError as exc:
            self._log(f"[tracking] could not save depth: {exc}")
            self.save_depth = False

    def _process(self, views, gripper_pose, trajectory_step, rgb_paths):
        report = TrackFrameReport(frame_idx=self.frame_idx, trajectory_step=trajectory_step,
                                  rgb_paths=dict(rgb_paths))
        report.gripper = self._gripper_state(gripper_pose, views)

        for name, target in self.targets.items():
            report.objects[name] = self._process_target(target, views, report.gripper)

        report.monitor = self.monitor(report)
        return report

    def _process_target(self, target, views, gripper=None):
        state = TrackedObjectState(name=target.name)
        z_eyes = {}

        # 1. update (or note the absence of) each camera's tracker
        for cam, view in views.items():
            track = CamTrack(cam=cam, seeded=target.seeded.get(cam, False))
            if track.seeded:
                self._update_camera_track(target, cam, view, track)
            state.cams[cam] = track
            if track.world_point is not None:
                _pixel, z_eye = geometry.project_world_to_pixel(view, track.world_point)
                z_eyes[cam] = z_eye
            track.health = score_camera(track, target.health[cam], z_eyes.get(cam))

        # 2. provisional fusion feeds the re-seed policy (a camera with no healthy peer can
        #    still be re-seeded from the previous fused estimate).
        fused, disagreement, _used = self._fuse(state, views, z_eyes, commit=False)
        fallback = fused if fused is not None else target.last_world_point

        # 3. cross-camera repair
        for cam, reason, donor, world_point in decide_reseeds(
                target.name, state.cams, target.health, self.frame_idx, fallback):
            event = self._reseed(target, cam, views.get(cam), world_point, reason, donor,
                                 pose=state.pose or target.last_pose)
            state.reseeds.append(event)
            if event.applied:
                track = state.cams[cam]
                self._update_camera_track(target, cam, views[cam], track, just_seeded=True)
                if track.world_point is not None:
                    _pixel, z_eye = geometry.project_world_to_pixel(views[cam], track.world_point)
                    z_eyes[cam] = z_eye
                track.health = score_camera(track, target.health[cam], z_eyes.get(cam))
                track.reseeded_reason = reason
            elif state.cams[cam].seeded:
                # The policy wanted this camera repaired but the donor point is not
                # imaged there (it is behind the occluder). Its own estimate therefore
                # points at whatever it latched onto - typically the occluding arm, with
                # a confidently rising template score - so it must not be fused until it
                # recovers. Health is zeroed so it also cannot act as a donor.
                state.cams[cam].status = "rejected"
                state.cams[cam].health = 0.0

        # 4. final fusion over the repaired tracks
        fused, disagreement, used = self._fuse(state, views, z_eyes)
        state.world_point = fused
        state.disagreement = disagreement
        state.confidence = float(np.mean([state.cams[c].confidence for c in used])) if used else 0.0
        state.lost = fused is None

        # 5. pose-driven repair (only providers that fit a pose ask for it)
        self._pose_repairs(target, state, views, gripper)

        if fused is not None:
            target.last_world_point = state.world_point
            target.lost_frames = 0
            for cam in state.cams:
                # Only a trusted track updates its own history: letting a camera that
                # jumped onto an occluder rewrite its reference point would make the very
                # next frame look perfectly continuous and hide the failure.
                if state.cams[cam].status == "ok" and not state.predicted:
                    target.health[cam].last_world_point = state.cams[cam].world_point
        else:
            target.lost_frames += 1
        return state

    def _pose_repairs(self, target, state, views, gripper):
        """Yellow force-correction, Red re-acquisition and Black gripper-FK coasting.

        * A measured pose is remembered (``target.last_pose``) so every later re-seed
          places the seed pattern *rotated*, not just shifted.
        * ``correct_cams`` (yellow): one camera's fit is good, the others disagree - re-seed
          them from the good pose, i.e. "project Cam1's points onto Cam2".
        * ``reacquire`` (red/black): no camera measures the object; re-seed any camera in
          which the *predicted* points are depth-verified visible, which is how tracking
          resumes on the object instead of on whatever the tracker drifted onto.
        * black + grasped: the pose follows the gripper (FK). Flagged ``source="fk"`` and
          ``predicted``, so monitors never read it as evidence that the object is attached.
        """
        pose = state.pose
        meta = state.lift_meta or {}
        if pose is None:
            return
        if pose.measured:
            target.last_pose = pose
            target.pose_mode = pose.mode
            if gripper is not None and gripper.world_pos is not None:
                target.grip_rel = self._relative_to_gripper(pose, target, gripper)
            else:
                target.grip_rel = None
        elif pose.mode == "black" and getattr(target, "grip_rel", None) is not None \
                and gripper is not None and gripper.world_pos is not None:
            fk_pose = self._pose_from_gripper(target, gripper)
            if fk_pose is not None:
                state.pose = fk_pose
                state.world_point = fk_pose.apply(target.seed_points_world.mean(axis=0))[0]
                meta["fk_coast"] = True

        cooldown = config.track_pose_reseed_cooldown
        wanted = []
        if meta.get("correct_cams"):
            wanted = [(c, "pose_correction") for c in meta["correct_cams"]]
        elif meta.get("reacquire"):
            wanted = [(c, "reacquire") for c in state.cams]
        for cam, reason in wanted:
            view = views.get(cam)
            health = target.health.get(cam)
            if view is None or health is None:
                continue
            if health.last_reseed_frame is not None and \
                    self.frame_idx - health.last_reseed_frame < cooldown:
                continue
            event = self._reseed_from_pose(target, cam, view, state.pose, reason)
            meta.setdefault("pose_reseed", {})[cam] = \
                ("ok: " if event.applied else "skip: ") + (event.detail or "")
            if event.applied:
                state.reseeds.append(event)

    @staticmethod
    def _grip_matrix(gripper):
        T = np.eye(4)
        T[:3, :3] = rotation_matrix(gripper.world_quat)
        T[:3, 3] = np.asarray(gripper.world_pos, dtype=float)
        return T

    def _relative_to_gripper(self, pose, target, gripper):
        """Object pose in the gripper frame, only while the object is actually in hand."""
        centre = pose.apply(target.seed_points_world.mean(axis=0))[0]
        if np.linalg.norm(centre - np.asarray(gripper.world_pos, dtype=float)) > \
                config.track_attach_max_dist:
            return None
        T_o = np.eye(4)
        T_o[:3, :3] = np.asarray(pose.R)
        T_o[:3, 3] = np.asarray(pose.t)
        return np.linalg.inv(self._grip_matrix(gripper)) @ T_o

    def _pose_from_gripper(self, target, gripper):
        T = self._grip_matrix(gripper) @ target.grip_rel
        return ObjectPose(R=T[:3, :3], t=T[:3, 3], source="fk", mode="black")

    def _reseed_from_pose(self, target, cam, view, pose, reason):
        """Re-seed ``cam`` at the pose-projected seed points that it genuinely images.

        Each seed point is kept only if the camera's depth at its projected pixel agrees
        with the point (Step 0.3 of the design) - so a re-seed never lands on the occluder
        that is still in front of the object, and correspondence stays exact.
        """
        event = ReseedEvent(cam=cam, obj=target.name, reason=reason, donor="pose")
        world = pose.apply(target.seed_points_world)
        pixels, indices = [], []
        for i, wp in enumerate(world):
            visible, px, _detail = geometry.is_visible(view, wp)
            if visible:
                pixels.append(px)
                indices.append(i)
        min_frac = (config.track_pose_join_min_frac if reason == "unseeded"
                    else config.track_pose_reseed_min_frac)
        min_pts = max(config.track_pose_min_points, int(np.ceil(min_frac * len(world))))
        if len(pixels) < min_pts:
            event.detail = f"{len(pixels)}/{len(world)} predicted points depth-verified"
            return event
        failure = self._install_seed(target, cam, view, pixels, indices)
        if failure:
            event.detail = failure
            return event
        event.applied = True
        event.detail = f"{len(pixels)}/{len(world)} point(s) from {pose.source} pose ({pose.mode})"
        self._log(f"[tracking] {reason}: re-seeded '{target.name}' in {cam} - {event.detail}")
        return event

    def _install_seed(self, target, cam, view, pixels, indices):
        """(Re-)initialise ``cam``'s tracker on ``pixels`` (seed ids ``indices``).

        Returns ``None`` on success or a failure message. Clears the camera's health history:
        a fresh track has no continuity with the one it replaces.
        """
        try:
            tracker = target.trackers[cam]
            tracker.reset()
            tracker.init(view.rgb, np.asarray(pixels, dtype=float), obj_id=target.name)
        except Exception as exc:
            return f"tracker init failed: {type(exc).__name__}: {exc}"
        if tracker.n_points != len(indices):
            # Correspondence would be silently wrong (row j != seed point j): refuse it.
            tracker.reset()
            return (f"tracker kept {tracker.n_points} of {len(indices)} seed points "
                    "(violates the one-row-per-seed contract)")
        target.seeded[cam] = True
        target.point_index[cam] = np.asarray(indices, dtype=int)
        health = target.health[cam]
        health.last_reseed_frame = self.frame_idx
        health.low_conf_frames = 0
        health.lost_frames = 0
        health.last_world_point = None
        return None

    def _timed_update(self, tracker, cam, rgb):
        """``tracker.update`` plus, inside :meth:`tick`, the timing the latency model needs:
        per-camera 2D-tracker time and the CoTracker window flushes (so a device profile
        can swap their CPU cost for the target GPU's). The process's first flush is the
        one-off cuDNN/allocator warm-up (~6 s) a deployed robot pays before it starts, so it
        is reported as ``warmup_s`` and not charged."""
        timing = self._timing
        if timing is None:
            return tracker.update(rgb)
        before = int(getattr(tracker, "_flushes", 0) or 0)
        t0 = time.perf_counter()
        result = tracker.update(rgb)
        dt = time.perf_counter() - t0
        timing["track_s"][cam] = timing["track_s"].get(cam, 0.0) + dt
        meta = getattr(result, "meta", None) or {}
        after = meta.get("flushes")
        if after is not None and int(after) != before:
            if not self._warmup_done:
                self._warmup_done = True
                timing["warmup_s"] = timing.get("warmup_s", 0.0) + dt
            else:
                timing["flushes"] += 1
                timing["flush_s"] += dt
        return result

    def _update_camera_track(self, target, cam, view, track, just_seeded=False):
        """Run one tracker update and turn its 2D points into a world point."""
        tracker = target.trackers[cam]
        if not tracker.initialised:
            track.seeded = False
            track.status = "unseeded"
            return
        result = self._timed_update(tracker, cam, view.rgb)
        track.seeded = True
        track.points_2d = np.asarray(result.points, dtype=float)
        track.visible = np.asarray(result.visible, dtype=bool)
        track.confidence = float(result.confidence)
        track.stale_frames = int((getattr(result, "meta", None) or {}).get("stale_frames", 0) or 0)
        if result.n_visible == 0:
            track.status = "lost"
            return
        world_points, valid = geometry.points_to_world(view, track.points_2d, track.visible)
        centroid, n_used = geometry.robust_centroid(world_points, valid)
        track.depth_valid = centroid is not None and n_used > 0
        track.world_point = centroid
        if centroid is None:
            track.status = "occluded"
        elif self._is_super_physical_jump(target, cam, centroid, just_seeded):
            # A metre-scale hop between two frames is not the object moving, it is the
            # tracker having latched onto something else (usually an occluder that just
            # slid in front). Templates match that occluder confidently for many frames,
            # so confidence alone would never catch it - the camera stays untrusted until
            # a re-seed repairs it.
            track.status = "jumped"
        elif track.confidence < config.track_reseed_conf:
            track.status = "low_confidence"
        else:
            track.status = "ok"

    @staticmethod
    def _is_super_physical_jump(target, cam, world_point, just_seeded):
        last = target.health[cam].last_world_point
        if just_seeded or last is None:
            return False
        return bool(np.linalg.norm(np.asarray(world_point, dtype=float)
                                   - np.asarray(last, dtype=float)) > config.track_max_jump_m)

    def _fuse(self, state, views, z_eyes, commit=True):
        """Delegate the 3D lift to the configured provider.

        ``z_eyes`` is no longer needed here - a provider that wants eye-space depth derives
        it itself - but the argument is kept so the call sites read the same.

        ``commit=False`` is the provisional pass that feeds the re-seed policy; a stateful
        provider (``pose_tracker``'s Kalman filter) must not advance twice per frame, so it
        is asked to roll its state back afterwards.

        A provider that fails outright must not lose the object: ``depth_fusion`` is always
        available and always correct-ish, so it is the fallback. That matters most for the
        experimental providers, where a checkpoint or a GPU could disappear mid-rollout.
        """
        try:
            extra = {} if commit or not getattr(self.tracker3d, "stateful", False) \
                else {"commit": False}
            result = self.tracker3d.lift(
                state.name, views, state.cams,
                point_index={cam: self.targets[state.name].point_index.get(cam)
                             for cam in state.cams} if state.name in self.targets else None,
                seed_world_points=(self.targets[state.name].seed_points_world
                                   if state.name in self.targets else None),
                **extra,
            )
        except Exception as exc:
            self.lift_errors += 1
            if self.lift_errors <= 3:
                self._log(f"[tracking] 3D lift '{self.tracker3d_name}' failed for "
                          f"'{state.name}': {type(exc).__name__}: {exc}")
            result = None

        if result is None or not result.ok:
            if self.tracker3d_name != "depth_fusion":
                result = self._fallback_lift(state, views, result)
            elif result is None:
                return None, None, []

        state.lift_meta = dict(result.meta)
        state.pose = getattr(result, "pose", None)
        state.predicted = bool(getattr(result, "predicted", False))
        return result.world_point, result.disagreement, list(result.used_cams)

    def _fallback_lift(self, state, views, failed):
        """Depth fusion as the safety net when an experimental provider produces nothing."""
        from providers.tracker3d.depth_fusion import DepthFusionTracker3D
        if getattr(self, "_fallback3d", None) is None:
            self._fallback3d = DepthFusionTracker3D(logger=self.logger)
        result = self._fallback3d.lift(state.name, views, state.cams)
        result.meta["fell_back_from"] = self.tracker3d_name
        if failed is not None:
            result.meta["why"] = failed.meta.get("why")
        return result

    def _reseed(self, target, cam, view, world_point, reason, donor, pose=None):
        """Project a donor world point into ``cam`` and re-initialise its tracker there.

        With a pose-fitting provider (``stateful``, i.e. ``pose_tracker``) the seed pattern is
        instead the *template* projected through the current pose and depth-verified point by
        point (:meth:`_reseed_from_pose`, design Steps 0.2-0.3). The generic path below snaps
        each point onto whatever surface the receiving camera renders, which is fine for a
        centroid but gives point ``i`` in this camera a different physical location than
        template point ``i`` - and a rigid fit over broken correspondences cannot converge
        (measured: 2-5 cm RMSE, red/black for the whole wrist-only phase of the hand-over).
        """
        if getattr(self.tracker3d, "stateful", False) and view is not None:
            if pose is None:        # before the first fit: the seed-time configuration
                pose = ObjectPose(R=np.eye(3), t=np.zeros(3), source="seed", mode="seed")
            event = self._reseed_from_pose(target, cam, view, pose, reason)
            event.donor = donor if donor is not None else event.donor
            return event
        event = ReseedEvent(cam=cam, obj=target.name, reason=reason, donor=donor)
        if view is None or world_point is None:
            event.detail = "no view or donor point"
            return event

        snapped, pixel, detail = geometry.surface_point(view, world_point)
        if snapped is None:
            event.detail = f"donor point not imaged by {cam}: {detail}"
            return event

        # Rebuild a point set around the donor estimate using the seed geometry, so
        # multi-point outlier rejection survives a re-seed. With a measured pose the
        # pattern is rotated too - a shifted-only pattern on a turned object (a door
        # handle swung 40 deg) lands the seed points off the object.
        offsets = target.local_offsets
        if target.last_pose is not None:
            offsets = offsets @ np.asarray(target.last_pose.R, dtype=float).T
        seed_world = world_point[None, :] + offsets
        pixels = []
        indices = []
        for i, wp in enumerate(seed_world):
            _snapped, px, _why = geometry.surface_point(view, wp)
            if _snapped is not None:
                pixels.append(px)
                indices.append(i)
        if not pixels:
            # Nothing but the donor centroid is imaged here. It has no seed-point identity,
            # so it is marked -1: a triangulator must not match it against another camera's
            # seed point 0, which is a different place on the object.
            pixels = [pixel]
            indices = [-1]

        failure = self._install_seed(target, cam, view, pixels, indices)
        if failure:
            event.detail = failure
            return event
        event.applied = True
        event.detail = f"{len(pixels)} point(s) at {np.round(np.mean(pixels, axis=0), 1).tolist()}"
        self._log(f"[tracking] re-seeded '{target.name}' in {cam} ({reason}"
                  f"{'' if donor is None else f', donor={donor}'})")
        return event

    # -- gripper -----------------------------------------------------------
    def _gripper_state(self, gripper_pose, views):
        gs = GripperState()
        if gripper_pose is not None:
            gs.world_pos = np.asarray(gripper_pose.get("position"), dtype=float) \
                if gripper_pose.get("position") is not None else None
            gs.world_quat = gripper_pose.get("orientation_q")
            gs.opening = gripper_pose.get("opening")
        elif self.robot is not None:
            try:
                pos, quat = self.robot.sim.get_link_pose(self.robot.id, self.robot.ee_index)
                gs.world_pos = np.asarray(pos, dtype=float)
                gs.world_quat = quat
            except Exception as exc:
                self._log(f"[tracking] FK gripper pose unavailable: {exc}")

        if not self.track_gripper or gs.world_pos is None:
            return gs

        # Visual cross-check: the gripper's FK position is known, so instead of tracking
        # it we simply verify that the cameras *see* something at that depth. A large
        # discrepancy means the depth/extrinsics disagree with FK, which invalidates every
        # object measurement from that camera - worth logging even though FK stays truth.
        estimates = {}
        for cam, view in views.items():
            visible, pixel, _detail = geometry.is_visible(view, gs.world_pos)
            if not visible:
                continue
            depth = geometry.sample_depth(view, pixel)
            if depth is None:
                continue
            world = geometry.deproject_pixel_to_world(view, pixel, depth)
            _px, z_eye = geometry.project_world_to_pixel(view, world)
            estimates[cam] = (world, geometry.camera_weight(1.0, z_eye, 4, True))
        fused, _disagreement, used = geometry.fuse_world_points(estimates)
        gs.visual_world_pos = fused
        gs.visual_confidence = 1.0 if used else 0.0
        return gs

    # -- lifecycle ---------------------------------------------------------
    def stop(self):
        self.active = False
        self.reporter.close()

    def summary(self):
        summary = self.reporter.summary()
        summary["aborted"] = self.aborted
        summary["abort_reason"] = self.abort_reason or summary.get("abort_reason")
        summary["frames"] = self.frame_idx
        summary["monitor_errors"] = self.monitor.error_count
        if self.pipeline is not None:
            summary["realtime"] = self.pipeline.describe()
        return summary

    def describe(self):
        return self.reporter.describe()

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)


def add_tracking_args(parser, include_enable=True):
    """The rollout-tracking CLI group, shared by ``main.py`` and the Genesis child process.

    ``run_simulation_environment`` falls back to these attributes when a START_TRACKING
    payload omits a field, so every process that hosts a session needs the same namespace.
    ``include_enable=False`` omits ``--tracking`` (only the agent process decides that).
    """
    import argparse

    from providers.tracker3d.factory import SUPPORTED as tracker3d_providers
    from providers.trackers.factory import SUPPORTED as tracker_providers

    group = parser.add_argument_group(
        "rollout tracking (tracking/, providers/trackers/, providers/tracker3d/)")
    if include_enable:
        group.add_argument("--tracking", action=argparse.BooleanOptionalAction,
                           default=config.tracking_enabled_default,
                           help="track the affordance object and gripper in 3D during "
                                "trajectory execution; exposes the track_objects tool to the "
                                "model and lets a monitor abort a sub-task mid-rollout "
                                "(default: off)")
    group.add_argument("--tracker-provider", dest="tracker_provider",
                       choices=list(tracker_providers), default=config.tracker_provider_default,
                       help="2D point tracker used by --tracking: 'template' (default, base "
                            "opencv), 'csrt' (needs opencv-contrib-python), 'cotracker' "
                            "(CoTracker3 online, needs torch + a one-off torch.hub download "
                            "into TORCH_HOME), 'remote' (stub)")
    group.add_argument("--tracker3d", dest="tracker3d", choices=list(tracker3d_providers),
                       default=config.tracker3d_provider_default,
                       help="3D lift: 'depth_fusion' (default, weighted per-camera average), "
                            "'pose_tracker' (per-point lift + epipolar/jump gates + Kabsch "
                            "6-DoF pose + Kalman filter, see docs/plans/3d_tracking.md §9), "
                            "'triangulate', 'weighted_triangulate', 'rigid_refine'")
    group.add_argument("--track-cameras", dest="track_cameras", nargs="+",
                       default=list(config.tracking_cameras), metavar="CAM",
                       help="cameras to track in (default: %(default)s; 'shoulder' is the "
                            "robot-base-mounted third camera)")
    group.add_argument("--track-interval", dest="track_interval", type=int,
                       default=config.track_interval,
                       help="legacy cadence (only with --track-camera-fps 0): run the tracker "
                            "every Nth recorded keyframe (1 = every keyframe)")
    group.add_argument("--track-camera-fps", dest="track_camera_fps", type=float,
                       default=config.track_camera_fps,
                       help="simulated camera rate in sim time; the tracker runs on this clock "
                            "whether or not the robot moves, and pays its compute as sim-time "
                            "latency (default: %(default)s; 0 = legacy motion-gated keyframes). "
                            "Costs ~43 ms of rendering per camera frame")
    group.add_argument("--track-latency", dest="track_latency", default=config.track_latency_mode,
                       help="sim-time latency charged per tracked frame: 'measured' (default, "
                            "this machine's wall time), 'zero' (lock-step) or a profile from "
                            "config.track_latency_profiles, e.g. 'a1000_fp16' "
                            f"({', '.join(sorted(config.track_latency_profiles))})")
    group.add_argument("--track-save-depth", dest="track_save_depth", action="store_true",
                       help="also dump the per-frame metric depth arrays used by tracking "
                            "(debugging; large)")
    group.add_argument("--track-log-dir", dest="track_log_dir",
                       default=config.tracking_output_dir,
                       help="root folder for tracking JSONL logs and summaries")
    return group


def make_session(robot, env, args, targets=None, monitor=None, track_gripper=True,
                 logger=None, run_id=None):
    """Build a session from parsed CLI ``args`` (the shape ``env.py`` receives)."""
    session = TrackingSession(
        robot=robot,
        env=env,
        cameras=getattr(args, "track_cameras", None),
        provider=args.tracker_provider,
        tracker3d=getattr(args, "tracker3d", None),
        monitor=monitor,
        track_gripper=track_gripper,
        interval=args.track_interval,
        logger=logger,
        run_id=run_id or time.strftime("%Y%m%d_%H%M%S"),
        camera_fps=getattr(args, "track_camera_fps", config.track_camera_fps),
        latency=getattr(args, "track_latency", None),
    )
    session.add_targets(targets)
    return session
