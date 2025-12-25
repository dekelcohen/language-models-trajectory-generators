Metaworld + pybullet providers
========================================
Run Metaworld server from another env:
* First time:
  * Install Metaworld + SciPy in a separate virtualenv.
    
	conda create --prefix /mnt/robotics/envs/metaworld python=3.11 -y	  	
	conda activate /mnt/robotics/envs/metaworld
	cd /mnt/robotics/
	git clone https://github.com/dekelcohen/Metaworld.git
	pip install -e .
	pip install packaging
* Every time:	
     How To Use:
	) General: The main process talks to the Metaworld server over JSON. Images and camera calibration flow back for accurate point-clouds.
	 cd /mnt/robotics/language-models-trajectory-generators 
    ) PyBullet (unchanged default):
      python main.py --sim pybullet --depth-format norm_1m
      Optional validation: set DEBUG_DIFF=1 to compare rotation math.
    ) Metaworld via main.py (headless):
	  where python	  
      set METAWORLD_PYTHON=C:\Users\dekelco\.conda\envs\metaworld\python.exe
      set METAWORLD_REPO=D:\NLP\Robotics\Simulators_Envs\Metaworld
      python main.py --sim metaworld --task sawyer_door_v3 --depth-format norm_zfar
    ) Viewer Metaworld - local sanity (viewer, independent of main.py):
	  cd /d D:\NLP\Robotics\VLM_Robotics\language-models-trajectory-generators
	  # Opens MuJoCo viewer, shows env at reset, runs a short passive loop
      %METAWORLD_PYTHON% providers\metaworld_server.py --env sawyer_door_v3 --viewer
    ) Tests
      cd /d D:\NLP\Robotics\VLM_Robotics\language-models-trajectory-generators
	  python -m unittest tests/test_metaworld_server.py -v

	  Config knobs you can tweak

	  - In config.py:
		  - trajectory_log_every = 5
		  - perception_log_first_n = 1
		  - perception_log_interval_frames = 0
	  - These keep logging minimal at runtime. Only every Nth trajectory frame is saved; perception overlays log once at the start unless you set a frame interval.	
	  
      
	  
	  
* Local Mujoco viewer for sanity (Windows dev box)
   - providers/metaworld_server.py: --viewer flag launches the MuJoCo passive viewer, resets the env, runs a few steps to show robot and environment, then exits. This is
   independent of main.py — you can invoke the server directly from your Metaworld Python environment:
          - METAWORLD_PYTHON C:\path\to\metaworld\venv\Scripts\python.exe
          - C:\path\to\metaworld\venv\Scripts\python.exe providers\metaworld_server.py --env sawyer_door_v3 --viewer
		  

Detailed summary of changes pybullet --> support also metaworld
===================================================================
 Here’s a precise, end‑to‑end path to adapt your current PyBullet pipeline to run the Metaworld “open door” task (sawyer_door_v3.py), keeping the same Pipe‑based API between
  main.py and the env process.

  Big Picture

  - Preserve main.py + api.py IPC and message semantics.
  - Swap out the current PyBullet run_simulation_environment for a Metaworld/MuJoCo adapter that implements the same commands (CAPTURE_IMAGES, EXECUTE_TRAJECTORY, etc.).
  - Adjust utils.py camera math and depth conversion to use MuJoCo camera intrinsics/extrinsics instead of PyBullet.
  - Use a simple Cartesian controller to translate your world‑space trajectory points to Metaworld’s 4‑D actions (dx, dy, dz, gripper).

  What Changes Where

  - env.py: replace or add a backend that boots SawyerDoorEnvV3, implements the same loop and message handling, and saves images to the same paths.
  - utils.py: add MuJoCo equivalents for camera intrinsics/extrinsics and world‑point projection.
  - config.py: align start poses and image sizes for Metaworld; optionally add a --sim flag to pick pybullet vs metaworld.
  - Optional: add env_metaworld.py to keep both backends side‑by‑side and switch in main.py.

  File references:

  - env.py:1
  - utils.py:1
  - main.py:1
  - api.py:1
  - config.py:1

  Interface Mapping (PyBullet → Metaworld/MuJoCo)

  - Physics stepping:
      - PyBullet: p.stepSimulation() inside env.update().
      - MuJoCo: call env.step(action); each call advances frame_skip internal steps. No sleep needed.
  - Cameras:
      - PyBullet: p.getCameraImage(...) with viewMatrix/projectionMatrix.
      - MuJoCo: render via env.mujoco_renderer.render(render_mode='rgb_array'/'depth_array', camera_id=...) or env.sim.render(...). Use specific camera names/ids.
  - Poses and transforms:
      - PyBullet: pull quaternion/matrix via pybullet.
      - MuJoCo: camera pose via env.data.cam_xpos[camera_id] and env.data.cam_xmat[camera_id].reshape(3,3). Convert to quaternion with
        scipy.spatial.transform.Rotation.from_matrix(...).as_quat().
  - End effector control:
      - PyBullet: IK → joint positions.
      - Metaworld: 4‑D action [-1,1]^4 ≈ delta XYZ (scaled by action_scale, default 1/100) + gripper effort. No wrist yaw control; ignore yaw or approximate via path.
  - Debug geometry:
      - PyBullet: p.addUserDebugLine/Points.
      - MuJoCo: no simple runtime debug draw; either skip or log. You can overlay in saved images if needed.

  Step‑By‑Step: Env Adapter

  1. Add a new mode/flag.

  - In main.py:1 add an arg --sim {pybullet,metaworld} default pybullet.
  - Branch the Process target to run_metaworld_environment when args.sim == 'metaworld'.

  2. Boot Metaworld door env.

  - Insert sys.path.append(r"D:\NLP\Robotics\Simulators_Envs\Metaworld") before import if not installed as a package.
  - Import and instantiate:
      - from metaworld.envs.sawyer_door_v3 import SawyerDoorEnvV3
      - env = SawyerDoorEnvV3(render_mode=None, width=config.image_width, height=config.image_height)
  - Task gating:
      - SawyerXYZEnv requires a task set before stepping. Use the built‑in task machinery:
          - Simple route: wrap with MT1/MT5/ML1 benchmark APIs and set_task(...) a sampled task, or
          - Direct route: mark _set_task_called = True and rely on reset_model() internal goals (works for V3 but is hacky).
      - Recommended: use the public wrappers in metaworld.__init__ / metaworld.wrappers to pick a task and call env.set_task(task).
  - Reset: obs, info = env.reset().

  3. Choose camera IDs.

  - Enumerate available cameras:
      - import mujoco as mj
      - cams = [env.model.name_camera(i) for i in range(env.model.ncam)]  # or id2name helper if available
  - Typical picks:
      - “Head” analogue: often “corner2” or “corner” (wide scene view).
      - “Wrist” analogue: often “gripperPOV”.
  - Store head_cam_id, wrist_cam_id by name lookup to avoid hardcoding.

  4. Implement CAPTURE_IMAGES.

  - Render RGB:
      - rgb_head = env.mujoco_renderer.render(render_mode='rgb_array', camera_id=head_cam_id)
      - rgb_wrist = env.mujoco_renderer.render(render_mode='rgb_array', camera_id=wrist_cam_id)
  - Render depth:
      - depth_head = env.mujoco_renderer.render(render_mode='depth_array', camera_id=head_cam_id)
      - depth_wrist = env.mujoco_renderer.render(render_mode='depth_array', camera_id=wrist_cam_id)
  - Convert depth to meters:
      - Near/far from MuJoCo: near = env.model.vis.map.znear, far = env.model.vis.map.zfar
      - Use OpenGL conversion: Z = near * far / (far - (far - near) * depth_buffer); clamp to [0,1] if you keep the downstream normalization.
  - Save to the same paths in config.py.
  - Return camera extrinsics:
      - Position: env.data.cam_xpos[camera_id] (3,)
      - Orientation: Rotation.from_matrix(env.data.cam_xmat[camera_id].reshape(3,3)).as_quat()

  5. Implement EXECUTE_TRAJECTORY.

  - A small Cartesian controller:
      - Query current EE pos: ee = env.get_endeff_pos().
      - For each target [x, y, z, yaw]:
          - Compute delta d = target[:3] - ee.
          - Convert to action space: a[:3] = clip(d / env.action_scale, -1, 1).
          - Gripper channel: use stored gripper_open flag — e.g., a[3] = +1.0 for open, -1.0 for closed. For door task the gripper isn’t required but keep semantics stable.
          - obs, reward, terminated, truncated, info = env.step(a.astype(np.float32)).
          - Iterate for N small steps until within config.margin_error or a step cap (like PyBullet’s loop).
      - Orientation/yaw: most SawyerXYZ envs don’t expose wrist rotation in this action API. Ignore yaw or just feed it forward into your path but don’t expect rotation
        control.
  - Between steps, call the same trajectory image saving as robot.move() does, using the MuJoCo render pipeline.

  6. Implement OPEN_GRIPPER / CLOSE_GRIPPER.

  - Maintain self.gripper_open in the adapter and apply the appropriate a[3] in subsequent steps.
  - If needed, run a few env.step calls with zero deltas and gripper effort to simulate the actuation settle (mirroring PyBullet’s wait).

  7. Implement ADD_BOUNDING_CUBES and ADD_TRAJECTORY_POINTS.

  - No MuJoCo runtime geometry debug lines. Keep the same message contract:
      - For ADD_BOUNDING_CUBES and ADD_TRAJECTORY_POINTS, simply log / no‑op and immediately send OK. Your perception pipeline only needs images and returned extrinsics; the
        lines are for visualization.

  8. Implement RESET_ENVIRONMENT.

  - env.reset() and reinitialize controller state (gripper_open = True, reset counters) and “settle” for a few steps if desired.

  9. Echo the ready message to the main process once set up.

  Step‑By‑Step: Camera + Projection Changes in utils.py

  1. Intrinsics/extrinsics for MuJoCo.

  - Add a new branch (or helper) alongside get_intrinsics_extrinsics(...) that takes MuJoCo camera pose and fovy:
      - fov_y_rad = np.deg2rad(env.model.cam_fovy[camera_id])
      - f = image_height / (2 * np.tan(fov_y_rad / 2))
      - K = [[f, 0, 0], [0, f, 0], [0, 0, 1]] and continue to use the “subtract image center” step like now (since K has zeros for principal point).
      - R = env.data.cam_xmat[camera_id].reshape(3,3), Rt = [[R, t],[0,0,0,1]] where t = env.data.cam_xpos[camera_id].
  - Return (K, Rt) like before.

  2. Depth conversion for MuJoCo.

  - In CAPTURE_IMAGES, convert depth by MuJoCo near/far, then normalize into [0,1] the same way your bullet path produces depth_array for downstream point cloud code.

  3. Wrist/head frame alignment.

  - Your current get_world_point_world_frame(...) flips axes depending on camera type. Keep the same sign conventions, but verify against test points:
      - Project a known 3D point (e.g., door handle center) and check that reprojected depth points land in the correct region of the image.

  4. Remove PyBullet imports from utils.py code paths used by MuJoCo (guard with if backend == 'metaworld').

  Config Alignment

  - End effector start in Metaworld: SawyerDoorEnvV3 uses hand_init_pos = [0, 0.6, 0.2]. Update or override config.ee_start_position to match Z≈0.2 when --sim=metaworld.
  - Image sizes: keep config.image_width/height consistent with env construction.
  - Optional: add config.sim_backend and use it in env.py/utils.py to select branches.

  IPC Semantics (Keep As‑Is)

  - env.py message switch remains identical:
      - CAPTURE_IMAGES returns head/wrist positions + orientations, writes RGB+depth images as before.
      - EXECUTE_TRAJECTORY consumes [(x, y, z, yaw), ...] in world frame and executes through the Cartesian controller.
      - OPEN_GRIPPER/CLOSE_GRIPPER toggle the controller’s gripper channel.
      - RESET_ENVIRONMENT calls env.reset() and reinitializes.

  - MuJoCo rendering on Windows:
      - If offscreen fails, set os.environ['MUJOCO_GL']='egl' (Linux w/ EGL) or use default GLFW on Windows with a created context. For pure headless, you may need a
        GPU‑capable context. Start with default; if it errors, try egl only if supported.
  - Camera names:
      - Verify cameras in your XML: list ids/names at runtime:
          - cams = [env.model.id2name(i, mujoco.mjtObj.mjOBJ_CAMERA) for i in range(env.model.ncam)]
          - Pick two stable cameras for “head” vs “wrist” analogues.
  - Door env specifics:
      - The gripper channel is present but not required for success. Your OPEN/CLOSE commands can remain no‑ops physically, but keep them for consistency.

  Validation Checklist

  - Run once with --sim=metaworld and command “open the door”:
      - Check that CAPTURE_IMAGES generates images/rgb_image_head.png, images/depth_image_head.png with plausible content.
      - Ensure utils.get_bounding_cube_from_point_cloud(...) produces points in the Metaworld world frame (e.g., door handle around x∈[−0.1,0.1], y∈[0.8,0.95], z≈0.15).
      - Confirm action loop moves EE toward given points.
  - If depth looks wrong:
      - Recheck near/far sources and depth conversion formula.
      - Verify that MuJoCo depth is not linear in z; it’s in OpenGL clip space; use the conversion above.

  Minimal Code Sketches (for reference)

  - Env boot (Metaworld):
      - env = SawyerDoorEnvV3(render_mode=None, width=config.image_width, height=config.image_height)
      - Optionally set task via metaworld task wrappers and env.set_task(task).
  - Action mapping:
      - a[:3] = np.clip((target[:3] - ee) / env.action_scale, -1, 1)
      - a[3] = 1.0 if gripper_open else -1.0
      - env.step(a.astype(np.float32))
  - Camera pose:
      - pos = env.data.cam_xpos[cam_id].copy()
      - quat = Rotation.from_matrix(env.data.cam_xmat[cam_id].reshape(3,3)).as_quat()

  Want me to scaffold a run_metaworld_environment alongside the current PyBullet loop and add a --sim flag so you can toggle between backends?


› yes - add new code for metaworld without affecting pybullet. you can create a providers model (for envs, robots, cameras ...) that switches between the 2 sims. later, we
  want to add other sims and envs. since there may be conflicting package versions between metaworld and pybullet (ex: numpy version), design it such that the metaworld
  process can be run from another env and communicate with the main.py process


• I’ll add a providers layer, a Metaworld server script using JSON over stdio, a subprocess connection adapter, and a --sim flag in main.py. I’ll also update utils.py to
  avoid PyBullet-only quaternion math so it works with both backends.
  

 I’m adding a pluggable providers layer and a Metaworld subprocess server, keeping PyBullet untouched. Then I’ll wire a backend flag so you can switch simulators and run
  Metaworld from a separate Python env.

  - Added a subprocess JSON bridge and Metaworld server
      - providers/subproc_connection.py: Lightweight newline‑delimited JSON adapter with send/recv/poll to mimic multiprocessing.Pipe. Lets us run a sim in a different Python
        environment.
      - providers/metaworld_server.py: Self‑contained Metaworld “door” server. Boots SawyerDoorEnvV3, renders RGB/depth, executes Cartesian deltas, and speaks the same
        command codes over stdio:
          - 1 CAPTURE_IMAGES: Renders head/wrist frames, saves to images/, returns camera poses.
          - 2 ADD_BOUNDING_CUBES: No‑op, returns OK.
          - 3 ADD_TRAJECTORY_POINTS: No‑op.
          - 4 EXECUTE_TRAJECTORY: Tracks each point with a simple (dx,dy,dz,grip) controller and saves trajectory frames.
          - 5/6 OPEN_GRIPPER/CLOSE_GRIPPER: Toggles gripper channel in subsequent steps.
          - 8 RESET_ENVIRONMENT: Resets and acknowledges.
      - Uses METAWORLD_REPO to add the Metaworld repo (D:\NLP\Robotics\Simulators_Envs\Metaworld) to sys.path, and METAWORLD_PYTHON to pick the Python interpreter for the
        server.
  - Added backend switch in main
      - main.py: Adds --sim {pybullet, metaworld}.
          - pybullet: start the existing run_simulation_environment process via Pipe (unchanged).
          - metaworld: start the stdio JSON server via SubprocessJSONConnection without changing api.py. The existing API keeps using main_connection.send([...]) and recv().
  - Made utils backend‑agnostic for camera math
      - utils.py: Removed dependency on PyBullet for quaternion→rotation; added _quat_to_rotmat and use it in get_intrinsics_extrinsics. This keeps the perception path
        compatible with either sim.

  What stays the same
  - main.py ↔ api.py messaging, command codes, file paths (images/...), and env.py (PyBullet) behavior are unchanged.
  - The LLM‑driven pipeline and Pipe semantics remain identical across backends.

  Limitations and notes
    perception.
  - Orientation control: Metaworld Sawyer XYZ action space does not include wrist yaw; the server tracks only position deltas and gripper.
  - Camera intrinsics: utils.get_intrinsics_extrinsics uses config.fov. Metaworld’s camera fovy can differ; if you need exact point cloud scale, we can extend the server to
    return the camera fov and near/far for precise reconstruction.

  How to run

  - PyBullet (current behavior):
      - python main.py --sim pybullet
  - Metaworld (separate env):
      - Ensure Metaworld and SciPy are installed in another Python env.
      - Set env vars:
          - METAWORLD_PYTHON to that interpreter, e.g. C:\path\to\metaworld\venv\Scripts\python.exe
          - METAWORLD_REPO to D:\NLP\Robotics\Simulators_Envs\Metaworld
      - Run: python main.py --sim metaworld
      - The server uses SawyerDoorEnvV3 and communicates via JSON. Images are saved under images/ as before, so the perception code runs unchanged.

  Changes summary

  - providers/subproc_connection.py:1
  - providers/metaworld_server.py:1
  - main.py:1 (add --sim, connect to providers)
  - utils.py:1 (remove PyBullet quaternion math)

  Next steps (optional)

  - Include camera fov/near/far in the IPC to strengthen point‑cloud scale for Metaworld.
  - Add provider selection for more tasks/simulators behind the same --sim flag.
  - Expand the server to choose envs by name (sawyer_door_v3, etc.) via a command‑line arg or an initial JSON “init” message.

  Want me to extend the server to pass camera intrinsics precisely and update utils to consume them for better 3D reconstruction under Metaworld?


› Q: What do these mean: Include camera fov/near/far in the IPC to strengthen point‑cloud scale for Metaworld. extend the server to pass camera intrinsics precisely and
  update utils to consume them for better 3D reconstruction under Metaworld?


• Why It Matters

  - Your point‑cloud and bounding‑box math assumes calibrated cameras.
  - In PyBullet you faked calibration via config.fov; with Metaworld/MuJoCo you should use the exact camera parameters to avoid scale drift and warped geometry.
  - Depth from MuJoCo uses OpenGL conventions that require near/far to recover metric depth.

  What To Include Over IPC

  - For each camera (head, wrist), send:
      - Intrinsics: fovy_deg, width, height, znear, zfar or directly K = [[fx,0,cx],[0,fy,cy],[0,0,1]].
      - Extrinsics: position (3,), orientation_quat (x,y,z,w) or directly Rt (4×4 camera→world).
  - Optionally send a flag depth_encoding = "opengl" or "metric" so the client knows whether to linearize depth.

  How To Get Them In MuJoCo

  - fovy_deg = env.model.cam_fovy[camera_id]
  - width, height are what you created the renderer with (e.g., 256×256).
  - znear, zfar are in the model viz settings (e.g., env.model.vis.map.znear/zfar).
  - Camera pose:
      - pos = env.data.cam_xpos[camera_id]

  Converting To Intrinsics

  - Compute fy = height / (2 * tan(fovy/2)), fx = fy * (width/height), cx = width/2, cy = height/2.
  - Build K = [[fx,0,cx],[0,fy,cy],[0,0,1]].

  Depth Linearization

  - MuJoCo “depth_array” follows OpenGL clip‑space mapping. Convert using near/far:
      - If d ∈ [0,1] is the buffer, metric depth along the camera ray is:
          - Z = (znear * zfar) / (zfar - d * (zfar - znear))
  - Without near/far, you can’t recover true Z; using an arbitrary normalization will mis‑scale point clouds and object dimensions.

  How utils Should Consume

  - Update get_intrinsics_extrinsics to accept:
      - Either the raw K (preferred), or {fovy_deg,width,height} and compute K.
      - Extrinsics from the server: use the provided Rt or reconstruct from pos, quat.
  - Ensure the principal point (cx, cy) is set to the image center; don’t leave zeros as in the current PyBullet fallback.
  - Use the provided znear/zfar (or metric depth flag) to linearize the depth image before back‑projection.

  Back‑Projection Consistency

  - Your pipeline assumes world = Rt @ [inv(K) * [u−cx, v−cy, 1] * Z; 1] where Rt maps camera→world.
  - With MuJoCo, cam_xmat/cam_xpos already give camera→world, so that formula remains correct.
  - Keep the small wrist/head sign flips only if validated; with true intrinsics you can usually drop ad‑hoc flips.

  Wire Format Example

  - Extend the CAPTURE_IMAGES response to include a calibration dict per camera:
      - [head_pos, head_quat, wrist_pos, wrist_quat, "OK", {"head":{"K":K,"znear":..,"zfar":..}, "wrist":{...}} ]
  - Or write sidecar JSON files next to the saved images (e.g., images/rgb_image_head.json) with the same fields; store in API and pass into utils.

  In short: “Include camera fov/near/far in the IPC” means have the Metaworld server transmit exact camera calibration (intrinsics + near/far and extrinsics) to the main
  process. “Extend the server to pass camera intrinsics precisely and update utils to consume them” means adjust both ends so your point‑cloud reconstruction uses true MuJoCo
  camera parameters, ensuring correct metric scale and geometry for segmentation and bounding‑cube estimates.


› ok. do that, but in utils, only when sim==pybullet, also call the original p.getMatrixFromQuaternion (and other p. funcs) you deleted, in order to compare their results
  with the new code, to test the code. place the p. calls + compare checks  under DEBUG_DIFF flag so it can be easily disabled)


• Planned change: add precise camera calibration over IPC and compare utils’ new math against PyBullet under a debug flag.

  What I added

  - Metaworld server now sends calibration
      - providers/metaworld_server.py: On CAPTURE_IMAGES, the server now returns:
          - [head_pos, head_quat, wrist_pos, wrist_quat, "OK", calib]
          - calib contains:
              - head.K, wrist.K (3×3 intrinsics with fx/fy and principal point at image center),
              - znear, zfar, width, height, fovy,
              - depth_encoding = "opengl".
          - Depth PNG stays in 0..1 OpenGL encoding; client linearizes using near/far.
  - API consumes calibration + linearizes depth
      - api.py: In detect_object, receives a 6th element (calibration). If depth_encoding == "opengl", depth is converted to linear Z and normalized (Z/zfar) before point
        cloud.
      - Passes K_override into utils.get_bounding_cube_from_point_cloud both for initial detection and XMem trajectory processing.
  - Utils updated, with debug compare for PyBullet
      - utils.py:
          - New _quat_to_rotmat and updated get_intrinsics_extrinsics(image_height, camera_position, camera_orientation_q, K_override=None):
              - Uses K_override if provided (from Metaworld server).
              - Builds K from config.fov when absent, keeping the legacy center-subtraction path intact.
              - Under DEBUG_DIFF=1, and if pybullet is importable, calls p.getMatrixFromQuaternion and prints max diff with the numpy result.
          - get_world_point_world_frame accepts K_override. For overrides, use direct pixel coordinates [u, v, 1] and supplied K; otherwise retain the PyBullet recentering
            and axis flips.
          - get_bounding_cube_from_point_cloud threads K_override through to the back-projection call.
  - main.py
      - Adds --sim {pybullet, metaworld} to choose backend.
  - Subprocess provider
      - providers/subproc_connection.py: A small stdio JSON adapter with send([CMD, args]), recv(), and poll(), mimicking Pipe.

  How this addresses your ask

    intrinsics.
  - “In utils, when sim==pybullet, also call original p.getMatrixFromQuaternion and compare”: Added a DEBUG_DIFF env flag. If set to 1 and pybullet is available, utils
    computes and compares the rotation matrix from PyBullet vs numpy, and logs the max difference.
  - Keep current behavior (PyBullet):
      - python main.py --sim pybullet
      - Optional: set DEBUG_DIFF=1 to verify quaternion to rotation math against PyBullet during detection/reconstruction paths.
  - Run Metaworld server from another env:
      - Install Metaworld + SciPy in a separate virtualenv.
      - Set:
          - METAWORLD_PYTHON to that env’s Python executable
          - METAWORLD_REPO to D:\NLP\Robotics\Simulators_Envs\Metaworld
      - Then: python main.py --sim metaworld
      - The main process talks to the Metaworld server over JSON. Images and camera calibration flow back for accurate point-clouds.

  Notes and follow‑ups

  - Depth normalization: I normalize Z by zfar to keep your downstream behavior (0..1). If you’d rather keep raw meters, we can change the utils and any thresholds
    accordingly.
  - Principal point: Using image center in the server-generated K. If Metaworld uses a different principal point, we can fetch it from MuJoCo XML; typical Sawyer cameras are
    centered.
  - Wrist vs head flips: With true K + Rt, the ad‑hoc axis flips are bypassed for the Metaworld path (we pass K_override); for PyBullet the legacy behavior remains.
  - If you want exact wrist camera calibration from MuJoCo too, it’s already included via wrist.K

• - Current behavior (PyBullet)
      - robot.py:151 computes linear Z from the depth buffer, then clamps and saves:
          - Converts OpenGL depth to metric Z: robot.py:187
          - Clips to [0, 1] and writes an 8‑bit image: robot.py:190
      - api.py reads the saved depth and normalizes to 0..1:
          - Loads depth_image_head_path: api.py:58
          - Normalizes to [0,1] by dividing by 255: api.py:59
      - Effect: You do not use raw metric Z downstream. You convert to meters, then clamp to a maximum of 1.0 m and store as 0..1; later you read it back as 0..1 and treat
        that as depth in “meters up to 1 m” during back‑projection.
  - Implication for a toggle
      - Today: PyBullet path is “normalized with 1 meter clamp.”
      - Metaworld path I added uses “normalized to zfar” (Z/zfar) after linearizing OpenGL depth with znear/zfar.
      - If we add a --depth_format CLI:
          - raw: store and consume linear Z in meters (no normalization).
          - norm_zfar: store Z/zfar (keeps full range, matches Metaworld server).
          - norm_1m: current PyBullet behavior (clip at 1 m, 0..1).
      - We can wire this flag to both providers so reconstruction is consistent across backends.


› 1) ok. do that.2) I also need to be able to install and run the metaworld env locally on my windows machine (without invoking main.py) for testing/sanity. does metaworld
  opens Mujoco viewer in this mode and displays the env and robot are at reset ? Keep in mind that actual usage of the whole main.py in in remote headless server. 3) also
  add the command line for selecting env/task name (in pybullet it will currenly has no effect, but important in metaworld).
  

 Planned updates: add depth-format toggle, selectable Metaworld env/task, and a local viewer path for sanity checks. Also keep PyBullet defaults intact and add debug
  comparisons in utils.

  What I changed

  - Depth format toggle
      - main.py: Adds --depth-format {norm_1m, norm_zfar, raw} (default norm_1m to preserve PyBullet behavior).
      - api.py: Applies the depth format when calibration is present:
          - raw: return linear Z in meters (clipped to [znear, zfar]).
          - norm_zfar: normalize Z by zfar to [0..1].
          - norm_1m: clip Z to [0, 1] meters for backward compatibility with PyBullet.
      - Note: PyBullet still writes normalized depth images the old way; when using Metaworld, the conversion uses the server’s near/far.
  - Selectable env/task for Metaworld
      - main.py: Adds --task (default sawyer_door_v3). PyBullet ignores it; Metaworld server receives it.
      - providers/metaworld_server.py: --env selects the environment class. Currently supports sawyer_door_v3 and falls back to it if unknown (extendable later).
  - Local Mujoco viewer for sanity (Windows dev box)
      - providers/metaworld_server.py: --viewer flag launches the MuJoCo passive viewer, resets the env, runs a few steps to show robot and environment, then exits. This is
        independent of main.py — you can invoke the server directly from your Metaworld Python environment:
          - METAWORLD_PYTHON C:\path\to\metaworld\venv\Scripts\python.exe
          - C:\path\to\metaworld\venv\Scripts\python.exe providers\metaworld_server.py --env sawyer_door_v3 --viewer
      - If viewer fails (e.g., headless), it prints the error and continues in server mode.
  - IPC calibration and utils testing hooks
      - providers/metaworld_server.py: CAPTURE_IMAGES now returns calib containing exact K, znear, zfar, etc.
      - utils.py: Uses K_override when provided for accurate back-projection. Under DEBUG_DIFF=1, compares PyBullet’s getMatrixFromQuaternion vs numpy rotation for quick
        validation during PyBullet runs without affecting Metaworld.
      - Environment variables:
          - METAWORLD_PYTHON: interpreter for the Metaworld server.
          - METAWORLD_REPO: repo path to add to sys.path for imports.
          - DEPTH_FORMAT: passed implicitly via main args and used in API.

  What remains the same

  - PyBullet env and robot control paths are untouched and remain the default.
  - API/LLM logic, command codes, and saved image paths are unchanged.
  
Perception
- - - - - - 
depth normalization and clipping at 1m
- - - - - - - - - - - - - - - - - - - - 
 - Current behavior (PyBullet)
      - robot.py:151 computes linear Z from the depth buffer, then clamps and saves:
          - Converts OpenGL depth to metric Z: robot.py:187
          - Clips to [0, 1] and writes an 8‑bit image: robot.py:190
      - api.py reads the saved depth and normalizes to 0..1:
          - Loads depth_image_head_path: api.py:58
          - Normalizes to [0,1] by dividing by 255: api.py:59
      - Effect: You do not use raw metric Z downstream. You convert to meters, then clamp to a maximum of 1.0 m and store as 0..1; later you read it back as 0..1 and treat
        that as depth in “meters up to 1 m” during back‑projection.
  - Implication for a toggle
      - Today: PyBullet path is “normalized with 1 meter clamp.”
      - Metaworld path I added uses “normalized to zfar” (Z/zfar) after linearizing OpenGL depth with znear/zfar.
      - If we add a --depth_format CLI:
          - raw: store and consume linear Z in meters (no normalization).
          - norm_zfar: store Z/zfar (keeps full range, matches Metaworld server).
          - norm_1m: current PyBullet behavior (clip at 1 m, 0..1).
      - We can wire this flag to both providers so reconstruction is consistent across backends.	  