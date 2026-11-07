# SLAM and mapping - concurrent mapping and localization for the mobile VLM robot

Research notes for the `Localizer` and the map layers used by semantic navigation.
- Plan: [../plans/semantic_navigation.md](../plans/semantic_navigation.md).
- Related: [navigation.md](navigation.md) (how maps are used for goals) and
  [mobile_manip.md](mobile_manip.md) (base odometry).

Compiled 2026-11-08 from these sources:
- arXiv IDs, checked with `docs/research_tools/paper_research.js arxiv`.
- GitHub star counts, checked on 2026-11-08.
- MIT News, 2025-11-05: "Teaching robots to map large environments".
- The @AIPaperSlop YouTube channel.

How to read the tags:
- **[unverified]** means the item comes from memory, which applies to most licenses, rates and venues.
- `n/a` means not found.

---

## 0. TL;DR: what our Localizer should be

| Setting | Recommended stack | Why |
|---|---|---|
| **Sim v1** | Wheel odometry with configurable noise and slip, plus an RGB-D occupancy grid built from `depth + odom pose`. It sits behind `Localizer.get_pose() -> (T_world_base, cov, lost)` | Deterministic and cheap. Uses no privileged state when noise is on. Same interface as on the real robot |
| **Real indoor, RGB-D + wheels (+IMU)** | **RTAB-Map** (ROS2) | Fuses wheel odometry, does appearance loop closure and multi-session maps, and outputs an occupancy grid plus a cloud |
| Real indoor with a 2D LiDAR | **slam_toolbox** for mapping, then **Nav2 AMCL** to localise on the saved map | The Nav2 default; supports lifelong and serialised maps |
| Localization on a prior tour map | RTAB-Map localization mode, AMCL, or visual relocalization against tour keyframes (topological) | The "with a map" mode |
| Outdoor | **FAST-LIO2 / LIO-SAM** (3D LiDAR + IMU) or **KISS-ICP** (LiDAR only). Camera-only: **ORB-SLAM3** stereo-inertial or **VINS-Fusion** + GNSS | RGB-D depth fails in sunlight |
| Research track: pixel-only, uncalibrated | **VGGT-SLAM** (MIT), **MASt3R-SLAM**, DROID-SLAM. Streaming feed-forward: **LingBot-Map** (~20 FPS) | Needs a GPU, has no guaranteed metric scale, drifts on robots. Good for building maps offline from a tour video |
| Semantic layer, on top of the Localizer pose | **ConceptGraphs / HOV-SG** (Python, offline or slow online), **Hydra / Clio / Khronos** (real-time C++ / ROS) | "Kitchen" is a room node. The semantic layer stays separate from the SLAM back end |

---

## 1. Concepts

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Front end / back end | Front end: odometry or tracking. Back end: pose-graph or bundle-adjustment optimisation | Defines where the `Localizer` plugin boundary sits | Cadena et al. 2016 survey |
| Pose graph | Nodes are poses; edges are relative constraints (odometry, loop closures) | Core of Cartographer, slam_toolbox and RTAB-Map | g2o, GTSAM |
| Loop closure | Recognising a revisited place to cancel drift | Long indoor routes; tour maps | DBoW2, NetVLAD, RTAB-Map |
| Relocalization / kidnapped robot | Recovering the global pose without a prior pose | Robot restart; "where am I" on a saved map | AMCL, 7-Scenes |
| Occupancy grid | 2D or 3D probability that each cell is free or occupied | Input to the path planner and frontier extraction | Elfes; Nav2 costmap |
| Metric vs topological map | Coordinates versus a graph of places | Metric for A* and frontiers; topological for drift-tolerant tour maps | ViNT, Mobility VLA |
| Multi-session / lifelong map | Merge and update maps across days | A persistent home map | RTAB-Map, slam_toolbox, Khronos |
| Scale ambiguity | Monocular systems cannot recover metric scale | Need depth, IMU or wheels to get scale | ORB-SLAM, DROID, VGGT |
| VIO / LIO | Tightly coupled IMU with camera or LiDAR | Robust to fast motion and outdoors | VINS-Fusion, LIO-SAM, FAST-LIO2 |
| 3D scene graph | Hierarchy of building, floor, room, place and object nodes | Room-level semantic goals | Kimera, Hydra, HOV-SG |
| Open-vocabulary map | Per-point or per-object CLIP features that can be queried with text | Free-text goals ("the red kettle") | ConceptFusion, ConceptGraphs |
| Feed-forward 3D foundation model | Poses and pointmaps from an image set in a single pass | Calibration-free mapping of a tour video | DUSt3R, MASt3R, VGGT, VGGT-Ω, LingBot-Map |
| Active SLAM / exploration | Choosing actions that reduce map uncertainty or reveal frontiers | The "no map" mode | Yamauchi '97, VLFM |

## 2. Techniques

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Depth → occupancy (height band) | Back-project depth, transform to the world frame, keep points within `[z_min, z_max]`, ray-cast free cells | Our `OccupancyMap.update()` | standard; VLFM |
| Scan matching (ICP / correlative) | Align successive scans | 2D/3D LiDAR odometry | Cartographer, slam_toolbox, KISS-ICP |
| Feature-based visual SLAM | ORB features, bundle adjustment, multi-map Atlas | CPU real-time camera SLAM | ORB-SLAM3 |
| Appearance-based loop closure + memory management | Bag-of-words loop closure with working / long-term memory | Bounded compute on large maps | RTAB-Map |
| Dense differentiable BA | Learned optical flow plus a dense BA layer | Accurate learned SLAM | DROID-SLAM |
| Submap alignment on SL(4) | Aligns uncalibrated feed-forward submaps up to a projective ambiguity | Scales VGGT to long sequences | VGGT-SLAM |
| Streaming geometric context | Anchor context + pose-reference window + trajectory memory in attention | ~20 FPS streaming reconstruction over 10k+ frames | LingBot-Map |
| Iterated error-state Kalman filter | Fast LiDAR-inertial odometry on raw points | Outdoor odometry | FAST-LIO2 |
| Factor graph / iSAM2 | Incremental smoothing with GPS and loop factors | Outdoor LIO with GNSS | LIO-SAM |
| AMCL | Adaptive Monte-Carlo localization on a static 2D map | Localization on a prior map | Nav2 |
| 3D Gaussian splatting map | Explicit, differentiable radiance-field map | Photoreal reconstruction, not needed for navigation | SplaTAM, MonoGS |
| Foundation segmentation lifted to 3D | SAM masks + CLIP / LLaVA captions fused across views | The open-vocabulary object layer | ConceptGraphs, HOV-SG, Clio |
| Task-driven clustering | Choose object granularity from the task list (information bottleneck) | Keeps the map relevant to the task | Clio |
| Spatio-temporal change detection | Detect added or removed objects across sessions | Keeps a stale home map up to date | Khronos, DynaMem |

## 3. Edge cases

| Name | What it is | Why it matters for us | Source papers |
|---|---|---|---|
| Dynamic objects / people | Moving things corrupt the map | Kitchens with people | Khronos |
| Textureless walls / glass / mirrors | Visual features fail; depth is invalid or wrong | Offices and homes | RTAB-Map + wheel odometry fusion |
| Perceptual aliasing | Identical corridors cause false loop closures | Pose-graph corruption | Robust PGO in Cartographer / RTAB-Map |
| Omni-wheel slip | Odometry error on kiwi or mecanum bases | **Our base**. Model slip noise in sim | [mobile_manip.md](mobile_manip.md) |
| Sunlight on IR depth | RGB-D depth saturates outdoors | Use stereo, LiDAR or monocular depth outdoors | n/a |
| Motion blur / rolling shutter | Breaks feature tracking | Cap the base speed during mapping | VINS-Fusion |
| Kidnapping / restart | No initial pose | Global relocalization | AMCL, RTAB-Map |
| Long-term change | Furniture moves | Multi-session update | Khronos, DynaMem |
| Manipulator in view | The arm pollutes the occupancy map | Mask robot pixels using forward kinematics before fusion | our engineering |
| Learned-SLAM metric scale | Monocular learned systems drift in scale | Fuse with wheel odometry or depth | VGGT-SLAM, MASt3R-SLAM |
| GPU budget | Learned SLAM competes with the VLM and segmentation for the GPU | Keep the default Localizer on CPU | our engineering |

## 4. Benchmarks

| Name | What it is | Metric | Use for us |
|---|---|---|---|
| TUM RGB-D | Handheld and robot RGB-D with ground truth | ATE / RPE | RGB-D localizer accuracy |
| EuRoC MAV | Visual-inertial, indoor | ATE | VIO |
| KITTI Odometry | Driving, LiDAR and stereo | Translational / rotational drift | Outdoor |
| Replica | Synthetic indoor | ATE, reconstruction | Neural / Gaussian SLAM |
| ScanNet / ScanNet++ | Indoor RGB-D scans | ATE, semantic mIoU | Open-vocabulary maps |
| TartanAir | Synthetic data in hard conditions | ATE | DROID-style training and evaluation |
| 7-Scenes | RGB-D relocalization | Pose error | Relocalization |
| Hilti SLAM Challenge | Construction sites, multi-sensor | Score | Robustness |
| uHumans2 | Unity scenes with ground-truth scene graphs | Graph accuracy | Kimera / Hydra |
| Newer College | Handheld LiDAR-inertial, outdoor | ATE | LIO |
| HM3D / MP3D ObjectNav | Embodied navigation using maps | SR / SPL | End-to-end semantic navigation |

## 5. Datasets

| Name | Content | Why it matters | Source |
|---|---|---|---|
| TUM RGB-D, EuRoC, KITTI, TartanAir, Newer College | Standard SLAM sequences | Evaluate the Localizer offline | Sturm '12, Burri '16, Geiger '12, Wang '20, Ramezani '20 |
| ScanNet / ScanNet++ / Replica | Indoor RGB-D | Evaluate open-vocabulary maps and scene graphs | Dai '17, Yeshwanth '23, Straub '19 |
| HM3D / HM3D-Semantics / MP3D | Building-scale scans | HOV-SG evaluation; navigation | Ramakrishnan '21 |
| uHumans2 | Photoreal sim with ground-truth scene graphs | Scene-graph evaluation | Rosinol '21 |
| Our rollouts | Sim RGB-D + odometry + ground-truth pose (eval only) | Regression for `nav/` | `outputs/<date>/...` |

---

## 6. System selection

Stars were checked on 2026-11-08. License, rate and ROS2 entries are **[unverified]** unless noted.

| System | Sensors | Map | Compute | ROS2 | License | Stars | Fit |
|---|---|---|---|---|---|---|---|
| **RTAB-Map** | RGB-D, stereo, LiDAR, wheel odometry, IMU | Cloud, mesh, 2D/3D occupancy | 1–30 Hz CPU | Yes (`rtabmap_ros`) | BSD-3 | 4,086 | **Real indoor default (camera)** |
| **slam_toolbox** | 2D LiDAR + odometry | 2D occupancy, pose graph | CPU real-time | Yes (Nav2 default) | LGPL-2.1 | 2,703 | **Real indoor with LiDAR** |
| Cartographer | 2D/3D LiDAR, IMU, odometry | Submap grid | CPU | Community | Apache-2.0 | 8,003 | OK; low maintenance |
| GMapping | 2D LiDAR + odometry | 2D grid | CPU | ROS1 | Non-commercial | n/a | Legacy; skip |
| ORB-SLAM3 | Mono / stereo / RGB-D ± IMU | Sparse | CPU real-time | Community wrappers | **GPL-3.0** | 9,227 | Visual-inertial; GPL |
| VINS-Fusion | Mono / stereo + IMU + GPS | Sparse | CPU | Community | GPL-3.0 | 4,778 | Outdoor VIO |
| FAST-LIO2 | 3D LiDAR + IMU | Cloud | 10–100 Hz CPU | Branch | GPL-2.0 | 5,352 | Outdoor (needs 3D LiDAR) |
| LIO-SAM | 3D LiDAR + IMU (+ GPS) | Cloud | CPU | Branch | BSD-3 | 4,971 | Outdoor + GNSS |
| KISS-ICP | 3D LiDAR only | Cloud | CPU | Yes | MIT | 2,372 | Simplest LiDAR odometry |
| DROID-SLAM | Mono / stereo / RGB-D | Dense depth | GPU, high VRAM | No | BSD-3 | 2,722 | Research |
| MASt3R-SLAM | Monocular, any camera model | Dense pointmaps | GPU, ~15 FPS | No | CC BY-NC-SA [unverified] | 3,237 | Research |
| VGGT-SLAM | RGB, uncalibrated | Submap clouds | GPU | No | BSD-2 | 1,176 | Research; MIT paper |
| LingBot-Map | Streaming RGB | Poses + points | GPU, ~20 FPS at 518×378 | No | n/a | n/a | Research; fast feed-forward |
| SplaTAM / MonoGS | RGB-D / mono | 3D Gaussians | GPU | No | research licenses | 2,219 / 2,191 | Reconstruction only |
| Kimera | Stereo / RGB-D + IMU | Mesh + scene graph | CPU | ROS1 | BSD-2 | 2,144 | Semantic SLAM |
| Hydra | Stereo / RGB-D + pose | 3D scene graph (rooms, places, objects) | Real time | ROS / ROS2 | BSD-2 | 1,193 | Real-time room graph |
| ConceptGraphs | RGB-D + poses | Open-vocabulary object graph | GPU, slow | No | MIT | 986 | **Semantic layer v1 (Python)** |
| HOV-SG | RGB-D + poses | Floor / room / object graph | GPU, offline | No | MIT | 565 | **Room goals from a tour** |
| Clio | RGB-D + poses + task list | Task-driven open-set graph | Real time | ROS | BSD [unverified] | 266 | Task-relevant objects |
| Khronos | RGB-D + poses | Spatio-temporal metric-semantic map | Real time | ROS | BSD [unverified] | 557 | Changing homes |

---

## 7. Papers

### 7.1 Learned / feed-forward mapping (research track)

* **VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold** (MIT SPARK, NeurIPS 2025)
  ) Goal: TLDR:
    - Input: uncalibrated monocular video.
    - Method: incremental VGGT submaps aligned by SL(4) homographies, with loop closure.
    - Output: a globally consistent dense map and poses.
  ) Problem: VGGT handles only about 60 frames per pass.
    - Similarity (Sim(3)) alignment is wrong for uncalibrated submaps, which have a projective ambiguity.
  ) Tags: SLAM, Feed-forward, Uncalibrated, Pixel-only
  ) Paper: https://arxiv.org/abs/2505.12549 / Code: https://github.com/MIT-SPARK/VGGT-SLAM (1,176★, BSD-2) /
    News: https://news.mit.edu/2025/teaching-robots-to-map-large-environments-1105
  ) Relevance: build a map offline from the **tour video** without calibration.
    - Needs a GPU, and the metric scale must come from wheel odometry or depth.

* **VGGT: Visual Geometry Grounded Transformer** (CVPR 2025 Best Paper)
  ) Goal: TLDR: images → feed-forward transformer → camera poses, depth, pointmaps and tracks in one pass.
  ) Paper: https://arxiv.org/abs/2503.11651 / Code: https://github.com/facebookresearch/vggt (14.6k★)
  ) Relevance: a geometry prior for tour mapping. VGGT features also help the VLM's spatial reasoning.

* **VGGT-Ω** (arXiv 2605.15195, 2026)
  ) Goal: TLDR: a scaled VGGT whose quality grows predictably with model and data size.
    - Improves accuracy and efficiency for static **and dynamic** scenes.
  ) Relevance: newest feed-forward reconstructor; handles dynamic scenes (people moving).

* **LingBot-Map: streaming 3D reconstruction with a geometric context transformer** (arXiv 2604.14141, 2026)
  ) Goal: TLDR: a video stream goes into a feed-forward model.
    - Its attention combines an anchor context (coordinate grounding), a pose-reference window (dense geometry) and a trajectory memory (drift correction).
    - It outputs poses and point clouds at **~20 FPS** (518×378) over sequences longer than 10k frames.
  ) Results: beats existing streaming methods and iterative-optimisation methods.
  ) Relevance: **closest learned candidate for an online camera-only Localizer**.
    - Check its license and metric scale first.

* **MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors** (CVPR 2025)
  ) Goal: TLDR: monocular video in, MASt3R two-view priors applied, and out come pointmap matching, tracking, loop closure and second-order global optimisation.
    - Works with any camera model, needing only a unique camera centre.
  ) Paper: https://arxiv.org/abs/2412.12392 / Code: https://github.com/rmurai0610/MASt3R-SLAM (3,237★)
  ) Relevance: robust on in-the-wild video. Reaches SOTA when the calibration is known.

* **DUSt3R** (CVPR 2024, arXiv 2312.14132) / **MASt3R** (ECCV 2024, arXiv 2406.09756; https://github.com/naver/mast3r, 3,159★)
  ) Goal: TLDR: pairwise pointmaps without calibration (DUSt3R), plus dense matching (MASt3R).
  ) Relevance: the base priors behind the learned SLAM systems above.

* **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** (NeurIPS 2021)
  ) Goal: TLDR: recurrent optical-flow updates plus a dense bundle-adjustment layer.
  ) Paper: https://arxiv.org/abs/2108.10869 / Code: https://github.com/princeton-vl/DROID-SLAM (2,722★)
  ) Relevance: an accuracy reference. It needs a lot of VRAM.

* **DPV-SLAM** (ECCV 2024) and **GO-SLAM** (ICCV 2023) [unverified IDs]
  ) DPV-SLAM: a lighter, patch-based deep visual SLAM with loop closure.
  ) GO-SLAM: globally optimised neural SLAM.

### 7.2 Neural / Gaussian maps (reconstruction quality; not needed for navigation v1)

* **SplaTAM** (CVPR 2024, arXiv 2312.02126; https://github.com/spla-tam/SplaTAM, 2,219★)
  ) Goal: TLDR: RGB-D frames are tracked and mapped into 3D Gaussians.
* **MonoGS: Gaussian Splatting SLAM** (CVPR 2024, arXiv 2312.06741; https://github.com/muskie82/MonoGS, 2,191★)
  ) Goal: TLDR: monocular or RGB-D input, mapped live into Gaussians.
* **iMAP / NICE-SLAM / Photo-SLAM** [unverified IDs]
  ) Goal: TLDR: implicit or neural RGB-D SLAM. Slow, and focused on reconstruction.
* Relevance: photoreal digital twins for evaluation (see also GS-Playground in [mobile_manip.md](mobile_manip.md)).

### 7.3 Classical visual / LiDAR SLAM (real-robot defaults)

* **RTAB-Map** (JFR 2019): appearance-based loop closure with memory management.
  - Supports multi-session maps and fuses RGB-D, stereo, LiDAR, wheel odometry and IMU.
  - Outputs 2D/3D occupancy.
  - https://github.com/introlab/rtabmap (4,086★). **Real indoor default.**
* **ORB-SLAM3** (T-RO 2021, arXiv 2007.11898): feature-based SLAM for mono, stereo and RGB-D, with or without an IMU.
  - Multi-map Atlas.
  - https://github.com/UZ-SLAMLab/ORB_SLAM3 (9,227★, GPL-3.0).
* **slam_toolbox** (JOSS 2021): 2D LiDAR pose-graph SLAM with lifelong and serialised maps; Nav2 default.
  - https://github.com/SteveMacenski/slam_toolbox (2,703★).
* **Cartographer** (ICRA 2016): real-time 2D/3D LiDAR SLAM with submaps and loop closure.
  - https://github.com/cartographer-project/cartographer (8,003★).
* **GMapping** (T-RO 2007): Rao-Blackwellized particle filter. Legacy.
* **Nav2 AMCL**: Monte-Carlo localization on a saved 2D map.
  - With a camera only, use `depthimage_to_laserscan`.
* **LIO-SAM** (IROS 2020, arXiv 2007.00258): tightly coupled LiDAR-inertial odometry on iSAM2, with GPS factors.
  - https://github.com/TixiaoShan/LIO-SAM (4,971★).
* **FAST-LIO2** (T-RO 2022, arXiv 2107.06829): direct-point iterated Kalman filter LIO.
  - https://github.com/hku-mars/FAST_LIO (5,352★).
* **KISS-ICP** (RA-L 2023, arXiv 2209.15397): tuning-free LiDAR odometry.
  - https://github.com/PRBonn/kiss-icp (2,372★, MIT).
* **VINS-Fusion** (2019): optimisation-based VIO for mono or stereo, with an IMU and GPS.
  - https://github.com/HKUST-Aerial-Robotics/VINS-Fusion (4,778★).

### 7.4 Semantic / open-vocabulary maps and 3D scene graphs

* **Kimera** (ICRA 2020, arXiv 1910.02490): metric-semantic VIO, mesh and scene graph.
  - https://github.com/MIT-SPARK/Kimera (2,144★).
* **Hydra** (RSS 2022, arXiv 2201.13360): real-time hierarchical 3D scene graph (places, rooms, objects) with loop closure.
  - https://github.com/MIT-SPARK/Hydra (1,193★).
  - Relevance: **real-time room segmentation** gives "kitchen" nodes.
* **Clio** (RA-L 2024, arXiv 2404.13696): task-driven open-set scene graph. It uses the task list to choose object granularity.
  - https://github.com/MIT-SPARK/Clio (266★).
  - Relevance: keeps the map to what the current task needs.
* **Khronos** (RSS 2024, arXiv 2402.13817): spatio-temporal metric-semantic SLAM in changing environments.
  - https://github.com/MIT-SPARK/Khronos (557★).
* **ConceptFusion** (RSS 2023, arXiv 2302.07241): pixel-aligned foundation features fused into 3D, queried by text, click, image or audio.
* **ConceptGraphs** (ICRA 2024, arXiv 2309.16650): SAM + CLIP/LLaVA objects with LLM-inferred relations.
  - https://github.com/concept-graphs/concept-graphs (986★, MIT).
  - **Our semantic layer v1.**
* **HOV-SG** (RSS 2024, arXiv 2403.17846): floor, room and object hierarchy for language-grounded navigation.
  - https://github.com/hovsg/HOV-SG (565★, MIT).
* **VLMaps** (ICRA 2023, arXiv 2210.05714) and **OpenScene** (CVPR 2023, arXiv 2211.15654): dense open-vocabulary grids or point features.
  - See [navigation.md](navigation.md).
* **DynaMem** (arXiv 2411.04999): a dynamic spatio-semantic memory for changing scenes.
  - See [mobile_manip.md](mobile_manip.md).

---

## 8. How this maps onto `nav/` (see the plan)

```
head RGB-D + K + T_base_cam ─┐
wheel odom (Plan A) ─────────┼─> Localizer (odom | rtabmap | orbslam3 | vggt_slam*)  -> T_world_base, cov, lost
                             │
                             ├─> OccupancyMap (5 cm, height band, robot-mask) -> frontiers, A*
                             └─> SemanticMap (seg_provider masks + CLIP) -> objects -> rooms (HOV-SG/Hydra-style)
                                                                          -> TopoGraph (keyframes, tour)
* research track
```
- The Localizer **never** reads `simenv.get_state()`.
- In sim, the ground-truth pose is logged only to measure ATE.
