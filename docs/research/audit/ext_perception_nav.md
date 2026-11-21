# External audit: perception and moving through space (C9, C10, D12 perception layer)

Audit date 2026-11-21. Scope: how Pinch should sense the world and move through it, checked against papers and
maintained code. Every paper below was fetched on arXiv this session (title, authors, year read from the page).
Every GitHub figure was read with `gh api` today. Anything not checked is marked UNVERIFIED.

## 0. What Pinch does today (evidence)

- Position comes from ONE printed ArUco marker (id 7) on the desk tray tab, found with OpenCV plus `solvePnP`:
  `robot/vision.py:27-28, 140-148`. Without a calibrated fisheye lens it refuses to give a position at all
  (`robot/vision.py:143-144`).
- Between marker sightings the position is dead reckoning from the walking command (assumes the gait achieves 80 % of
  the commanded speed, `robot/localizer.py:21-22, 38`) or from "ground flow", a phase correlation of the desk patch in
  front of the camera (`robot/vision.py:197-236`, `robot/localizer.py:57-71`). Drift is modelled as 30 % (command) or
  10 % (flow) of distance walked, capped at 6 cm (`robot/localizer.py:18-19`).
- Objects are found by four fixed colour ranges (pen, eraser, cap, ball) with known heights, projected to the desk
  plane: `robot/vision.py:30-36, 168-188`. Faces: the old Haar cascade (`robot/vision.py:190-199`).
- The desk is a hard-coded rectangle of 0.70 x 0.45 m with a foot-aware edge guard (`pinch/world.py:14-21, 29-59`).
  The twin only gives the mind objects inside a field of view, plus noisy "unknown" props
  (`pinch/twin/sensors.py:79-112`). Edge sensing is the five downward beams (`pinch/twin/sensors.py:104-112`).
- The research note already says: no RL for navigation at desk scale; geometric planning plus visual servoing
  (`docs/research/TRAINING_RESEARCH.md:352-365`), and borrows Nav2's progress checker and recovery ladder
  (`docs/research/TRAINING_RESEARCH.md:105-122`).

The structure is sensible (separate "where am I" from "am I really moving", distrust vision without calibration).
The weak point is not the code quality. It is that every part assumes the desk, the marker and four coloured toys.

## 1. Main table

Where: NPU = RK3566 NPU, CPU = RK3566 Cortex-A55 cores, LAP = laptop.

| Topic | Pinch today | Best current paper (verified) | Best maintained code (gh) | Where | Verdict | Adopt |
|---|---|---|---|---|---|---|
| (a) Vision on RK3566 | OpenCV only (no NPU) | none needed; see benchmark table in section 2 | airockchip/rknn_model_zoo 2,916 stars, pushed 2026-10-20; airockchip/rknn-toolkit2 3,564 stars, pushed 2026-10-20 | NPU | missing | Convert 2 to 3 small INT8 models to RKNN: YOLOv8n-style detector, RetinaFace-mobile, optional pose. Never ViT models. |
| (b) Own motion/position without markers | marker + command dead reckoning + ground flow | Youm et al., "Legged Robot State Estimation With Invariant EKF Using Neural Measurement Network", arXiv 2402.00366, 2024; Wisth, Camurri, Fallon, "VILENS", IEEE T-RO 2023 (arXiv 2107.07243) | rpng/open_vins 3,197 stars, pushed 2025-11-30, GPL-3.0; UZ-SLAMLab/ORB_SLAM3 9,266 stars, pushed 2024-07-24, GPL-3.0 | CPU (filter); LAP (optional SLAM) | improve | Make the proprioceptive filter (leg kinematics + IMU + learned velocity) the base. Keep ground flow. Keep markers as a bonus anchor only. No on-robot VIO. |
| (c) Edges, drops, obstacles | 5 downward VL53L0X beams; desk size hard-coded | Yang et al., "Depth Anything V2", NeurIPS 2024, arXiv 2406.09414 (25M to 1.3B params) | DepthAnything/Depth-Anything-V2 9,187 stars, pushed 2026-03-24, Apache-2.0 (Small model only) | beams: robot CPU reflex; mono depth: LAP | improve | Keep the beams as the safety reflex (confirmed). Add ONE 8x8 multizone ToF looking forward-down. Mono depth only as laptop help, never for safety. |
| (d) Local map | hard-coded desk rectangle + cliff memory | Macenski, Moore, Lu, Merzlyakov, Ferguson, "From the Desks of ROS Maintainers", Robotics and Autonomous Systems 2023 (arXiv 2307.15236): layered costmaps | ANYbotics/grid_map 3,274 stars, pushed 2026-11-15, BSD-3 (ROS-bound, so copy the idea, not the code) | CPU | missing | Robot-centred 2.5D grid (about 1 x 1 m, 1 cm cells) + a persistent list of "danger places" in a world frame that is reset when picked up. Plain numpy. |
| (e) General objects | 4 colour ranges, fixed heights | Cheng et al., "YOLO-World", arXiv 2401.17270, 2024; Wang et al., "YOLOE: Real-Time Seeing Anything", ICCV 2025, arXiv 2503.07465; Minderer, Gritsenko, Houlsby, "Scaling Open-Vocabulary Object Detection" (OWLv2), arXiv 2306.09683 | AILab-CVC/YOLO-World 6,617 stars, pushed 2025-02-26, GPL-3.0; THU-MIG/yoloe 2,341 stars, pushed 2025-06-26, AGPL-3.0; ChaoningZhang/MobileSAM 5,923 stars, pushed 2026-05-05 | NPU: closed-set detector. LAP: open-vocab + masks | replace | Two tiers: on-robot detector for a handful of classes (person, ball, cup, pen-like, "other thing" via motion + blob); laptop open-vocab naming of unknown things, cached. |
| (f) Scene/object memory | `WorldModel` dict by name | Gu et al., "ConceptGraphs", arXiv 2309.16650, 2023; Liu et al., "DynaMem", arXiv 2411.04999, 2024 | concept-graphs/concept-graphs 995 stars, pushed 2025-10-16; ok-robot/ok-robot 619 stars, pushed 2024-03-04; DynaMem: official repo not found (UNVERIFIED) | CPU (store) + LAP (labels) | missing | Lightest design in section 5: a small table of "things" with place, size, label, last seen, confidence, plus a place list. Not a 3D scene graph. |
| (g) People | Haar face cascade only | Guo, Deng, Lattas, Zafeiriou, "SCRFD", arXiv 2105.04714, 2021; Jiang et al., "RTMPose", arXiv 2303.07399, 2023 | airockchip RetinaFace on NPU (zoo); deepinsight/insightface 30,278 stars, pushed 2026-10-19; open-mmlab/mmpose 8,070 stars, pushed 2025-08-04 (stale); google-ai-edge/mediapipe 37,769 stars, pushed 2026-11-21 | NPU: face. CPU: 2-mic bearing, VAD, wake word. LAP: recognition, ASR | replace face detector; add rest | RetinaFace-mobile on NPU; YOLOv8n-pose only if needed; Silero VAD + openWakeWord on CPU; 2-mic bearing via GCC-PHAT; speech recognition only on laptop. |
| (h) Route planning / control | unmerged branch: A*, progress checker, recovery ladder in plain Python | Macenski et al. 2023 (above) for the architecture | ros-navigation/navigation2 5,091 stars, pushed 2026-11-21 | CPU | confirmed (no ROS) | Keep plain Python; copy Nav2's interfaces and behaviour names. Do not run Nav2 on the robot. |
| (i) Visual-nav foundation models | not used | Shah et al., "ViNT", CoRL 2023, arXiv 2306.14846; Sridhar et al., "NoMaD", arXiv 2310.07896; Cheng et al., "NaVILA", arXiv 2412.04453 | robodhruv/visualnav-transformer 1,350 stars, pushed 2024-09-15 (not updated since); NaVILA code: AnjieCheng/NaVILA 736 stars, pushed 2025-08-20 (official status UNVERIFIED) | LAP at best | confirmed (skip) | Overkill. Revisit only as a laptop "explore" helper after the foundation works. |
| (j) Following | face/colour bearing -> walking command | ByteTrack, Zhang et al., arXiv 2110.06864 (2021, still the tracker inside most current stacks) | ifzhang/ByteTrack 6,823 stars, pushed 2024-06-19; roboflow/supervision 51,682 stars, pushed 2026-11-20 (UNVERIFIED that it includes ByteTrack in current form) | CPU | improve | Detector box -> simple tracker (ByteTrack-style or IoU+Kalman) -> bearing/size servo. Classical visual servoing is the standard. |
| (k) VLM on laptop | local LLM for speech only | Ahmad, Hafeez, Zaidi, "Vision-Language Models on the Edge for Real-Time Robotic Perception", arXiv 2601.14921, 2026 (a 2B model runs sub-second at the edge with lower accuracy) | huggingface/transformers 169,336 stars, pushed 2026-11-21; ggml-org/llama.cpp 135,898 stars, pushed 2026-11-21 | LAP | missing | Ask the laptop VLM "what is this, what can I do with it" about a cropped, tracked object; cache the answer; the robot behaves normally with a generic label when the laptop is absent. |

Note: no verified paper on a small robot pet using a laptop VLM was found (one search). The evidence for (k) is
general edge-VLM latency work, not pet robots.

## 2. (a) What can run on an RK3566

Source: the benchmark table in the airockchip/rknn_model_zoo README (read today). Column "RK3566 RK3568", frames per
second, INT8 unless noted, single model running alone on the NPU.

| Model (input) | RK3566 fps | Use for Pinch |
|---|---|---|
| mobilenetv2 (224) | 192.2 | tiny classifier (place recognition, "is this a person") |
| RetinaFace_mobile (320) | 148.9 | faces, clearly fits |
| yolov8n (640) | 35.1 | object detector, fits |
| yolov8n-pose (640) | 23.7 | body pose, fits at low rate |
| yolov5n-seg / yolov8n-seg (640) | 34.4 / 28.9 | instance masks, fits |
| yolov10n / yolo11n / yolo26n (640) | 22.5 / 22.6 / 16.7 | newer but slower here than yolov8n |
| deeplabv3 mobilenet (513) | 11.7 | free-space/floor segmentation, borderline |
| ppseg lite (512) | 6.6 | too slow |
| yolo_world_v2s (640) | 5.9 | open vocabulary, only for an occasional "look" |
| clip image encoder (224, FP16) | 2.3 | one look per second at best |
| mobilesam encoder tiny (448, FP16) | 0.8 | NOT usable on the robot (laptop) |
| whisper base, 20 s audio (FP16) | real-time factor 1.207 | slower than real time: not usable |
| zipformer streaming (FP16) | real-time factor 0.183 | streaming ASR possible, but the example model is bilingual Chinese-English |

Platform support, read from the READMEs:

- `rknn-toolkit2` lists RK3566/RK3568 as supported (README, "Support Platform"). This is the converter and runtime
  to use.
- `rknn-llm` lists RK3588, RK3576, RK3562 and RV1126B only. RK3566 is NOT supported, so no on-robot LLM or VLM on the
  NPU. (1,776 stars, pushed 2026-09-29.)
- `sherpa-onnx` (16,074 stars, pushed 2026-11-21, Apache-2.0): its Rockchip page lists RK3566 among boards known to
  work, but the published RKNN models there are for RK3588 (UNVERIFIED that any model runs well on RK3566). The CPU
  path (ONNX, int8) is the safe default.

Realistic frame rates with one camera while the servo loop shares the CPU. These are my estimates, not measurements:
the NPU does the heavy maths, but resize, colour conversion, post-processing and non-max suppression run on the CPU
cores.

- The zoo numbers are one model alone; assume 40-60 % of them in a live system: about 10-15 fps for a 640 detector.
  A smaller input (e.g. 320) should be several times faster (UNVERIFIED scaling).
- Budget: camera 15 fps; detector 10 fps; face 5 fps; pose only when a person is near, 5 fps; ground flow 10-15 fps
  (already cheap: a 0.25 scale phase correlation, `robot/vision.py:206`).
- Pin the servo loop to one core and the vision threads to the other three. Measure on the real board before choosing
  numbers (the 2 GB RAM also limits how many RKNN models stay loaded).

Pick: YOLOv8n (INT8) plus RetinaFace-mobile (INT8), both on the NPU. Fallback: the same models through ncnn on CPU
(Tencent/ncnn, 24,082 stars, pushed 2026-11-21) if RKNN conversion breaks.

## 3. (b) Knowing its own motion without markers

Findings.

- Visual-inertial odometry (VIO, tracking position from a camera plus an IMU) on legged robots is hard because the gait
  shakes the camera and blurs frames. WALK-VIO (Lim et al., arXiv 2111.15164, 2021) and VILENS (above) fuse leg
  kinematics with vision for exactly that reason. A 2025 quadruped paper uses leg odometry to restart a lost visual
  system (arXiv 2505.02272; from a search snippet only: UNVERIFIED).
- Compute: the only embedded number found is ORB-SLAM3 monocular-inertial at 10 Hz and 25-30 % CPU on a Raspberry
  Pi 4 (a quadcopter paper, arXiv 2504.15305, search snippet only: UNVERIFIED). An RK3566 has four slower A55 cores.
  No Pi-class number was found for OpenVINS or Basalt (basalt-mirror 1,034 stars, pushed 2026-03-22). VINS-Fusion has
  not been updated since 2024-05-23 (4,800 stars) and ORB-SLAM3 not since 2024-07-24.
- Pinch's camera is 3-4 cm above a mostly textureless desk. Feature-based VIO needs texture and a camera that looks at
  the scene. The ground-flow code itself notes a weak correlation peak on a plain desk (`robot/vision.py:197-200`). I
  judge on-robot VIO as not worth the risk. This is my assessment, not a measurement.
- Learned proprioceptive estimators: a small network on joint and IMU history outputs body velocity and contact
  probability, fused in a filter (Youm et al., arXiv 2402.00366; trained only in simulation, which suits Pinch's
  MuJoCo-first pipeline). A newer preprint: TRACE, Kong, Kim, Hwangbo, arXiv 2608.05975, submitted to RA-L August 2026
  (page gives no robot size or compute). Pinch's own research note already planned this: "Proprioceptive body-velocity
  and contact estimator, trained alongside the walker" (`docs/research/TRAINING_RESEARCH.md:74`).
- What small robots use: Anki Vector has a single-point forward ToF, four cliff sensors, a 120-degree camera and four
  microphones (Vision Systems article via search: UNVERIFIED). Its SDK keeps poses per "origin id": when Vector is
  picked up it starts a new coordinate frame and old poses cannot be compared; its nav map is a quad-tree grid (.NET
  SDK docs via search: UNVERIFIED). Open Duck Mini (4,763 stars, pushed 2026-01-31) runs learned walking policies on a
  Pi Zero 2W; its page mentions no camera in the current version and no state estimation. Robot vacuums commonly
  integrate gyro yaw and use SLAM with an image sensor (patent text, not model-specific: UNVERIFIED).

Is relying on printed markers acceptable as a foundation? No. A marker is acceptable as an optional anchor (the dock,
the desk "home"), like Vector's charger, because it gives a drift-free fix. It must not be the only source, because the
owner wants "a desk or a room floor". Today `locate()` returns None without the marker, and the only position source
left is a command integral with 30 % assumed drift.

Pick: a small filter (extended Kalman filter over x, y, yaw) fed by (1) leg-kinematic and learned body velocity,
(2) IMU yaw rate and tilt, (3) ground-flow velocity when its correlation is good, (4) marker or landmark fixes when
seen, (5) the Vector idea: a new "origin id" (drop the map, keep the danger list) on pickup or a long loss. Keep
`Localizer`'s split between `pose()` and `measured_xy()` (confirmed: right idea). Fallback: run ORB-SLAM3 or OpenVINS
offline on recorded video, only as a checker of the filter's drift in the twin and on the bench.

## 4. (c) Edges, drops, obstacles

- Monocular depth: Depth Anything V2 (arXiv 2406.09414) gives RELATIVE depth unless a metric fine-tune is used. The
  Small model is Apache-2.0; larger ones are not (official README, via search). No speed figure for the RK3566 or a
  laptop was found. Zoo evidence: ViT-type encoders on the RK3566 NPU run at about 1-2 fps (MobileSAM 0.8, CLIP 2.3),
  so a depth transformer there would be far too slow for reflexes (UNVERIFIED for the depth model itself: measure on
  the laptop). Verdict: laptop only, for "what is that bump", never for the edge.
- A table edge is a texture-poor, shadowed boundary; a single image is the worst case for guessing a drop. Small
  commercial robots use cheap physical sensors for this (Vector: cliff sensors under the corners). Pinch's five beams
  are the same approach: confirmed. Patent text notes that cliff sensors under the body see the edge only once part of
  the body is already over it (search result: UNVERIFIED). Pinch's foot-aware guard (`pinch/world.py:29-59`) is the
  right answer to that problem.
- The standard step up from single points is a multizone ToF (time-of-flight: distance from light travel time)
  sensor. VL53L5CX: 8x8 zones, up to 400 cm, up to 60 Hz, 65-degree diagonal field of view (Adafruit product page).
  Adafruit board, product 6503: 25.4 x 17.7 x 4.6 mm, 1.9 g, USD 19.95 one-off (USD 15.96 at 100+). The VL53L7CX
  (wider view) could not be verified: the ST page timed out. UNVERIFIED: the L5CX needs a firmware upload over I2C at
  boot and shares the default I2C address with the VL53L0X (the LPn pin reassigns it). Check both in the datasheet
  before wiring.
- Mounting idea: one L5CX at the front, tilted down about 30 degrees. Its 8x8 grid gives obstacle height ahead, the
  edge ahead (a row of zones that suddenly reads far) and a coarse shape for a toy. Keep the five VL53L0X beams as the
  reflex ring.

## 5. (d) Local map and (f) memory: the lightest design

Recommended design (my synthesis; the references supply only the ideas):

- Layered costmaps as in Nav2 (Macenski et al. 2023): a layer for remembered things (the danger list), a layer for
  what is seen right now (ToF + camera), and an inflated "keep your feet this far away" layer. Pinch already has the
  third as `fit_feet` and `feet_clear` (`pinch/world.py:34-59`).
- Robot-centred 2.5D grid, about 1 x 1 m at 1 cm: each cell holds height (from ToF), "drop" or "unknown". Roll it as
  the robot moves. This replaces the hard-coded `Desk` rectangle (`pinch/world.py:14-21`) with something learned by
  walking: when a beam says drop, remember that point. The desk becomes whatever the crab found, so a room floor works.
- A "things" table, not a 3D scene graph: id, label, last place, size (from ToF and image), a few attributes from the
  laptop (soft, rolls, light), when seen, confidence. Forget by age. ConceptGraphs (arXiv 2309.16650) and DynaMem
  (arXiv 2411.04999) are the research versions; they need depth cameras, GPUs and 3D fusion, so take only the idea of
  attaching a label and a place to each object. OK-Robot (Liu et al., arXiv 2401.12202, RSS 2024) reports 58.5 %
  success in real homes with a large stack: the full systems are much heavier than a pet needs.
- No code dependency: `hello-robot/stretch_ai` (251 stars) is archived and `ok-robot/ok-robot` has not been pushed
  since 2024-03-04.

## 6. (e) General objects, size and what can be done with them

- On the robot: a closed-set detector (YOLOv8n) for a few useful classes plus a generic "unknown thing" found by
  motion or by a blob that differs from the floor (what `perceive_props` in the twin already imitates:
  `pinch/twin/sensors.py:79-112`). Size and distance from the ToF grid and the floor-plane projection that
  `Camera.to_desk` already does (`robot/vision.py:107-118`).
- On the laptop: open-vocabulary naming of the crop (YOLO-World, YOLOE or OWLv2). YOLO-World on the RK3566 NPU is
  5.9 fps in the zoo table, so it is possible only for an occasional look. YOLOE (ICCV 2025) claims 1.4x the speed of
  YOLO-World-S with better accuracy and adds segmentation, but its code is AGPL-3.0 and last pushed 2025-06-26.
  MobileSAM masks: the paper says about 10 ms on a GPU; on the RK3566 it is 0.8 fps.
- Affordances (what can I do with it: push, roll, grasp): no verified lightweight method exists. Practical rule for
  Pinch: estimate size and roundness from the observation, then LEARN by trying (push, watch if it moves, record it in
  the things table). That matches the project's "remember which escape worked" idea.
- Tracking: ByteTrack-style association between frames on the CPU (cheap). It gives each object a stable id for
  memory and following.

## 7. (g) People

- Face: replace the Haar cascade (`robot/vision.py:128-130, 190-199`) with RetinaFace-mobile on the NPU (148.9 fps in
  the zoo table). The zoo's RetinaFace entry is listed as "Face Key Points", which also supports a head-turn estimate
  for "is this person looking at me".
- Recognition (who is it): insightface (30,278 stars; its licence shows as null in the API, and pretrained weights
  have their own terms: check before use, UNVERIFIED) with a MobileFaceNet-class model. Run on the laptop, or on the
  NPU only if conversion works. Keep a tiny list of known faces on the robot.
- Body pose: YOLOv8n-pose at 23.7 fps on the NPU is enough for "a person is standing there / waving". RTMPose (arXiv
  2303.07399; mmpose last pushed 2025-08-04) is not needed.
- Attention: face direction from landmarks over a second or two; no special model.
- Sound: two microphones give only a bearing with a front-back ambiguity (a pair cannot tell mirrored positions;
  from a review snippet, UNVERIFIED as a fetched paper). Use GCC-PHAT (a standard time-delay method) for a coarse
  bearing, and resolve the mirror with the camera or by turning. ODAS (introlab/odas, 1,096 stars, pushed 2026-10-26)
  targets 8 to 16 microphone arrays and its README states no two-mic case: not worth adopting. Silero VAD
  (snakers4/silero-vad, 10,730 stars, pushed 2026-11-20, MIT) for "someone is speaking" on CPU. openWakeWord (2,963
  stars, pushed 2025-12-30, Apache-2.0): its README says one Raspberry Pi 3 core runs 15-20 models in real time, so one
  wake word on one A55 core is safe. Speech recognition (whisper.cpp 55,384 stars; sherpa-onnx): whisper base on the
  RK3566 NPU is slower than real time (RTF 1.207), so run speech recognition on the laptop. With no laptop the pet
  reacts to sound level, bearing and tone only, which fits a pet that never speaks.

## 8. (h) Route planning: own Python versus ROS 2 plus Nav2

Evidence. Nav2 (ros-navigation/navigation2, 5,091 stars, pushed 2026-11-21) has the architecture the project cites:
planner server, controller server, progress checker, behaviour-tree recovery with contextual recoveries first
(`docs/research/TRAINING_RESEARCH.md:105-122`; Macenski et al. 2023). Resource evidence (forum posts from one search,
not fetched: UNVERIFIED but consistent): a Raspberry Pi 4 ran Nav2 near 100 % CPU with poor success, a Pi 5 about
60-75 % CPU, and one user could not start the MPPI controller on a Pi 4. Pinch's board has four A55 cores (slower
than a Pi 4's A72), plus a servo loop, vision and 2 GB RAM.

Verdict: the plain-Python re-implementation is the right call, with conditions.

- Nav2 is built for wheeled robots on metre-scale maps with lidar; costmaps, DDS messaging and lifecycle nodes cost
  far more than a 70 cm desk planner needs. The useful parts (A*, progress checker, recovery order) are a few hundred
  lines.
- Conditions: (1) keep Nav2's NAMES and contracts (planner, controller, progress checker, recovery behaviours,
  costmap layers) so the code stays comparable and the research note holds; (2) test the planner as data against the
  twin worlds; (3) keep the "pet" layer separate: a pet needs interruptible, expressive movement rather than shortest
  paths.
- Comparable non-ROS robots: Vector keeps its own map (quad-tree, origin ids; UNVERIFIED as above). Open Duck Mini and
  Stanford Pupper (stanfordroboticsclub/StanfordQuadruped, 1,835 stars, pushed 2024-10-24) do not navigate on their own
  (I did not verify Pupper's page; UNVERIFIED). No verified example of a small non-ROS pet with a full planner was found.
- Escape hatch: if the owner later wants room-scale maps, run ROS 2 with Nav2 or slam_toolbox (2,723 stars, pushed
  2026-11-02) on the LAPTOP and send velocity commands. Do not put it on the robot.

## 9. (i) Visual-navigation foundation models and (k) VLM on the laptop

- ViNT (Shah et al., CoRL 2023) and NoMaD (Sridhar et al., 2023) are trained on many wheeled-robot datasets with
  GPU-class networks. NaVILA (Cheng et al., 2024) is a vision-language-action model for legged robots that outputs
  mid-level commands like "move forward 75 cm" to a separate low-level walker. They target metre-scale robots with
  eye-height forward cameras. Pinch's camera is low and room-scale route following is not yet needed. Verdict: skip
  for now. The idea worth keeping from NaVILA is the interface (short commands above, a learned walker below), which
  Pinch already uses.
- VLM on the laptop: the 2026 edge-VLM paper (arXiv 2601.14921) finds a 2B model answers in under a second with lower
  accuracy while an 11B model is more accurate and slower. Pinch measured 18-37 s per decision for a laptop-CPU language
  model (a local design note (not in the repository)), so a VLM call on a cropped object is a background, seconds-long, optional service.
  Rules: (1) never in the control loop; (2) call it once on a NEW tracked object and cache; (3) send a small crop, not
  the stream; (4) without the laptop the object keeps a generic label and behaviour falls back to simple rules
  (approach, nudge, observe). This design is my proposal; no published small-pet robot with this fallback was found.

## 10. (j) Following a moving thing

Standard practice is detect, track, servo: the detector gives a box, the tracker keeps its id, and a controller turns
bearing error and box size (a distance proxy) into turn and forward speed, with a dead-band and a lost-target timeout
(the project's note already says this: `docs/research/TRAINING_RESEARCH.md:352-365`). With ToF, distance comes from
the grid rather than box size. Pick: ByteTrack-style (arXiv 2110.06864) association on detector output at 10 Hz and a
proportional controller. Fallback: the existing colour blob for one known toy when the detector fails.

## 11. Recommended minimal perception and estimation stack

1. **Reflex layer (always on, no laptop).** Five VL53L0X beams plus the foot-aware guard (keep) + IMU tilt and pickup
   detection + STS3032 load (stuck). One 8x8 ToF (VL53L5CX) at the front.
2. **State estimator.** An EKF fusing leg-kinematic and learned body velocity, IMU, ground flow, and optional
   marker/landmark fixes. New frame on pickup.
3. **Local map.** Robot-centred 2.5D grid + persistent danger list + things table.
4. **Vision on the NPU.** YOLOv8n INT8 (objects, person) and RetinaFace-mobile INT8 (faces, head turn), with a
   ByteTrack-style tracker. No ViT models.
5. **Audio on the CPU.** Silero VAD, openWakeWord, GCC-PHAT bearing. Sound level and direction drive reactions.
6. **Laptop services (optional).** Open-vocabulary naming and "what could I do with it" (VLM), face recognition,
   speech recognition, monocular depth, offline SLAM check. All asynchronous, cached, with timeouts.

## 12. What runs where, at what rate

| Function | Where | Rate | Model/method |
|---|---|---|---|
| Cliff beams + foot guard | robot CPU (or MCU) | 50-100 Hz | 5 x VL53L0X |
| Front multizone ToF grid | robot CPU | 15-30 Hz | VL53L5CX 8x8 |
| IMU + leg + flow state filter | robot CPU | 50 Hz (IMU), 10 Hz (flow) | EKF + small learned velocity net |
| Object detection | NPU | 10 Hz | YOLOv8n INT8 (35 fps alone in the zoo table) |
| Face detection / head turn | NPU | 5 Hz | RetinaFace-mobile INT8 (148.9 fps alone) |
| Pose (only when a person is near) | NPU | 5 Hz | YOLOv8n-pose INT8 (23.7 fps alone) |
| Tracking | robot CPU | 10 Hz | ByteTrack-style |
| Local map update + planner | robot CPU | 10 Hz map, 2-5 Hz replan | numpy grid + A* |
| Voice activity, wake word, bearing | robot CPU | 30 ms frames | Silero, openWakeWord, GCC-PHAT |
| Open-vocabulary naming, VLM, face ID | laptop | new objects only; seconds | YOLO-World/OWLv2, small VLM |
| Speech recognition | laptop | on demand | whisper.cpp / sherpa-onnx |
| Monocular depth | laptop | 1-5 Hz, advisory | Depth Anything V2 Small |
| SLAM checker (offline) | laptop | offline | ORB-SLAM3 / OpenVINS |

Rates are targets. Measure each on the real board with the servo loop running.

## 13. The single most valuable sensor addition

One VL53L5CX 8x8 multizone time-of-flight board at the front, tilted forward-down. Price USD 19.95 on Adafruit (about
INR 1,700-2,500 with shipping and duty, my estimate). Weight 1.9 g, size 25.4 x 17.7 x 4.6 mm. It turns "five dots
pointing down" into a 64-zone depth image up to 4 m, so Pinch can see walls, edges, a ball rolling in and the height of
toys, without a GPU. A small depth camera would cost more in money, weight and RK3566 compute. UNVERIFIED: whether the
VL53L7CX's wider view would be better; check its ST datasheet.

## 14. What to replace, what to keep

Replace or demote:
- Printed marker as the only position source (`robot/vision.py:140-148`): demote to an optional anchor.
- Four hard-coded colour ranges and heights (`robot/vision.py:30-36`): replace with a detector plus an "unknown thing"
  path; keep colour blobs only as a fallback and a test fixture.
- The Haar face cascade (`robot/vision.py:128-130`): replace.
- The hard-coded 0.70 x 0.45 m desk as the world (`pinch/world.py:14-21`): make it a learned map; keep the rectangle
  as one test world.
- The 30 % command-drift model alone (`robot/localizer.py:19`): replace with the learned velocity filter.

Keep:
- Downward beams as the safety reflex, and the foot-aware guard (`pinch/world.py:29-59`).
- The `pose()` versus `measured_xy()` split and the freeze-while-stuck rule (`robot/localizer.py:7-16`).
- Ground flow (`robot/vision.py:197-236`) as one filter input.
- "Uncalibrated lens means no position" (`robot/vision.py:143-144`).
- Same vision code in the twin and on the robot (`robot/vision.py:9-10`); extend it to the NPU models.
- The plain-Python planner with Nav2-named behaviours; no RL for navigation (`docs/research/TRAINING_RESEARCH.md:352-365`).

## 15. Open questions (unverified, need measurements)

- Frame rates of every model on the real Radxa ZERO 3W with servos running.
- Whether sherpa-onnx, RetinaFace and YOLOv8n convert and run cleanly on RK3566 with current toolkit versions.
- Speed of Depth Anything V2 Small on the owner's laptop.
- VL53L5CX details: firmware upload time, I2C address handling, whether it sees low objects at 3-4 cm height.
- Official repositories for DynaMem and NaVILA (only unclear results found).
