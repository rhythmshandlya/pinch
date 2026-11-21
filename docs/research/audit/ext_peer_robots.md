# External audit: Pinch against its closest peer robots

*2026-11-21. One focused pass (about 30 tool calls). Stars, push dates, archive flags and licences were read with `gh api` today. Papers were fetched from arxiv.org today. Anything I could not confirm is marked UNVERIFIED. Pinch claims cite `path:line` in this repository.*

## 0. What I checked and what I dropped

Verified to exist (gh stats today, none archived):

| Repo | Stars | Last push | Licence |
|---|---|---|---|
| apirrone/Open_Duck_Mini | 4,763 | 2026-01-31 | Apache-2.0 |
| apirrone/Open_Duck_Playground (training) | 205 | 2025-08-11 | none declared (files carry Apache-2.0 headers) |
| apirrone/Open_Duck_Mini_Runtime | 200 | 2026-09-07 | none declared |
| apirrone/Open_Duck_reference_motion_generator | 127 | 2026-02-09 | none declared |
| Rhoban/bam (actuator models) | 629 | 2026-11-10 | Apache-2.0 |
| Rhoban/onshape-to-robot | 650 | 2026-11-19 | MIT |
| hshi74/toddlerbot | 807 | 2026-11-16 | MIT |
| HybridRobotics/Berkeley-Humanoid-Lite | 2,017 | 2026-03-10 | MIT (hardware assets CC-BY-SA-4.0) |
| KingKongRobotics/jumper (crab, 22 DoF) | 3,233 | 2026-10-30 | Apache-2.0 |
| MakeYourPet/hexapod | 1,837 | 2023-07-27 | MIT |
| PetoiCamp/OpenCat | 5,572 | 2026-11-18 | MIT |
| mangdangroboticsclub/QuadrupedRobot (Mini Pupper) | 1,204 | 2026-11-06 | Apache-2.0 |
| stanfordroboticsclub/StanfordQuadruped (Pupper v1/v2) | 1,835 | 2024-10-24 | MIT |
| huggingface/lerobot | 29,009 | 2026-11-21 | Apache-2.0 |
| pollen-robotics/reachy_mini | 1,678 | 2026-11-19 | Apache-2.0 |
| google-deepmind/mujoco_playground | 2,326 | 2026-11-21 | Apache-2.0 |
| mujocolab/mjlab | 3,437 | 2026-11-21 | Apache-2.0 |
| kscalelabs/ksim | 228 | 2025-10-29 | MIT |
| kercre123/wire-pod (Vector) | 848 | 2026-11-17 | MIT |
| digital-dream-labs/vector | 119 | 2022-05-28 | NOASSERTION |
| anki/cozmo-python-sdk | 728 | 2022-11-26 | NOASSERTION |
| lukehollis/sesame-ml | 21 | 2026-08-17 | Apache-2.0 |
| Stanford-TML/robot_keyframe_kit | 179 | 2026-03-27 | MIT |

Dropped or not found: `Nate711/Pupper-v3`, `anki/vector`, `RealAnt/realant`, `Rhoban/rustypot`, `Rhoban/Rhoban-Quadruped`, `stanfordroboticsclub/pupper-v3-documentation` all return 404 under those names. Pupper v3 exists for me only as a docs site (see 1.5). I did not research Loona, EMO, Eilik, Lovot or Moflin: they are closed products with no engine code, so they add nothing to a build comparison (anything said about them would be UNVERIFIED). I found no open-source CRAB with learned walking at Pinch's scale apart from Jumper (a web search returned a sprawling salamander-style robot, arXiv 2504.14103, which I did not open, and RealAnt, arXiv 2011.03085, from 2020).

Papers fetched and read (abstract level unless noted):
- ToddlerBot: Shi, Wang, Song, Liu. arXiv 2502.00893 (v4, Oct 2025). Full HTML read for numbers.
- Berkeley Humanoid: Liao et al., arXiv 2407.21781 (2024).
- Demonstrating Berkeley Humanoid Lite: Chi, Liao, Long, Huang, Shao, Nikolic, Li, Sreenath, arXiv 2504.17249, RSS 2025.
- Extended Friction Models for the Physics Simulation of Servo Actuators (the paper behind BAM): Duclusaud, Passault, Padois, Ly, arXiv 2410.08650, ICRA 2025 (venue from the BAM README).
- Impact of Static Friction on Sim2Real in Robotic RL: Hu et al., arXiv 2503.01255 (2025).
- Design and Control of a Bipedal Robotic Character (Disney BD-X): Grandia et al., RSS 2024, arXiv 2501.05204.

## 1. The peers, how they are actually built

### 1.1 Open Duck Mini v2 (closest in servo class and approach)
- **What**: about 42 cm tall, a mini version of Disney's BD-X; README says BOM "under $400". Mass: UNVERIFIED.
- **Actuators**: Feetech STS3215 serial-bus servos (docs name BAM params `feetech_sts3215_7_4V`). The runtime sets 14 gains (`kps = [..] * 14`, Runtime `scripts/v2_rl_walk_mujoco.py` lines 175-179), so 14 servos including the head. Servo library: Rhoban rustypot (`rustypot_position_hwi.py`); its repo was not found by gh, so stats UNVERIFIED.
- **Computer**: Raspberry Pi Zero 2 W, Raspberry Pi OS Lite. Pinch's Radxa ZERO 3W is the same class (4x A55 vs 4x A53, plus a 0.8 TOPS NPU).
- **Sensors**: IMU, feet contact switches (`feet_contacts.py`); camera, speaker, mic marked optional in the checklist; antennas, LED eyes, projector.
- **Simulator / RL**: MuJoCo Playground (MJX + Brax PPO) in their own Playground repo. Reward: Disney's imitation reward, with reference motions from a parametric walk engine (reference_motion_generator repo). Recipe: `docs/sim2real.md`.
- **Actuator identification**: BAM (pendulum rig, recorded trajectories, extended friction models), exported to MuJoCo `damping, kp, frictionloss, armature, forcerange`. Their "current win" is a backlash task variant: `runner.py --task flat_terrain_backlash --num_timesteps 300000000` (Playground README).
- **Randomised** (`playground/common/randomize.py`, based on Berkeley Humanoid): floor friction U(0.5,1.0), frictionloss x U(0.9,1.1), armature x U(1.0,1.05), torso CoM +-5 cm, link masses x U(0.9,1.1), torso mass +-0.1 kg, qpos0 offset +-0.03, kp. Action delay 0-3 control steps and IMU delay 0-3 steps (`joystick.py` lines 62-65).
- **Rates**: sim_dt 0.002, control dt 0.02 = 50 Hz (`joystick.py` lines 51-52). Runtime `control_freq=50`, `action_scale=0.25`, low-pass action filter, ONNX inference (`v2_rl_walk_mujoco.py` lines 30-66, 82, 290-300). Python, no ROS.
- **Power**: not documented in the files I read (UNVERIFIED); there is a `check_voltage.py` script.
- **Personality**: sounds (beeps, "happy" wavs), antennas, eyes, head puppeteering, Xbox gamepad. No mind or memory layer. README: expression features "can be added after building the robot".
- **Order**: hardware design first ("mechanical design is pretty much finalized", Jan 2025), then RL ("now working on making it walk with RL"), then sim2real progress videos (Feb 2025).
- **Lessons shown**: (a) "cheap servomotors that are hard to model and not overly powerful"; "It's critical that the simulator simulates the motors accurately ... at worst, produce chaotic movements" (`docs/sim2real.md`). (b) They override CAD masses with the slicer's infill-aware estimate (same file). (c) Loctite on every metal-to-metal screw because vibration loosens them (`docs/assembly_guide.md`). (d) FTDI latency timer set to 1 ms (Runtime README). (e) Most-commented issues: BOM and servo-spec clarifications, 3D-print part problems, install errors. Playground issues are mostly CUDA/JAX/Brax version breakage.
- **Maintenance flag**: Playground has had no push since 2025-08 and declares no licence. Do not depend on it.

### 1.2 ToddlerBot (the best-documented sim-to-real pipeline)
- **Size/mass/cost**: 0.56 m, 3.4 kg, "under 6,000 USD", about 90% of it motors and computer (arXiv 2502.00893).
- **Actuators**: 30 active DoF, five Dynamixel models, full-state feedback at 50 Hz.
- **Computer**: Jetson Orin NX 16 GB. RL policy on CPU at 50 Hz; a 300M-parameter diffusion policy on GPU at 10 Hz.
- **Simulator / RL**: MuJoCo MJX via `train_mjx.py` with gin configs. Keyframe animations authored in `robot_keyframe_kit` and replayed or used as RL references.
- **Actuator identification** (paper plus `docs/tools/02_sysID.rst`): friction loss and damping by back-driving the motor at constant rpm against a torque sensor; armature by free spin-down; then chirp position tracking to fit 9 parameters jointly, final mean tracking error 1.3 degrees. Rig: a printed pendulum with 21700 cells as the load, Optuna for optimisation. "Motor dynamics are nearly identical across instances of the same model", so a second robot needed no new identification.
- **Randomised**: the paper text I read describes none (read through a summariser, so UNVERIFIED). Their claim is that identification, not randomisation, closed the gap.
- **Sim-to-real numbers**: walking tracking error sim vs real: position 0.082 vs 0.133 m, linear speed 0.016 vs 0.032 m/s, yaw rate 0.056 vs 0.113 rad/s.
- **Power**: about 75 Wh per hour of continuous walking; the paper says "3-5 hours" intermittent and "about 2 hours" in practice (the paper's own figures disagree). Packs: 2000 mAh LiPo (215 g) or 4x21700 (330 g).
- **Order**: design for reproducibility, then hardware and electronics, then the twin by zero-point calibration and identification, then control; "over a year of testing on two instances". The robot existed before the twin was trusted.
- **Lessons shown**: longest fall-free run 19 minutes because rising motor temperature left the training distribution; the model ignores temperature; plastic parts break on impact. Tips page (`docs/sections/05_tips_and_tricks.rst`): use a voltage tester or the pack over-discharges; XC330 cannot sustain 4 Mbps (use 2); a loose cable near an ankle cut the bus mid-motion; stripped screw holes in prints; gear wear; keep the Jetson fan uncovered.
- **Licence**: MIT, active. Newest paper in README: "Locomotion Beyond Feet", arXiv 2601.03607 (not read).

### 1.3 Berkeley Humanoid Lite
- "Under $5,000" humanoid, modular 3D-printed cycloidal gearboxes, zero-shot RL transfer, hardware and training frameworks open (arXiv 2504.17249). DoF count is not on the abstract page. Repo follows Isaac Lab; only `lowlevel/` ships to the robot; sim2sim validation is a named step. Licence MIT, hardware CC-BY-SA-4.0. The earlier Berkeley Humanoid paper (arXiv 2407.21781) credits a "narrow sim-to-real gap" to the design and a simple RL controller with "light domain randomization". Far larger than Pinch, but the cleanest "sim2sim first, only low-level code on the robot" example.

### 1.4 Jumper by KingKong Robotics (the only comparable crab)
- 22 DoF, 400 x 400 x 200 mm, 1.8 kg, RK3576 (6 TOPS NPU, 4 GB), 25.2 V 3000 mAh Li-ion with USB-C PD charging, 54x42 dToF array, 5 MP fisheye, two round TFT eyes, 4 mics (`docs/HARDWARE.md`). Apache-2.0, 3,233 stars.
- Training: mjlab + rsl_rl + MuJoCo Warp, exported to ONNX/RKNN and a Rust controller on CycloneDDS (DDS, no ROS). The motor model is a measured torque-speed curve loaded at import and enforced every sim step, with tests that compare the spec file against the MJCF (`assets/jumper/motor/README.md`).
- Caveat in its own docs: "`rknn_run` on real hardware is not yet verified" and "The control loop has never run against a bus" (`deploy/README.md`, section Open; `docs/PROJECT_GUIDE.md` lines 89-90). That describes this open-source controller. The company sells a physical robot, so I cannot say how much of this code the product uses (UNVERIFIED).
- Behaviour: AI-authored skins, motions and scenes via coding-assistant prompts. No pet mind layer found.
- Lesson: even a funded crab team publishes a simulated stack before its controller has driven a real bus, and says so in writing.

### 1.5 Stanford Pupper and Pupper v3
- Pupper v1/v2 paper: arXiv 2110.00736 (2.1 kg, 12 DoF, off-the-shelf parts; only the search summary read). `StanfordQuadruped` 1,835 stars, last push 2024-10. The Pupper v3 docs site says about USD 2,000; a secondary listing says Raspberry Pi 5, brushless GIM4305 motors and an RL walking policy (UNVERIFIED: not on the official page I could read). Takeaway: the education platform moved from hobby servos to brushless actuators for RL.

### 1.6 Petoi OpenCat (Bittle) and Mini Pupper
- OpenCat: 5,572 stars, MIT, active. Bittle kit about USD 269-339 (Tom's Hardware: $329), 10 custom servos in the base kit, about one hour walking, battery flagged as short in the review. Gaits are authored skill files run by a microcontroller board; the README mentions sim2real support (extent UNVERIFIED). Mini Pupper repo: 1,204 stars, Apache-2.0, pushed 2026-11-06 (ROS-based; not read in detail). They are the nearest pet-class robots by size and cost, and neither relies on learned walking by default.

### 1.7 Reachy Mini (Pollen / Hugging Face)
- Expressive desktop robot, Apache-2.0, 1,678 stars. Wireless version: Raspberry Pi CM4, battery, Wi-Fi. A daemon on the robot owns hardware I/O, safety checks and a REST + WebSocket API; client code (Python, JavaScript over WebRTC, or an LLM app) runs anywhere. The SDK clamps to limits (head pitch/roll +-40 degrees; 65 degrees maximum difference between head and body yaw). A MuJoCo simulation mode lets apps be built without hardware; apps are shared through Hugging Face Spaces. This is the closest model for Pinch's A3 (safe without the laptop, secured API).

### 1.8 Anki Vector / Cozmo (production pet)
- Vector's engine source was published by Digital Dream Labs (`digital-dream-labs/vector`, 119 stars, last push 2022, licence NOASSERTION). Community docs describe a behaviour tree stored as config files under `behaviorComponent` and a manual chapter "Emotion Model, Behaviour Engine, Actions and Animation Engine" (randym32 docs, seen only as a table of contents, so details UNVERIFIED). `wire-pod` (848 stars, MIT, active) replaces Anki's cloud for voice. Takeaway: a production pet is an animation-driven behaviour tree plus a mood model on a small onboard computer, with hosted language optional.

### 1.9 LeRobot (servo bus code)
- `huggingface/lerobot` (29,009 stars, Apache-2.0, pushed today) has `src/lerobot/motors/{feetech,dynamixel,...}`, a shared `motors_bus.py` and `calibration_gui.py`. Feetech STS3215 is the SO-100/SO-101 arm servo (my knowledge, not checked today), so this is the most widely exercised Feetech bus code. Pinch has its own `robot/feetech_bus.py` (file exists; I did not audit it).

## 2. Comparison table

| | **Pinch (today)** | Open Duck Mini | ToddlerBot | Jumper | Petoi Bittle / OpenCat | Reachy Mini |
|---|---|---|---|---|---|---|
| Build order | Simulation, 98 decisions; no hardware | Hardware design first, then RL, then sim2real | Hardware + electronics, then twin by identification; a year of tests | Simulated stack published; open controller not yet run on a bus | Hardware and authored gaits first | Hardware first; sim added for app builders |
| Actuator model | Not yet identified (listed as step 0, `docs/research/TRAINING_RESEARCH.md:73`); BAM has no STS3032 parameters | BAM fit of STS3215, backlash variant | 9-parameter chirp fit, 1.3 deg error | Measured torque-speed curve, tested against MJCF | None (authored gaits) | None |
| Learning stack | MJX + Brax PPO, mjlab planned; numpy export | MJX + Brax (Playground), imitation reward, ONNX | MJX + gin; keyframe references; diffusion for manipulation | mjlab + rsl_rl + MuJoCo Warp | None by default | None |
| On-robot runtime | Plain Python on RK3566, no ROS | Python, ONNX on Pi Zero 2 W, no ROS, 50 Hz | Python on Jetson Orin NX, no ROS, 50 Hz | Rust + DDS, 1 kHz publish, NPU | Microcontroller firmware | Daemon + REST/WebSocket on Pi CM4 |
| Perception | Fisheye cam, 5 ToF, IMU, touch, 2 mics | IMU, foot switches; camera/mic optional | Stereo depth + skill classifier | dToF array, fisheye, 4 mics | IMU; touch/camera add-ons | Camera, mics |
| Behaviour layer | Large (mind, memory, activities) but only in sim | Sounds, antennas, eyes; gamepad | Keyframes + skill classifier | AI-authored motions | Skill files, sounds | App store, LLM conversation |
| Power | 2S 850 mAh, 27-40 min measured in sim, hand charging | Not found in files read | ~1 h continuous, ~2 h practical | 25.2 V 3000 mAh, USB-C PD | ~1 h, built-in charging | Battery (wireless version) |

No peer in the table uses ROS 2 for its learned-walking path; Jumper uses DDS directly. Pinch's "no ROS" choice is the norm among the small learned-walking robots (Mini Pupper's repo is ROS-based, but it is not a learned-walking stack).

## 3. The five most important things peers do that Pinch has not

1. **Put real servos on a bench and fit the model before training.** Open Duck (BAM) and ToddlerBot (chirp fit, 1.3 degree error) both did; ToddlerBot found the fit carries to a second robot. Pinch lists it as step 0 and has not done it (`TRAINING_RESEARCH.md:73`). BAM ships STS3215 parameters only, so the STS3032 needs its own run. Cost: one pendulum rig and a day.
2. **Randomise delay and the friction the model leaves out.** Open Duck randomises action and IMU delay by 0-3 control steps, plus kp, armature and frictionloss. A grep for "delay" and "latency" over `training/*.py` today finds nothing, and only `grasp_env.py` and `export_policy.py`-style files mention friction/kp/armature; I did not read them, so treat this as a lead, not a verdict. Hu et al. (arXiv 2503.01255) report real-robot policies "significantly worse" when static friction was missing from randomisation.
3. **Run a real policy on a lifted or tethered robot early, and publish the gap.** ToddlerBot published sim vs real tracking error; Open Duck keeps a hardware checklist; Jumper writes down what it has not verified. Pinch has no measurement of this kind.
4. **Reuse tested servo-bus and calibration code.** LeRobot's Feetech bus and Open Duck's scripts (`check_motors.py`, `find_soft_offsets.py`, `calibrate_imu.py`, `check_voltage.py`) have run on real hardware; Pinch's `robot/feetech_bus.py` and `robot/calibrate.py` have not.
5. **Build expressive behaviour as an authored layer over learned low-level skills, on a body that already works.** Disney BD-X (RSS 2024) feeds an animation engine's commands to an RL controller; Open Duck added expression after the robot walked. Pinch built the mind first and judged it only from simulated video, and the owner rated it 5% of the goal.

## 4. Things Pinch does that no peer does

| Pinch feature | Strength or warning |
|---|---|
| 98 recorded decisions and a golden-trace behaviour test with no robot | **Warning.** Excellent discipline, but all evidence is simulation; no peer reached this volume before a robot walked. |
| Strict layers (training never imported by robot) and policies as `.npz` | **Strength.** Matches Berkeley Humanoid Lite's `lowlevel/`-only deploy and Open Duck's ONNX-only runtime. Keeps JAX off the robot. |
| A pet mind with memory, boredom and personality on a 0.8 TOPS board | **Strength in design, warning in timing.** Vector shows it works, on a tested body. Peers add it after the body walks. |
| A GPU job launcher with ledger (`ops`) | **Neutral.** Peers use Colab or one workstation. It is infrastructure the robot does not yet need. |
| Feedback-free PWM arm servos for claws | **Warning.** No peer uses servos without position feedback for manipulation; Pinch's own INSPIRATION.md notes arm angles are only commanded. Expect trim drift and an unmeasurable claw state. |
| Local LLM on a laptop with a safe fallback | **Strength.** Reachy Mini and Vector (wire-pod) both treat language as a service the robot survives without. |

## 5. Code, assets and tools Pinch can reuse

| Item | Licence | Stats (today) | Use |
|---|---|---|---|
| Rhoban/bam | Apache-2.0 | 629 stars, pushed 2026-11-10 | Fit an STS3032 friction and PD model; `bam.to_mujoco`, MuJoCo CPU and Warp |
| huggingface/lerobot `motors/feetech` + calibration | Apache-2.0 | 29,009 stars, pushed today | Feetech bus reads/writes and calibration; compare with `robot/feetech_bus.py` |
| Open_Duck_Mini_Runtime scripts and checklist | no licence declared | 200 stars, pushed 2026-09-07 | Read as patterns only; cannot copy until a licence appears |
| mujoco_playground and Open Duck `randomize.py` | Apache-2.0 (headers) | 2,326 stars; Playground repo 205 stars | Delay and friction randomisation recipe for `training/` |
| hshi74/toddlerbot `docs/tools/02_sysID.rst` and rig CAD | MIT | 807 stars, pushed 2026-11-16 | Pendulum rig design and Optuna procedure |
| Stanford-TML/robot_keyframe_kit | MIT | 179 stars, pushed 2026-03-27 | Keyframe posing for emotes (alternative to INSPIRATION.md idea 6) |
| Rhoban/onshape-to-robot | MIT | 650 stars, pushed 2026-11-19 | Only if CAD moves to Onshape; slicer-mass override trick |
| mujocolab/mjlab | Apache-2.0 | 3,437 stars, pushed 2026-11-21 | Already the planned stack; Jumper is a worked example |
| pollen-robotics/reachy_mini daemon + SDK pattern | Apache-2.0 | 1,678 stars, pushed 2026-11-19 | Architecture reference for A3 (daemon owns hardware and safety, thin client) |

## 6. Is "98 decisions in simulation, no hardware" normal?

No. Among the peers I read, the robot exists first or alongside the simulation work, and the twin is calibrated against it:
- ToddlerBot: the twin is built from "zero-point calibration and motor system identification" (arXiv 2502.00893), so its model comes from measurements of the built robot, tested for over a year on two units.
- Open Duck: CAD frozen in January 2025 before RL began; the authors call the servos "hard to model" and warn a wrong motor model gives "chaotic movements" (`docs/sim2real.md`).
- Jumper is nearest to Pinch (large simulated stack) and lists "never run against a bus" as open in its deploy README. It still has a physical product to test against; Pinch does not.
- Berkeley Humanoid (arXiv 2407.21781) credits its "narrow sim-to-real gap" to a design built for clean simulation, a simple RL controller and "light domain randomization".
- Hu et al. (arXiv 2503.01255): missing static friction in randomisation made a real policy "significantly worse"; an actuator network fixed flat ground but failed on stairs. Cheap-servo friction is the usual hidden gap.

Implication: simulation is trustworthy for architecture, reward design, test infrastructure and ranking policies against each other. It is not trustworthy for absolute behaviour on hobby servos until the actuator is measured. No peer claims otherwise. I found no peer that states a numeric trust limit; the evidence is gaps (ToddlerBot's real tracking error is about double the sim's), not a threshold.

## 7. Cost sanity check

- Pinch: INR 88,566 total (`hardware/BOM.md` header). 14 x STS3032 = INR 46,586.40 (BOM servo row), 53% of the bill. At about INR 83 per USD (my assumption) the total is about USD 1,070.
- Open Duck Mini: "under $400" (README) for 14 STS3215 and a Pi Zero 2 W; bigger (42 cm) yet cheaper. Pinch pays INR 3,328 (about USD 40) per servo from Evelta; cheaper sources for the same class probably exist (UNVERIFIED, prices not checked).
- ToddlerBot: under USD 6,000 (90% motors and computer). Berkeley Humanoid Lite: under USD 5,000. Petoi Bittle: about USD 270-340. Pupper v3: about USD 2,000 (docs claim).
- Verdict: Pinch is inside the range for its servo count; 14 servos at INR 46.6k is not an outlier. Open Duck's example suggests the whole robot could cost less. Spares are 2 STS3032 (about INR 6.7k): buying 12 first and spares after the first bench test would defer that spend (savings UNVERIFIED).

## 8. Suggested order for the owner

1. Buy one STS3032 and a few GOTECK servos, build a pendulum rig, fit a BAM-style model, measure bus latency (Section 3 item 1; `TRAINING_RESEARCH.md:73`).
2. Add delay and friction randomisation (item 2) before adopting more policies.
3. Build one leg on the bench with the Feetech bus; run LeRobot's calibration against it and compare with `robot/feetech_bus.py`.
4. Keep the 16-piece plan but order it so A2 (servo truth) and a lifted-robot test come before more mind and expression work.
