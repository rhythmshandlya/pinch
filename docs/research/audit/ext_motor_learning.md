# External audit: motor learning and sim-to-real (2026-11-21)

Scope: `docs/research/TRAINING_RESEARCH.md` (called "the note" below) and `docs/ARCHITECTURE.md` section 7
(line 128 onward, the tool table). Papers were fetched from arXiv (abstract pages, and full text where a number
mattered). GitHub numbers were read with `gh api` today. Anything I could not read is marked UNVERIFIED.
Terms: **sim-to-real** = making a policy trained in simulation work on the real robot. **sysID** = fitting the
simulator's motor model to measurements. **BAM** = "Better Actuator Models", Rhoban's servo-fitting tool.

## 0. Headline findings (ranked)

1. **Keep the plan "new environments on mjlab", but the reasons change, and a better template exists.** The closest
   published recipe to Pinch is `pollen-robotics/microduck_rl` (mjlab, BAM actuators, backlash model, delay and
   voltage randomisation, stand-up, ball kick, hot-swappable policy family). The note never mentions it.
2. **MuJoCo Playground is NOT the better fit.** Playground v1.0 (2026-11-06) was rewritten on PyTorch + MuJoCo Warp +
   RSL-RL; its old JAX/MJX legged environments moved to a frozen `rss2025` folder. The Open Duck training repo that
   used the old Playground has had no push since 2025-08-11 (205 stars). Both roads now lead to the mjlab stack.
3. **Two facts in the note are wrong.** DribbleBot trained for 7 billion steps (about 48 hours), not 77 billion
   (note lines 53, 321, 421). `HybridRobotics/whole_body_tracking` (BeyondMimic code) is archived and "retired",
   not "active" (note lines 295-296, 432); the maintained version is mjlab's tracking task.
4. **BAM does not support the STS3032.** Its Feetech entry is the STS3215 (7.4 V and 12 V parameter sets). The BAM
   paper itself identifies Dynamixel MX-64/MX-106 and eRob80 actuators, not Feetech. Pinch must run BAM's pendulum
   rig on its own STS3032 and write a small actuator class.
5. **"Newton on the Isaac Lab roadmap" is stale.** Isaac Lab v3.0.0 (2026-10-20) ships Newton, including a
   MuJoCo-Warp solver path. Isaac Lab is a more credible fallback than the note says.
6. **mjlab ships no small-servo robot and no crab.** Built-in robots: Unitree G1, Go1, and the YAM arm. Its
   tracking task is G1-only. Everything for Pinch is a custom robot (MJCF), as microduck_rl and Jumper show.
7. **Real evidence for the crab shape exists and was missed:** `KingKongRobotics/jumper` (22-DoF crab robot, trained
   with mjlab, 3.2k stars) trains `tripod`, `ripple` and `tetrapod` velocity gaits and converts policies for a
   Rockchip NPU. It is larger (400 x 400 x 200 mm, 1.8 kg) and has its own servos, but it is the best code
   template for a crab body in an RL framework.
8. The note is thin on **randomisation terms** and **latency, voltage and backlash**; section 2(i) fills this from
   microduck_rl.

## 1. Verification of the load-bearing claims

| # | Claim in the note | Verdict | What is actually true (source read) |
|---|---|---|---|
| 1 | Rudin et al., "Learning to Walk in Minutes", game-inspired curriculum, minutes on one GPU (S21, line 417) | CONFIRMED | arXiv 2109.11978, Rudin, Hoeller, Reist, Hutter, CoRL 2021. Abstract: single workstation GPU, flat terrain in under 4 min, uneven in 20 min. |
| 2 | Walk These Ways conditions on gait, body height, step height (S20, line 416) | CONFIRMED (paper); the exact list of conditioning inputs is UNVERIFIED from the abstract, which says only "variable footswing, posture, and speed" | arXiv 2212.03238, Margolis and Agrawal, CoRL 2022 oral. Repo `Improbable-AI/walk-these-ways`: 1,474 stars, last push 2024-06-16, licence NOASSERTION. Not maintained. The note's "last push 2024" is correct. |
| 3 | Disney BD-X: animation engine plus perpetual, periodic and episodic policies; actuator identification (S30, line 429) | CONFIRMED | arXiv 2501.05204 (Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer), RSS 2024. Full text: one perpetual policy (standing with head and torso commands), one periodic (walking), several episodic (one clip each). Policy at 50 Hz. Actuator models from first principles with parameters from sysID experiments, including static and dynamic friction and a backlash term; randomised within the measured range. |
| 4 | Olaf: temperature input, thermal reward, impact reward -13.5 dB (S31, line 430) | CONFIRMED | arXiv 2512.16705 (Müller, Knoop, Mylonopoulos, Serifi, Hopkins, Grandia, Bächer), v2 April 2026. Text: temperatures fed to the policy, control-barrier rewards for thermal and joint limits, impact rewards that cut foot sound by 13.5 dB. |
| 5 | BAM: pendulum-bench identification beats Coulomb-viscous; Rhoban/bam 629 stars (S42, line 436) | CONFIRMED with corrections | arXiv 2410.08650, Duclusaud, Passault, Padois, Ly; the venue is **ICRA 2025** (the note gives only arXiv). The paper validates four servos: Dynamixel MX-64, MX-106, eRob80:50, eRob80:100. Repo: 629 stars, pushed 2026-11-10, Apache-2.0. |
| 6 | "Open Duck Mini uses BAM" (S41, line 435) | CONFIRMED, but by the Open Duck README only | The README says "We used Rhoban's BAM". The BAM paper does not mention Open Duck. Repo `apirrone/Open_Duck_Mini`: 4,763 stars, pushed 2026-01-31. Its training repo `Open_Duck_Playground` is stale (205 stars, pushed 2025-08-11). Open Duck's servo is the Feetech STS3215, which is BAM's Feetech entry. |
| 7 | ToddlerBot: transferable motor sysID, zero-shot RL walking (S43, line 437) | CONFIRMED; venue "CoRL 2025" UNVERIFIED (arXiv page lists none) | arXiv 2502.00893, Shi, Wang, Song, Liu. Text: Dynamixel motors, a chirp signal on a motor test bed, an optimisation-fitted actuation model, policies transferred to a second robot without extra sysID, RL walking. Repo 807 stars, pushed 2026-11-16, MIT. |
| 8 | Jeon et al. 2023 legged pushing, 93.6 %, 5 Hz command policy, about 15 GPU-hours (S24, lines 51-52, 420) | CONFIRMED, with fine print | arXiv 2308.16820 (Jeon, Jung, Choi, Kim, Hwangbo). High-level policy sends velocity at 5 Hz to a 100 Hz low-level controller; reward includes `log(||goal-object|| + 0.05)` and an object-velocity term; training about 15 h on an RTX 3070. **93.6 % is one of five success criteria (the strictest, 0.03 m and 5 deg), in simulation.** The robot is the 27 kg Raibo (about 44 times Pinch's mass); real objects weigh 15-19 kg. |
| 9 | DribbleBot: 77 billion steps (lines 53, 321, 421) | **WRONG** | arXiv 2304.01159 (Ji, Margolis, Agrawal), ICRA 2023. Full text: "converges after 7 billion timesteps, or about 48 hours of training". Ball velocity as a command is confirmed. Fix to 7 billion; the contrast with push2-v1's 30M still holds (about 230 times, not 2,500 times). |
| 10 | HoST (humanoid standing-up): multi-critic, smoothness, implicit speed bound (S37, line 246) | CONFIRMED | arXiv 2502.08378, Huang, Ren et al., RSS 2025, deployed on the Unitree G1. Also relevant, missing from the note: He, Dong, Chen, Gupta, "Learning Getting-Up Policies for Real-World Humanoid Robots", arXiv 2502.12152, RSS 2025 (stage 1 finds a motion with few constraints, stage 2 refines it to slow and smooth). Both are humanoid-scale. |
| 11 | mjlab arXiv 2601.22074 ships velocity, tracking and manipulation tasks (S50, line 443) | CONFIRMED, plus limits | Zakka, Liao, Yi, Le Lay, Sreenath, Abbeel (UC Berkeley, Sorbonne), v2 25 Feb 2026. Three reference tasks: velocity tracking (G1 and Go1), motion imitation (**G1 only**), cube lifting (YAM arm). Repo: 3,437 stars, pushed today, v1.6.1 (2026-11-15), Apache-2.0. |
| 12 | Barkour: specialists plus a distilled Locomotion-Transformer; generalist slightly lower score, smoother transitions (S22, line 418) | CONFIRMED | arXiv 2305.14654 (Caluwaerts, Iscen, Tan et al., Google DeepMind). Specialist score 0.77; generalist maximum 0.71; the text claims smoother transitions from the single generalist. |
| 13 | ABS, learned reach-avoid value (S12, line 238) | CONFIRMED, but misapplied | arXiv 2401.17583 (He, Zhang, Xiao, He, Liu, Shi), RSS 2024; `LeCAR-Lab/ABS` 633 stars, pushed 2024-07-11, no declared licence. ABS is about **collision avoidance with obstacles** at 2-3 m/s with onboard sensing. Roadmap row 7 (note line 80) uses it for "edge, fall, wedge"; the paper does not show that. Keep optional. |
| 14 | DreamWaQ (you asked) | EXISTS, not in the note | arXiv 2301.10602, Nahrendra, Yu, Myung, ICRA 2023 per the arXiv listing (I read the abstract only). Proprioception-only locomotion with an implicit terrain "context" estimate. Relevant to 2(j). |
| 15 | Isaac Lab arXiv 2511.04831 (S52), MuJoCo Playground arXiv 2502.08844 (S54), BeyondMimic arXiv 2508.08241 (S35), AMOR 2505.23708 (S32) | UNVERIFIED | I did not fetch these four ids. The repos are checked in section 3. |
| 16 | BeyondMimic code "2.4k stars, active" (lines 295-296, 432) | **WRONG** | 2,443 stars, but **archived=true**. README: "retired and no longer maintained"; it points to mjlab's tracking task. Last push 2026-10-16. |

Two smaller errors: (a) note lines 65-66 and 409 say Newton integration is "on the roadmap" for Isaac Lab; it shipped in
v3.0.0. (b) ARCHITECTURE.md section 7 rejects "Isaac Lab 3.0 as the main stack: PhysX by default would differ from the
MuJoCo twin"; Isaac Lab 3.0 now has a kit-less Newton path with a MuJoCo-Warp solver, so that reason is weaker
(`isaac-sim/IsaacLab` v3.0.0 release notes).

## 2. Topic table

What Pinch does today: Brax PPO on MJX; presets in `configs/train/presets.toml` (gait 60M steps and 2048 envs, push2
30M, grasp 30M); environments in `training/registry.py:34-62` (residual walking on a scripted trot `gait`, `gait_v2`;
`terrain`, `terrain_hard`, `terrain_push`; `push2`; `emote`; `righting`; `grasp`). Policies run as numpy `.npz`.

| Topic | What Pinch does or plans | Best current paper (verified) | Best maintained code (gh) | Verdict | What to adopt |
|---|---|---|---|---|---|
| (a) Velocity-commanded walking, small servo robot | Residual policy over a scripted trot (48 fixed references); plan: continuous commands through the JAX gait (note line 272) | Rudin 2109.11978 (recipe); Margolis 2212.03238 (commands). No paper for Microduck or Open Duck, code only | `pollen-robotics/microduck_rl`: 2,937 stars, pushed 2026-11-17, Apache-2.0 (mjlab, 50 Hz, ONNX export, "usable gait in 1-2 h at 4096 envs"). Fallback: `apirrone/Open_Duck_Playground` (stale) | **Improve** (plan right, template missing) | Use microduck_rl's velocity task as the skeleton for locomotion v3: command block, zero-command sampling, turn-in-place bucket, dead-weight rule, unfiltered actions. |
| (b) Reference gait vs pure reward shaping | Residual on references; imitation planned for emotes | Grandia 2501.05204 (imitation reward on references). Hexapod: Liu et al. 2511.03167 (motion priors, real hexapod, Nov 2025) | `apirrone/Open_Duck_reference_motion_generator` (127 stars, pushed 2026-02-09) with the Playground `USE_IMITATION_REWARD` switch (polynomial reference over commands). Jumper keeps a gait clock | **Confirmed** (a reference prior is right for a 12-joint crab) | Keep the reference, but generate it continuously over commands as Open Duck does. Gate the gait clock off at zero command (Jumper's tutorial warns that a free-running clock makes the robot step in place). |
| (c) One conditioned policy vs a library; switching | Library of 4-5 policies under an authored selector; distil only if glitches (note lines 399-402) | Barkour 2305.14654 | microduck_rl: **hot-swaps walk, recover and trick policies behind one shared 61-D observation**, unused command slots zero-padded, never deleted; also a merged walk + fall-recovery task (`VelStand`). Jumper: manifest of slots with ramped switches | **Confirmed, plus a concrete mechanism** | Adopt the shared-observation-contract rule now. It makes switching a runtime swap and removes the need to distil. |
| (d) Expressive tracking of authored animation | One tracking policy on mjlab's tracking task (note line 76) | Grandia 2501.05204; Olaf 2512.16705; BeyondMimic 2508.08241 (id UNVERIFIED) | mjlab `tasks/tracking` (maintained BeyondMimic). The original repo is **archived**. Jumper does dance and gesture on a crab | **Improve** | mjlab's tracking task is humanoid (G1) only, so "use it as template" means rewriting the tracking terms for Pinch's joints. Microduck's per-clip episodic policies (`--kind episodic`) are a closer template and match BD-X. Low priority until the walker is done. |
| (e) Fall recovery, get-up | Learned righting (decision 76); smoother variant planned with HoST ideas | HoST 2502.08378; He et al. 2502.12152 (both humanoid) | microduck_rl `StandUp` (face-down, face-up, sitting) and `VelStand`, with a curated floor-contact collision model and a written reward-hacking playbook | **Confirmed**, template improved | Use microduck's rules: pay progress (change in uprightness), never "being low"; one fixed target pose with generous tolerance; add smoothness only after the skill is found; check every penalty's sign in the logs. |
| (f) Pushing, nudging, kicking | Learned push steering over a frozen walker (push v3), log-distance reward | Jeon 2308.16820; DribbleBot 2304.01159 | microduck_rl `BallKick` (70 mm, 15 g ball, **actor is ball-blind**). No maintained code for Jeon | **Improve** | Push v3 design is sound. Add a cheaper first experiment: a kick or poke **primitive** that a scripted layer aims, as in BallKick. Fix the DribbleBot figure. |
| (g) Grasp with a tiny open-loop claw | RL grasp, v6 adopted (11/24 on the pick bench) | None for a 15-23 mm, no-feedback claw. Nearest: Alexiev et al. 2609.31323 (blind grasp reflex for dexterous hands; needs hand proprioception) | None matching | **Missing evidence; lean scripted plus servoing** | At this scale with no servo feedback I found no evidence that RL beats scripted approach plus camera alignment plus a fixed closing motion. Keep RL for closing and lifting only if it beats a scripted baseline on the pick bench. |
| (h) Servo identification, STS series | Step 0: one hour of chirps and steps on a pendulum, BAM-style fit (note line 73) | Duclusaud 2410.08650 (ICRA 2025). Alternatives: Kovalev 2604.10351 (Apr 2026); Kovalev 2508.04696 (IROS 2025) | `Rhoban/bam` (library: STS3215, no STS3032). Playground v1 has `tasks/sysid` using `mujoco.sysid` | **Improve** | See 2(h). |
| (i) What to randomise for hobby servos | "Randomise around the fit", no terms or numbers (note line 381) | Peng 1710.06537, Tan 1804.10332 (old, still the method); Hu et al. 2503.01255 (static friction) | microduck_rl `FrictionDRBamActuator` | **Improve** | See 2(i). |
| (j) State estimation in the policy, no foot sensors | Estimator head or history input (note line 279) | Ji 2202.05481 (RA-L 2022); RMA 2107.04034; DreamWaQ 2301.10602 | `leggedrobotics/rsl_rl` v6.0.0 (distillation, symmetry); microduck_rl `distill.py` | **Confirmed** | Keep. History-conditioned policy plus RSL-RL distillation; the estimator head is the cheaper first step. |
| (k) Blind walking over small obstacles, 20 cm robot | Terrain curriculum with cables, hurdles, wedges (note line 275) | Lee 2010.11251 (note S8); Zhu et al. 2409.07409 "Robust Robot Walker: tiny traps" (Sept 2024, blind, contact encoder, two stages) | microduck_rl rough and slope terrains; no maintained code for the tiny-traps paper found | **Confirmed**, add one reference | Add 2409.07409. Scale obstacle height to leg length: a 15 mm bar on a 20 cm crab is a much larger fraction of its size than 2 cm on a 40 cm quadruped. |

### 2(h) Servo identification in detail

- **BAM and the STS3032:** BAM's library list is Dynamixel MX-64, MX-106, XL-320, XL330-M288-T, eRob80:50,
  eRob80:100, Feetech STS3215. Its parameter folders include `feetech_sts3215_7_4V` and `waveshare_st3025`.
  `bam/feetech/actuator.py` defines `STS3215Actuator` (voltage-controlled, firmware `kp=32`, `error_gain=0.166`
  measured with an oscilloscope; the firmware rate-limits the target). No issue mentions STS3032. The two servos
  share the Feetech protocol, so a new subclass with different torque constant, resistance and inertia ranges is the
  expected route (my inference, untested).
- **Rig:** a pendulum on the output shaft, a set of light arms of different lengths, a set of weights, a bracket,
  and a bus adapter (Pinch's FE-URT-1 fits). The arm must swing +/-90 degrees.
- **Procedure and time:** four 6-second trajectories (`sin_time_square`, `lift_and_drop`, `up_and_down`,
  `sin_sin`), played at several P-gains for every mass and length combination. Recording is minutes per
  combination. The fit (`bam.fit`) uses CMA-ES with a default of 100,000 trials and a held-out P-gain for
  validation; the BAM paper quotes minutes and about 4,000 iterations for one model. Plan about one day including
  checks. The note's "one hour" is plausible for recording only.
- **Output:** one JSON per model variant (M1 to M6) with motor constants, friction terms (base, viscous, Stribeck,
  load-dependent) and a position offset, plus MuJoCo CPU and MuJoCo Warp actuator classes. `bam/mjlab.py` exists,
  so it plugs into mjlab.
- **Alternatives, 2024-2026:** Kovalev et al. 2604.10351 (actuator models from encoder motion alone, through a
  differentiable simulator, no torque or current sensing; reports held-out position error 14.20 to 7.54 mrad and
  46 % farther travel in a real locomotion test); Kovalev et al. 2508.04696 (IROS 2025, differentiable MJX sysID
  inside the RL loop); ToddlerBot's chirp test bed. None names a Feetech servo. They suit servos that report only
  position, load and temperature.
- **Arm servos (GS-4308MG, PWM, no feedback):** BAM does not apply. Measure deadband, speed limit and gear play with
  the camera or a protractor rig, as the note already says.

### 2(i) What to randomise (terms from microduck_rl, a working small-servo recipe; its servo is a Dynamixel XL330)

I read the list of terms, not the numeric ranges. Take the numbers from Pinch's own bench fit.

- Battery voltage and voltage sag under load (microduck randomises both). Pinch's 2S LiPo droops, and servo torque
  scales with supply voltage.
- Command delay (bus plus loop latency), per environment, around the measured value.
- Friction magnitude **inside the actuator model**. Hu et al. 2503.01255 report that leaving static friction out of
  randomisation hurt real performance. Under BAM, MuJoCo's `dof_frictionloss` is zeroed, so randomising it is a
  silent no-op (microduck AGENTS.md).
- Backlash: a +/-1 degree unactuated hinge in series with each servo, with the **encoder read after the play**;
  they keep a plain and a backlash twin of each task. Pinch's arms have about 1.5 degrees (note line 383).
- IMU misalignment, encoder bias, observation noise. Never let randomisation accumulate across resets (their bug).
- **Temperature:** I found no source that randomises it. Olaf uses it as an observation and a reward term
  (2512.16705). Treat it as a long-session input, not a randomised parameter.
- Train and deploy with the same action filtering: microduck's policies are unfiltered, and a deployment-only
  low-pass filter breaks transfer.

## 3. Frameworks as of 2026-11-21 (gh)

| Framework | Stars | Last push | Latest release | Licence | Relevant shipped tasks | GPU and platform notes |
|---|---|---|---|---|---|---|
| google/brax | 3,265 | 2026-10-20 | v0.14.2 (2026-03-15) | Apache-2.0 | PPO, SAC | **Maintenance mode**: the last five commits are lint and type-checker suppressions; releases 2-4 months apart. Used by current Pinch envs and old Playground. |
| mujocolab/mjlab | 3,437 | 2026-11-21 | v1.6.1 (2026-11-15) | Apache-2.0 | Velocity (G1, Go1), tracking (G1), cube lift (YAM); community: microduck_rl, Jumper, Spiderbot | NVIDIA GPU only; FAQ recommends RTX 40-series or newer on Linux; Windows and WSL "preliminary, not guaranteed stable". No VRAM guidance in the docs. MJCF robots only. |
| google-deepmind/mujoco_playground | 2,326 | 2026-11-21 | v1.0.1 (2026-11-09) | Apache-2.0 | v1: SO-101, YAM, Trossen arms, G1 whole-body tracking, sysid, teleop. Old legged envs (Go1, Berkeley humanoid, Barkour, OP3, T1, Spot) live in frozen `rss2025` (JAX/MJX) | v1 is PyTorch + MuJoCo Warp + RSL-RL on new `reaf-sim` (24 stars). No small servo legged robot in v1. |
| google-deepmind/mujoco_warp | 1,646 | 2026-11-20 | v4.0.1 (2026-11-11) | Apache-2.0 | engine | Native Windows and Linux, CUDA 12.4 or newer. Memory grows with worlds and collider type (CCD costs more); docs suggest `testspeed --memory`. I found **no verified VRAM figure** for a 6 GB card. |
| isaac-sim/IsaacLab | 8,693 | 2026-11-21 | v3.0.0 (2026-10-20) | NOASSERTION | locomotion, manipulation, many robots | v3.0 runs kit-less on Newton (Warp physics; solver paths include MuJoCo-Warp, VBD, MPM). Python 3.12, CUDA 13.0, RSL-RL 5.5.1. Heavy install. |
| newton-physics/newton | 5,977 | 2026-11-21 | v1.8.0 (2026-11-11) | Apache-2.0 | engine | Warp-native; shared by Isaac Lab 3.0. |
| Genesis-Embodied-AI/Genesis | 30,281 | 2026-11-20 | v1.4.7 (2026-11-17) | Apache-2.0 | locomotion examples | Not evaluated beyond stats; no small-servo sim-to-real evidence found. |
| leggedrobotics/rsl_rl | 3,152 | 2026-11-20 | v6.0.0 (2026-11-20) | NOASSERTION | PPO, distillation, symmetry, RND | Used by mjlab, Playground v1, Isaac Lab. |
| Toni-SM/skrl | 1,103 | 2026-11-20 | 2.1.2 (2026-11-15) | MIT | many algorithms | Multi-framework; no reason to add it. |

**ONE recommendation: build new environments on mjlab, with `pollen-robotics/microduck_rl` as the structural
template, and freeze the existing MJX/Brax environments.** The plan stands. Reasons, in order:

1. Playground is no longer an alternative. v1.0 moved to the same Warp + RSL-RL stack and dropped legged tasks from
   the main tree; the Open Duck repo on the old Playground is stale.
2. A small servo robot with BAM actuators, backlash, delays, stand-up, kick and runtime policy swapping already runs
   on mjlab (microduck_rl, pushed 4 days ago). Jumper (a crab) and Spiderbot (hexapod, arXiv 2609.26989, Sept 2026)
   also train in mjlab. This is the evidence the note lacked.
3. BAM has mjlab support (`bam/mjlab.py`), and Playground v1's sysid flow uses `mujoco.sysid`.
4. Brax is in maintenance mode. The mjlab report (section 1) argues Playground's monolithic style makes multi-robot,
   multi-task codebases hard to maintain, which fits Pinch's 8+ environment variants.

Risks: mjlab is young (API moves at minor versions); Windows is "preliminary"; a 6 GB RTX 4050 is below the
documented comfort zone. Use the laptop for 64-env smoke tests (microduck's advice: a 5-iteration smoke test catches
about 95 % of config errors) and train on the remote GPUs. Run `testspeed --memory` on Pinch's 18-joint model before
choosing `num_envs`. Export must bake the observation normaliser into the policy (microduck documents that skipping
it works in the viewer and fails on hardware); Pinch's `.npz` export needs the same check.

## 4. Sprawled-leg specifics

Verified (abstract pages unless noted; all but RealAnt are 2024-2026):

- Liu et al., arXiv 2511.03167 (Nov 2025): real hexapod, adversarial motion priors, complex terrain, no vision.
  Posture not stated.
- Qu, Li, Zakhor, Yu, Zhang, arXiv 2412.10628 (Dec 2024): hexapod, teacher-student, runs on cheap hardware without
  joint feedback; uses a depth camera.
- Liu, Liu, Fang, Guo, arXiv 2511.08299 (Nov 2025, v2 June 2026): salamander-like robot, 22 learned gaits from a
  phase variable per limb, no reference motions, symmetry by data augmentation. Sim-to-real not stated.
- Tian, Fu, Ning, Pey, Ijspeert, arXiv 2603.16683 (March 2026): salamander robot, whole-body RL, real-to-sim
  matching, real flat and uneven terrain.
- Boney et al., arXiv 2011.03085 (RealAnt, 2020, v2 2022): EUR 350 ant-shaped quadruped, learns to walk in under
  10 minutes of real experience. Old; it is the RealAnt shape you asked about.
- Hu et al., arXiv 2503.01255 (Mar 2025): static friction in joints is a major sim-to-real term.
- Code: `KingKongRobotics/jumper` (crab, mjlab, tripod, ripple and tetrapod velocity tasks; 3,233 stars, pushed
  2026-10-30). No accompanying paper found.
- Sideways walking: PubMed 35439747 ("Sideways crab-walking is faster and more efficient than forward walking for a
  hexapod robot") appeared in a search. I could not open it (cookie wall), so authors, journal and figures are
  UNVERIFIED. The search summary said about 75 % faster and 40 % lower cost of transport; treat as unconfirmed.

I found no paper that trains RL walking on a sprawled crab-like robot with sideways as the preferred direction.
The note admits this gap (line 456).

What probably differs from upright quadrupeds (my reasoning, not from a paper I read):

1. Static stability is easy (low centre of mass inside a wide support polygon). The risk moves from falling to
   snagging and belly drag, so body-height and foot-clearance terms matter more than anti-fall terms.
2. Lateral walking is the cheap direction, so the command sampler should cover vy as densely as vx. Microduck found
   that uniform sampling gave turn-in-place about 2 % of experience and it never trained; expect the same for rare
   command corners.
3. Leg-to-leg and shin-to-shell contacts are common, so curate collision geometry (microduck's `groundcontact`
   model idea).
4. Gait choice is a design lever. Jumper's tripod, ripple and tetrapod tasks differ "in nothing else", so run gaits
   as a bench A/B.
5. Both left-right and front-back mirror symmetry exist: use symmetry augmentation (Mittal 2403.04359, already in
   the note; also used in the salamander paper).

## 5. What is missing from the note (8 items)

1. **microduck_rl and Jumper.** The two most relevant open recipes are absent. Add them as the reference
   implementations for locomotion v3 and the crab body.
2. **A shared observation contract for all motor policies** (microduck's rule), written into ARCHITECTURE.md. It
   turns the "switch policies per tick" question into a runtime swap.
3. **Reward-hacking rules** from microduck's AGENTS.md that apply directly to push v3, get-up and emotes: no
   jackpots for reaching a goal state; never gate a positive reward on a bad state; pay progress, not position;
   multiplicative composites at goal states; check penalty signs in the logs; add smoothness after the skill is
   found; keep a small non-zero range for every command so its weights stay alive; train zero command and rare
   command buckets explicitly. Several explain decision 88 independently of Ng et al.
4. **Delay, voltage and backlash terms** in the randomisation list, with "the encoder reads through the backlash".
5. **STS3032 is not in BAM.** Step 0 assumes it is. Plan a new actuator class and contribute the fit upstream. Name
   the encoder-only differentiable methods (2604.10351, 2508.04696) as a fallback.
6. **Fix the errors in section 1** (DribbleBot 7B not 77B; BeyondMimic repo archived; Newton shipped; BAM venue ICRA
   2025; the 93.6 % is the strictest of five criteria on a 27 kg robot). The project's own rule is verified sources.
7. **Deployment compute.** Jumper's tutorial converts policies for a Rockchip NPU and records a bug found only by an
   on-board check. Pinch's RK3566 NPU (0.8 TOPS) is unmeasured; a small MLP on the CPU is surely adequate, but the
   note's "well under 1 ms" (line 183) is an estimate. Measure it.
8. **An honest grasp section.** The note says RL-over-scripted evidence is "still missing" (line 350) and keeps RL
   anyway. Add a scripted-plus-servoing baseline on the pick bench so RL must beat something, and record that no
   paper supports RL for a 15-23 mm open-loop claw.

## 6. UNVERIFIED list

arXiv ids 2511.04831, 2502.08844, 2508.08241, 2505.23708 (not fetched); ToddlerBot venue; Walk These Ways
conditioning list (abstract is vague); the PubMed sideways-walking paper; numeric randomisation ranges in
microduck_rl; any GPU memory figure for mjlab, MuJoCo Warp or Genesis on Pinch's model (no source gives one).
