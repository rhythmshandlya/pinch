# Training research: getting unstuck, and how to train a robot like Pinch

*Research note, 2026-11-20. It answers two questions from the owner:*

- *Q1: how do research and production systems detect and recover from being stuck, wedged, tangled or blocked,
  and which combination fits Pinch?*
- *Q2: how should a small legged robot like Pinch be trained overall?*

*It builds on [APPROACH_PLAN.md](../APPROACH_PLAN.md) (2026-11-19) and on what decisions 50–97 in
[DECISIONS.md](../DECISIONS.md) measured. Sources are numbered [S1], [S2] and so on and listed in the tables in
§Q1.5 and §Q2.8. Repository stars and last-push dates were read with `gh api` on 2026-11-20. Where the evidence is
thin or comes from a much bigger robot, the text says so.*

---

## Executive summary

1. **Getting stuck is mainly a detection-and-escalation problem, not a learning problem.** Production stacks
   handle it in three parts [S1][S2][S3]:
   - a cheap progress check that compares commanded motion with measured motion;
   - an escalating ladder of recoveries, from local and contextual to global and system-level;
   - a memory of where things went wrong.

   Learning adds two things on top. It makes the *low-level reflexes* robust, so fewer situations become "stuck" at
   all [S8][S9][S12]. And it gives *better estimates* of body motion and contact from proprioception
   [S14][S15][S16]. Pinch already has the skeleton: `StuckSense`, the `Unstick` activity and the danger map.
   What it lacks:
   - a detector that works without camera fixes;
   - a per-leg "caught" signal;
   - more than one escape move per stuck type;
   - escape moves that the walking policy has actually been trained to do.
2. **Learn the body, author the mind.** APPROACH_PLAN's conclusion still holds. The best expressive legged robots
   train a **small library of motor policies** driven by an authored animation and behaviour layer. Examples are
   Disney's BD-X and Olaf, which keep separate standing, walking and per-clip policies [S30][S31]. No published
   system learns the pet-level choices end to end better than authored logic [APPROACH_PLAN §4.3].
3. **Pinch's biggest unaddressed sim-to-real risk is the actuator model.** Every small-servo success identified its
   servos from bench logs:
   - Open Duck Mini uses Rhoban's BAM [S41][S42];
   - ToddlerBot runs a sysID pipeline [S43];
   - Disney identifies PD, friction and backlash [S30];
   - ANYmal learned an actuator network [S40].

   Pinch's servo-bench item is still open (docs/TODO.md). Until it is done, no policy can be trusted on the real
   robot, however good its sim bench looks.
4. **push2-v1 failed for a known, fixable reason.** Its reward paid for *being near* the object. That is a
   textbook local optimum. Successful legged pushing uses a different reward:
   - a log-distance object-to-goal term;
   - a term for the object's velocity toward the goal;
   - proximity terms that are bounded and switched off near the goal.

   That design reached 93.6 % success in simulation [S24]. Two more tools apply: potential-based shaping, which
   pays nothing for hovering [S27], and a reverse curriculum that starts episodes close to success [S28]. The
   budget was also small: 30M steps, against about 15 GPU-hours for [S24] and 77 billion steps for DribbleBot [S25].
5. **The single-walker approach hit interference.** terrain-push-v2 gained the eraser but lost the cap
   (decision 96). Walk These Ways [S20] makes the regime an explicit *command* (gait, body height, step height), so
   push-crouch and high-step climbing stop competing for the same uncommanded behaviour. Barkour [S22] shows that
   specialists beat a distilled generalist slightly on score, while the generalist transitions more smoothly. So:
   condition first, distil only if switching glitches appear.
6. **Tool stack.** Build *new* environments on mjlab (MuJoCo Warp + RSL-RL), starting with the motion-tracking
   policy. The existing Brax/MJX environments stay until they need retraining. Reasons:
   - mjlab ships velocity, motion-tracking and manipulation task templates [S50];
   - RSL-RL brings teacher–student distillation, symmetry augmentation and RND curiosity out of the box [S51];
   - it keeps the MuJoCo physics the twin uses.

   Isaac Lab remains the fallback: PhysX by default differs from the twin; Newton integration is on its roadmap
   [S52]. **Uncertain:** mjlab's throughput on Pinch has not been measured. Benchmark it as decision 97 did for
   MJX before committing.

### Recommended training roadmap (ordered)

| # | What | Method, reward and curriculum | Why this order | Gate |
|---|---|---|---|---|
| 0 | **Servo bench sysID → actuator model** (STS3032 legs; GS-4308MG arms) | One hour of logged chirps and steps on a pendulum rig. Fit a friction model, e.g. BAM's extended Stribeck/load-dependent model [S42], or a small actuator MLP on position-error history [S40]. Measure bus latency. Randomise both around the fit [S44][S45]. PWM arms: model them open-loop (no feedback), with measured gear play | Every policy below inherits this. It is the cheapest high-value step, at CPU-minutes, as noted in [APPROACH_PLAN §6] | Sim and real step responses agree; re-bench gait-v2, righting and grasp with the new model |
| 1 | **Stuck stack v2** (mostly not RL) | (a) Proprioceptive body-velocity and contact estimator, trained alongside the walker [S14]. (b) Per-leg residual from STS3032 position, load and temperature (momentum-observer idea [S16]; entanglement [S17]). (c) An escalation ladder with several escape primitives and retries [S2][S3]. (d) Per-place, per-type memory of which escape worked [S19]. Details in §Q1.4 | Cheap, testable with the existing fault injection (decision 71), and it covers a gap the PRD calls core: a pet that gets itself out of trouble | Fault-injection matrix: wedged, caught leg, belly on edge, blocked path, weak leg. Measure escape rate and time, and false "stuck" calls per hour on the soak runs |
| 2 | **Locomotion v3: one command-conditioned walker** | Continuous commands from the JAX gait (decision 66): vx, vy, yaw rate, plus Walk-These-Ways behaviour parameters (body height, step height, gait frequency, "push crouch" flag) [S20]. Terrain curriculum promoted by success [S21] over desk, hurdles, ramps, cables and *wedge* worlds with shin and shell collisions. Push references sampled in balance with plain walking. Mirror-symmetry augmentation [S53]. History or estimator input [S8][S14] | Fixes decision 96's interference by making regimes explicit. The escape moves (high step, crouch, back up) become *commands* the policy was trained for. Mood styles (slow and low when sad, quick and high when excited) come for free | Hurdle capability ≥ terrain-v1, tidy cap ≥ gait-v2, robustness worlds 0 falls |
| 3 | **Generic motion-tracking policy** (emotes, claw expression, idle layers) | Imitation reward on keyframed clips plus layered offsets, as in BD-X's episodic and perpetual policies [S30]. Smoothness and thermal terms (STS3032 reports temperature) [S31][S32]. mjlab's tracking task as the template (BeyondMimic-style) [S35][S50] | New emotes then cost an animation, not a training run. The Runpod 4090 run is already earmarked for it (decision 97) | Emote robustness bench, servo-speed checks (`sim/check.py`), ≥ current per-emote policy |
| 4 | **Push v3: learned steering over the frozen walker** | 5–10 Hz command policy as in [S24]. Extrinsic reward `log(‖goal−object‖+ε)`; reward for object velocity toward the goal; proximity bounded or potential-based [S27] and switched off near the goal [S24]. Reverse curriculum: short pushes first, then 8–25 cm [S28]. Asymmetric critic with privileged object mass and friction [S29]. Alternative: command the *object velocity*, as DribbleBot does [S25], and keep the scripted planner for the goal | The cap at 9–11/24 is the weakest important skill (decisions 65, 84, 96). The failure mode is understood | Tidy bench cap > 11/24 with no rise in knock-offs |
| 5 | **Grasp v7+ teacher→student** | Continue grasp-v7. Distil a privileged teacher into a student that sees only robot-observable inputs [S8][S51] | Already adopted v6 (11/24). The gains depend on the claw redesign | Pick bench |
| 6 | **Get-up smoothing** | Action-rate cost plus an implicit joint-speed bound (HoST [S37]); actuator model from step 0 | Decision 85 measured near bang-bang targets (p95 17–20 rad/s), and righting-smooth-v1 did not get gentler. HoST names the fix | Righting bench with commanded joint speed p95 ≤ swing × k |
| 7 | *(optional)* **Learned safety value / recovery switch** (ABS-style) | A reach-avoid value network that predicts imminent failure (edge, fall, wedge) and switches to a recovery policy [S12][S13] | Only if the authored supervisor and stuck stack show misses on long soaks. It adds a learned component to a safety path, which APPROACH_PLAN warns against | Soak runs: misses before against after |
| — | **Not to train with RL:** desk navigation, following, behaviour selection, emotion | Geometric planning plus visual servoing on the 70×45 cm desk; authored utility selector (§Q2.4, §Q2.6) | Small, fully observed workspace. Classical methods are exact, explainable and cheap on the RK3566 | — |

---

## Q1. Getting stuck: detection and recovery

### Q1.1 What "stuck" means for Pinch

| Kind | Example on the desk | Signal Pinch can read |
|---|---|---|
| Blocked, no progress | Walking into the tray wall or a book | Commanded vs measured motion |
| High-centred, belly on an edge | Shell on a cable or book edge, feet unloaded | IMU tilt, leg loads (STS3032 load) |
| Leg caught or entangled | Shin under a cable, foot in a gap | Per-leg position error and load during swing |
| Foot over the drop | Desk edge | Cliff beams, feet margin (exists) |
| Flipped | On its back | IMU (exists; learned get-up) |
| Livelock | Planner keeps re-approaching the same failing line-up | Repeated failure at the same place and goal |
| Degraded body | Weak or dead servo, hot servo | Load, temperature, tracking error |

`StuckSense` (pinch/touch.py) already covers the first, second and fourth kinds: tilt, feet loaded, feet margin,
and "walking but not moving". The progress check needs camera marker or ground-flow fixes, so with no
measurement it gives no judgement on progress. `Unstick` then tries `_recover` twice and asks for help. `_recover`
does one of two things: shuffle toward the middle of the desk, or run the high-step climb in the walking
direction.

### Q1.2 Classical and production practice

- **Progress checking.** Nav2's controller server runs a progress checker. The default declares failure if the robot
  has not moved 0.5 m within 10 s. A pose variant also counts rotation [S1]. The thresholds are
  platform-specific: slow robots must extend the window [S1]. Mars Exploration Rover drivers used the same idea.
  Visual odometry was compared with wheel odometry to detect slip, and driving stopped beyond a slip limit [S6].
  This is exactly Pinch's "commanded vs measured" principle.
- **Recovery behaviour trees.** Nav2's default navigate-to-pose tree wraps navigation in a `RecoveryNode`. On failure
  it runs a recovery subtree with retries (default 6), in a fixed order [S2]:
  1. clear the local and global costmaps;
  2. spin;
  3. wait;
  4. back up.

  Nav2 separates *contextual* recoveries, such as re-planning or clearing one costmap, from *system-level* ones,
  and recommends exhausting the contextual ones first [S2]. Behaviour trees are the standard way to express this
  [S3][S4]. Macenski et al. describe the design and its field use [S5].
- **Local controllers.** MPPI samples many perturbed control sequences through a model and weights them by cost
  [S7]. Nav2 ships an MPPI controller [S1]. Sampling with a short horizon gets out of some local minima that
  gradient or DWA-style controllers sit in, but not out of geometric traps. Those still need the recovery layer or a
  global re-plan.
- **Escaping local minima.** Bug algorithms follow the obstacle boundary until the line to the goal is free, with
  completeness guarantees in the plane [S18]. This is a cheap, explainable fallback when a direct approach keeps
  failing.
- **Body–terrain physics.** Cockroaches and robots pushing through cluttered beams get trapped in "basins" of a
  potential-energy landscape. Leg-driven *body oscillation* supplies the energy to cross into an easier mode, such
  as rolling instead of pitching [S10]. For a small robot: when wedged, a deliberate rocking or jitter, and a
  change of body attitude, can free it where pushing harder does not.
- **Damage and adaptation.** Intelligent trial and error lets a hexapod with a damaged leg find a working gait in
  about a minute. It tries a handful of behaviours from a precomputed map, guided by what worked [S19]. The lesson
  for Pinch's stuck stack: keep a repertoire of escapes, and *remember which worked where*.

### Q1.3 Learned approaches

- **Robust reflexes reduce how often the robot gets stuck.**
  - Teacher–student proprioceptive policies trained over a terrain curriculum stayed robust to impediments never
    seen in training [S8]. The paper's analysis also describes emergent foot-trapping reactions; this is in the main
    text, not the abstract.
  - RMA encodes recent proprioceptive history into a latent estimate of the environment and adapts within a
    fraction of a second [S9].
  - Both rely on the same thing: give the policy history, and train it on situations where feet catch.
- **Learned recovery controllers.**
  - Lee, Hwangbo and Hutter trained a get-up and recovery controller for ANYmal: a behaviour selector over learned
    recovery, standing and locomotion policies [S11b]. Pinch's learned get-up follows this.
  - HoST adds multi-critic training, a terrain curriculum, smoothness regularisation and an implicit speed bound so
    that learned stand-ups are not violent on hardware [S37].
- **Failure prediction and switching.**
  - ABS trains an agile policy, a recovery policy, and a reach-avoid value network that decides when to switch and
    guides the recovery policy [S12].
  - Recovery RL is the general form: learn a safety critic from offline violations, and hand control to a recovery
    policy when it predicts danger [S13].
  - PrePARE learns a predictor from proprioception that flags slips up to 720 ms before a fall, using anomaly
    detection, expert labels and an ensemble [S15].
  - These help most when failures are fast, such as falls. Being wedged is slow, and a progress check suffices.
- **Contact and collision from proprioception.**
  - A state estimator trained alongside the policy predicts body velocity, foot height and contact probability from
    proprioception. It reached 3.75 m/s on a Mini Cheetah [S14].
  - Classical collision detection uses the residual between expected and measured joint motion or momentum [S16].
  - Yim et al. sense leg entanglement from proprioception and free the leg during its swing, succeeding in 14 of 16
    lab trials [S17].
  - PPL trains a proprioceptive policy for crawl spaces whose estimator predicts collision states [S38].
  - Jin et al. treat perception failures as invisible obstacles and pits, and learn to react to body collisions
    [S39].
- **Learned local navigation for legged robots.**
  - ViNL decouples a vision navigation policy, which outputs velocity commands, from a locomotion policy that steps
    over clutter [S33].
  - Miki et al. use a two-layer hierarchy with voxel input to walk in confined spaces [S34].
  - Parkour systems use either one end-to-end policy [S46][S47] or a skill library chosen by a navigation module
    [S48].
  - Hierarchical RL navigation ran kilometre-scale missions on a wheeled-legged robot [S49].
  - All of these use depth cameras or LiDAR and large robots. Pinch has one wide RGB camera and a desk it can map
    almost completely, so these are **not a fit** beyond the decoupled command interface.

### Q1.4 What fits Pinch

The design follows the Nav2 shape (progress check, then an escalating recovery tree), adds learned estimation and
reflexes, and keeps the danger map:

1. **Detection** (all cheap on the RK3566. An MLP estimator should be well under 1 ms, an *estimate* not yet
   measured on the board.)
   - **Velocity without the camera.** Train a body-velocity and foot-contact estimator alongside the next walker
     [S14]. Export it with the policy. `StuckSense` then has a progress signal on every tick, and camera fixes only
     confirm it.
   - **Per-leg "caught".** Compare each leg's commanded and measured STS3032 position and load during swing. A
     persistent residual on one leg means it is caught [S16][S17]. The arms' PWM servos report nothing, so arm
     snags can be inferred only from body motion.
   - **Livelock.** Count failures per (place, goal). The danger map already learns from "stuck"; add "the same
     line-up failed N times", the Nav2 retry count [S2].
   - **Thresholds.** Nav2's 0.5 m in 10 s is for large robots [S1]. Pinch's 8 mm in 2.5–3 s is already scaled.
     Tune it on soak runs for the false-alarm rate, the way PrePARE tunes its predictor's operating point [S15].
2. **Recovery ladder** (authored, in `Unstick`; contextual before system-level [S2])
   1. **Reflex in the policy.** The walker, trained on wedge, cable and gap worlds with shin and shell contact, lifts
      or retracts a caught foot by itself [S8][S17].
   2. **Contextual primitives, chosen by stuck type, each with a timeout and a retry budget:**
      - back up along its own recent trail (Nav2 BackUp [S2]);
      - turn in place (Spin);
      - high-step climb (exists);
      - raise or lower the body and *rock* side to side to leave an energy basin [S10];
      - switch gait;
      - for a caught leg, lift and retract that leg only [S17].

      In locomotion v3 these are commands to the same policy (body height, step height, frequency [S20]), not
      separate controllers.
   3. **Re-plan.** Mark the spot in the danger map (the costmap analogue), approach from another side, or
      boundary-follow (bug-style [S18]).
   4. **System level.** Ask for help (exists). Remember which primitive freed it for this stuck type and place, and
      try that first next time (trial-and-error memory [S19]; a bandit over primitives, as for other preferences
      [APPROACH_PLAN §3.12]).
3. **Learning, later and optional.** An ABS-style reach-avoid value as a learned "about to fail" monitor [S12],
   only if long soaks show the authored layers missing failures. Keep it advisory to the Simplex supervisor, never
   a replacement.

**Uncertain.**
- No paper studies stuck recovery on a hand-sized hexapod-like crab. The closest small-robot evidence is
  damage adaptation on a hexapod [S19] and cockroach-scale terradynamics [S10].
- The thresholds and escape order above are engineering choices to be tuned on the fault-injection matrix, not
  values from the literature.

### Q1.5 Sources for Q1

| # | Title | Authors | Venue / year | Link | Takeaway | Relevance to Pinch |
|---|---|---|---|---|---|---|
| S1 | Nav2 controller server: SimpleProgressChecker / PoseProgressChecker; MPPI controller | Nav2 maintainers | ros-navigation/navigation2 (5.1k ★, pushed 2026-11-19) | https://api.nav2.org/nav2-jazzy/html/classnav2__controller_1_1SimpleProgressChecker.html | Fail if not moved `required_movement_radius` (0.5 m) within `movement_time_allowance` (10 s); pose variant counts rotation | Same commanded-vs-measured idea as StuckSense; thresholds must be scaled |
| S2 | Nav2 behaviour trees: detailed walkthrough; nav-to-pose with recovery | Nav2 docs | docs.nav2.org, 2025–26 | https://docs.nav2.org/behavior_trees/overview/detailed_behavior_tree_walkthrough.html | RecoveryNode with retries (6); recovery = clear costmaps → Spin → Wait → BackUp; contextual before system-level | The escalation ladder for `Unstick` |
| S3 | Behavior Trees in Robotics and AI | Colledanchise, Ögren | CRC Press 2018 / arXiv 2017 | https://arxiv.org/abs/1709.00084 | BTs give reactive, modular fallback sequences | How to structure retries and fallbacks |
| S4 | BehaviorTree.CPP | Faconti et al. | GitHub (4.3k ★, pushed 2026-11-13) | https://github.com/BehaviorTree/BehaviorTree.CPP | The BT engine under Nav2 | Reference design only; Pinch's activities stay Python state machines |
| S5 | The Marathon 2: A Navigation System | Macenski, Martín, White, Ginés Clavero | IROS 2020 | https://arxiv.org/abs/2003.00368 | Nav2 design: BT navigator, plugins, field-tested recoveries | Production precedent for the ladder |
| S6 | Two years of visual odometry on the Mars Exploration Rovers | Maimone, Cheng, Matthies | J. Field Robotics 24(3), 2007 | https://doi.org/10.1002/rob.20184 | VO against wheel odometry detects slip; drives stop beyond a slip limit | Measured progress, not commanded, is what catches "stuck" |
| S7 | Information theoretic MPC for model-based RL (MPPI) | Williams, Wagener, Goldfain, Drews, Rehg, Boots, Theodorou | ICRA 2017 | https://doi.org/10.1109/ICRA.2017.7989202 | Sampling-based MPC escapes some local minima | Not needed on a 70×45 cm desk; noted for completeness |
| S8 | Learning quadrupedal locomotion over challenging terrain | Lee, Hwangbo, Wellhausen, Koltun, Hutter | Science Robotics 2020 | https://arxiv.org/abs/2010.11251 | Teacher–student proprioceptive policy with terrain curriculum; robust to unseen impediments | Train the walker on catch and wedge worlds with history input |
| S9 | RMA: Rapid Motor Adaptation for legged robots | Kumar, Fu, Pathak, Malik | RSS 2021 | https://arxiv.org/abs/2107.04034 | History → latent extrinsics; adapts in a fraction of a second | Same mechanism for a caught foot or a weak servo |
| S10 | An energy landscape approach to locomotor transitions in complex 3D terrain | Othayoth, Thoms, Li | PNAS 2020 | https://arxiv.org/abs/2006.12717 | Body oscillation helps cross energy barriers out of trapped modes | Add a "rock or jitter and change attitude" escape |
| S11b | Robust recovery controller for a quadrupedal robot using deep RL | Lee, Hwangbo, Hutter | arXiv 2019 | https://arxiv.org/abs/1901.07517 | Learned recovery plus behaviour selection between policies | Already followed by Pinch's get-up |
| S12 | Agile But Safe (ABS) | He, Zhang, Xiao, He, Liu, Shi | RSS 2024 | https://arxiv.org/abs/2401.17583 | Agile and recovery policies switched by a learned reach-avoid value; code LeCAR-Lab/ABS (633 ★, last push 2024-07) | Optional learned "about to fail" monitor |
| S13 | Recovery RL | Thananjeyan, Balakrishna et al. | RA-L 2021 | https://arxiv.org/abs/2010.15920 | Safety critic plus recovery policy, learned from offline violations | Same idea, general form |
| S14 | Concurrent training of a control policy and a state estimator | Ji, Mun, Kim, Hwangbo | RA-L 2022 | https://arxiv.org/abs/2202.05481 | Estimator predicts base velocity, foot height and contact from proprioception | Camera-free progress signal for StuckSense |
| S15 | PrePARE: predictive proprioception for failure event detection | Dey, Fan, Schmid, … Agha-mohammadi | IROS 2022 | https://arxiv.org/abs/2208.00322 | Anomaly detector → labelled slips → ensemble predicts slips up to 720 ms early | How to build and tune a learned failure predictor if needed |
| S16 | Robot collisions: a survey on detection, isolation and identification | Haddadin, De Luca, Albu-Schäffer | IEEE T-RO 2017 | https://doi.org/10.1109/TRO.2017.2723903 | Momentum and residual observers detect collisions from joint signals | Per-leg "caught" detector from STS3032 position and load |
| S17 | Proprioception and reaction for walking among entanglements | Yim, Ren, Ologan, Garcia Gonzalez, Johnson | IROS 2023 | https://arxiv.org/abs/2304.02129 | Sense a caught leg; free it during swing; 14/16 lab trials | Caught-leg primitive |
| S18 | Path-planning strategies for a point mobile automaton moving amidst unknown obstacles (Bug) | Lumelsky, Stepanov | Algorithmica 1987 | https://doi.org/10.1007/BF01840369 | Boundary following escapes local minima with guarantees | Re-plan fallback |
| S19 | Robots that can adapt like animals | Cully, Clune, Tarapore, Mouret | Nature 2015 | https://arxiv.org/abs/1407.3501 | A hexapod recovers from damage in about a minute by trying a few behaviours from a prior map | Escape repertoire plus memory of what worked |
| S37 | Learning humanoid standing-up control across diverse postures (HoST) | Huang, Ren et al. | RSS 2025 | https://arxiv.org/abs/2502.08378 | Smoothness and an implicit speed bound make learned stand-ups hardware-safe | Fix for decision 85's violent get-up |
| S38 | PPL: point-cloud supervised proprioceptive locomotion in crawl spaces | Ma, Xu, Qi et al. | RA-L (arXiv 2508.09950) | https://arxiv.org/abs/2508.09950 | Proprioceptive policy whose estimator predicts collision states | Collision-aware estimator, no depth needed at run time |
| S39 | Resilient legged local navigation | Jin, Zhang, Frey, Rudin, Mattamala, Cadena, Hutter | arXiv 2023 | https://arxiv.org/abs/2310.03581 | Learn reactions to body collisions with "invisible" obstacles | Supports collision-heavy training worlds |
| S33 | ViNL: visual navigation and locomotion over obstacles | Kareer, Yokoyama, Batra, Ha, Truong | ICRA 2023 | https://arxiv.org/abs/2210.14791 | Decoupled nav (velocity) and locomotion policies | Keeps Pinch's command interface; learned nav not needed |
| S34 | Learning to walk in confined spaces using 3D representation | Miki, Lee, Wellhausen, Hutter | ICRA 2024 | https://arxiv.org/abs/2403.00187 | Two-layer hierarchy with voxel input | Needs 3D sensing Pinch lacks |
| S49 | Learning robust autonomous navigation and locomotion for wheeled-legged robots | Lee, Bjelonic, Reske, Wellhausen, Miki, Hutter | Science Robotics 2024 | https://arxiv.org/abs/2405.01792 | Hierarchical RL navigation over a learned locomotion policy | Large-robot evidence for the hierarchy |

---

## Q2. How to train a robot like Pinch

### Q2.1 Locomotion

- **What the evidence says.**
  - *Massively parallel PPO with a game-inspired terrain curriculum.* Robots are promoted to harder terrain when
    they succeed and demoted when they fail. This trains a quadruped walker in minutes on one GPU [S21].
  - *Behaviour multiplicity.* Walk These Ways conditions one policy on gait frequency, phase, body height, foot
    swing height and stance. Those parameters are then tuned at run time for new tasks [S20].
  - *Specialists or a generalist.* Barkour trains specialist skills, then distils them into a transformer
    generalist. The generalist scores slightly lower but transitions more smoothly [S22].
  - *Small robots.* Open Duck Mini (Pi Zero 2W, MuJoCo Playground, BAM servo models) [S41] and ToddlerBot [S43]
    walk zero-shot with RL. MuJoCo Playground reports minutes-to-hours training on one GPU [S54].
  - *Priors.* Residual policies on a trajectory generator are a proven way to keep a readable gait [APPROACH_PLAN R9].
- **What it implies for Pinch.**
  - gait-v2 is a residual policy over 48 recorded references, and gait_tp2 showed interference (decision 96). The
    next walker should:
    1. sample **continuous** commands through the JAX gait (decision 66) instead of fixed references;
    2. add **behaviour parameters** as commands: body height, step height, gait frequency and a push-crouch flag
       [S20];
    3. use a **success-promoted terrain curriculum** [S21] over flat, rough, hurdles, ramps, cables and wedge worlds,
       with push episodes kept at a fixed share;
    4. use **mirror-symmetry augmentation**, since Pinch is left-right symmetric and data augmentation beat a
       mirror loss [S53];
    5. take an **observation history or estimator head** [S9][S14].
  - Behaviour parameters also give mood-dependent motion: "sad" means low and slow, "excited" means high and quick
    steps.
  - **Uncertain:** whether one conditioned policy beats separate walk and push policies on the tidy bench. Barkour
    suggests specialists edge out generalists on score [S22]. Body can already switch per tick (decision 84), so
    let the bench decide.

### Q2.2 Expressive and character motion

- **What the evidence says.**
  - **DeepMimic** tracks reference clips with an imitation reward plus task terms [S23].
  - **AMP** replaces hand-written style rewards with a motion discriminator [S26]. On a real quadruped, AMP
    replaced complex reward functions [S26b].
  - **ASE** learns reusable latent skills [S26c].
  - **MaskedMimic** unifies many control modes in one tracker through masked motion inpainting (NVlabs/ProtoMotions,
    2.5k ★, active) [S36].
  - **BeyondMimic** tracks highly dynamic clips on hardware, then distils them into a guided diffusion controller.
    Code: HybridRobotics/whole_body_tracking, 2.4k ★, active; mjlab ships a tracking task in this style [S35].
  - **Disney's BD-X** uses an animation engine and separate RL policies: perpetual (standing), periodic (walking)
    and episodic (per clip), tracking artist animation [S30]. **Olaf** adds actuator temperature as an
    observation, a thermal reward and a foot-impact reward (−13.5 dB) [S31]. **AMOR** makes reward weights an
    input that can be changed after training [S32].
- **What it implies for Pinch.**
  - Train **one motion-tracking policy** over all keyframed emotes plus procedural layers (breathing, gaze, mood
    style from decision 92), in the BD-X episodic and perpetual style [S30]. It replaces per-emote balance
    training, so new emotes cost an animation, not a run.
  - Reward: pose and keypoint tracking plus balance, action rate and thermal terms. STS3032 reports temperature,
    and Olaf shows the term works [S31].
  - AMP is optional, for "crab-like" walking style [S26][S26b]. It needs a reference clip set; Pinch would have
    to author those.
  - **Uncertain:** the hobby PWM arm servos give no feedback, so arm tracking on the real robot is open-loop.
    Tracking quality there depends on the step 0 actuator model.

### Q2.3 Loco-manipulation (push, pick, claws)

- **What the evidence says.**
  - **Hierarchical pushing** with a quadruped's body: a 5 Hz policy outputs (vx, vy, yaw) to a frozen locomotion
    controller [S24]. It reached 93.6 % success in simulation and moved 15–19 kg objects in the real world. The
    reward design is in the first bullet of the implications below. The authors report that pure extrinsic reward
    stalls during approach, and that proximity terms left on near the goal hurt accuracy [S24].
  - **DribbleBot** commands the *ball velocity* and rewards velocity tracking, with distance and alignment terms.
    It trained for 77 billion steps [S25].
  - **Curiosity (RND)** lets loco-manipulation learn from sparse task rewards without heavy shaping [S27b]. RSL-RL
    ships an RND extension [S51].
  - **Pedipulation** [S55], **legs as manipulators** [S56] and **LocoMan** (leg-mounted 3-DoF manipulators) [S57]
    show that limbs used for walking can also manipulate. LocoMan is the closest morphology to a crab with claws on
    front limbs, though it is model-based whole-body control, not RL.
  - **DexPBT** uses population-based training to tune hard manipulation rewards and hyperparameters [S58].
  - **Classic shaping theory.** Only *potential-based* shaping, `F = γΦ(s') − Φ(s)`, provably keeps the optimal
    policy unchanged, so a "near the object" potential pays nothing for hovering [S27]. A reverse curriculum
    starting near the goal solves long-horizon manipulation that forward exploration does not [S28]. Asymmetric
    actor–critic gives the critic privileged state and keeps the actor deployable [S29].
- **Why push2-v1 learned to hover** (decision 88). It had three of the failure signs above:
  - a dominant "near the thing" term;
  - no term for object motion toward the goal;
  - full-length episodes from the start.

  It also had a small budget: 30M steps.
- **What it implies for Pinch (push v3).**
  - Keep the 10 Hz command policy over the frozen walker. That is the same structure as [S24], and it is sound.
  - Replace the reward:
    - `−log(‖goal − object‖ + 0.05)`-style extrinsic progress [S24];
    - object velocity toward the goal [S24][S25];
    - proximity as a *potential difference* [S27], dropped once the object is near the goal [S24];
    - an arrival bonus.
  - Curriculum: goals 2–4 cm away first, widening to 8–25 cm as success rises [S28].
  - Critic sees object mass and friction [S29].
  - Budget: at least 5–10× push2-v1's steps before judging. That number is *uncertain*; [S24] and [S25] are bigger
    robots and tasks.
  - Ablate RND on a sparse arrival-only reward as a second arm [S27b].
- **Grasp.** Keep scripted approach plus a learned closing, distilled teacher→student, as in grasp-v6 and v7
  [APPROACH_PLAN §3.5]. Strong RL-over-scripted grasp evidence for cheap 1-DoF claws is still missing.

### Q2.4 Navigation and following

- **What the evidence says.** Learned navigation pays off in large, partially observed spaces with depth sensing
  [S33][S34][S49]. If navigation *is* learned, a kinematic, low-fidelity simulator for the high level transferred
  better than a physics one [S59]. High-level command policies at 5–10 Hz over a learned walker are the standard
  interface [S24][S33].
- **What it implies for Pinch.**
  - The desk is 70×45 cm, mapped by the cliff ring and camera.
  - Use geometric planning on the danger and occupancy map, and visual servoing for *follow* and *investigate*:
    detection bearing and range become walking commands. Pinch already does this with its own `_goto`,
    `guard_edge` and Investigate activities (decision 95).
  - Do not train navigation with RL. It has no measured benefit at this scale, and it costs explainability and
    replay.
  - Spend the robustness budget on the walker so commands are tracked well under clutter.

### Q2.5 Sim-to-real for cheap servos

- **What the evidence says.**
  - An actuator network trained on logged data was key to ANYmal's transfer [S40].
  - Coulomb–viscous friction misses servo behaviour. Extended friction models identified on a pendulum bench do
    better, with code in Rhoban/bam (629 ★, active) [S42]. Open Duck Mini uses BAM [S41].
  - ToddlerBot's sysID transfers across units of the same motor [S43].
  - Disney identifies PD, friction and backlash, and runs policies at 50 Hz [S30].
  - Latency and actuator modelling were decisive for Minitaur [S44].
  - Dynamics randomisation [S45] and history-based adaptation [S9] cover what remains.
  - Thermal state matters for small actuators [S31].
  - An IMU frame mismatch broke Bittle's transfer [APPROACH_PLAN R54].
- **What it implies for Pinch.**
  - Step 0 of the roadmap: log STS3032 step, chirp and load responses at the bus rate.
  - Fit BAM-style friction plus PD plus latency, and randomise ± the fitted spread.
  - Add temperature to the observation for long emote sessions [S31].
  - The GS-4308MG arm servos need a separate open-loop model: deadband, gear play (~1.5°) and speed limit.
  - Re-run every adopted policy's bench with the identified model before any hardware test.

### Q2.6 Architecture: generalist or skill library, and the pet layer

- **What the evidence says.**
  - One conditioned policy covers a family of gaits [S20].
  - Distillation into a generalist trades a little score for smoother transitions [S22].
  - Parkour systems split between end-to-end [S46][S47] and a skill library chosen by a navigation module [S48].
  - DeepMind's OP3 soccer distilled separately trained skills (get-up, scoring) into one agent
    [APPROACH_PLAN R6].
  - Character robots keep separate motor policies under an animation engine [S30][S31].
  - Pets (aibo, Vector) use authored behaviour systems with an emotion model [APPROACH_PLAN R30, R31, R33].
- **What it implies for Pinch.**
  - Keep a **library of about 4–5 motor policies**: locomotion (with push and climb modes as commands), get-up,
    motion tracker, grasp, and possibly push steering.
  - Put them under the **authored utility selector, activities, reflexes and Simplex supervisor**. The research plan
    already chose this, and nothing found here contradicts it.
  - Distil into one network only if per-tick switching between policies shows glitches on the benches [S22].

### Q2.7 Tool stack

| Option | Evidence | For Pinch |
|---|---|---|
| Brax PPO on MJX (current) | google/brax 3.3k ★, last push 2026-10-20; MuJoCo Playground 2.3k ★, active [S54]; measured 14–27k steps/s (decision 97) | Keep for existing envs (gait, righting, grasp) until they need retraining. Brax's last push is older than the rest |
| **mjlab** (MuJoCo Warp + RSL-RL) | mujocolab/mjlab 3.4k ★, pushed today; tech report Jan 2026 [S50]; ships velocity, tracking and manipulation tasks. RSL-RL 3.2k ★, active: PPO, distillation, SAC, RND, symmetry [S51]. MuJoCo Warp 1.6k ★ | **New envs here**: motion tracking first, then locomotion v3 and push v3. Same physics family as the twin. Benchmark throughput first (*unmeasured*) |
| Isaac Lab | isaac-sim/IsaacLab 8.7k ★, active; paper 2025 [S52] | Fallback. PhysX would diverge from the MuJoCo twin; Newton integration is on its roadmap [S52] |
| legged_gym | 3.2k ★, last push 2025-05 | Superseded by Isaac Lab and mjlab; reference only [S21] |

### Q2.8 Sources for Q2

| # | Title | Authors | Venue / year | Link | Takeaway | Relevance to Pinch |
|---|---|---|---|---|---|---|
| S20 | Walk These Ways | Margolis, Agrawal | CoRL 2022 | https://arxiv.org/abs/2212.03238 | One policy, many gaits via behaviour parameters (Improbable-AI/walk-these-ways, 1.5k ★, last push 2024) | Regime and mood as commands; fixes decision 96's interference |
| S21 | Learning to walk in minutes using massively parallel deep RL | Rudin, Hoeller, Reist, Hutter | CoRL 2021 | https://arxiv.org/abs/2109.11978 | Game-inspired terrain curriculum; minutes on one GPU (legged_gym 3.2k ★) | Success-promoted terrain curriculum |
| S22 | Barkour | Caluwaerts, Iscen, Tan et al. | arXiv 2023 / Google | https://arxiv.org/abs/2305.14654 | Specialists plus a distilled Locomotion-Transformer; generalist slightly lower score, smoother transitions | Condition first, distil later |
| S23 | DeepMimic | Peng, Abbeel, Levine, van de Panne | SIGGRAPH 2018 | https://arxiv.org/abs/1804.02717 | Imitation reward on reference clips (xbpeng/DeepMimic 3.1k ★) | Basis of the motion tracker |
| S24 | Learning whole-body manipulation for quadrupedal robot | Jeon, Jung, Choi, Kim, Hwangbo | arXiv 2023 | https://arxiv.org/abs/2308.16820 | 5 Hz velocity-command policy over a frozen walker; log-distance extrinsic + object-velocity intrinsic reward; 93.6 % in sim | Template for push v3 |
| S25 | DribbleBot | Ji, Margolis, Agrawal | ICRA 2023 | https://arxiv.org/abs/2304.01159 | Commanded ball velocity; velocity-matching rewards; 77B steps | Object-velocity command alternative; budget reality |
| S26 | AMP: adversarial motion priors | Peng, Ma, Abbeel, Levine, Kanazawa | SIGGRAPH 2021 | https://arxiv.org/abs/2104.02180 | Discriminator style reward | Optional crab style |
| S26b | AMP make good substitutes for complex reward functions | Escontrela et al. | IROS 2022 | https://arxiv.org/abs/2203.15103 | AMP on a real quadruped | Same |
| S26c | ASE: adversarial skill embeddings | Peng, Guo, Halper, Levine, Fidler | SIGGRAPH 2022 | https://arxiv.org/abs/2205.01906 | Reusable latent skills (nv-tlabs/ASE 1.1k ★) | Later, if many skills share one controller |
| S27 | Policy invariance under reward transformations | Ng, Harada, Russell | ICML 1999 | https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf | Only potential-based shaping preserves the optimal policy | Stops "hover near the thing" |
| S27b | Curiosity-driven learning of joint locomotion and manipulation tasks | Schwarke, Klemm, van der Boon, Bjelonic, Hutter | CoRL 2023 | https://proceedings.mlr.press/v229/schwarke23a.html | RND with sparse task reward instead of shaping | Second arm for push v3 |
| S28 | Reverse curriculum generation for RL | Florensa, Held, Wulfmeier, Zhang, Abbeel | CoRL 2017 | https://arxiv.org/abs/1707.05300 | Start near the goal, expand outward | Short pushes first |
| S29 | Asymmetric actor critic for image-based robot learning | Pinto, Andrychowicz, Welinder, Zaremba, Abbeel | RSS 2018 | https://arxiv.org/abs/1710.06542 | Privileged critic, deployable actor | Critic sees object mass and friction |
| S30 | Design and control of a bipedal robotic character | Grandia, Knoop, Hopkins, … Bächer | RSS 2024 | https://arxiv.org/abs/2501.05204 | Animation engine + perpetual, periodic and episodic RL policies; actuator identification | Pinch's motor architecture and the tracker |
| S31 | Olaf: bringing an animated character to life | Disney Research | arXiv 2512.16705, 2025 | https://arxiv.org/abs/2512.16705 | Temperature in the observation, thermal and impact rewards (−13.5 dB) | STS3032 temperature; quiet feet on a desk |
| S32 | AMOR | Alegre, Serifi, Grandia, Müller, Knoop, Bächer | SIGGRAPH 2025 | https://arxiv.org/abs/2505.23708 | Reward weights as inputs, changeable after training | Tune smoothness against expressiveness without retraining |
| S35 | BeyondMimic | Liao, Truong, Huang, Tevet, Liu, Sreenath | arXiv 2025 | https://arxiv.org/abs/2508.08241 | Dynamic motion tracking on hardware; diffusion distillation (2.4k ★) | Tracker recipe; mjlab tracking task |
| S36 | MaskedMimic | Tessler, Guo, Nabati, Chechik, Peng | SIGGRAPH Asia 2024 | https://arxiv.org/abs/2409.14393 | One tracker for many control modes (ProtoMotions 2.5k ★) | Future unification of emotes and gaze |
| S40 | Learning agile and dynamic motor skills for legged robots | Hwangbo, Lee, Dosovitskiy, … Hutter | Science Robotics 2019 | https://arxiv.org/abs/1901.08652 | Actuator network from logs | Step 0 option |
| S41 | Open Duck Mini | Pirrone et al. | GitHub (4.8k ★, last push 2026-01) | https://github.com/apirrone/Open_Duck_Mini | Pi Zero 2W, MuJoCo Playground, BAM servo models | Same compute class and approach |
| S42 | Extended friction models for the physics simulation of servo actuators (BAM) | Duclusaud, Passault, Padois, Ly | arXiv 2410.08650 (v4 2025) | https://arxiv.org/abs/2410.08650 | Pendulum-bench identification beats Coulomb–viscous (Rhoban/bam 629 ★) | Step 0 tool |
| S43 | ToddlerBot | Shi, Wang, Song, Liu | CoRL 2025 | https://arxiv.org/abs/2502.00893 | Transferable motor sysID; zero-shot RL walking (hshi74/toddlerbot 807 ★) | Sysid once per servo model |
| S44 | Sim-to-Real: learning agile locomotion for quadruped robots | Tan, Zhang, Coumans, … Vanhoucke | RSS 2018 | https://arxiv.org/abs/1804.10332 | Actuator model and latency were key | Randomise bus latency |
| S45 | Sim-to-real transfer of robotic control with dynamics randomization | Peng, Andrychowicz, Zaremba, Abbeel | ICRA 2018 | https://arxiv.org/abs/1710.06537 | Dynamics randomisation | Around the fitted model |
| S46 | Robot Parkour Learning | Zhuang et al. | CoRL 2023 | https://arxiv.org/abs/2309.05665 | Skill policies distilled into one vision policy (1.1k ★) | Evidence on generalist vs library |
| S47 | Extreme Parkour with legged robots | Cheng, Shi, Agarwal, Pathak | ICRA 2024 | https://arxiv.org/abs/2309.14341 | One end-to-end policy from depth (1.2k ★, inactive since 2023) | Same |
| S48 | ANYmal parkour | Hoeller, Rudin, Sako, Hutter | Science Robotics 2024 | https://arxiv.org/abs/2306.14874 | Perception + skill library + navigation selector | Skill library under a selector |
| S50 | mjlab: a lightweight framework for GPU-accelerated robot learning | Zakka et al. | arXiv 2601.22074, 2026 | https://arxiv.org/abs/2601.22074 | Isaac Lab-style API on MuJoCo Warp, RSL-RL training | Recommended stack for new envs |
| S51 | RSL-RL | Robotic Systems Lab, ETH | GitHub (3.2k ★, pushed 2026-11-20) | https://github.com/leggedrobotics/rsl_rl | PPO, distillation, SAC, RND and symmetry extensions | Teacher–student, curiosity, mirror augmentation |
| S52 | Isaac Lab | Mittal et al. (NVIDIA) | arXiv 2511.04831, 2025 | https://arxiv.org/abs/2511.04831 | GPU sim framework; Newton integration on the roadmap (8.7k ★) | Fallback stack |
| S53 | Symmetry considerations for learning task symmetric robot policies | Mittal, Rudin, Klemm, Allshire, Hutter | ICRA 2024 | https://arxiv.org/abs/2403.04359 | Data augmentation beats a mirror loss | Pinch is left-right symmetric |
| S54 | MuJoCo Playground | Zakka et al. | RSS 2025 | https://arxiv.org/abs/2502.08844 | MJX envs with sim-to-real examples, minutes to hours on one GPU | Current stack's basis |
| S55 | Pedipulate | Arm, Mittal, Kolvenbach, Hutter | ICRA 2024 | https://arxiv.org/abs/2402.10837 | RL foot-position tracking for manipulation, robust to interaction forces | Claw and leg poke or push play |
| S56 | Legs as manipulator | Cheng, Kumar, Pathak | ICRA 2023 | https://arxiv.org/abs/2303.11345 | Separate walk and manipulate skills plus a planner | Same |
| S57 | LocoMan | Lin et al. | arXiv 2403.18197, 2024 | https://arxiv.org/abs/2403.18197 | Leg-mounted 3-DoF manipulators; whole-body control (105 ★) | Closest morphology to claws on front limbs |
| S58 | DexPBT | Petrenko, Allshire, State, Handa, Makoviychuk | RSS 2023 | https://arxiv.org/abs/2305.12127 | Population-based training tunes rewards and hyperparameters for hard manipulation | If push or grasp still stalls |
| S59 | Rethinking sim2real: lower fidelity simulation leads to higher sim2real transfer in navigation | Truong, Rudolph, Yokoyama, Chernova, Batra, Rai | CoRL 2022 | https://arxiv.org/abs/2207.10821 | A kinematic sim for high-level navigation transfers better | If navigation is ever learned |

### What is still uncertain

- **Scale.** Most evidence comes from 10–50 kg quadrupeds and humanoids. The small-robot evidence is Open Duck
  Mini, ToddlerBot, Bittle and the OP3 [S41][S43][APPROACH_PLAN R6, R54]. None is a 12+6-DoF crab with
  sideways gaits.
- **Push budget.** How many steps push v3 needs is a guess. The comparable systems are much bigger [S24][S25].
- **Throughput and maturity.** mjlab's throughput on Pinch and its API stability are unmeasured. It is a young
  framework (report Jan 2026) [S50].
- **Generalist or specialists.** A conditioned single walker may still lose to separate specialists on the tidy
  bench [S22]. The bench decides, not this note.
