# How each of Pinch's behaviours should be built

*Research note, 2026-11-19. It covers which technique fits each behaviour area, what sensing each one needs and
whether that runs on the Radxa ZERO 3W, how to test each one automatically in the MuJoCo twin, how much work it is,
and which parts are worth training with reinforcement learning (RL) or imitation learning (IL). Sources are listed
at the end as [R1], [R2] and so on. Where the evidence is thin, the text says so. Effort sizes are **S** (a few days),
**M** (1–3 weeks) and **L** (a month or more). Training cost classes are **CPU-min** (CPU minutes), **GPU-h**
(hours on a single GPU) and **GPU-many** (days of GPU time).*

## The short answer

1. **Learn the body. Write the mind.** Every published case where learning clearly beat hand design on a small
   robot is a *motor* problem: walking over rough ground, getting up after a fall, recovering from pushes, and
   tracking artist animation while staying balanced [R1][R4][R6][R12][R13]. No published system learns
   *which behaviour to show* end-to-end and does it better than authored logic. Disney's newest characters
   still leave high-level choices to a human puppeteer or to authored logic [R12][R14]. Anki and Sony built
   their pets on authored behaviour systems with an emotion model [R30][R31][R33]. RL for social behaviour
   needed about 14 days of real interaction to learn a single handshake decision [R37].
2. **Use online statistics, not RL, for things that depend on the owner.** That covers which people Pinch
   knows, which toys and games a person likes, and what is still interesting. Bandits, Rescorla–Wagner
   updates, habituation and learning-progress estimates learn in seconds, can be explained, and replay
   exactly from a seed [R25][R26][R39].
3. **Keep the layers already in the code and make them stricter:**
   - an independent safety supervisor, Simplex-style [R44][R45];
   - reflexes (startle, cliff stop);
   - utility selection over activities, with each activity a small state machine inside;
   - an animation engine that layers procedural idle motion, keyframed clips, gaze offsets and affect "style"
     (as in Disney's BD-X [R12]);
   - learned motor policies that turn those commands into servo targets.
4. **"Believable" can be partly measured in simulation**, using repetition and entropy, gaze-target agreement
   and smoothness/jerk. Humans still have to judge it in the end, with the Godspeed or RoSAS questionnaires on
   short videos [R47][R48]. Evidence that the automatic proxies predict human ratings is **thin**. Treat them as
   regression guards, not as proof.

---

## 1. Summary table

| Area | Recommended approach | Perception needed (fits RK3566?) | How to test automatically in sim | Effort | Key citation |
|---|---|---|---|---|---|
| Idle aliveness & anti-repetition | Procedural layer: breathing, Perlin-noise sway, blinks and fidgets, with randomised timing and an anti-repeat memory. Authored, not learned | None beyond its own state (yes) | n-gram entropy of idle events, rate of the same clip within 60 s, coefficient of variation of gaps, amplitude caps | S | Perlin & Goldberg [R27]; Asselborn et al. [R28]; Terzioglu et al. [R29] |
| Attention | Saliency map (motion, faces, sound direction, touch) plus habituation and inhibition of return; winner-take-all with hysteresis. Authored | Frame-difference motion on CPU, YuNet faces (NPU or CPU), two-mic GCC-PHAT, MPR121 (all yes) | Gaze-target agreement against scripted stimuli, orienting latency, return-to-target interval, habituation curve | M | Breazeal & Scassellati [R19]; Ruesch et al. [R20] |
| Curiosity & exploration | Novelty (habituation filter) plus learning-progress interest, choosing among *authored* explore and inspect activities. Online statistics | Novelty score from the camera embedding or detector, odometry, cliff sensors (yes) | Desk coverage over time, approach latency for a new versus a familiar object, interest decays and then recovers | M | Oudeyer et al. [R25]; Marsland et al. [R26] |
| Locomotion (all directions, curves, scuttle/crawl, transitions, terrain, pushes) | **RL**: one command-conditioned policy (velocity, yaw rate, gait frequency, body height, step height), keeping the scripted gait as a reference or prior; actuator model; domain randomisation | IMU, joint positions, cliff sensors (yes; an MLP policy takes well under 1 ms on an A55, *estimate*) | Tracking error per direction and curvature, terrain-curriculum success, push-recovery rate against impulse, slip, servo-load proxy | M–L | Margolis & Agrawal [R7]; Iscen et al. [R9]; Rudin et al. [R3] |
| Claws: expression, inspection, push, grasp, play | Expression: keyframed plus procedural. Push/nudge: scripted primitive with visual servoing. Grasp: scripted approach plus **RL residual or teacher–student** for closing, *after the claw redesign* | Object detection or blob on the NPU, claw contact via the touch pad or servo current (partly) | Grasp success over randomised objects and poses, force caps, push accuracy, drop rate | M (push), L (grasp) | Johannink et al. [R10]; Cheng et al. [R17]; Zhao et al. (ACT) [R18] |
| Face & gaze | Procedural eye model: saccades, smooth pursuit, blinks, pupil and lid shape from affect; eyes lead head and body. Authored | Attention target (from above) | Gaze lands on the target within N ms, eye-head order, blink-interval distribution, no frozen gaze longer than X s | S–M | Admoni & Scassellati [R21]; Ruhland et al. [R22] |
| Touch, bonding, familiarity | Small **learned touch-gesture classifier** (pat, stroke, poke, hold) on MPR121 time series; authored responses; Rescorla–Wagner familiarity per person (exists) | MPR121 at 50–100 Hz, face ID (yes) | Classifier accuracy on recorded gestures, response latency, familiarity curves for simulated owners | S–M | Jung et al. (CoST) [R23]; Leite et al. [R40] |
| Emotion model | Low-dimensional affect state (valence, arousal, stance, Kismet/PAD-like) driven by authored appraisal rules, decaying to a mood baseline; it modulates utility weights and animation style. Authored | Events from the other modules (yes) | Scenario tests (stimulus gives the expected sign of change), bounded and decaying trajectories, no oscillation, observers can label the expressed emotion | M | Breazeal [R32]; Becker-Asano & Wachsmuth [R34] |
| Play (self, with people, chase, hide-and-seek) | Authored game state machines (rules, turn-taking) on top of learned locomotion; game choice by preference bandit; hiding spots from learned object relations, as Trafton et al. did | Person detection and tracking, sound direction, touch (yes for chase; hide-and-seek is limited by a 10 cm viewpoint) | Rule-conformance tests, chase capture rate and latency, turn-taking timing, games chosen in proportion to simulated preference | M–L | Trafton et al. [R41]; Gordon et al. [R39] |
| Startle & defence | Reflex layer: fast fixed responses (flinch, claws up, back off) with habituation and sensitisation; bypasses selection and is overridden only by safety. Authored | IMU jolt, loud-onset detector, looming (fast optical expansion), sudden touch (yes) | Stimulus-to-motion latency, habituation over repeats, recovery after rest, never crosses a cliff | S | Arkin et al. [R31]; Marsland et al. [R26] |
| Sounds | Procedural non-verbal utterances (pitch, contour, tempo from affect) plus authored sound events tied to animation keys; rate limiter. Authored | None (output only) | Sync with animation events, rate limits, no overlap, affect-to-contour mapping | S | Read & Belpaeme [R42]; Anki sound design interview [R33] |
| Memory & preferences | Event memory plus per-person and per-object statistics; Thompson-sampling or UCB bandits for "what does this person like"; saved to disk. Online statistics | Identities from the other modules (yes) | Regret against simulated owners with hidden preferences, persistence across restart, identical replay from a seed | S–M | Gordon et al. [R39]; Leite et al. [R40] |
| Daily routines | Clock-driven homeostatic drives (energy, sleepiness) with a circadian gate; authored wake, settle and sleep sequences | Clock, battery, ambient light (camera mean) (yes) | Accelerated 24 h sim runs: activity-by-hour histogram, never sleeps during interaction, wakes on cue | S | Arkin et al. (aibo) [R31]; Fujita [R30] |
| Failure recovery (stuck, blocked, flipped) | Detectors: commanded versus measured motion residual, IMU orientation. Responses: authored escape repertoire with backtracking; **RL self-righting** policy | IMU, joint error and current, odometry (yes) | Fault-injection matrix (flipped at every orientation, wedged leg, blocked path, dead servo): success rate and time | M | Lee, Hwangbo & Hutter [R5]; Haarnoja et al. [R6] |
| Safety supervisor | Separate process with a simple verified rule set (joint limits, torque and temperature, cliff, tilt, watchdog, pinch-force cap) that can override any command (Simplex). **Never learned** | Raw sensors on their own path (yes) | Property and fuzz tests: random brain output can never break the limits; watchdog fires on a stalled brain; timing tests | M | Sha [R44]; Brunke et al. [R45] |
| Sim/test infrastructure | Deterministic clock and seeded replay (exists); scenario DSL; fault injection; simulated-days soak runs; behaviour-quality observer that computes all the metrics above online; motion-realism scorer; human video studies | n/a | The observer itself gets golden tests; CI runs scenario and soak suites | M–L | Fremont et al. (Scenic) [R49]; Lee et al. (AST) [R50]; Bartneck et al. [R47] |

---

## 2. The overall architecture

```
 sensors ─┬─────────────────────────────────────────────► SAFETY SUPERVISOR (separate process, simple rules,
          │                                                 can clamp / stop anything)      [R44][R45]
          ├─► perception: motion, faces, sound DOA, touch class, novelty
          │        │
          │        ▼
          │   attention (saliency + habituation + inhibition of return)
          │        │
          ├─► REFLEXES (startle, cliff stop, flinch) ───────────────┐  pre-empts the selection layer
          │        │                                                │
          │   affect state ◄── appraisal rules ◄── events           │
          │        │  (biases utilities and motion style)           │
          │        ▼                                                │
          │   UTILITY SELECTOR over ~20+ activities                 │
          │   (each activity = small hierarchical state machine)    │
          │        │ commands: walk(v, ω, gait), look_at, clip, claw, sound, face
          │        ▼                                                ▼
          │   ANIMATION ENGINE: layers = procedural idle + clips + gaze offsets + affect style  [R12]
          │        │ reference poses and commands
          │        ▼
          └─► LEARNED MOTOR POLICIES (locomotion, motion tracking, self-righting, grasp residual)
                   │
                   ▼  servo targets ──► supervisor clamp ──► servos
```

**Why this split.** It follows the structure of the best-documented expressive robot, Disney's BD-X character
[R12]. In that robot an animation engine blends a looping background animation, triggered clips and joystick
offsets into commands. Separate RL policies then execute them: one "perpetual" policy for standing, one
"periodic" policy for walking, and one "episodic" policy per triggered clip. Show functions (eyes, lamps,
antennas) are driven directly. The paper states that high-level decisions are left to a puppeteer. Olaf (2025)
keeps this pattern and adds thermal-aware and impact-reducing rewards [R14]. Pinch replaces the puppeteer with
the utility selector, so the selector is the part that has to be authored carefully.

**Utility AI versus behaviour trees versus state machines.**
- Keep **utility selection at the top**. Pinch already has it. It handles "many competing wants" smoothly, and
  every decision can be explained as a table of scores [R24].
- Use **small hierarchical state machines inside each activity**. Each one has a handful of states, a timeout
  and an exit condition. They are easy to unit-test and replay.
- Behaviour trees [R43] are a reasonable alternative for the inside of activities if they grow large. Their
  advantage is reactive interruption and reuse of modules. Pinch does not need to switch: what matters more
  is that every node logs why it ran, which the decision log already does.
- Keep reflexes **outside** both layers so they cannot be delayed by a slow decision. That is the layering
  Arkin et al. used for aibo [R31].

---

## 3. Details per area

### 3.1 Idle "aliveness" and anti-repetition

- **Approach.** Use a procedural base layer that is always running. It has three parts:
  - breathing (a slow body-height and claw oscillation);
  - low-amplitude Perlin-noise sway on the shoulders and eyes [R27];
  - stochastic micro-events (blink, claw click, weight shift, glance), each with a *refractory period* and a
    weighted random choice that down-weights recent events.
- **What the evidence says.** Idle movement raises perceived lifelikeness and friendliness compared with a static
  robot [R28][R29]. *High-intensity* idle movement disrupts people, and a Jibo home deployment found constant
  motion intrusive [R28][R46]. So amplitude and frequency must fall when someone is working nearby or when the
  affect state is calm.
- **Learned?** No. There is nothing to optimise that a designer cannot tune. RL would only add risk.
- **Perception.** None beyond its own state. To lower the intensity, it uses attention's "person present /
  engaged" signal.
- **Tests.**
  - Shannon entropy of idle-event 2-grams and 3-grams over a 30 min sim run, with a floor set by the designer.
  - Maximum count of the same micro-event in any 60 s window.
  - Coefficient of variation of the gaps between events. It must not be near 0, because evenly spaced events
    read as mechanical.
  - Peak joint velocity and amplitude caps.
  - These thresholds are the project's own choices. No paper validates specific numbers (evidence thin).
- **Effort.** S. **Depends on** the animation engine's layering.

### 3.2 Attention (select, track, switch, return)

- **Approach.** Use the classic social-robot attention system. Several feature maps (motion, face, sound
  direction, touch location, "my name"/voice onset) are combined with weights that the current activity and
  affect set. Two more pieces sit on top:
  - **habituation**, which suppresses a stimulus the longer it is attended [R19];
  - **inhibition of return**, which keeps a short memory of locations already looked at so attention moves on
    and comes back later [R20].
  Kismet's design adds top-down weighting: the active behaviour raises the weight of the features it cares
  about [R19]. Add **hysteresis** (a minimum dwell time and a switch margin) so the gaze does not jitter
  between two targets. Make the inhibition **lower for liked people**, so Pinch returns to its owner sooner;
  this was proposed for social robots in the iCub/IBPRIA line of work [R20].
- **Learned?** The detectors are learned (faces). The selection logic is authored. A learned saliency model is
  not needed at a 10 cm desk scale.
- **Perception on RK3566.**
  - Frame differencing or sparse optical flow at 160×120 on the CPU is cheap.
  - YuNet face detection (already integrated) runs on the CPU or NPU.
  - Two-microphone GCC-PHAT gives an azimuth but cannot tell front from back. The standard fixes are a turn
    of the head or body and a second listen, or a third microphone [R51].
  - With the microphones roughly 6–8 cm apart (assumed), the maximum delay is only about 0.2 ms, so direction
    is coarse (tens of degrees off-axis). That is good enough to orient, then let vision take over.
- **Tests.** Script stimuli in sim, such as a face appearing at bearing θ, a sound at φ or a moving object. Then
  measure:
  - (a) gaze-target agreement: the fraction of time the gaze ray is within N° of the stimulus the rules say
    should win;
  - (b) orienting latency;
  - (c) the interval before returning to a previously attended target;
  - (d) a habituation curve: the response falls with repeats and recovers after a rest.
- **Effort.** M. **Depends on** perception feeds and the face/gaze module.

### 3.3 Curiosity and exploration

- **Approach.** Two signals feed the utility of authored "explore", "inspect object", "poke" and "follow edge"
  activities:
  - a **novelty** signal: a habituation-based novelty filter, i.e. how far the current input is from things
    already seen [R26];
  - a **learning-progress** signal: how fast Pinch's predictions about an object or activity are improving.
    Seek activities where progress is high, and drop both the boring and the unlearnable ones [R25].
  The play model already uses learning progress. The novelty filter can use the face/object embeddings already
  computed.
- **Why not RL curiosity (ICM and the like [R52])?** Those methods are exploration bonuses that need millions of
  steps in a single environment. They teach a *policy* to explore. They do not make a creature *look* curious,
  and that look comes from authored approach-sniff-retreat sequences.
- **Perception.** An embedding from the existing face or object models, or a tiny CNN on the NPU, gives the
  novelty score. Cliff sensors and odometry give coverage.
- **Tests.**
  - Desk coverage (percentage of the free area visited) over 30 simulated minutes.
  - A novel object added at time t: approach within T s, inspect, then lose interest (inspection rate decays).
  - Familiar objects are not re-inspected within the inhibition window.
- **Effort.** M. **Depends on** attention, locomotion and memory.

### 3.4 Locomotion: all directions, curves, scuttle/crawl, transitions, terrain, disturbances

- **Approach.** Use **RL**. The evidence is strongest here.
  - Train a single policy conditioned on a command vector: forward speed, lateral speed (crabs walk sideways),
    yaw rate, gait frequency, body height and swing height. Walk These Ways showed that one policy with
    "behaviour parameters" yields crouching, fast trotting, bracing against shoves and so on, all tunable
    after training [R7].
  - "Quick scuttle" and "slow cautious crawl" then become *commands* from the selector, not separate
    controllers.
  - Gait transitions come out of that same policy. Learned gait transitions were shown to beat fixed gaits on
    energy and speed [R8].
  - Keep the scripted trot as a **reference or prior**, as in Policies Modulating Trajectory Generators [R9] and
    residual RL [R10][R11]. Pinch's residual policy already followed that recipe (69%→86% speed tracking).
    The next step widens its command space instead of adding more scripted gaits.
  - For a crab "style", add an Adversarial Motion Priors (AMP) style reward from a few keyframed crab-walk clips
    [R15][R16].
- **Evidence that RL beats hand design (legged, small and large):**
  - ANYmal: a learned controller with an actuator network beat the model-based controller on speed and
    precision, and added fall recovery [R1].
  - ANYmal: rough-terrain RL policies outperformed the hand-designed controller in the wild [R2].
  - 4,000 parallel envs train a walker in about 20 minutes on one GPU [R3].
  - RMA adapts online to payloads and terrain [R4].
  - Small servo robots:
    - Robotis OP3 (Dynamixel servos, about 0.5 m): learned soccer skills walked 181% faster and turned 302%
      faster than the scripted baseline, and got up in 63% less time [R6].
    - ToddlerBot (Dynamixel, 0.56 m) and Open Duck Mini (Feetech STS3215 hobby servos, Raspberry Pi Zero 2W) walk
      with zero-shot sim-to-real RL policies [R35][R36].
    - Gait modulation trained with augmented random search (ARS) on hobby-servo quadrupeds [R53] and work on the
      Petoi Bittle [R54] are the closest in size.
  - Boston Dynamics added RL into Spot's locomotion stack, combined with MPC, in its 4.0 release [R56]. RAI
    Institute ran an end-to-end RL policy on Spot at higher speed than the stock controller [R57].
- **Perception.** Proprioception, IMU and cliff sensors. No vision is needed for "blind" rough-terrain walking
  [R2]. The policy is a small MLP, so CPU inference is negligible (estimate, not measured on this board).
- **Tests.**
  - Tracking error per direction (0–360°) and per curvature.
  - Success on a terrain curriculum (hurdle, ramp and rough worlds already exist).
  - Push-recovery success against impulse size and direction.
  - Foot slip.
  - Torque and thermal proxy: the summed squared torque over the run, since hobby servos overheat. Olaf added
    a thermal reward for exactly this reason [R14].
  - Transition jerk and SPARC [R55] when the commands change.
- **Effort.** M for the policy, L to polish it on hardware. **Depends on** the actuator model (§5, step 1).

### 3.5 Claws: expression, inspection, pushing, grasping, play

- **Approach, split by sub-skill:**
  - *Expression* (wave, click, raise in threat or joy): keyframed clips run through the motion-tracking
    policy [R12]. Authored.
  - *Gentle inspection and pushing*: scripted primitives (approach, align, slow contact, nudge) closed around
    visual servoing on an object blob or box, with force or current limits. This is reliable and explainable.
    "Legs as manipulator" and "Pedipulate" show that RL helps when *legs* do the manipulation [R17][R58], but
    pushing a desk object with a dedicated claw does not need it.
  - *Grasping*: the current PPO grasp is unreliable, and the project notes say the claw itself needs a
    redesign. Fix the **hardware first** (wider jaws, compliant pads), then use a **scripted approach plus a
    learned closing and residual policy**. Train it teacher–student: a teacher policy with privileged
    object-pose information, distilled into a student that uses only onboard observations [R2][R10].
    Imitation from teleop demos (ACT, diffusion policy [R18][R59]) works well on low-cost arms but needs a
    teleop rig and 50+ demos per task; it is optional. Large-scale RL grasping (QT-Opt) needed hundreds of
    thousands of real grasps [R60]. That rules out end-to-end vision-based grasp RL here.
  - *Play with objects*: authored games (bat a ball, push a cap off the desk edge and stop at the cliff)
    built from those primitives.
- **Perception.** An object detector for a few desk toys. YOLOv5n-class int8 models run at roughly 10–20 FPS on
  the RK3566 NPU in community benchmarks [R61]; these are indicative figures, so measure on the board. Contact
  comes from claw servo current or a pad.
- **Tests.** Grasp, push and drop rates over randomised object size, mass, friction and pose; peak contact force
  below the supervisor cap; push-to-target error.
- **Effort.** M for push and inspection; L for grasp, which is blocked on the claw redesign.

### 3.6 Face and gaze expression

- **Approach.** Use a procedural eye model, the same idea as Cozmo/Vector's procedural face.
  - Saccades to the attention target, smooth pursuit for moving targets, and micro-saccades while fixating.
  - Blinks timed stochastically and also triggered by gaze shifts.
  - Lid and pupil shapes interpolated from the affect state.
  - Keyframed "emotes" for big moments.
- **The rules that matter from the reviews** [R21][R22]:
  - the eyes lead and the head or body follows;
  - mutual gaze holds for a limited time before a glance away;
  - gaze aversion signals thinking or shyness;
  - blink rate rises with arousal.
- **Principles of animation.** Disney-style principles (anticipation, follow-through, slow-in/slow-out, and
  secondary action) make robot state more readable [R38][R62][R63]. Build them into the clip library and the
  procedural layer, for example ease curves on every gaze shift, and an anticipation squint before a startle
  turn.
- **Learned?** No. A recent study shows expressive movement improves engagement even on a non-humanoid lamp
  [R64], but those movements were *designed*, not learned.
- **Tests.**
  - The gaze reaches the target within N ms of an attention switch.
  - Eye-before-head ordering in the logs.
  - The blink-interval distribution is not periodic.
  - No staring longer than X s unless the activity is "watch".
  - The logged eye state is consistent with the affect state.
- **Effort.** S–M. **Depends on** attention and affect.

### 3.7 Touch, affection, bonding, familiarity

- **Approach.**
  - **Recognising** the touch gesture is a genuine learning problem. A small classifier (random forest or a tiny
    1-D CNN) on MPR121 channel time series separates poke, pat, stroke, hold and rough handling.
  - Reported accuracies on the 14-class social-touch corpus are only about 55–61% [R23]. Use **4–5 coarse
    classes**, which this project's sensor can realistically separate, and record its own data with the real
    pads. The sim cannot produce realistic capacitive data, so this needs hardware.
  - **Responses** are authored (purr-like sound, lean in, eyes close).
  - **Bonding** keeps the Rescorla–Wagner-style familiarity that already exists, with valence from touch type
    and interaction quality. Long-term HRI reviews say what keeps people engaged is *personalised memory and
    changing responses*, while novelty wears off within days or weeks [R40].
- **Perception.** MPR121 at 50–100 Hz plus face identity. Feasible on the CPU.
- **Tests.** Classifier confusion matrix on recorded data (hardware). In sim, simulated owners with fixed touch
  habits check familiarity curves, the response chosen per touch class, and that rough handling lowers valence
  and raises avoidance.
- **Effort.** S for the logic, M including data collection. **Depends on** the hardware pads.

### 3.8 Internal emotional state that shapes the whole body

- **Approach.** Use a small continuous affect space, as in Kismet's arousal, valence and stance space [R32] or
  PAD [R65], driven by **appraisal rules**: "unexpected + loud → arousal up, valence down", "familiar person +
  stroke → valence up". Keep a slower **mood** that the fast emotion relaxes back toward. That is the WASABI
  design, with primary and secondary emotions [R34]. Affect then reaches the body through three channels:
  1. it **re-weights activity utilities**;
  2. it sets **style parameters** on all motion: speed, amplitude, body height, gait frequency, claw openness;
  3. it shapes the eye and sound parameters.
  Channel 2 is how the emotion becomes a whole-body state rather than a face. Command-conditioned locomotion
  (§3.4) makes it cheap: "nervous" means a low body, high step frequency and short steps.
- **Industry practice.** Sony's aibo used ethological drives plus an emotion model to select behaviour
  [R30][R31]. Anki's Cozmo had a "mood manager" that biases behaviour choice, as Anki's sound designer
  describes [R33]. Community documentation of Vector shows an emotion model alongside a behaviour tree [R66].
- **Learned?** No. The evidence for RL-tuned emotion models is lab-scale, and they become hard to explain.
- **Tests.** Scenario tables (stimulus, then the expected sign of change on each axis); bounded values;
  exponential return to baseline; no limit cycles under constant input; the style parameters reach the motion
  commands; observers in a human study can label the intended emotion from video.
- **Effort.** M. Mostly design and tuning; the affect/needs model already exists.

### 3.9 Play: self-directed and with people (chase, hide-and-seek)

- **Approach.** Each game is an **authored state machine** with explicit rules, turn-taking and an end
  condition, built on learned locomotion and attention.
  - **Chase**, in both directions, is visual servoing on the tracked person or hand plus scuttle commands.
  - **Hide-and-seek** at desk scale: Trafton et al. made a robot play a credible game by learning *object
    relations* such as "behind" and "under" for hiding places, using a cognitive model rather than RL [R41].
    Multi-agent RL hide-and-seek produced interesting strategies only after hundreds of millions of episodes
    in simulation [R67], which does not apply here.
  - **Which game, when, and with whom**: a per-person bandit over the games, with reward from engagement
    signals (time the person stays, touches, laughter-like sound onset). That is the personalisation approach
    Gordon et al. used with a reinforcement learner over a small, designed action set [R39].
- **Perception.** Person or hand tracking (faces plus a detector), sound direction and touch. Hide-and-seek is
  limited by the low viewpoint, so keep it simple: peek-a-boo behind an object.
- **Tests.** Rule-conformance unit tests per game; in sim, a scripted "player" agent gives chase capture rate,
  time to capture and turn-taking latency; bandit choices converge to a simulated player's preferred game.
- **Effort.** M per game for the first two; L for a real hide-and-seek.

### 3.10 Startle and defensive responses

- **Approach.** Use reflexes, as already done in `Reflexes`.
  - Triggers: a loud sound onset (energy jump), a looming visual expansion, a sudden IMU jolt, or a touch
    while unaware.
  - Response: a fixed fast reaction (eyes wide, claws up, back off) with intensity scaled by arousal.
  - **Habituation**, so repeated harmless stimuli lose their effect, and **sensitisation** after a real fall.
  - It pre-empts selection but never the safety supervisor. Arkin et al.'s aibo architecture puts reflexive
    behaviours below the deliberative layer for exactly this reason [R31].
- **Learned?** No. It must be fast, predictable and testable.
- **Tests.** Stimulus-to-first-motion latency (one budget for reflexes, set by the project); the habituation
  curve; the backing-off move stops at a cliff (supervisor test); and an anti-pattern check that it does not
  fire on Pinch's own sounds.
- **Effort.** S.

### 3.11 Sounds

- **Approach.** Pinch has no speech by design. Use **procedural non-linguistic utterances**: chirps and clicks
  whose pitch contour, tempo and roughness come from affect, plus authored sound events keyed to animation
  frames. People do read affect from such sounds, and they sort them into prototypical emotions [R42].
  Anki's Cozmo made its audio *procedural and tied to animation and mood* rather than playing fixed clips
  [R33].
- **Tests.** The sounds stay in sync with animation keys; per-minute rate limits; no overlapping sounds; the
  affect-to-contour mapping matches a unit-test table.
- **Effort.** S.

### 3.12 Memory and preference learning

- **Approach.** Keep three stores:
  - an event log;
  - per-entity statistics (people, objects, places): familiarity, valence and last seen;
  - preference estimates, as Beta or Gaussian posteriors per person × activity, chosen by Thompson sampling.
  This is online statistical learning. It updates after each interaction, its state is a few numbers you can
  print, and it replays exactly from a seed [R39][R40].
- **Tests.**
  - Simulated owners with hidden preferences; measure regret over N sessions.
  - Persistence across a restart.
  - Replay determinism.
  - "Forgetting" is bounded: old preferences decay slowly, not suddenly.
- **Effort.** S–M.

### 3.13 Daily routines: wake, rest, sleep

- **Approach.** Homeostatic drives (energy, battery, sleepiness) plus a circadian gate, with authored
  transitions (stretch on waking, a settle-down routine, sleeping eyes and breathing). The aibo line used
  instinct and homeostasis to drive behaviour [R30][R31]. The evidence on *which* routines people find
  charming is thin; tune them in a human study.
- **Perception.** Clock, battery and ambient light (camera mean brightness).
- **Tests.** Accelerated 24 h runs, checking:
  - the activity-by-hour histogram;
  - no falling asleep in the middle of an interaction;
  - waking on touch or voice within T s;
  - a battery-low path that always ends at rest.
- **Effort.** S.

### 3.14 Failure recovery: stuck, obstructed, self-righting

- **Approach.**
  - *Detect.* "Stuck" means the commanded body velocity is not matched by the estimated velocity (from IMU
    and leg odometry) for longer than T. That residual can be a small learned dynamics model, or a threshold
    to start with. "Flipped" comes from the IMU orientation.
  - *Respond.* An authored escape repertoire: back up, turn, wiggle, try another approach, with memory of
    where Pinch got stuck.
  - **Self-righting is an RL case.** ETH's learned recovery on ANYmal succeeded in over 97% of 100+ trials
    [R5]. The OP3 learned get-up was 63% faster than the scripted one [R6]. Pinch's current searched-swing
    righting is a solid baseline to measure an RL policy against.
- **Tests.** A fault-injection matrix in sim:
  - start upside down and on each side, sweeping roll and pitch;
  - a leg wedged in a gap;
  - a path blocked by a wall;
  - one dead servo (zero torque) and one weak servo (half torque).
  Report the success rate and time-to-recover per cell.
- **Effort.** M. **Depends on** the actuator model and the fault injection harness.

### 3.15 Safety supervisor, independent of everything

- **Approach.** Use the **Simplex** pattern [R44]. A small, simple controller with verified rules sits between
  every command source and the servos and can clamp or stop anything:
  - joint limits;
  - torque or current and temperature limits (servo stall is the main hardware risk);
  - claw pinch-force cap (fingers);
  - cliff, tilt and fall limits;
  - a watchdog on the brain's heartbeat;
  - a low-battery limit.
  It runs in its own process or thread with its own sensor reads. Learned policies stay *inside* the envelope
  it enforces [R45]. It must **never** be learned.
- **Tests.** Property-based fuzzing (random or adversarial command streams can never violate the limits);
  watchdog tests (kill or freeze the brain, and the robot reaches a safe pose within T); fault tests on sensor
  dropout.
- **Effort.** M.

### 3.16 Simulation and test infrastructure

| Piece | What it is | Evidence / precedent | Effort |
|---|---|---|---|
| Deterministic replay | One sim clock, seeded randomness, and a decision log (both already exist). Extend it to perception stubs | Standard practice; decision logs give explainability that RL policies lack [R68] | S |
| Scenario language | Declarative scenarios ("face at 30°, then loud clap at t=5 s") with expected-behaviour assertions | Scenic, a probabilistic scenario language for testing [R49] | M |
| Failure injection | Servo dead, weak or slow; sensor dropout; latency spikes; flip; wedge | Domain-randomisation literature [R1][R13]; adaptive stress testing searches for likely failures automatically [R50] | M |
| Long-duration soak | Simulated days at accelerated time, watching for drift, starvation of drives, repetition and memory growth | Long-term HRI studies show problems appear only over days [R40] | S once the observer exists |
| Behaviour-quality observer | A process that computes all the per-area metrics online: entropy, repetition, gaze agreement, latencies, jerk/SPARC [R55], tracking error | Project-specific. The metrics are reasonable, **but the evidence linking them to human believability ratings is thin** | M |
| Motion-realism score | A discriminator trained to tell Pinch's motion from the reference clip library (AMP-style) [R15]; low scores flag "robotic" motion | AMP uses this as a *training* reward. Using it as an *evaluation* metric is this plan's own proposal, not validated | M |
| Human evaluation | 20–60 s videos rated online with Godspeed (animacy, likeability) [R47] or RoSAS (warmth, competence, discomfort) [R48]; A/B against ablations | The standard HRI instruments. RoSAS was built to fix overlapping Godspeed factors [R48] | S per study |

---

## 4. RL-specific findings

### 4.1 Where RL demonstrably beat hand design

| Domain | Result | Scale relevance |
|---|---|---|
| Agile locomotion and fall recovery | ANYmal: learned policy plus actuator network was faster and more precise than the model-based controller; also learned to recover from falls [R1] | Large robot, but the actuator-network idea transfers directly |
| Rough terrain | Proprioceptive RL beat the hand-tuned controller over mud, snow and rubble [R2]; perceptive version in the wild [R13] | Same recipe works at small scale with randomised terrain |
| Small servo humanoid (OP3) | Walk +181%, turn +302%, get up 63% faster, kick +34% compared with scripted skills [R6] | **Closest evidence**: Dynamixel servos, about 3.5 kg |
| Self-righting | Learned recovery behaviours, over 97% success in 100+ trials [R5] | Directly applicable |
| Hobby-servo robots | Open Duck Mini (Feetech servos, Pi Zero 2W) and ToddlerBot walk with RL policies zero-shot [R35][R36]; ARS gait modulation on hobby servos [R53]; Petoi Bittle [R54] | Same servo class and compute as Pinch |
| Expressive motion | Disney BD-X: RL policies track artist animation robustly and ran about 10 h in public "without a single fall" [R12]; RobotMDM generates text-conditioned motions that an RL tracker can execute [R70]; AMOR lets weights be retuned after training [R69] | The pattern for Pinch's emotes |
| Manipulation with legs | Legs as manipulator / Pedipulate: RL learns pressing and pushing with feet [R17][R58] | Relevant for claw-plus-leg play |
| Small-gripper grasping | **No strong evidence** that RL beats scripted grasping on cheap 1-DoF claws. Low-cost manipulation wins come from imitation (ACT) [R18] | Keep grasp scripted-first |

### 4.2 Sim-to-real practices that matter at this scale

1. **Model the actuator, or learn it.** Hobby and smart servos have deadband, backlash, PD gains inside the
   firmware, velocity-dependent torque limits and heat. Every successful small-servo project does servo system
   identification (sysID):
   - Disney BD-X: PD, friction, backlash and torque-speed model from the bench, randomised within the measured
     range [R12];
   - ToddlerBot: sysID once per motor model, which transferred to other units [R35];
   - Open Duck Mini: published STS3215 identification [R36];
   - ANYmal: an actuator *network* trained on logged data [R1].
2. **Latency and control rate.** Model the policy period and the communication delay, and randomise them. Tan
   et al. found that latency and actuator modelling were key for Minitaur [R13b]. Disney runs the policy at
   50 Hz with a filtered first-order hold to a 600 Hz actuator loop [R12].
3. **Domain randomisation** of mass, friction, motor strength, sensor noise and pushes [R71]. Train the policy to
   adapt online by encoding its recent history (RMA) [R4].
4. **Residual on a scripted gait** (PMTG and residual RL) [R9][R10][R11]. This gives fast, safe early training
   and keeps the gait readable. Pinch already does this. Next, widen the command space (§3.4).
5. **Smoothness and thermal terms.** Action-rate and torque penalties; Olaf's thermal reward [R14]; AMOR's
   finding that raising the smoothness weight reduced jitter and the sim-to-real gap [R69].
6. **GPU-parallel MuJoCo.** MuJoCo Playground (MJX) trains locomotion for OP3-class and quadruped robots in
   minutes to hours on one GPU, with sim-to-real examples [R72]. That fits msi-1 or Modal.
7. **IMU frame and sign checks.** The Bittle sim-to-real work found a simulated-versus-real IMU frame mismatch
   [R54]. Add a unit test for it.

### 4.3 Has anyone learned believability or behaviour selection end-to-end?

- **Not convincingly.**
  - The review of RL in social robotics finds that most work learns small, designed action sets (choosing
    among authored behaviours or tuning parameters) from human-derived rewards, mostly in short lab studies
    [R73].
  - End-to-end deep RL on Pepper needed **14 days** of real interaction to learn whether to offer a handshake
    [R37].
  - Disney leaves high-level decisions to a puppeteer [R12].
  - LLM-generated expressive behaviour (GenEM [R74]) composes *authored* primitives, and it runs off-board.
- **Versus authored systems:**
  - Behaviour trees and state machines win on modularity, reactivity, readability and testability [R43].
  - Explaining RL policies is still an open problem [R68].
- **Where learning helps inside an authored system:**
  - tuning parameters (bandits over activities, as in [R39]);
  - tuning utility weights against the observer's metrics offline (Bayesian optimisation; this is the plan's
    proposal, with little direct evidence);
  - generating motion assets [R70].
- **Recommendation:** keep behaviour selection authored and logged; learn its parameters.

---

## 5. Dependency-ordered build plan

| Phase | Work | Needs first | Effort |
|---|---|---|---|
| **0. Foundations** | (a) Harden the safety supervisor into a separate process with fuzz tests. (b) Behaviour-quality observer with the metrics from §3. (c) Fault-injection hooks and a scenario DSL. (d) Servo bench sysID → actuator model in the twin (needs one real servo on a bench rig) | Existing replay and decision log | M + M + M + S–M |
| **1. Motor core** | (a) Locomotion v2: command-conditioned policy (direction, curvature, gait frequency, height) over the residual gait, with terrain and push curriculum. (b) RL self-righting, compared against searched swing. (c) Generic motion-tracking policy for clips, replacing per-emote balance | 0d (actuator model), 0b, 0c | M, M, M |
| **2. Expression layer** | Animation engine with layers (procedural idle §3.1 + clips + gaze offsets + affect style); procedural eyes §3.6; procedural sounds §3.11 | 1c | M |
| **3. Perception and attention** | Saliency, habituation and inhibition-of-return attention §3.2; GCC-PHAT plus turn-to-disambiguate; novelty score; touch-gesture classifier (after hardware) | 2 for gaze output | M |
| **4. Mind** | Affect with appraisal and style channels §3.8; startle reflex with habituation §3.10; curiosity §3.3; routines §3.13; preference bandits §3.12; stuck detector and escape repertoire §3.14 | 1, 2, 3 | M–L in total |
| **5. Play and manipulation** | Chase and peek-a-boo state machines §3.9; push and inspect primitives; claw redesign, then grasp residual / teacher–student §3.5 | 1a, 3, 4; hardware for grasp | M–L |
| **6. Validation** | Simulated-days soak suite in CI; motion-realism scorer; first online video study (Godspeed/RoSAS) comparing the full system with ablations (no idle layer, no inhibition of return, no affect style) | Everything above | M |

---

## 6. What to train next, in priority order

| # | Component | Method | Expected benefit | Cost class |
|---|---|---|---|---|
| 1 | **Servo / actuator model** from bench logs (parametric fit, or a small actuator network) | Supervised / sysID | The biggest single sim-to-real factor in every small-servo success [R1][R12][R35][R36]. Makes everything below transfer | CPU-min (needs about an hour of bench logging) |
| 2 | **Locomotion v2: command-conditioned, all-direction policy** (lateral, curves, gait frequency, body height, step height), residual over the scripted gait, terrain and push curriculum, AMP crab-style reward optional | PPO | One policy covers scuttle, crawl, curves, transitions, terrain and push recovery; emotions become gait styles [R7][R9][R15] | GPU-h (a few hours to a first policy; GPU-many to polish, as Disney's 30 min preview versus 22 days to final shows [R12]) |
| 3 | **Self-righting / get-up policy** | PPO with randomised start orientations | Expect faster and more reliable righting than the searched swing (OP3 −63% time [R6], ANYmal >97% [R5]); turns a fall into a character moment | GPU-h |
| 4 | **Generic motion-tracking policy** (track any keyframed clip plus layered gaze and idle offsets while balancing) | PPO imitation reward, perpetual/episodic split | Replaces per-emote balance work; new emotes cost an animation, not a training run [R12][R69] | GPU-h to GPU-many |
| 5 | **Stuck / obstruction detector** (dynamics residual model) | Supervised on sim rollouts with injected faults | Earlier, fewer false "stuck" calls; feeds the escape repertoire | CPU-min |
| 6 | **Touch-gesture classifier** (4–5 classes) | Random forest / tiny 1-D CNN on own MPR121 recordings | Makes affection responses specific (stroke versus poke versus rough) [R23] | CPU-min (needs hardware data) |
| 7 | **Desk-object detector fine-tune** (a handful of toy classes) and a small sound-event classifier (clap, knock, voice onset) | Fine-tune YOLO-n / small audio CNN; RKNN int8 | Gives curiosity, play and attention real object identities [R61] | GPU-h (detector), CPU-min to GPU-h (audio) |
| 8 | **Claw grasp residual** (after the redesign): scripted approach plus learned closing, teacher–student | PPO teacher → student distillation; optionally ACT from ~50 teleop demos | Reliable pick-up of light desk toys; enables fetch/play [R10][R18] | GPU-h |

**Not worth training with RL:** behaviour selection, emotion, attention, routines, play rules and the safety
supervisor. These stay authored, with online statistics (bandits, Rescorla–Wagner, learning progress,
habituation) where Pinch must adapt to its owner. Optionally, run offline Bayesian optimisation of utility
weights against the observer metrics (CPU-hours). That is low priority, because no study validates it.

---

## 7. References

- [R1] Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. *Learning agile and dynamic motor skills for legged robots.* Science Robotics, 2019. https://arxiv.org/abs/1901.08652
- [R2] Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning quadrupedal locomotion over challenging terrain.* Science Robotics, 2020. https://arxiv.org/abs/2010.11251
- [R3] Rudin, Hoeller, Reist, Hutter. *Learning to walk in minutes using massively parallel deep RL.* CoRL 2021. https://arxiv.org/abs/2109.11978
- [R4] Kumar, Fu, Pathak, Malik. *RMA: Rapid Motor Adaptation for legged robots.* RSS 2021. https://arxiv.org/abs/2107.04034
- [R5] Lee, Hwangbo, Hutter. *Robust recovery controller for a quadrupedal robot using deep RL.* 2019. https://arxiv.org/abs/1901.07517
- [R6] Haarnoja et al. (DeepMind). *Learning agile soccer skills for a bipedal robot with deep RL.* Science Robotics, 2024. https://arxiv.org/abs/2304.13653 (figures from v2; v1 reports smaller walk and kick gains)
- [R7] Margolis, Agrawal. *Walk These Ways: tuning robot control for generalization with multiplicity of behavior.* CoRL 2022. https://arxiv.org/abs/2212.03238
- [R8] Yang, Zhang, Coumans, Tan, Boots. *Fast and efficient locomotion via learned gait transitions.* CoRL 2021. https://arxiv.org/abs/2104.04644
- [R9] Iscen, Caluwaerts, Tan, Zhang, Coumans, Sindhwani, Vanhoucke. *Policies modulating trajectory generators.* CoRL 2018. https://arxiv.org/abs/1910.02812
- [R10] Johannink et al. *Residual reinforcement learning for robot control.* ICRA 2019. https://arxiv.org/abs/1812.03201
- [R11] Silver, Allen, Tenenbaum, Kaelbling. *Residual policy learning.* 2018. https://arxiv.org/abs/1812.06298
- [R12] Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer (Disney Research). *Design and control of a bipedal robotic character.* RSS 2024. https://arxiv.org/abs/2501.05204
- [R13] Miki, Lee, Hwangbo, Wellhausen, Koltun, Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild.* Science Robotics, 2022. https://arxiv.org/abs/2201.08117
- [R13b] Tan, Zhang, Coumans, Iscen, Bai, Hafner, Bohez, Vanhoucke. *Sim-to-Real: learning agile locomotion for quadruped robots.* RSS 2018. https://arxiv.org/abs/1804.10332
- [R14] Müller, Bächer et al. (Disney Research Imagineering). *Olaf: bringing an animated character to life in the physical world.* 2025. https://arxiv.org/abs/2512.16705
- [R15] Peng, Ma, Abbeel, Levine, Kanazawa. *AMP: adversarial motion priors for stylized physics-based character control.* SIGGRAPH 2021. https://arxiv.org/abs/2104.02180
- [R16] Escontrela et al. *Adversarial motion priors make good substitutes for complex reward functions.* IROS 2022. https://arxiv.org/abs/2203.15103
- [R17] Cheng, Kumar, Pathak. *Legs as manipulator: pushing quadrupedal agility beyond locomotion.* ICRA 2023. https://arxiv.org/abs/2303.11345
- [R18] Zhao, Kumar, Levine, Finn. *Learning fine-grained bimanual manipulation with low-cost hardware (ACT).* RSS 2023. https://arxiv.org/abs/2304.13705
- [R19] Breazeal, Scassellati. *A context-dependent attention system for a social robot.* IJCAI 1999. https://www.ijcai.org/proceedings/1999/163 (also https://groups.csail.mit.edu/lbr/mars/pubs/kismet-attn.pdf)
- [R20] Ruesch, Lopes, Bernardino, Hörnstein, Santos-Victor, Pfeifer. *Multimodal saliency-based bottom-up attention: a framework for the humanoid robot iCub.* ICRA 2008. https://www.robotcub.org/misc/papers/08_Ruesch_Lopes_Hornstein_Victor_Pfeifer.pdf
- [R21] Admoni, Scassellati. *Social eye gaze in human-robot interaction: a review.* JHRI 6(1), 2017. https://www.ri.cmu.edu/app/uploads/2018/01/admoni2017jhri.pdf
- [R22] Ruhland et al. *A review of eye gaze in virtual agents, social robotics and HCI: behaviour generation, user interaction and perception.* Computer Graphics Forum, 2015. https://doi.org/10.1111/cgf.12603
- [R23] Jung, Poel, Poppe, Heylen. *Automatic recognition of touch gestures in the corpus of social touch.* J. Multimodal User Interfaces, 2017. https://link.springer.com/article/10.1007/s12193-016-0232-9
- [R24] Graham. *An introduction to utility theory.* Game AI Pro, ch. 9, 2013. http://www.gameaipro.com/GameAIPro/GameAIPro_Chapter09_An_Introduction_to_Utility_Theory.pdf
- [R25] Oudeyer, Kaplan, Hafner. *Intrinsic motivation systems for autonomous mental development.* IEEE Trans. Evolutionary Computation, 2007. https://doi.org/10.1109/TEVC.2006.890271
- [R26] Marsland, Nehmzow, Shapiro. *Novelty detection on a mobile robot using habituation* (and later *On-line novelty detection for autonomous mobile robots*, RAS 2005). https://arxiv.org/abs/cs/0006007
- [R27] Perlin, Goldberg. *Improv: a system for scripting interactive actors in virtual worlds.* SIGGRAPH 1996. https://doi.org/10.1145/237170.237258
- [R28] Asselborn, Johal, Dillenbourg. *Keep on moving! Exploring anthropomorphic effects of motion during idle moments.* RO-MAN 2017. https://infoscience.epfl.ch/record/231097/files/keep_on_moving.pdf
- [R29] Terzioglu, Mutlu, Şahin. *Designing social cues for collaborative robots: the role of gaze and breathing in human-robot collaboration.* HRI 2020. https://aperta.ulakbim.gov.tr/records/3247
- [R30] Fujita. *AIBO: toward the era of digital creatures.* IJRR 20(10), 2001. https://doi.org/10.1177/02783640122068092
- [R31] Arkin, Fujita, Takagi, Hasegawa. *An ethological and emotional basis for human–robot interaction.* RAS 42(3-4), 2003. https://doi.org/10.1016/S0921-8890(02)00375-5
- [R32] Breazeal. *Emotion and sociable humanoid robots.* Int. J. Human-Computer Studies 59, 2003. https://doi.org/10.1016/S1071-5819(03)00018-1
- [R33] Designing Sound. *Sound for robots: an interview with Anki senior sound designer Ben Gabaldon.* 2016. https://designingsound.org/2016/11/15/sound-for-robots-an-interview-with-sr-sound-designer-ben-gabaldon/
- [R34] Becker-Asano, Wachsmuth. *Affective computing with primary and secondary emotions in a virtual human (WASABI).* AAMAS Journal, 2010. https://doi.org/10.1007/s10458-009-9094-9
- [R35] Shi, Wang, Song, Liu. *ToddlerBot: open-source ML-compatible humanoid platform for loco-manipulation.* CoRL 2025. https://arxiv.org/abs/2502.00893
- [R36] Pirrone et al. *Open Duck Mini* (open-source, RL-walking small biped with Feetech servos; sim2real notes). https://github.com/apirrone/Open_Duck_Mini
- [R37] Qureshi, Nakamura, Yoshikawa, Ishiguro. *Robot gains social intelligence through multimodal deep reinforcement learning.* Humanoids 2016. https://sites.google.com/seecs.edu.pk/ahqureshi/social-intelligence
- [R38] Takayama, Dooley, Ju. *Expressing thought: improving robot readability with animation principles.* HRI 2011. https://www.wendyju.com/publications/151r-takayama.pdf
- [R39] Gordon, Spaulding, Kory Westlund, Lee, Plummer, Martinez, Das, Breazeal. *Affective personalization of a social robot tutor for children's second language skills.* AAAI 2016. https://ojs.aaai.org/index.php/AAAI/article/view/9914
- [R40] Leite, Martinho, Paiva. *Social robots for long-term interaction: a survey.* IJSR 5, 2013. https://doi.org/10.1007/s12369-013-0178-y
- [R41] Trafton, Schultz, Perznowski, Bugajska, Adams, Cassimatis, Brock. *Children and robots learning to play hide and seek.* HRI 2006. https://apps.dtic.mil/sti/pdfs/ADA480331.pdf
- [R42] Read, Belpaeme. *People interpret robotic non-linguistic utterances categorically.* IJSR 8(1), 2016. https://doi.org/10.1007/s12369-015-0304-0
- [R43] Iovino, Scukins, Styrud, Ögren, Smith. *A survey of behavior trees in robotics and AI.* RAS, 2022. https://arxiv.org/abs/2005.05842
- [R44] Sha. *Using simplicity to control complexity* (Simplex architecture). IEEE Software 18(4), 2001. https://engineering.purdue.edu/dcsl/reading/2007/foob-using_simplicity_to_control_complexity.pdf
- [R45] Brunke et al. *Safe learning in robotics: from learning-based control to safe reinforcement learning.* Annual Review of Control, Robotics, and Autonomous Systems, 2022. https://arxiv.org/abs/2108.06266
- [R46] Arias et al. (Jibo dormitory deployment, HRI 2020 design workshop). https://malulu.github.io/HRI-Design-2020/assets/pdf/Arias%20et%20al.pdf
- [R47] Bartneck, Kulić, Croft, Zoghbi. *Measurement instruments for the anthropomorphism, animacy, likeability, perceived intelligence, and perceived safety of robots (Godspeed).* IJSR 1, 2009. https://doi.org/10.1007/s12369-008-0001-3
- [R48] Carpinella, Wyman, Perez, Stroessner. *The Robotic Social Attributes Scale (RoSAS).* HRI 2017. https://doi.org/10.1145/2909824.3020208
- [R49] Fremont et al. *Scenic: a language for scenario specification and scene generation.* PLDI 2019. https://arxiv.org/abs/1809.09310
- [R50] Lee, Mengshoel, Saksena, Gardner, Genin, Silbermann, Owen, Kochenderfer. *Adaptive stress testing: finding likely failure events with reinforcement learning.* JAIR, 2020. https://arxiv.org/abs/1811.02188
- [R51] *A review on sound source localization in robotics: focusing on deep learning methods.* 2025. https://arxiv.org/abs/2507.01143
- [R52] Pathak, Agrawal, Efros, Darrell. *Curiosity-driven exploration by self-supervised prediction.* ICML 2017. https://arxiv.org/abs/1705.05363
- [R53] Rahme, Abraham, Elwin, Murphey. *Dynamics and domain randomized gait modulation with Bezier curves for sim-to-real legged locomotion.* 2020. https://arxiv.org/abs/2010.12070
- [R54] Jabbour, Neuman, Mazumder. Sim-to-real on the Petoi Bittle (RSS 2022 Sim2Real workshop). https://sim2real.github.io/assets/papers/2022/jabbour.pdf
- [R55] Balasubramanian, Melendez-Calderon, Roby-Brami, Burdet. *On the analysis of movement smoothness (SPARC).* J. NeuroEngineering and Rehabilitation, 2015. https://doi.org/10.1186/s12984-015-0090-9
- [R56] Boston Dynamics. *Starting on the right foot with reinforcement learning.* Blog, 2024. https://bostondynamics.com/blog/starting-on-the-right-foot-with-reinforcement-learning/
- [R57] RAI Institute. *High-performance reinforcement learning on Spot: optimizing simulation parameters with distributional measures.* 2025. https://arxiv.org/abs/2504.17857
- [R58] Arm et al. *Pedipulate: enabling manipulation skills using a quadruped robot's leg.* ICRA 2024. https://arxiv.org/abs/2402.10837
- [R59] Chi et al. *Diffusion policy: visuomotor policy learning via action diffusion.* RSS 2023. https://arxiv.org/abs/2303.04137
- [R60] Kalashnikov et al. *QT-Opt: scalable deep RL for vision-based robotic manipulation.* CoRL 2018. https://arxiv.org/abs/1806.10293
- [R61] Q-engineering. *YoloV5-NPU* (RK3566/RK3588 int8 benchmarks; community, indicative). https://github.com/Qengineering/YoloV5-NPU
- [R62] Ribeiro, Paiva. *The illusion of robotic life: principles and practices of animation for robots.* HRI 2012. https://doi.org/10.1145/2157689.2157814
- [R63] van Breemen. *Bringing robots to life: applying principles of animation to robots.* CHI 2004 workshop (Philips iCat). No stable link found; cited via [R62].
- [R64] Hu, Huang, Sivapurapu, Zhang (Apple). *ELEGNT: expressive and functional movement design for non-anthropomorphic robot.* DIS 2025. https://arxiv.org/abs/2501.12493
- [R65] Mehrabian. *Pleasure-arousal-dominance: a general framework for describing and measuring individual differences in temperament.* Current Psychology, 1996. https://doi.org/10.1007/BF02686918
- [R66] Community Vector documentation (behaviour tree, emotion model; reverse-engineered, not official Anki material). https://randym32.github.io/Anki.Vector.Documentation/
- [R67] Baker et al. (OpenAI). *Emergent tool use from multi-agent autocurricula.* ICLR 2020. https://arxiv.org/abs/1909.07528
- [R68] Milani, Topin, Veloso, Fang. *Explainable reinforcement learning: a survey and comparative review.* ACM Computing Surveys, 2023. https://arxiv.org/abs/2211.06665
- [R69] Alegre, Serifi, Grandia, Müller, Knoop, Bächer (Disney Research). *AMOR: adaptive character control through multi-objective RL.* SIGGRAPH 2025. https://arxiv.org/abs/2505.23708
- [R70] Serifi, Grandia, Knoop, Gross, Bächer (Disney Research). *Robot motion diffusion model: motion generation for robotic characters.* SIGGRAPH Asia 2024. https://la.disneyresearch.com/publication/robot-motion-diffusion-model-motion-generation-for-robotic-characters/
- [R71] Peng, Andrychowicz, Zaremba, Abbeel. *Sim-to-real transfer of robotic control with dynamics randomization.* ICRA 2018. https://arxiv.org/abs/1710.06537
- [R72] Zakka et al. *MuJoCo Playground.* 2025. https://arxiv.org/abs/2502.08844
- [R73] Akalin, Loutfi. *Reinforcement learning approaches in social robotics.* Sensors 21(4), 2021. https://arxiv.org/abs/2009.09689
- [R74] Mahadevan et al. *Generative expressive robot behaviors using large language models (GenEM).* HRI 2024. https://arxiv.org/abs/2401.14673

**Where the evidence is thin, stated plainly:**
- No published study links automatic repetition, entropy or gaze metrics to human believability ratings.
- No published work evaluates RL grasping on tiny hobby claws.
- Loona/EMO publish no technical documentation of their behaviour systems, so they are not used here.
- Modern aibo's learning features are described only in Sony marketing; the citations here are the original
  aibo architecture papers.
- The RK3566 NPU throughput figures come from community benchmarks.
- Daily-routine design has little HRI evidence beyond the aibo line.
