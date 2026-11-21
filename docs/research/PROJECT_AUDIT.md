# Project audit: Pinch against research and open-source practice

*Audit of 2026-11-21, made before planning the foundation. It checks what the project has built, and how it plans
to proceed, against research papers, production pet robots and well-maintained GitHub projects. The nine detailed
reports behind it are in [audit/](audit/). "The twin" means the MuJoCo simulation of the robot and its desk.*

## 1. Verdict in one page

The engineering habits are strong: every decision is measured and recorded, runs replay exactly, the code is
layered, and one command checks everything. The direction has three faults that those habits could not catch,
because all of the evidence so far comes from one simulator that has never been compared with anything real:

1. **The simulated servos are guesses, and the training leaves out delay.** Every small servo robot we found that
   walks first measured its real servos on a bench.
2. **The pet's mind is tested on information the robot will not have.** The twin tells the mind exactly where
   everything is. On the real robot most of those inputs are missing or far weaker.
3. **The world is one desk and four named objects.** The requirements and the owner both describe a room, any
   object, and people.

Four more findings follow from the same root: the mind has no declared structure, the training tools have moved
on upstream, many comparisons used too few trials to mean anything, and the robot's own software is not yet safe to
leave running.

### What is sound and should stay

| Choice | Why it is right |
|---|---|
| Learned skills reach the robot only as plain number files; strict code layers | Berkeley Humanoid Lite ships only its low-level code to the robot; Open Duck Mini runs exported policies only |
| No ROS on the robot | None of the learned-walking peers runs ROS 2 on board (section 4) |
| The same mind and body code in the twin and on the robot, with only the hardware layer swapped | Anki's Vector was built exactly this way, with simulator tests on every change (section 4) |
| A scripted gait that guides the learned walker | Disney's BD-X and Open Duck Mini both train against reference motions |
| Policies trained in MJX are judged in the separate CPU twin | A simulator-to-simulator check; Berkeley Humanoid Lite makes the same check a named step. It is not a check against reality |
| The motion sensor (BNO055) in its gyro-plus-accelerometer mode at 50 Hz | Open Duck Mini feeds its policy the same way |
| Downward distance beams plus the foot-aware edge guard | Vector senses desk edges with four cliff sensors |
| Route planning in plain Python (the unmerged branch) | Vector plans with A* on a grid and an escape planner; Nav2 is too heavy for this computer |
| The laptop language model as an optional helper | Rockchip's on-board language-model toolkit does not support this chip |
| A 27-40 minute battery | Normal for the class; the fix is a dock, not a bigger pack |
| The session recorder, realism measures, decision log and run records | Better than most of the projects compared |

### The seven findings, most important first

**1. Servo truth is missing, and it blocks trust in every learned skill.**
- *Our code:* only stall torque, no-load speed and rated torque come from a datasheet. Stiffness, friction, rotor
  inertia, gear play, delay and top speed are guesses (`crab/params.py:89-126`). No code fits the model to
  measurements. Training applies no command delay, no sensor delay and no motion-sensor bias; the delay line in
  `sim/latency.py` is used by two design scripts only.
- *Practice:* Open Duck Mini fitted its Feetech servos with Rhoban's BAM tool and warns that a wrong motor model
  gives "chaotic movements". ToddlerBot fitted nine parameters per motor and still saw about twice the tracking
  error on the real robot. Disney identified friction and gear play before training BD-X. Hu et al. (2025) found
  that leaving static friction out of training made real policies clearly worse.
- *Consequence:* "0 falls" and "200 of 200 get-ups" are statements about the guessed model. The get-up policy
  already asks for joint speeds of 17 rad/s where the robot allows about 8.

**2. The twin is an oracle, so "done in the twin" overstates what the robot can do.**
- *Our code:* the twin gives the mind each object's true position plus 4 mm of noise
  (`pinch/twin/sensors.py:48-66`), the robot's exact pose, and exact tilt readings. No camera image is processed in
  any bench. The twin's camera is an 80-degree pinhole; the real lens is a 160-degree fisheye (`sim/desk.py:175`).
  People are flags and function calls, and speech is typed text.
- *On the robot:* position comes only from one printed marker on the tray. Nothing produces the observations that
  feed curiosity and startle. There is no speech recogniser. Grabbing cannot succeed at all: the two functions
  that tell the body what the claw sees and holds are set only in the twin (`pinch/twin/core.py:72-74`).
- *Practice:* Vector's simulator replaced only the hardware layer; the same vision and behaviour code ran on
  simulated sensors, and those tests ran on every change.
- *Consequence:* about half of the requirements are marked implemented and tested, but that is true in the
  oracle twin. Play, claws, and face and gaze are the weakest areas.

**3. The world is one desk; the goal is a room.**
- *Our code:* a 70 x 45 cm desk, a tray, a fixed seat for the owner and four named, coloured objects are constants
  (`pinch/world.py:14-21`). Sixteen of the 26 activities depend on that geometry.
- *The requirements:* they describe a room-scale home with people, not one desk. They contain no tidying,
  carrying or pick-and-place item; what they ask of the claws is expressive, gentle interaction. Yet tidying,
  grasping and pushing took at least 18 of the 98 recorded decisions and about half of the training runs.
- *Research:* no paper supports learning to grasp with a 15-23 mm claw whose servos report nothing.
- *Consequence:* the owner's first video review ("a tidy-up robot, not a pet", "5 %") is what this produces.

**4. The mind has no declared structure.**
- *Our code:* priority exists only as numbers spread over 26 files (get-up 6.0, low battery 5.0, unstick 4.0 ...
  idle 0.25). Sensing, deciding, planning and acting share one object. The body takes one task at a time, the
  face shows whatever was written last, and idle life stops when anything else starts. About 730 numbers that
  shape behaviour sit in Python.
- *Practice:* Vector's engine documents describe behaviours as data-defined instances in a tree, actions that lock
  the motors they use, a separate animation process with layered tracks, and tests that walk the tree to enforce
  design rules. Their emotion note says an emergent, emotion-scored chooser was replaced with explicit control
  because it could not be directed. BD-X and Reachy Mini use the same layered playback.
- *Consequence:* a new activity can silently outrank safety, the pet cannot look at something while walking,
  and behaviour cannot be tuned, compared or rolled back without editing code.

**5. The learning tools have moved.** Brax's own README says only its training code is still maintained. MuJoCo
Playground 1.0 (November 2026) moved to PyTorch, MuJoCo Warp and RSL-RL and froze its JAX environments. The
maintained path for a small servo robot is mjlab, and two templates exist that our notes missed:
`pollen-robotics/microduck_rl` (a small servo robot) and `KingKongRobotics/jumper` (a crab).

**6. Many comparisons were too small to decide anything.** With 24 trials, a result of 11 successes has a 95 %
range of 28-65 %. A difference of 19 against 14 out of 48 is not distinguishable from chance (p = 0.39). Telling
46 % from 67 % needs about 86 trials each. Each policy was trained once, and the same test layouts were reused
for every candidate. Two adoptions do hold up: the learned grab against the scripted one (11 against 2 of 24,
p = 0.008) and the rejection of the combined walker (2 against 10 of 24, p = 0.017).

**7. The robot's own software is not ready to be left on.** There is no way to charge without a person. The
control port listens on the whole network with no login and the service runs as root (`robot/http_api.py:74`, `deploy/pinch.service:12`).
The installer copies a folder that no longer exists (`deploy/install.sh:15`). There is no watchdog and no way back
from a bad update. The robot's self-tests are not run by the automatic checks.

### What this means for the four decisions that were waiting

| Decision | What the audit says |
|---|---|
| Room floor as well as the desk | Yes. The requirements themselves describe a room. The desk stays as one generated world |
| A charging dock | Yes: a contact dock with the charger in the dock, not in the shell. About INR 1,100-1,700 of parts. Fix the contact position in the CAD before anything is printed |
| Buy two leg servos, an arm-servo pair and the bus adapter now (about INR 8,600) | Yes, first. Then a small bring-up kit (computer, motion sensor, battery monitor, one battery). Hold the full order until the five risks in section 5, piece A3, are checked |
| Unpause Modal | Not yet. No paid training is useful until the new walking environment and a measured servo model exist |

## 2. How the audit was done

- **Code, read in full by three reviewers:** body and training; mind; robot software and twin. Every claim carries
  a file and line. Six of the most serious claims were then re-checked by hand against the source, and all six
  held.
- **Outside evidence, six reviewers:** comparable robots; motor learning and simulation-to-reality; perception and
  navigation; pet behaviour and character; twin, test world and statistics; robot software, power and safety.
- **Verification:** every cited paper and repository was checked against the live source by script. All 97 arXiv
  papers resolve to the titles cited, and 86 repositories exist with the stars, last-push date and licence
  given. Four cited repositories are archived and are marked so. The statistics in finding 6 were recomputed
  independently.
- **Read first-hand for this audit:** the published Anki Vector engine's architecture and behaviour documents
  (`digital-dream-labs/vector`), the MuJoCo Playground 1.0 README and its servo-fitting example, the Brax README,
  Rockchip's model benchmark table, the `microduck_rl` README and agent notes, and the DribbleBot paper text.
- **Recency:** references are 2022-2026 where the field has moved. Older ones are used only where they are still
  the method inside current systems, and the reports say why.
- **Limits:** no hardware exists, so nothing about real behaviour could be measured. Tests, benches and training
  were not run for this audit. Mechanical design, the electronics layout and the power budget under peak load were
  not audited here; a separate audit session on the same day covered them, and its findings are not repeated.

## 3. Where the project stands

| Area | State on 2026-11-21 |
|---|---|
| Code | About 19,500 lines of Python in the robot, pet, firmware, training and simulation packages; 98 recorded decisions; about 90 automatic tests |
| Hardware | None built or bought. Bill of materials INR 88,566, of which 14 leg servos are INR 46,586 |
| Learned skills in use | Walking (a correction on a scripted trot), get-up, emote balance, grab; all trained with Brax on MJX, each once |
| Worlds | One desk with a tray and four objects, plus three unknown props |
| Requirements | About half implemented and tested, in the oracle twin. Weakest: play, claws, face and gaze, long-run and realism tests |
| Unmerged work | Three branches: route planning and stuck recovery; grab-fit checks; activity balance (section 8) |

## 4. How comparable robots are built

| Robot | What it is | Servo model | Learning stack | Software on the robot | Lesson for Pinch |
|---|---|---|---|---|---|
| Open Duck Mini (4.8k stars) | 42 cm biped, 14 Feetech STS3215, Pi Zero 2 W, under USD 400 | Fitted on a bench with BAM; a gear-play variant won | MJX + Brax (old Playground), imitation of reference walks | Plain Python at 50 Hz, exported policy, no ROS | Hardware frozen before learning started; delay randomised 0-3 control steps |
| ToddlerBot (0.8k) | 56 cm humanoid, 30 Dynamixel servos | Nine parameters per motor from chirp tests, 1.3 degree error | MJX | Python at 50 Hz, no ROS | Real tracking error still about twice the simulated one |
| microduck_rl (2.9k) | Small servo biped by the Reachy Mini makers | BAM, with gear play modelled and the encoder read through it | mjlab (MuJoCo Warp + RSL-RL) | Policies swap at run time behind one shared observation list | The closest training template |
| Jumper (3.2k) | 22-joint crab, 1.8 kg, RK3576 | Measured torque-speed curve, tested against the model | mjlab | Rust controller; its own notes say it has not yet driven a real bus | A crab on the stack we plan to use |
| Disney BD-X (RSS 2024) | Expressive biped character | Identified, with friction and gear play | RL policies per motion, imitating authored references | An animation engine layers background, triggered clips and operator input over the policies | Authored character over learned motor skills |
| Anki Vector (shipped product) | Desk pet, four small CPU cores, one camera, four cliff sensors, one forward distance sensor | n/a (wheels) | none | Three processes: behaviours and vision (60 ms), animation (33 ms), motor control (5 ms). Behaviours are data in a tree. Simulator tests on every change. Charger with a printed marker | The nearest production design for the mind, the map and the dock |
| Reachy Mini (1.7k) | Expressive desktop robot | n/a | none | A daemon owns the hardware and safety; clients connect to it; a language model picks among recorded clips | Pattern for the robot service and for using a language model |

Among these, none built this much on an unmeasured simulation. The two nearest cases, Jumper's open controller
and Anki's simulator work, both had real robots beside them. Pinch's cost is inside the peer range.

## 5. Audit by foundation piece

Each piece: today's state, the one reference to follow, what to reuse, and the verdict.

### A. Physical robot

**A1 Power for a whole day: missing.**
- *Today:* no charging path. The 27-40 minute figure comes from a model with an assumed 4 W for the computer
  (`sim/power.py:25-30`). Servos are never switched off at rest.
- *Reference:* Vector's pattern (short active spells, then a self-found charger with a printed marker) and the
  five phases of Nav2's docking server: staging pose, detect, controlled approach, confirm charging, retry.
- *Do:* a contact dock with spring pins, magnets, a funnel and a printed tag; the charger in the dock; rest lying
  down with the servos off. Plan on about 35 minutes active and 70 docked.

**A2 Servo truth: missing. The first hardware step.**
- *Today:* see finding 1.
- *Reference:* Duclusaud et al., "Extended Friction Models for the Physics Simulation of Servo Actuators",
  ICRA 2025 (arXiv 2410.08650).
- *Reuse:* `Rhoban/bam` (629 stars, Apache-2.0). It ships a model for the larger STS3215, not our STS3032, so we
  run its pendulum test ourselves. Also `mujoco.sysid`, already inside our MuJoCo 4.0.1; MuJoCo Playground 1.0
  has a worked example that fits Feetech-servo joints and measures command delay.
- *Do:* bench two leg servos and the arm pair; fit the model; measure bus timing, holding current and delay.

**A3 Robot runtime: improve.**
- *Today:* see finding 7. The compute budget is an assumption; nothing has been timed on the real chip.
- *Reference:* Vector's three-process split; the PREEMPT_RT timing study (arXiv 2604.19275) for what Linux can
  hold.
- *Reuse:* the Open Duck Mini runtime as the pattern; systemd's watchdog; Tailscale Serve for the control port.
- *Do:* separate control, mind and control-port processes, with control on its own core; watchdog and a
  two-folder update with rollback; a login token; servo protection limits written and checked at start.
- *Five risks to check before the full order:* the charging contacts in the CAD; heat inside a closed shell at
  35-40 C (PLA softens near 55-60 C); bring-up of the computer (six PWM outputs, watchdog, shared sensor bus,
  camera); bus timing through the USB adapter at 1 Mbps; behaviour when the battery cuts out mid-write.

### B. Twin and test world

**B4 The twin matches the robot: the largest software gap.**
- *Today:* see finding 2. Training also uses 4 solver passes where the twin uses about 100.
- *Reference:* Dao and Fern, 2026 (arXiv 2604.11090): score "the twin matches the robot" by comparing the spread
  of joint readings from the same short motions on both, with under five minutes of robot data.
- *Do:* add delay, noise and the encoder's step size to the twin; render the fisheye camera and run the same
  detector on it; keep the oracle as a switch and label every result "oracle" or "perceived". Until this passes,
  use the twin to rank ideas and find crashes, not for absolute numbers.

**B5 Procedural worlds: missing.**
- *Reference:* RoboCasa, RSS 2024 (arXiv 2406.02523), for everyday object models; HuNavSim 2.0
  (arXiv 2507.17317) for scripted human behaviour.
- *Reuse:* MuJoCo's own model-building interface (`MjSpec`) for a small generator; RoboCasa's object set
  (1.9k stars; assets CC BY 4.0). Not the full kitchens.
- *Do:* generate desks, floors, rooms, rugs, slopes, lighting and up to ten random objects; give people bodies
  that move (a hand, a face at desk height, walking legs).

**B6 Tests as data and sound statistics: improve. The cheapest valuable item.**
- *Today:* see finding 6. Scenarios are Python code.
- *Reference:* Snyder et al., RSS 2025 (arXiv 2503.10966), on comparing two policies with the fewest trials.
- *Reuse:* scipy.
- *Do:* print rate, range and count for every result; compare on the same layouts; say "inconclusive" when it is;
  use 100 or more layouts and a held-back set for adoption; write scenarios as a timeline plus checks in a data
  file.

**B7 Measurement and pipeline: confirmed, with one gap.** The robot's self-tests and the pinned behaviour trace
do not run in the automatic checks (the trace is exact only on Windows).

### C. Body

**C8 Motor skills: rebuild on a maintained stack.**
- *Today:* the walker is a correction of at most 0.15 rad on a scripted trot, trained on flat ground for 48 fixed
  commands. Each skill has its own input layout. Hand-over between skills takes one 20 ms step. The grab runs
  only in the twin.
- *Reference:* Grandia et al., "Design and Control of a Bipedal Robotic Character", RSS 2024 (arXiv 2501.05204):
  separate standing, walking and per-clip policies that imitate authored references, on measured actuators.
- *Reuse:* `pollen-robotics/microduck_rl` (2.9k stars, Apache-2.0) on `mujocolab/mjlab` (3.4k stars): measured
  servo models, gear play, delay and voltage variation, stand-up, a ball kick, and one shared input layout so
  skills swap at run time. `KingKongRobotics/jumper` for the crab body.
- *Do:* freeze the current environments. Build walker v3 with continuous commands, small obstacles, the measured
  servo model and delays. Add a kick-and-nudge primitive that a scripted layer aims. Stop grab training; keep a
  scripted grab as the baseline anything learned must beat.
- *Risk:* mjlab calls its Windows support preliminary, and both of our GPUs are 6 GB laptops. Measure before
  committing.

**C9 State estimation and a world map: replace.**
- *Today:* position only from the tray's marker; otherwise a sum of commands with 30 % assumed drift
  (`robot/localizer.py:19`). Four colour ranges for objects. No map.
- *Reference:* Youm et al., 2024 (arXiv 2402.00366): a filter that fuses leg motion, the motion sensor and a
  small learned speed estimate trained in simulation.
- *Reuse:* `airockchip/rknn_model_zoo` (2.9k stars). On this chip a small detector runs at 35 frames a second
  alone and a face detector at 149; open-vocabulary and transformer models are too slow (6, 2 and 1).
- *Do:* an estimator that needs no marker, with the marker as an optional anchor at the dock; a robot-centred
  grid, a table of things and a list of dangerous places; small detectors on the chip; the laptop names unknown
  things when it is there. One sensor addition: an 8 x 8 distance sensor facing forward and down (about USD 20,
  1.9 g).

**C10 Moving safely: design confirmed, not yet measured.**
- *Today:* on the main branch, a straight walk, one side waypoint and two recovery moves. The unmerged branch has
  A* on a cost map, a progress checker and a ladder of recoveries.
- *Reference:* Macenski et al., 2023 (arXiv 2307.15236), the Nav2 design; Vector's planner does the same on a
  grid, with replanning and an escape step when it starts inside an obstacle.
- *Do:* measure the branch on generated worlds; replace the desk rectangle with the learned map.

**C11 Safety supervisor: improve. It was listed as done; it is not.**
- *Today:* joint limits, a per-step clamp, temperature and voltage checks exist (`robot/crab_robot.py:107-199`),
  inside the process they protect. No watchdog. Servo overload limits are never written. A killed process leaves
  the servos powered. `pinch/safety.py` is the pet's fear memory, not a supervisor.
- *Reference:* the Simplex idea (a small trusted layer under a complex one) is still how this is done; recent
  learned safety filters such as Gameplay Filters (arXiv 2405.00846) are too heavy for this computer.
- *Do:* layers that survive a hung program: watchdogs, limits stored in each servo, the battery protection
  board, lift detection that cuts torque, and a temperature ladder.

### D. Mind

**D12 Architecture: restructure around the existing activities.**
- *Today:* see finding 4.
- *Reference:* the Vector engine's behaviour-system documents; Ghzouli et al., IEEE TSE 2023 (arXiv 2208.04211),
  on behaviour trees and state machines in robotics.
- *Reuse:* no library. A behaviour-tree library would mean rewriting 26 activities for no gain.
- *Do:* a declared table of states and priorities as data; an arbiter that says who owns legs, claws, gaze,
  face and sound (reflex over command over reaction over activity over ambient life); an always-on ambient
  layer; tests that enforce the design rules.

**D13 Character: improve.**
- *Today:* five fixed numbers, never saved and never changing. No written character.
- *Reference:* Vector's emotion note: a few named moods that decay to neutral and mostly choose animations.
- *Do:* write the character first; add a slow mood over hours and days, and traits that drift with treatment.

**D14 Expression engine: replace the structure, keep the content.**
- *Today:* 22 keyframed motions with good mood styling. They cannot layer on walking, are not synchronised with
  face and sound, and have no authoring tool.
- *Reference:* the animation engine in Grandia et al. 2024; Vector's animation document (tracks, layers,
  alternates per trigger, a preview tool).
- *Reuse:* `Stanford-TML/robot_keyframe_kit` (179 stars, MIT) or Blender for authoring.

**D15 Behaviour settings as data: missing.**
- *Reference:* Vector's "how data drives behaviours": a code class plus data-defined instances and conditions.
- *Reuse:* TOML files checked by pydantic (29.6k stars); a settings fingerprint in every log.

**D16 Memory and learning: improve.**
- *Today:* one JSON file of 400 episodes. The call that learns where things live is never made. No version
  number. Mood is not saved.
- *Reference:* Park et al., "Generative Agents", UIST 2023 (arXiv 2304.03442): recall by recency, importance and
  relevance, with a periodic reflection step.
- *Reuse:* SQLite from the standard library; nightly summaries by the laptop model. No memory server.

## 6. Reuse instead of write

| Need | Project | Stars, licence | Use |
|---|---|---|---|
| Servo model fitting | Rhoban/bam | 629, Apache-2.0 | Pendulum test and friction model for the leg servo |
| Servo model fitting | `mujoco.sysid` (in MuJoCo) and the Playground 1.0 example | official | Fit joint parameters and delay from logs |
| Training framework | mujocolab/mjlab | 3.4k, Apache-2.0 | New environments |
| Training template | pollen-robotics/microduck_rl | 2.9k, Apache-2.0 | Structure, randomisation, shared input layout, reward rules |
| Crab on that stack | KingKongRobotics/jumper | 3.2k, Apache-2.0 | Gaits and export for a Rockchip chip |
| Servo bus reference | huggingface/lerobot | 29k, Apache-2.0 | Compare with our own bus code and calibration |
| On-board vision models | airockchip/rknn_model_zoo | 2.9k | Small detector and face detector |
| Object models for worlds | robocasa/robocasa | 1.9k, MIT code, CC BY assets | Everyday objects only |
| Statistics | scipy | 15k, BSD | Ranges and tests in every bench |
| Settings | pydantic | 29.6k, MIT | Checked behaviour settings |
| Control port | tailscale | 39k, BSD | Private access with identity |
| Design reference only | digital-dream-labs/vector | 119, no clear licence | Read for the design; copy no code |

## 7. Corrections to our own notes

- `TRAINING_RESEARCH.md` said DribbleBot trained for 77 billion steps. The paper says 7 billion, about 48 hours.
  Corrected.
- It called the BeyondMimic repository active. It is archived; the maintained version is mjlab's tracking task,
  which supports one humanoid only. Corrected.
- It said Newton was on Isaac Lab's roadmap. Isaac Lab 3.0.0 (October 2026) ships it. Corrected. The reason
  `ARCHITECTURE.md` gives for rejecting Isaac Lab is weaker for it, though mjlab remains the better fit.
- It planned to fit our leg servo with BAM as if BAM supported it. BAM supports the larger STS3215 only.
- The 93.6 % pushing result it cites is the strictest of five measures, in simulation, on a 27 kg robot.
- Yesterday's foundation list marked the safety supervisor and the measurement pipeline as done. The first is
  partial (piece C11); the second has the gaps in pieces B6 and B7.
- A local design note (not in the repository) describes a folder and design that no longer exist.

## 8. The unmerged branches

| Branch | What it adds | Audit view |
|---|---|---|
| Route planning and stuck recovery | A* on a cost map, a progress checker, a ladder of recoveries, following a moving toy | The design matches Vector and Nav2. Its commits report no numbers. Measure it, then merge it as the start of piece C10 |
| Grab-fit checks | Refuses grabs that cannot fit; smoother claw targets | Useful ideas, but the grab cannot run on the robot and tidying is outside the requirements. Park it; keep the smooth hand-over change for piece C8 |
| Activity balance | Caps on how long any activity and chores can run; a comparison of ten companion robots | A standard device that fixes the tidy-heavy day, with ten more numbers in code. Fold it into pieces D12 and D15; use the comparison for D13 |

## 9. What could not be verified

- Anything about real hardware: timing on the chip, bus behaviour of the real servo, camera, heat, power draw.
- How aibo, Lovot, Moflin, EMO and Loona work inside; only makers' descriptions exist.
- Whether the route-planning branch beats the main branch; no numbers were recorded and none were run today.
- Several figures the reports mark "search only", such as Vector's 25 minutes of active time and the price of
  parts in India. They are flagged in the reports and none carries a conclusion alone.
- No paper was found for learned walking on a sideways-walking crab, or for a language-model-driven small pet
  running for weeks. Those parts of the plan rest on neighbouring evidence.

## 10. What this means for the plan

This is input for the foundation plan, not the plan. It needs the owner's approval before anything is built.

1. **Now, with no hardware:** the statistics helper and held-back test set; the robot's self-tests in the
   automatic checks; the installer fix; "oracle" and "perceived" labels on results.
2. **Hardware truth, in two small purchases:** the servo bench first, then the computer bring-up kit. Both use
   parts already in the bill of materials. The full order waits for the five risks in piece A3.
3. **An honest twin and a wider world:** delay and noise, generated worlds with people, perception in the loop.
4. **The body on the maintained stack:** one shared input layout, walker v3, get-up, kick and nudge, the
   position estimator.
5. **The mind's structure:** the written character, the state and priority table, the channel arbiter with
   ambient life, settings as data, memory.

Items 3, 4 and 5 can run side by side; item 2 feeds item 4. Tidying leaves the foundation and returns later, if
wanted, as one optional habit.

## 11. Fixed together with this audit

These defects in the project's own setup were fixed in the same change as the audit. They are infrastructure, not
design, so they did not wait for the plan:

- **The automatic checks on GitHub had failed since the recorder merge of 2026-11-20.** A renderer probe aborted
  the whole test process on a machine with no display. The checks now run with a virtual display, and the probe
  runs in a child process. (This was found while re-verifying the setup after the audit; the audit itself had
  taken the checks to be passing.)
- The robot's two self-tests now run in `python -m ops check` and in the automatic checks.
- The installer no longer copies the missing folder, and a test checks its folder list.
- The robustness bench exits with a failure when a verdict is FAIL, so `ops eval` cannot report success for it.
- Benches print each success count with its 95 % range, and `sim/benches/stats.py` compares two results.
- The training seed can be set per experiment or with `--seed`, and each run records it.

Everything else in sections 1 and 5 is open and waits for the foundation plan.
