# Code audit: body and training (pillar C, training side of pillar B)

Audited: `master` (commit 3bbc1a9) plus two UNMERGED branches, labelled "unmerged" everywhere they appear.
Method: read the files in the brief in full and cited `path:line`. Nothing was run (no tests, benches or training).

Terms, defined once. **Residual policy**: a neural network that adds small corrections to a hand-written controller.
**Domain randomisation**: training on many slightly different simulated robots so the policy does not overfit one.
**Privileged information**: simulation facts (true position, contacts) that the real robot cannot measure.

---------------------------------------------------------------------------------------------------------------

## 1. Motor skills (piece C8)

### 1.1 Common to every skill

- Control rate 50 Hz (`crab/policy_io.py:17` `CTRL_DT = 0.02`). Physics 500 Hz (`sim/models/crab.xml:4` timestep 0.002;
  training envs default `sim_dt=0.002`, `training/gait_env.py:108`). The scripted gait integrates at 500 Hz inside each
  20 ms tick (`pinch/body/common.py:17` `SUB = 10`).
- Network: Brax PPO MLP, swish hidden layers, tanh on the mean (`crab/policy_np.py:190-198`). Exported to numpy
  (`training/export_policy.py:329-344`) and checked to 1e-4 against Brax (`:347-367`). Observations are normalised with
  running mean/std and clipped to +-5 (`crab/policy_np.py:191-192`). **No history stacking** anywhere except the last
  action and (walk/emote) a 0.1 s look-ahead of the reference.
- PPO settings shared by all runs (`training/train.py:160-174`): `seed=0` fixed, observation normalisation on, clipping
  0.2, GAE lambda 0.95, grad-norm 1.0, reward scaling 1.0. The value network sees the SAME observation as the actor (no
  asymmetric critic). Per-skill settings in `configs/train/presets.toml`.
- Rewards are multiplied by the control step (0.02 s) then clipped to [-1, 10] (`training/emote_env.py:194`,
  `grasp_env.py:328`). So the walking fall penalty of -10 becomes -0.2, about two steps of good reward.
- Brax draws ONE randomised robot per parallel env at the start of a run and keeps it for all episodes
  (`brax/training/agents/ppo/train.py:97-113`, read in `.venv-rl`). A run sees 2048 or 4096 distinct robots, not a new one
  per episode. The eval env gets the same randomisation function with a different key (`:760-777`).

### 1.2 Per-policy table

| | Walk (gait-v2, `policies/gait.npz`) | Emote balance (emote-v1) | Get-up (righting-v1) | Grasp (grasp-v6) | Push steering (push2) |
|---|---|---|---|---|---|
| Status | adopted, decision 84 | adopted | adopted, decision 85 | adopted, decision 93, **sim only** (section 8) | **failed, not used** (decision 88); no `push.npz` |
| Env code | `training/gait_env.py:107-118` (inherits `emote_env.py`) | `training/emote_env.py:78-222` | `training/righting_env.py:41-132` | `training/grasp_env.py:175-358` | `training/push2_env.py:113-240` |
| Observation size | 74 | 74 | 60 | 60 | 16 |
| Action | 12 leg residuals x 0.15 rad added to the scripted gait's targets | 12 leg residuals x 0.15 rad added to the emote keyframe | 18 ABSOLUTE targets over each joint's full range (`policy_io.py:155-157`), no default pose | 12 leg residuals x 0.15 rad around a crouch + 3 absolute right-arm targets, arm eased at 2/2/4 rad/s (`policy_io.py:93-116`) | 3: walking command vx, vy, wz scaled into -0.035..0.05 m/s, +-0.03, +-0.6 (`push2_env.py:41-42,186`) at 10 Hz |
| Default pose | the scripted gait output at that instant | the keyframe pose | none (STAND only subtracted in the observation) | `GRASP_NOMINAL` (`policy_io.py:91`) | n/a |
| Episode | 600 steps = 12 s; reference clips are 6 s, then a new command is drawn | 500 steps = 10 s, emotes chained | 200 steps = 4 s | 200 steps = 4 s | 150 steps x 0.1 s = 15 s |
| Network (actor / critic) | 128x3 / 256x3 | 128x3 / 256x3 | 128x3 / 256x3 | 256,256,128 / 256x3 | 128x2 / 256x2 |
| PPO | 2048 envs, batch 256, 32 minibatches, unroll 20, 4 epochs, lr 3e-4, entropy 0.005, discount 0.97 (`presets.toml:84-98` of the gait_v2 block) | 4096 envs, entropy 0.005, discount 0.97 | 4096 envs, entropy 0.01, discount 0.97 | 4096 envs, entropy 0.01, discount 0.97 | 2048 envs, entropy 0.01, discount 0.99 |
| Steps trained | 150,000,000 (`policies/registry.toml:8`) | 40,000,000 | 39,813,120 (preset says 60M, experiment 40M) | 30,638,080 of 40M (stopped at a 3 h cap, decision 93) | 30,000,000 |
| Recorded bench result | robustness desk/rough/stuck/terrain/terrain_hard tracking 86/85/81/83/80 %, 0 falls; tidy cap 11/24 (`registry.toml:12`) | "22 emotes, 0 falls" (`registry.toml:29`) | 200/200 random falls, 0.54 s mean (`registry.toml:21`) | pick bench, 24 layouts: eraser 11/24 (scripted 2/24), pen 10/24 (9/24) (`registry.toml:45`) | reward rose but progress stayed 0, no episode succeeded (DECISIONS 88) |
| Training physics | 2 ms | **4 ms** (`emote_env.py:79`, registry env "emote") | 2 ms, 6 solver iterations (`righting_env.py:38`) | 2 ms, 6 iterations | 2 ms, 6 iterations |

(PPO block line numbers: gait_v2 preset `configs/train/presets.toml` section `[gait_v2]`; sections `[emote]`, `[righting]`,
`[grasp]`, `[push2]` give the other columns. The registry maps env names to presets in `configs/experiments/*.toml`.)

Observation terms (sizes):

- **Walk and emote (74)**, `policy_io.py:51-54,57-70`: gyro x0.25 (3), gravity unit vector (3), leg position error vs
  reference (12), leg velocity x0.05 (12), last action (12), reference legs minus default (12), reference 0.1 s ahead (12),
  reference arms (6), progress sin/cos (2). There is no explicit velocity command: the command is implied by the reference
  trajectory. The robot can measure all of these (BNO055 gyro/gravity, STS3032 position/velocity).
- **Grasp (60)**, `policy_io.py:96,119-137`: gyro 3, gravity 3, leg error 12, leg velocity 12, last action 15, arm targets 3,
  grab point relative to the claw gap (3, x20), object axis sin2a/cos2a (2), kind one-hot (3), size (3, x50), time fraction
  (1). The object terms are NOT produced by any robot code (section 8).
- **Righting (60)**, `policy_io.py:143-152`: gravity 3, gyro 3, all 18 joint positions minus STAND, all 18 joint velocities,
  last action 18. The 6 arm joints are PWM servos with no feedback (`robot/crab_robot.py:91-100`): on the robot their q is
  the last command and qd is 0. Training feeds true values (`righting_env.py:127-132`).
- **Push2 (16)**, `push2_env.py:45,232-240`: gyro, gravity, last command, thing and goal relative to the body, thing velocity,
  time left. Thing position/velocity are ground truth plus 1 % noise.

Reward terms (weights):

- **Walk (gait-v2)**, `gait_env.py:38-39` over `emote_env.py:32-43`: track_legs 1.0, track_attitude 1.5, track_height 0.5,
  **track_vel 2.5**, ang_vel_xy -0.02, foot_slip -1.0, action_rate -0.1, torque_limit -1.0, residual -0.1, termination -10.
  Gaussian kernels, velocity sigma 1.5 cm/s on a low-passed body velocity (`emote_env.py:178-182`).
- **Terrain variants** (`terrain_env.py:35-36`, not the adopted walker): action scale 0.25, track_legs 0.4, track_attitude
  0.8, track_height 0.2, residual -0.05.
- **Emote**: same table, track_vel 0, track_legs 2.0, residual -0.2.
- **Righting** (`righting_env.py:31`): upright 1.0, height 1.0, stood 2.0, action_rate -0.01, energy -0.0005, no fall penalty.
- **Grasp** (`grasp_env.py:61-62`): reach 5, align 1, grasp 4, lift 10, success 20, push -2, tilt -2, action_rate -0.01,
  termination -10.
- **Push2** (`push2_env.py:43-44`): progress 40, near 0.05, success 0.5, tilt -0.2, action_rate -0.01, lost -0.3, termination -1.

Termination: walk/emote = gravity z > -0.7 (about 46 deg tilt) or body below 37 % of stand height (`emote_env.py:192`).
Righting ends only on NaN (`righting_env.py:115,124`). Grasp ends on a fall (gravity z > -0.6 or height < 30 %), on
shoving the object over 4 cm without holding it, or NaN (`grasp_env.py:323-325`). Push2 ends on a fall.

Curriculum: none in the usual sense. Walk: 48 commands plus 9 crouched-push commands x 2 arms as fixed 6 s reference clips
(`gait_env.py:43-54,160-167`), shoves up to 0.12 m/s every 2.5 s (`registry.py:32` `_WALK_PUSH`). Terrain (not adopted): half the
robots get a gentle course, half up to `HARD` (`terrain_env.py:81-83`). Righting start: rolled 180 +-30 deg, pitched +-25
deg, any heading (`righting_env.py:77-79`). Grasp: kind 60 % stick / 40 % block / 0 % ball (`grasp_env.py:55`), position
noise +-8 mm, yaw noise +-0.25 rad (`:176`).

### 1.3 How the scripted gait relates to the learned walker

It is **reference plus residual with an imitation reward**, not pure RL. `crab/gait.py` (trot, Raibert-style foot placement)
produces the reference. `training/gait_env.py:72-109` plays it on a nominal CPU model for a grid of commands and records
joint targets, body attitude and height. The policy action is added to the reference targets (`policy_io.py:73-80`,
`emote_env.py:158`); with zero output it IS the scripted gait. Reward = follow the reference legs/attitude/height
(imitation) plus walk at the commanded speed (`track_vel`). `crab/gait_jax.py` is the same gait in JAX but only the failed
push2 env uses it; the adopted walker trains on recorded references (decisions 66-67 intended a continuous-command walker;
not built). Consequences: the learned part can only correct by +-0.15 rad; it cannot find a different gait; the reference
grid covers 48 commands, not a continuous range.

On the robot the live gait runs 0.1 s ahead on a deep copy to give the look-ahead (`pinch/body/locomotion.py:78-102`,
`GAIT_LOOK = 5` at `common.py:41`). The progress input is `(t mod 6 s)/6` (`locomotion.py:98`), matching the 6 s clips.

### 1.4 How skills are commanded and switched (pinch/body)

- Interface: the mind sends a tuple such as `("goto", xy, tol, ...)`, `("walk", vx, vy, wz, s)`, `("emote", name)`,
  `("pick", obj)`, `("push", obj, goal)`, `("drop", goal)`, `("recover",)`. `Body.start(step)` finds `_<kind>` and stores a
  Python generator; every 20 ms `tick()` takes the next joint target from it (`pinch/body/core.py:55-63,80-100`). One task
  at a time; `busy` and `ok` are the only result (`core.py:62,89`).
- Which learned policy runs is decided per tick: the walker only when arms are free and nothing is carried; while crouched
  pushing, the push policy if any, else the walker if its `covers` includes "push"; otherwise (carrying) the pure scripted gait
  (`locomotion.py:69-76`). Emotes run the emote policy unless holding something (`pinch/body/expression.py:61`). The get-up
  policy runs inside `_right_self` (`locomotion.py:190-221`), the grasp policy inside `_learned_grasp`
  (`manipulation.py:228-253`).
- **Blending: almost none.** A policy start resets its last action to zero (`locomotion.py:85-86`). When the walker is switched
  off the residual vanishes in one tick (`locomotion.py:82-84` returns the raw reference). Poses blend with a smoothstep from
  the last command (`core.py:103-109`, 0.5 to 1.5 s). A reflex (cliff, edge) cancels the running generator and starts its own
  step at once (`core.py:65-78`); afterwards `hold` is sent, which can be far from where the joints are. The only physical
  protection is the clamp in `robot/crab_robot.py:117` (0.158 rad per 20 ms).
- Evidence the hand-over is a known weakness: the UNMERGED grasp-fit branch adds "fade walking corrections in and out"
  (commit db0b241) and an idle ease-to-hold at 2.5 rad/s because "the claw jumped 0.9 rad in one tick to the hold pose"
  (unmerged diff of `pinch/body/core.py`).

---------------------------------------------------------------------------------------------------------------

## 2. Simulation actuator model (A2 / B4)

Defined in `crab/params.py`, written into `sim/models/crab.xml` by `sim/build_model.py:219-232`, adapted for MJX by
`training/robot_model.py:26-49`.

- **Type**: MuJoCo `<position>` actuator (a spring toward the target), `kv=0` (`params.py:48-50`), force clamp = `forcerange`
  (`crab.xml:30` STS3032: kp 2.0, +-0.3675 N m; `crab.xml:22` GS4308: kp 0.8, +-0.059 N m). The torque-speed line is made with
  joint damping equal to the back-EMF slope (stall / no-load speed) plus a small viscous term (`params.py:61-62`;
  `crab.xml:29` damping 0.044 N m s/rad).
- **Joint extras**: armature 5e-4 kg m2 (STS3032), 6e-5 (GS4308); friction loss 0.012 / 0.002 N m (`crab.xml:21,29`).
- **Backlash**: a second un-actuated hinge in series with each joint, range +-0.5 deg (legs) or +-1.5 deg (arms), damping
  0.00141, armature 0.000731 (`build_model.py:51-53,229-230`; `crab.xml:32-34`). Those damping and armature values are
  `0.01*S^3` and `0.01*S^4`, arbitrary.
- **Velocity limit**: not in the physics. Only the software clamp `max_cmd_speed` 9.5 rad/s x 5/6 at 5 V (`params.py:80-86,99`),
  applied by `Crab.tick` (`robot/crab_robot.py:27,117`), NOT in the training envs.
- **Delay**: `command_delay` 5 ms (legs) / 20 ms (arms) (`params.py:99,125`) is used only in design scripts `sim/check.py:29,45`
  and `sim/walk.py:27,94` through `sim/latency.py:77-86`. It is NOT in the twin's `SimBackend` (`robot/backends.py:107-111`
  writes `ctrl` directly), NOT in training, NOT in the main benches.
- **Solver**: the CPU twin uses MuJoCo defaults (Newton, up to 100 iterations); MJX training uses 4 iterations and 8
  line-search iterations (`robot_model.py:37-38`), 6 for righting/grasp/push2. Contact softness is default except grasp
  (`grasp_env.py:53`). For walking only the feet collide (`robot_model.py:39-48`).

Where the numbers come from (`params.py:89-126`):

| Value | STS3032 (12 legs) | GS-4308 (6 arm joints) | Origin |
|---|---|---|---|
| stall torque | 0.441 N m at 6 V, 0.3675 at 5 V | 0.059 N m | Datasheet / vendor listing (4.5 kg cm; arm "0.6 kg cm at 5 V" interpolated) |
| no-load speed | 11.6 rad/s (0.09 s per 60 deg) | 7.8 | Datasheet |
| rated torque | 0.147 | 0.020 | Datasheet (arm: **assumed** one third of stall, `params.py:119-120`) |
| kp | 2.4 (2.0 at 5 V) | 0.8 | **Guess**, no source (`params.py:94`: "No published system identification exists") |
| viscous, friction loss, armature | 0.006, 0.012, 5e-4 | 0.0005, 0.002, 6e-5 | **Guess** |
| backlash, delay, max command speed | 0.5 deg, 5 ms, 9.5 rad/s | 1.5 deg, 20 ms, 6.5 | **Guess** (arm backlash is a hobby-servo rule of thumb) |
| 5 V scaling | linear in voltage | linear | Reasoned, unmeasured |

Only the retired ST3215 (`params.py:128-133`) has measured numbers (Rhoban BAM). **There is no system-identification code**:
a search of the `.py` files for chirp, sysid, curve_fit, least_squares finds nothing in `sim/`, `robot/`, `training/`.
`docs/research/TRAINING_RESEARCH.md:73` lists "servo bench sysID" as step 0 of the roadmap; it is not done. The real servo's
own position loop (PID plus speed/acceleration registers, `robot/feetech_bus.py:50`) is not modelled: the sim is a
spring-damper with a torque clamp.

---------------------------------------------------------------------------------------------------------------

## 3. Domain randomisation

### 3.1 What is randomised (training)

Shared function `training/robot_model.py:68-105` (walk, emote, righting, terrain, push2 use it):

| Quantity | Range | Line |
|---|---|---|
| Friction coefficient (ONE scalar for every geom, so floor and feet move together; contact takes the larger) | 0.3 to 1.25 | 76 |
| Body masses (independent per body; inertia and centre of mass NOT scaled) | x0.85 to x1.15 | 77 |
| Servo stiffness (gain and bias together, per actuator) | x0.75 to x1.25 | 79-81 |
| Servo strength (force limit) with back-EMF damping together (damping gets x0.85-1.15 extra), per actuator | x0.65 to x1.15 | 85-91 |
| Joint armature (every dof) | x0.8 to x1.2 | 92 |
| Gravity direction ("desk tilt") | 0 to 5 deg, random azimuth | 94-96 |
| Observation noise (Gaussian, same std on every element) | 0.02 (grasp 0.01) | `emote_env.py:222`, `righting_env.py:132`, `grasp_env.py:358` |
| Random push (velocity kick on base x,y every 3 s; gait-v2 every 2.5 s) | +-0.08 m/s (gait-v2 +-0.12) | `emote_env.py:161-163`, `registry.py:32` |
| Initial leg joint offset | +-0.03 rad | `emote_env.py:143` |
| Righting start | roll 180+-30 deg, pitch +-25 deg, any yaw, joints +-0.2 rad | `righting_env.py:77-83` |
| Obstacles (terrain envs only) | hurdles 3-10 mm (x1.5 hard), ramp 0-14 deg (x1.5), 4 obstacles in a 13-20 cm ring | `crab/terrain.py:283-309`, `terrain_env.py:81-83` |
| Grasp objects: stick radius 3-5.5 mm and half length 2.5-7.5 cm, block 6-12 x 3.5-6 x 3-5.5 mm half sizes, mass 2-12 g, friction x0.75-1.15, servo force x0.7-1.15, kp x0.75-1.25, tilt 0-3 deg | as listed (no armature or damping randomisation in this env) | `grasp_env.py:128-172` |
| Push2 thing: half size 8-16 mm, mass 3-10 g, friction 0.3-0.9 | | `push2_env.py:94-107` |

The adopted walker (gait-v2) trains on FLAT ground: terrain is not in `gait_v2` (`registry.py:41`).

### 3.2 NOT randomised

| Item | Status |
|---|---|
| Floor friction | randomised, but the foot-to-floor ratio is never varied (one scalar for all geoms) |
| Link masses | randomised; inertia and centre of mass are not (battery placement is the real risk) |
| Centre of mass offset | **no** |
| Actuator gains and strength | yes |
| Joint friction (`frictionloss`) | **no** (fixed 0.012 N m, itself a guess) |
| Joint armature | yes |
| Backlash size | **no** |
| Action delay | **no** in training. The benches delay only the learned correction by one tick in half the walk episodes (`robustness.py:87,172`). `sim/latency.py` is not used in any loop |
| Observation delay | **no** |
| Observation noise | partly: one Gaussian level for everything; no per-sensor levels, no bias, no dropout, no encoder quantisation |
| IMU bias / drift | **no** |
| Random pushes | yes, xy velocity kicks only; no yaw kicks, no force pulses on legs, no payload |
| Initial state | partly (small joint offsets, stand-still start) |
| Terrain | only in the terrain envs; the adopted walker never saw a bump |
| Contact softness and solver settings | **no**; training uses 4 iterations, the bench default 100 |
| Supply voltage / battery sag | not separately; folded into strength |
| Reference gait | generated once on a nominal robot (`gait_env.py:72-109`) |

The bench draws its own randomisation of the same kind: friction 0.35-1.2, masses +-20 % (inertia also fixed), strength
-30..+15 %, stiffness +-25 %, tilt 0-4 deg (`sim/benches/robustness.py:77-86`). It is independent in draws but identical in
kind and on the same MuJoCo model, so it measures in-distribution robustness only.

---------------------------------------------------------------------------------------------------------------

## 4. State estimation and world model (piece C9)

**Own pose.** The runtime pose is `robot/localizer.py`. Inputs, in priority order:
1. ArUco marker id 7, 40 mm, on the desk tray's tab, found by OpenCV then `solvePnP` (`robot/vision.py:27-28,140-163`). Each
   fix pulls the pose with weight 0.6 (`localizer.py:45-53`).
2. Ground optical flow of the desk just ahead (phase correlation, `vision.py:197-236`), accepted below 0.1 m/s
   (`localizer.py:55-71`). The class says a textureless desk gives "no answer at all" (`vision.py:200-203`).
3. Dead reckoning from the COMMANDED velocity x 0.8 and commanded yaw rate x 0.8 (`localizer.py:19,36-43`). The IMU yaw is
   not used, so heading drifts with the commanded turn rate.
Velocity is never estimated; only the commanded velocity and the flow reading exist.

**No marker in view.** `locate()` returns None (`vision.py:143-148`; also None if the lens is uncalibrated). The pose then
follows flow or the commanded velocity, with an error budget growing 0.3 m per m (command) or 0.1 (flow), capped at 6 cm
(`localizer.py:18-19,76-77`). `Body._edge_guard` adds that sigma to its 2 cm foot margin (`locomotion.py:41-43`), so a lost
robot keeps away from the edge. `StuckSense` judges progress only from `measured_xy` (marker or flow); with neither it makes
"no judgement on progress" (`pinch/touch.py:235,262-269`).

**Desk and edges.** Constants: desk half sizes 0.35 x 0.225 m, tray at (0.18, 0.08), owner at (0, -0.30), home (-0.05, 0.08)
(`pinch/world.py:14-20`). The comment "the robot learns its own with the cliff sensors" (`world.py:15`) is not implemented (no
code writes `half_x`). Edges are also sensed by five VL53L0X downward beams (`crab/params.py:246-252`); the reflex
(`pinch/touch.py:327-389`) stops and backs off when a beam sees no desk while walking toward its side, and a failed beam
blocks that direction.

**Objects.** Four colour blobs (blue pen, pink eraser, red cap, yellow ball) by HSV threshold, the blob's lowest point
projected onto the desk plane at a fixed height per name (`vision.py:30-36,165-183`). The firmware stores xy only, yaw stays
0, `graspable=False` (`robot/firmware/senses.py:123-125`; `pinch/world.py:127-143`). Unknown things reach the mind only
through the twin (`pinch/twin/sensors.py:74-101`); the real pipeline does not produce them yet
(`firmware/senses.py:129-130`).

**People.** A Haar face detector gives found/centring (`vision.py:185-194`); `pinch/faces.py` identifies people;
`OwnerBelief` holds booleans and timestamps (`world.py:100-111`). No person position, no tracking.

**What `pinch/world.py` holds** (189 lines): `Desk` geometry and edge/tray maths, `ObjectBelief` (name, xy, yaw, last seen,
rest xy), `OwnerBelief`, battery, pose, and a rule list `situations()` (owner_down, object_near_edge, battery_low ...).

**Map.** None. No occupancy grid, no landmarks, nothing outside the desk rectangle. The only spatial memory is the danger
map: up to 40 Gaussian fear spots (`pinch/safety.py:17-76`).

**Twin versus robot.** In the twin `pose()` is the simulator's true pose (`pinch/twin/world.py:15-19`), object poses are true
(`:21-22`), perception adds 4 mm noise inside an idealised field-of-view rule (`twin/sensors.py:54-66`), the feet-down count
is true contact (`twin/world.py:49-64`), IMU and servo state are noise-free, `load` is the true actuator force and
temperature is fixed at 35 C (`robot/backends.py:113-128`). Every whole-pet bench number (tidy, pick, scenarios, soak)
therefore assumes perfect localisation.

---------------------------------------------------------------------------------------------------------------

## 5. Navigation and stuck recovery (piece C10)

### 5.1 On master

- **Path choice**: none. `Body._goto` walks a straight line with a proportional controller (gain 1.6, speed cap 6 cm/s, turn
  cap 0.8 rad/s) toward the target, clamped to the desk and fitted so every foot is 2.5 cm inside the edge
  (`pinch/body/locomotion.py:111-153`).
- **Things**: `_detour` inserts one side waypoint if the straight segment passes within 10-12 cm of one thing; a pen counts as
  a line (`locomotion.py:160-186`). A cost term keeps feet and working arm 3.5 cm from other things when choosing a stand spot
  (`manipulation.py:328-350`). Unknown flat things (cable, book edge) are simply walked over.
- **Edges**: `_edge_guard` trims velocity so no foot gets within 2 cm (plus localizer sigma) of the edge in the next 0.4 s
  (`locomotion.py:27-56`); the cliff beams latch and back off (`touch.py:327-389`).
- **Stuck detection**: `StuckSense` (`pinch/touch.py:228-283`): tilt over 7 deg while standing, 1-2 feet loaded, a foot over
  the edge, or "walking but not moving" (commanded over 1.2 cm/s, measured displacement under 8 mm in a 3 s window);
  debounced 1.5 s on, 2 s off.
- **Recovery**: the mind's `Unstick` activity (`pinch/activities/recovery.py:38-72`) grumbles, runs `("recover",)` up to
  twice, then begs up to 4 times at 20 s gaps. `_recover` has two moves: foot over the edge, shuffle toward the desk centre;
  otherwise `_climb`, a 20 mm high-step crawl for 6 s in the direction it was going (`locomotion.py:223-260`). The danger
  map learns the spot (`pinch/mind/senses.py:134-142`).
- Nothing detects an unreachable goal, a loop between two blocked headings, or a person in the way.

### 5.2 UNMERGED navigation branch (`worktree-agent-a1511a78f21d14995`, 6 commits, 17 files, +1809 lines)

- `pinch/body/navigation.py` (948 lines): A* over a 1.5 cm grid with 8 headings (about 11k states) and a Navigation2-style
  inflation costmap (lethal footprint including claws, soft margin, inflation). Obstacles: desk edges, tray, known and unknown
  things, danger-map no-go spots. The grid path is string-pulled into straight legs (`navigation.py:1-22,91-131`). Turning in
  place is allowed only where the half-way heading keeps the feet on the desk (commit 6894dec).
- `ProgressChecker` (`:385-436`): stuck = less than 2 cm gain toward the goal for 3 s AND less than 1.5 cm of movement (commit
  a399b94 added the second condition because slow lining-up tripped it).
- Recovery ladder (`:449-455,649-714`): back_up, high_step, rock, sidestep, turn; `free_leg` first if one leg's servo load is
  1.8x the others'; the escape that worked within 8 cm is tried first; at most 8 recoveries then a clean give-up; stuck spots
  kept out of plans for 3 min and written into the danger map on repeat.
- `Follow` step and activity (keep a moving toy 15-20 cm away); benches `sim/benches/navigation.py`, `stuck.py`;
  `tests/test_navigation.py`.
- **Reported numbers**: the commit messages contain NO result numbers; the first commit says the golden trace was "not
  re-recorded". The tests assert `reached` or `gave_up` per case (`tests/test_navigation.py:236-246`). UNVERIFIED whether the
  branch beats master; benches not run (not allowed here).
- Limits even if merged: world = the desk rectangle plus one tray; the checker's docstring admits the twin's position is the
  truth (`navigation.py:388-391`); on the robot the position moves only with marker fixes or flow.

### 5.3 UNMERGED grasp-fit branch (`worktree-agent-ace6b1d6a0023aad2`, 4 commits, all titled WIP)

Adds `claw_fits`, `grasp_width`, `jaw_axis` (open claw 16 mm; pen thickness against yaw), refuses picks that cannot fit
(`pick_no_fit`), `_square_up` (turn the jaws to a nudged thing), a keep-clear cost for at-risk things, arm easing and fading of
walking corrections. Constants `CLAW_OPEN = 0.016`, `JAW_AXIS0 = 0.262` and `GRASP_HALF` (pen, eraser) are scene-specific. No
numbers in the commits.

---------------------------------------------------------------------------------------------------------------

## 6. Safety (piece C11)

**Naming trap**: `pinch/safety.py` is NOT a supervisor. It is the fear map (Rescorla-Wagner danger spots) and battery hunger
(`pinch/safety.py:1-81`). The real limits are in `robot/crab_robot.py`.

What `Crab.tick` enforces on every 20 ms target (`crab_robot.py:107-133`): non-finite or wrongly sized targets become "hold"
(`:113-115`); joint ranges (CAD) clip (`:116`); per-tick step clamp = servo speed limit (`:117`); shin-versus-hip coupling
(`:119`); the two arms may not collide (`:122-124`). `_check_health` every 0.2 s (`:170-190`): servo temperature over 65 C, bus
voltage under 92 % for 1 s or pack under 6.4 V gives torque off and `SafetyStop` (`:192-199`). A servo bus read failure
stops (`:84-87`). A user emergency stop pauses torque (`:148-156`; `robot/firmware/core.py:148,160`). SIGTERM sits down and
saves (`core.py:274`). Above the runtime: edge guard, cliff reflex, stuck sense, fear map (section 5). Learned policies are
only clamped, not supervised.

NOT covered (searches for watchdog, overrun, torque_limit, overload in `robot/` and `pinch/body` find nothing):
- no servo overload or stall protection: the STS torque-limit register exists in the map (`robot/feetech_bus.py:50`) but
  nothing sets it; load is only read for gestures (and, unmerged, a caught leg);
- no tilt or fall torque cut and no protective pose before impact; falls are handled by the mind (`RightSelf`);
- no hardware or software watchdog: a hung Python process leaves all servos at their last goal; no loop-deadline monitor;
- no check that a learned policy behaves sanely (joint-speed envelope, oscillation, saturation);
- no force limit near a person; PWM arms have no feedback and are assumed where last sent (`crab_robot.py:91-100`);
- INA219 current is not used for protection; there is no thermal model, only a 65 C cut;
- no safe mode, rollback or brown-out handling beyond the voltage cut (piece A3, not built).

---------------------------------------------------------------------------------------------------------------

## 7. Generality check (desk-scene assumptions)

Counts are lines naming a scene entity (pen, eraser, cap, ball, tray, owner/home positions, desk half sizes, marker, colour
table), non-comment, by `grep -w`. Indicative, not exhaustive.

| File | Lines | Examples (path:line) |
|---|---|---|
| `pinch/body/manipulation.py` | 29 | tray special-casing in push `:30-45`; tray entrance logic `:34-44`; pen laid lengthwise in the tray `:361-366`; pen/eraser grasp offsets `:150,189` |
| `pinch/world.py` | 26 | desk 0.35 x 0.225 `:15-16`; owner, tray, tray size, home `:17-20`; `in_tray`, `tray_gap`, `feet_clear` `:53-80` |
| `sim/benches/scenarios.py` | 19 | cap, eraser, ball positions `:105-183` |
| `robot/vision.py` | 19 | marker id 7 on the tray tab `:27-28,132-148`; colours and heights `:30-36` |
| `sim/benches/pick.py` | 11 | pen and eraser only, drop into the tray `:31-48,60` |
| `sim/benches/longrun.py` | 10 | `:92,163-166,211-223` |
| `sim/benches/tidy.py` | 9 | cap and eraser only, fixed start coordinates `:32-34` |
| `pinch/body/common.py` | 8 | `LONG_THINGS` `:25`, pen half length `:22`, per-name extents `:35-38`, `V_MAX/W_MAX` `:19` |
| `training/grasp_env.py` | 5 | kinds stick/block/ball by size, fairly general `:55-57`; `GRASP_KINDS` `policy_io.py:95` |
| `sim/benches/soak.py` | 4 | `:44,79,92` |
| `pinch/twin/*.py` | 12 | twin knows `OBJECTS` `twin/core.py:83-86`; `twin/world.py` 4 |
| `pinch/body/core.py`, `locomotion.py` | 3 + 3 | `Desk()` default `core.py:22`; **fixed centre-of-mass offset 0.015*0.52** `core.py:36` |
| `training/push2_env.py` | 1 | one box "thing" `:57-60` |
| `crab/policy_io.py`, `crab/gait.py` | 1 + 1 | grasp kinds; gait com offset |
| Unmerged navigation | n/a | planner grid is the desk rectangle plus the tray (`navigation.py:94-131`); per-name `RADIUS` table `:45` |

About 160 scene-bound lines in body, bench and vision code. The **walking, emote and righting policies themselves are
scene-free** (they see only IMU, joints and a reference). The scene lives in: the `Desk` rectangle and single tray, the
tray-mounted marker (without it the robot has no pose), four named colours, the pen/eraser grasp shapes, the bench layouts
and the `sim/desk.py` world they run in.

---------------------------------------------------------------------------------------------------------------

## 8. Privileged information

Code meant to run on the robot that uses simulation truth, or observations the sensors cannot give:

| Where | What | Robot equivalent |
|---|---|---|
| `pinch/twin/world.py:15-22` | pose and object poses from `qpos` | marker + flow + dead reckoning; objects xy only, yaw 0 (`firmware/senses.py:125`) |
| `pinch/twin/sensors.py:15-30` | grasp view: grab point and axis relative to the claw gap, 2 mm noise | **not implemented on the robot**: `grasp_view_fn` is set only in the twin (`twin/core.py:74`), absent from `robot/firmware/core.py`, so the learned grab cannot run there |
| `pinch/twin/sensors.py:32-46` | `gripping()` from contacts | the robot has `grip_fn=None`; the fallback compares object height `o[1] > oz+0.004` (`manipulation.py:219`) but the robot's belief height is a constant 0.0045 (`firmware/core.py:104-108`), so "held" is never true: **no grasp confirmation on the robot** |
| `pinch/twin/world.py:49-64` | feet-down from contacts (stuck and lift detection) | STS load readings (the code comment says so) |
| `robot/backends.py:113-128` | sim IMU noise-free, servo load = true actuator force | BNO055, STS load |
| `training/grasp_env.py:340-358` | object relative position (3 mm noise), axis angle (0.05 rad), size (10 %) from true bodies | a camera pipeline that does not exist |
| `training/push2_env.py:232-240` | thing position and velocity from `xpos` (1 % noise) | camera; policy failed anyway |
| `training/righting_env.py:127-132` | true arm joint angles and speeds | PWM arms report nothing (`crab_robot.py:91-100`); the bench fakes it (`benches/righting.py:75-76`) but training did not |
| `training/emote_env.py:214-215`, `grasp_env.py:344-345` | leg angle includes the backlash hinge | assumes the STS encoder is on the output shaft; UNVERIFIED |
| Rewards (`emote_env.py:166-190`, grasp) | true body velocity, foot contact, actuator force | training-only, acceptable |

Not privileged: walk and emote observations are otherwise realistic (IMU gyro and gravity, joint encoders, own last action).
Actor and critic share one observation, so nothing leaks through a privileged critic.

---------------------------------------------------------------------------------------------------------------

## 9. Training pipeline quality

- **Seeds**: training `seed=0`, single run per policy (`train.py:170`). Variance across training seeds is unmeasured. Bench seeds
  are fixed: robustness 1000+i (`robustness.py:249-253`), righting 5000+i (`benches/righting.py:41`), tidy 1000+seed
  (`tidy.py:25`), pick `seed`. They are disjoint from training randomness but are the SAME seeds reused for every candidate, i.e.
  also the selection set. No frozen held-out set.
- **Seeds per result**: robustness 40 episodes; righting 200; tidy 24 per case; pick 24 per object; hurdles 3 robots per case
  (`hurdles.py:2-3`); emote 40 episodes of 5 emotes. No confidence intervals in the registry or decisions. Differences such as
  pen 10/24 against scripted 9/24 are inside noise; eraser 11 against 2 of 24 is real.
- **Determinism**: CPU MuJoCo, seeded numpy generators, one BLAS thread per worker (`benches/common.py:10-12`): reruns of one
  commit should repeat. Across code versions the authors report chaotic divergence (`tidy.py:26-28`; decision 65).
- **Distribution**: benches and training share simulator, actuator model and randomisation families (section 3), so they
  measure in-distribution robustness. No second physics engine, no real data.
- **What an RL reviewer would object to**:
  1. one training seed per policy, no learning-curve variance;
  2. tuning on the evaluation seeds, no frozen test set;
  3. training solver (4 iterations; 4 ms for emote) differs from the evaluation solver (100 iterations, 2 ms); decision 41
     measured the 4 ms gap: scripted gait tracking 32 % at 4 ms against 67 % at 2 ms;
  4. the adopted emote policy was trained at 4 ms (`registry.toml:27`, env "emote") while `configs/experiments/emote.toml`
     now points to `emote2`, so the adopted policy is not reproducible from the current experiment file;
  5. checkpoint metadata is wrong for some slots: `policies/righting.json` and `grasp.json` carry `action_scale 0.15` and
     `covers ["walk"]`, defaults written by `training/train.py:118-121`;
  6. fall penalty tiny after multiplying by dt (section 1.1);
  7. grasp and righting were adopted when training stopped on a time cap or step count below the preset, not at a convergence
     test (30.6M of 40M; 39.8M of 60M preset);
  8. latency, sensor bias, backlash and friction-loss are fixed at unmeasured guesses, so "0 falls" does not transfer by itself;
     the get-up bench reports 17 rad/s commanded joint speed (decision 85) while the robot clamps at about 8 rad/s, and the
     bench does not apply that clamp (`benches/righting.py:79-84`);
  9. the bench's "20 ms latency" delays only the learned correction by one tick, not the servo command
     (`robustness.py:172,195`).
- **Good practice already present**: run records with git sha, config and host (`train.py:123-127`); NaN guard
  (`train.py:149-150`); export equivalence check; styled emote variants in the bench; fault injection in the twin;
  a golden trace that pins behaviour.

---------------------------------------------------------------------------------------------------------------

## 10. Ranked weak points (what an experienced legged-robot engineer would flag)

1. **Servo model is unmeasured.** kp, damping, friction, armature, backlash and delay are guesses (`crab/params.py:94-100`) and
   there is no system-ID code. Consequence: every learned policy is tuned to a motor nobody has measured, so sim-to-real success
   is unknown until piece A2 is done.
2. **No latency, bias or per-sensor noise in training or in the main benches** (`training/robot_model.py:68-105`;
   `sim/latency.py` unused in any loop; `robot/backends.py:107-128`). Consequence: high-gain policies (the get-up above all) can
   oscillate on a real 5-20 ms bus.
3. **All whole-pet benches run on ground truth** (`pinch/twin/world.py:15-22`, `twin/sensors.py:54-66`, noise-free IMU).
   Consequence: tidy, pick and follow pass rates assume a perfect pose the robot only gets while the tray marker is visible.
4. **The learned grasp cannot run on the robot**: `grasp_view_fn` and grasp confirmation exist only in the twin; yaw is always 0
   and `graspable=False` on the robot (`firmware/senses.py:125`). Consequence: "learned grab 11/24" is a simulator-only number.
5. **Localisation hangs on one tray marker plus commanded-velocity dead reckoning; heading ignores the IMU; no map**
   (`robot/localizer.py:35-43`, `robot/vision.py:140-148`). Consequence: off the desk, away from the tray or in the dark, the
   robot has no pose and edge safety degrades to a 6 cm margin guess.
6. **The scene is hard-wired** (section 7): one 70 x 45 cm rectangle, one tray, four named colours. Consequence: a room floor, a
   second desk or an unmarked table is unsupported without rewriting `world.py`, the planner and the vision.
7. **The walker is a +-0.15 rad patch on a scripted trot trained on flat ground only** (`training/gait_env.py`,
   `training/registry.py:41`). Consequence: it cannot discover recovery or terrain gaits; adopting it cost hurdle ability
   (decision 84: bars forward to 6 mm only).
8. **The get-up policy is near bang-bang and was never trained or benched with the robot's speed clamp or with missing arm
   feedback** (`training/righting_env.py:127-132`, `sim/benches/righting.py:79-84`, `robot/crab_robot.py:117`). Consequence:
   expect failure or servo damage in the first hardware tests.
9. **Safety is thin around learned skills**: no overload or stall limit set, no watchdog, no tilt cut, no policy-envelope monitor
   (section 6). Consequence: a policy fault or a hung process leaves 18 servos driving to the last goal.
10. **Evaluation rigour**: one training seed, reused eval seeds, n = 24 comparisons without intervals (section 9). Consequence:
    small "improvements" adopted today may be noise.
11. **Skill hand-over is abrupt**: one-tick residual drop, reflex cancel then `hold` (`pinch/body/locomotion.py:82-84`,
    `pinch/body/core.py:65-78`). Consequence: jerks and falls at transitions; the unmerged grasp-fit branch already patches two
    symptoms.
12. **Navigation on master is a straight walk with one side waypoint** (`locomotion.py:111-186`) and a two-move recovery
    (`:223-260`); the A* and ladder exist only unmerged, without recorded numbers. Consequence: the owner's "stuck near a tray"
    observation is the expected behaviour of master.

---------------------------------------------------------------------------------------------------------------

## Could not determine

- Whether the real STS3032 encoder sits on the output shaft (so "link angle including backlash" in observations matches the
  bus reading). Not documented in the repo.
- Real-robot behaviour of any policy: no hardware exists.
- The unmerged navigation branch's benefit: no numbers in the commits; benches not run (instructed not to).
- Whether Brax on the GPU hosts redraws randomisation per episode (read only the local `.venv-rl` copy of Brax).
- The sim-to-real gap for friction (floor is a MuJoCo plane; the real desk surface is not measured).
- Behaviour of the walker for command values between the 48 reference points (no test found; the bench uses 6 fixed commands,
  `robustness.py:45-46`).
- Compute cost of the 74-input policy plus the look-ahead gait copy on the RK3566 (decision 62 says the tick fits; not
  re-measured).
