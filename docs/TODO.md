# TODO

*Ordered by the owner's priorities: (1) clean code and architecture, (2) a framework that can meet every
requirement, (3) the important features done and trained. Updated 2026-11-19, evening. Decisions referenced
are in [DECISIONS.md](DECISIONS.md); the research behind the order is in [APPROACH_PLAN.md](APPROACH_PLAN.md).*

## Running now

- [ ] **Modal:** walking v2 `gait-v2` (48 commands + pushing; 48M of 150M steps), learned push steering
      `push2-v1` (slow: 3.4M steps, no successes yet), learned self-righting `righting-v1` (76).
- [ ] **msi-1 GPU:** grasp-v6, then righting2-v1 (no energy cost) (queue: `C:/Users/armaa/pinch_queue.txt`).
- [ ] **msi-1 CPU:** soak, 6 simulated days × 4 h (`sim/benches/soak.py`).

## 1. Architecture

- [x] Split body, activities and mind into packages by responsibility; golden trace identical (61, 63).
- [x] Robot geometry in `crab/`; layering enforced by `tests/test_architecture.py` (68).
- [x] One test runner, `python tests/run.py` (replay, golden, states, observer, supervisor, faults,
      architecture).
- [x] Split `robot/pinch_main.py` into `robot/firmware/` + `localizer.py` + `http_api.py` (75).
- [x] Split `pinch/sim_runner.py` into `pinch/twin/` (core, world, sensors, stimuli, faults) (78).
- [x] Benches in `sim/benches/` with one shared runner and `--json` on each (81).
- [x] Removed retired code: the from-scratch walker, the old push env, the rake, the Colab path;
      `training/robot_model.py` holds what the envs share; `pinch/` and `robot/` may not import `training/` (77).
- [ ] Continuous integration: `tests/run.py` on every push (GitHub Actions, Ubuntu, CPU) (79). Written; pushing it
      needs the owner to grant the `workflow` permission (`gh auth refresh -s workflow`).

## 2. Framework

- [x] Exact replay (one simulation clock), decision log, explicit state layer (state, phases, timeouts,
      resume) (61, 63).
- [x] Behaviour-quality observer (oscillation, alternation, failure loops, state lock, repetition) (64).
- [x] Emergency stop (HTTP, voice) with resume; fuzz-tested servo supervisor (69).
- [x] Fault injection in the twin: vision, flaky detection, audio, dead cliff beam, weak leg (71).
- [x] Long soak runs with a verdict (`sim/benches/soak.py`).
- [x] Video clips as evidence (`sim/clips.py`).
- [x] **Attention:** salience, habituation, inhibition of return, sound direction → turn to look (72).
- [x] **Idle life layer:** breathing, weight shifts, varied micro-actions scaled by arousal, no repeats (74).
- [ ] **Animation layering:** clips plus gaze offsets plus affect "style" (speed, amplitude).
- [ ] Scenario language: write scenarios as data (stimulus timeline plus assertions) instead of code.
- [x] Daily routine: sleepier at night (Rest ends the evening), quiet hours by voice or HTTP, kept in memory (82).
- [x] User controls: see what it remembers, forget by kind or person, face recognition off, quiet hours (82, 83).
- [ ] Realism metrics in the observer (entropy, gaze-target agreement, jerk) with trends over versions.

## 3. Features and training

- [x] Walking: terrain-v1 in use (0 falls; 81–87 % tracking across five worlds; bars to 8–12 mm, ramps to
      15–20°) (70).
- [x] Emote balance policy (0 falls).
- [x] Self-righting (learned get-up, 0.7 s in the twin; searched swing as the fallback).
- [x] Face recognition (YuNet + SFace) and names from voice.
- [ ] **Tidying (weakest important skill):** cap 3–5/48, eraser 23–26/48 on the 48-layout bench.
  - [ ] Learned push steering (push2-v1): bench it, and wire it into Body if it beats the scripted pusher.
  - [ ] Keep the toy ball on the desk (brushed off in about half the runs).
  - [ ] Erasers slipping out of the claw in transit.
- [ ] Walking v2 (curves, diagonals, pushing): bench it against terrain-v1.
- [x] RL self-righting policy: 60/60 starts vs the swing's 34/60, 0.66 s; in Body with the swing as fallback (76, 80).
  - [ ] Bench the final righting-v1 and righting2-v1 checkpoints; keep the best.
- [ ] General motion-tracking policy for emotes (replaces per-emote balance training).
- [ ] Stuck / obstruction detector learned from fault-injected runs (CPU).
- [ ] Desk-object detector fine-tune and a sound-event classifier (later: needs real camera data).

## Needs the owner / hardware

- [ ] Measure the OLED module (hole positions, glass thickness) before printing.
- [ ] Servo bench logs for an actuator model: the biggest sim-to-real factor (research §6, item 1).
- [ ] Face recognition with two different people; touch-gesture data from the real pads.
