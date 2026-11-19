# Roadmap: idea → one order → a walking, curious pocket crab

**Strategy:** do everything possible in software first, then place **one combined order** (servos,
electronics, 3D prints, screws, cables and a beginner tool kit). Real servos aren't tested before
buying, so the simulator uses each servo's datasheet, scaled to our 5 V bus, keeps torque margins, and
the learned policy trains across a wide range of motor strengths.

## Phase A: order-ready (everything except GPU training and assembly)

| # | Milestone | Done when | Status |
|---|---|---|---|
| A1 | Simulation | Stands, 6 emotes within servo limits (`sim/check.py`) | ✅ |
| A2 | Scripted walking | Trot + crawl, 6 directions, within servo limits (`sim/walk.py`) | ✅ (crawl re-tuned for the mini) |
| A3 | Motor model | Per-servo torque-speed, back-EMF, friction, rotor inertia, backlash, latency; heat check | ✅ datasheet models (STS3032, GS-4308MG; older SCS0009 / HD-1910 kept) |
| A4 | CAD, hand-sized | 12 parts / 26 pieces, STL + STEP; walls ≥ 1.6 mm; servo fit; collision sweep (`cad/check.py`) | ✅ |
| A5 | Sim from CAD | Meshes + mass/inertia per body (0.61 kg); CAD-derived joint limits; A1/A2 still pass | ✅ |
| A6 | Electronics | [electronics.md](../hardware/electronics.md), [wiring.svg](../hardware/wiring.svg): block diagram, wiring, Radxa pin map, power budget, safety | ✅ (board positions not yet fitted in the CAD: A12) |
| A7 | Robot software | `robot/`: own Feetech driver, arm servos on the Radxa's hardware PWM, calibration (legs measured, leg directions checked, arms trimmed by eye with a travel check), safety (limits, speed, shin limit, battery, bus sag, arm guard), emotes, gait, policy, `--check` | ✅ 32/32 self-tests against emulated bus, PWM and MuJoCo |
| A8 | RL pipeline | MJX + Brax PPO, domain randomisation, Modal GPUs and the msi-1 queue, CPU smoke test | ✅ |
| A9 | Brain + pet mind | Gemma 4 (Ollama) sees + hears; reflexes / heart / mind layers; tones only, never speech | ✅ in sim ([PET_MIND.md](PET_MIND.md)) |
| A10 | Order pack | [BOM.md](../hardware/BOM.md): Evelta + robu + Amazon.in, prices incl. GST, stock status | ✅ everything in stock on 2026-11-18 (₹85k + prints) |
| A11 | Beginner docs | [ASSEMBLY.md](../hardware/ASSEMBLY.md): order → Radxa setup → servo setup → mechanics → first power-up | ✅ |
| A12 | Electronics layout in CAD | Every board modelled as a box inside the body and checked for fit (the deck is tight); dome or tray adjusted if needed | ⏳ next |

## Phase B: needs a GPU (done on Modal and msi-1)

- B1: train on a cloud GPU: `python -m modal run --detach training/modal_train.py::launch --env <env> --preset <preset> --run <run>`,
  then `fetch` and `training/export_policy.py` (see [DECISIONS.md](DECISIONS.md), 50 onwards).
- B2: bench it: `python sim/benches/robustness.py` and `python sim/benches/hurdles.py`; walking uses `policies/gait.npz`.

## Phase C: physical build

- C1: **order** ([BOM.md](../hardware/BOM.md)). Re-check stock in the carts first.
- C2: measure the servos with the calliper (the GS-4308MG especially) and update `cad/servo.py` if needed. Print the parts (robu service).
- C3: set up the Radxa (overlays, audio check), set leg servo IDs, then build, calibrate and run `robot/run.py --check` ([ASSEMBLY.md](../hardware/ASSEMBLY.md)).
- C4: emotes first, then the scripted trot, then the learned policy.
- C5: **senses layer for the pet mind**:
  - camera novelty and faces
  - microphone loudness and keyword spotting
  - touch pads
  - distance sensors (cliff → reflex stop)
  - IMU (lifted / tipped)

  These all feed the same `Heart` that already runs in the simulator.
- C6: measure the real servos (a simple swing test per joint) and update `crab/params.py`. Retrain if the
  sim and the robot disagree (closing the sim-to-real gap).
- C7: speech understanding (a small on-device speech-to-text model) so it reacts to its name. It still
  only answers in tones.
- C8: features borrowed from Jumper / MakeYourPet / Sesame, best value first: measured servo models and
  randomised delays in the sim, a stale-data / tilt / no-command hold, idle "life", one structured
  `{action, emote, sound}` message from Gemma, a keyframe studio, mood-picked gait styles, printed
  calibration gauges, swappable shells ([INSPIRATION.md](INSPIRATION.md)).

## Why this order

Each step catches mistakes before they cost money:
- **Hand-sized redesign:** the simulator showed the micro servo was too weak for the thighs (94% of
  rated torque just standing), so the thighs got a bigger servo.
- **Price check:** found the key parts sold out, so the crab was redesigned around in-stock parts. The
  simulator then showed heavy arm servos overloading the front thighs (82%), so the arms got 5.8 g servos
  (33%).
- **CAD sweep:** found the front legs hitting the arms, so the arms now point further forward and the
  joint limits come from bisection.
- **Real CAD masses:** the robot came out heavier than the estimate, and the crawl gait was re-tuned for it.
- **Pin check:** the Radxa's six hardware PWM outputs turned out free of the audio pins, so the arm servos
  need no PWM board, which wouldn't have fit in the body.
