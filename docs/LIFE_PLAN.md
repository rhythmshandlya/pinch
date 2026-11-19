# Pinch: making it alive, and what the side audits found

This document gathers the research done on 2026-11-19 by background agents (sources are in their reports;
the key links are below), what each one means for Pinch, and the order to build it in. It also lists every
mistake the independent audits found, and whether it is fixed yet.

## 1. Mistakes found by the side audits

| # | Found | Status |
|---|---|---|
| A1 | The twin's cliff sensors looked **backwards** (−130°), so the edge reflex never fired | **fixed** (sim/desk.py; range now 1.2 m) |
| A2 | Pinch's edge limit kept the body *centre* 7 cm in, but its feet reach ~10 cm. In a 10-minute run a rear foot ended up over the front edge and Pinch stood there, lopsided, unaware | **fixed**: a foot-aware edge guard on every walking command, a foot-over-the-edge reflex, and a stuck sense with a grumble → try → ask-for-help activity. Tested in the twin: notices within 1.4 s and no longer falls off (it fell off before). It can't yet free a hanging foot by itself (it needs a high step), so it asks for help |
| A3 | The robot's install used Debian's OpenCV 4.6, which has no ArUco detector (localization would crash) | **fixed** (pip OpenCV). **Still open:** pin it below 5, because OpenCV 5 drops CascadeClassifier |
| A4 | The ball had no rolling resistance in the twin, so it rolled forever and once left the desk | **fixed** (condim 6) |
| A5 | The face visor couldn't be assembled (closed pocket, screws running along the board), and its walls were too thin | **fixed**: open back, 1.6 mm bezel, four M2 screws through spacer posts |
| A6 | grasp and push training borrowed the walking randomization, which overwrote the grippy pads' friction; Brax's auto-reset left stale episode bookkeeping behind; there was no NaN guard | **fixed** (decision 25) |
| A7 | **Firmware blockers** (software audit): <br>• the libgpiod v1/v2 API mismatch disables the cliff sensors, then the reflex walks backwards forever <br>• the reflex uses `min()` and keys on task names <br>• it runs uncalibrated without a warning <br>• the 50 Hz loop would run at ~10 Hz on the RK3566 (the gait math, vision in the loop, servos read one at a time) <br>• one I2C error kills the process <br>• an I2C race between threads <br>• the fisheye camera is uncalibrated | **open**: needs a firmware-fix pass before the robot is built |
| A8 | On the real robot, position between marker sightings is dead-reckoned from the walking command, so a stuck Pinch "thinks" it is moving. The no-progress check and the edge guard need camera-measured motion | **open** |
| A9 | The twin's collision shape leaves out the visor and models the shell half as wide as it is, so results for flips and getting high-centred are meaningless | **open** |
| A10 | Neither gait crosses a 5 mm cable, a fixed pen, the tray wall or 8–12 mm steps; a crawl with 20 mm foot lift crosses all of them | **open**: add the crawl as the first recovery reflex, and rough-terrain training (~$0.6) |
| A11 | Pinch can't right itself from its back (the visor makes it worse) | **decision**: accept "beetle asks for help", or reshape the shell before printing |
| A12 | My Modal spend tally left out grasp-v2/v3 | corrected in the morning report |

The hardware/BOM audit and the sensor review are still running; their findings will be added here.

## 2. People and relationships (your idea 1)

- **Seeing:**
  - YuNet (230 KB) finds faces and SFace (39 MB) recognizes them, both from OpenCV's model zoo (MIT/Apache).
  - They run on the Radxa CPU at a few frames per second, on fisheye frames rectified first.
  - The installed OpenCV already has both classes, so only the model files are missing.
- **Identity:**
  - Up to 8 face embeddings per person; never photos.
  - It takes 3 agreeing votes to recognize someone and 5 "unknown" votes to create a stranger. Strangers who look alike are merged.
  - Enrolment: "my name is X" for you, "Pinch, this is Alex" for others. "Forget me" deletes a person at once.
- **The relationship, per person:**
  - **familiarity F:** grows with exposure, so a visitor becomes familiar in ~3 days
  - **warmth W:** petting and play raise it, scolding and rough handling lower it
  - **fear R:** rough handling raises it, and it fades in 3 days
- **Behaviour:**
  - the favourite person gets a warmer greeting the longer they were away, plus its own little tune
  - a Shy activity for strangers
  - a Hide activity for someone feared
  - with a trusted person nearby, fear fades twice as fast
- **Twin test:** simulated identities with noisy recognition across 14 compressed days, plus a webcam test on the laptop.
- **Needs your OK:** downloading the two model files (230 KB and 38.7 MB). Should unnamed strangers be remembered for 14 days?

## 3. Self-directed play (your idea 2)

- **A PlayBook of games:** verb × toy × target, built from skills Pinch already has (push to, carry to, put on, nudge, show).
- **How it chooses:**
  - **learning progress:** Oudeyer's curiosity from the AIBO "Playground" experiments
  - **novelty:** new toys and new games
  - **competence:** the most fun is a game it wins about half the time
  - **your praise**
  - **boredom** with mastered games

  The scoring formula and its numbers are in the research report.
- **The LLM as imagination:** used at most 8 times a day, and only when something is new or Pinch is bored. It proposes up to 3 games as constrained JSON, and code filters them for safety: toys at least 10 cm from any edge, no pushing anything off the desk.
- **Memory:** favourite games are remembered across days. Learning a genuinely new motor skill is §5.

## 4. Self-preservation (your idea 3)

- **A danger map of the desk:** spots where near-falls happened, learned with a Rescorla–Wagner rule. Two near-falls make a spot no-go. The fear fades over ~8 awake hours, and "peeking" at the spot unlearns it faster.
- **Learning the cause:** Pinch learns whether the place, the action, the object or the person caused a bad outcome. It works out that the corner was the problem, not the ball.
- **Fear of heights when lifted:** it scales with how high and how long. A trusted person calms it.
- **Battery anxiety:**
  - **40 %:** a rumbly tummy
  - **25 %:** it goes home and begs to be charged
  - **10 %:** it saves its memory and says goodnight
  - **charging:** relief
- **Servo heat:** from 50 °C it walks slower; from 55 °C it takes a rest; at 62 °C it goes limp.
- **A charging dock** (magnetic pogo pins, ArUco-guided parking, about ₹1.2k) is worth adding. **Needs your OK to spend.**

## 5. Learning new motor skills over time (your question about the cloud while charging)

- **Never let an LLM drive the motors.** The local LLM takes ~4 s per decision and the control loop runs at 50 Hz. Published "LLM walks the robot" work needed GPT-4 at $2,000 in simulation only.

  When Pinch is in trouble, it escalates:
  1. reflexes
  2. learned recovery skills
  3. the LLM picks one of those
  4. it asks you
- **Nightly learning while charging:**
  1. Pinch logs the goals it failed at.
  2. At night, the LLM writes a JSON training spec from a tested library of reward terms (the Language-to-Rewards/Eureka idea).
  3. Training runs on a GPU (your RTX 4050 under WSL2, or Modal).
  4. Automatic safety gates must all pass: a quick CPU check, a NaN check, ≥80 % success, a robustness bench in the full twin, and every existing test.
  5. Then Pinch asks you "can I try?" on a mat, at reduced torque.

  About $2–4 per shipped skill on Modal, or free on your GPU. Only code and physics numbers leave the house.
- **A cheaper fix for the sim-to-sim gap:** train on the twin's own contact model.

## 6. Simulator: MuJoCo, not Isaac (your question; independently checked)

- **Why not Isaac:**
  - Isaac Lab wants 16 GB+ of VRAM and 32 GB of RAM.
  - On Modal it runs on 4×L40S at about $7.80/h.
  - It uses a different physics engine (PhysX) from the MuJoCo twin, which would add an engine-to-engine gap.
- **Where MuJoCo itself is heading:** its docs now point contact-heavy work to **MuJoCo Warp**. Warp has the cylinder–box, mesh and heightfield contacts MJX lacks, and it is 1.5–2× faster on manipulation. mjlab puts an Isaac-Lab-style interface on top of it.
- **Closest precedent:** Microduck, a small servo robot on the same RK3566 chip, trains in mjlab and checks its policies in CPU MuJoCo.
- **Plan:**
  - **Step 0 (free):** find out how much of the twin gap comes from our training model's own shortcuts.
  - **Step 1 (~$0.30):** add an `impl='warp'` switch to the environments.
  - **Step 2:** port to mjlab when the rough-terrain, self-righting or nightly-learning work starts.
- **Your RTX 4050 (6 GB):** enough for this under WSL2 (Ubuntu). Isaac Sim can't run on it.

## 7. Build order

1. **Firmware-fix pass (A7, A8).** No real robot works without it.
2. **Twin collision shape (A9), crawl reflex and rough-terrain bench (A10).**
3. **Self-preservation (§4).** It builds on the new stuck sense and edge guard.
4. **People (§2),** once you OK the model download.
5. **Self-directed play (§3).**
6. **Nightly skill learning (§5),** once GPU access is set up.
7. **Grasping:**
   - the learned grab (grasp-v3) holds things in training but does worse than the scripted grab on pens in the full twin (2/10 vs 5/10; erasers 4/6 vs 2/6), so it is kept as `policies/grasp_v3_experimental.npz` and not used
   - next: train on the twin's contact model, or fine-tune with the twin's settings
