# What we borrow from other robots

Direction: **the crab keeps its hand-sized body (about 18–20 cm)**, but gets as close as it can to
the features of these robots. Researched 2026-11-18. Links point at the file where each idea lives.

| Robot | What it is | Licence |
|---|---|---|
| [KingKong Jumper](https://github.com/KingKongRobotics/jumper) | 22-DoF crab, 40 × 40 cm, 1.8 kg, RK3576 with NPU, 25 V. RL policies (mjlab + rsl_rl + MuJoCo Warp) exported to a Rust controller that runs on the robot, in the browser and in Python. Its own docs say that controller hasn't driven the real servos yet. | Apache-2.0 |
| [MakeYourPet hexapod](https://github.com/MakeYourPet/hexapod) ("Chica") | 18-servo hexapod much bigger than ours. Servo2040/Maestro board, an old Android phone as the brain. Gait presets in a text config, printed calibration jigs. | MIT |
| [Sesame](https://github.com/dorianborian/sesame-robot/) | 8 × MG90S, ESP32, OLED face. Keyframe studio, browser simulator, web/JSON API. Its sister repo [sesame-ml](https://github.com/lukehollis/sesame-ml) trains in MuJoCo Playground with Brax PPO on a PCA9685 + IMU robot, almost exactly our stack. | Apache-2.0 |

## Ideas worth borrowing (best value first)

| # | Idea | From | Effort | What it means for the crab |
|---|---|---|---|---|
| 1 | **Measured servo models + randomised delays in the sim** | Jumper [motor model](https://github.com/KingKongRobotics/jumper/blob/main/assets/jumper/motor/README.md), sesame-ml [simulation.md](https://github.com/lukehollis/sesame-ml/blob/main/docs/simulation.md) | M | Log STS3032 step responses (they report position and load) and fit the sim to them; give the PWM arm servos random lag, 0–60 ms delay and a slew limit. The biggest factor in whether trained walking works on the real robot. |
| 2 | **Safety rules first in the robot loop** | Jumper [BUNDLE.md](https://github.com/KingKongRobotics/jumper/blob/main/deploy/BUNDLE.md), MakeYourPet config | M | Hold a pose if servo data is stale, the body tips too far, or no command arrives for 500 ms. We already have joint limits, speed limits and battery stops. Optional hardware: a wire from the Radxa to the PCA9685's OE pin as a kill switch. |
| 3 | **Train only on what the robot can sense, and gate before hardware** | Jumper [deploy/README.md](https://github.com/KingKongRobotics/jumper/blob/main/deploy/README.md), sesame-ml | S | Policies see leg encoder readings, but only *commanded* arm angles (the arm servos can't report). Reject any "walking" policy that stands still. First real test: robot lifted off the ground. |
| 4 | **Idle "life"** | Sesame [firmware README](https://github.com/dorianborian/sesame-robot/blob/main/firmware/README.md) | S | Random small claw twitches, body sway and chirps every few seconds when nothing is happening, fed into the pet mind's boredom. No display needed. |
| 5 | **One structured message from the LLM** | Sesame [companion app](https://github.com/dorianborian/sesame-companion-app) | S | Gemma returns `{action, emote, sound}`; the pet mind decides whether to do it. |
| 6 | **Keyframe studio** | [Sesame Studio](https://github.com/dorianborian/sesame-studio) | M | Pose the crab in the MuJoCo viewer, retime keyframes, save our emote format, "play on robot" button. |
| 7 | **Mood-picked gait styles** | MakeYourPet [config](https://github.com/MakeYourPet/hexapod/blob/main/chica-config-2040.txt) | S | Named presets (body height, step height, speed): *sneak*, *scuttle*, *proud*, chosen by mood. |
| 8 | **Calibration jigs + stance checker** | MakeYourPet `STL/calibration-*.stl`, Jumper `calibrate_home.py` | S–M | A printed angle gauge for the arm trim (no encoders there); a sim check that all four feet share the weight in the stand pose and no joint sits near its limit. |
| 9 | **One controls file + gamepad** | Jumper [CONTROLS.md](https://github.com/KingKongRobotics/jumper/blob/main/docs/CONTROLS.md) | S–M | Same input map for the sim viewer and the robot; optional Bluetooth gamepad. |
| 10 | **Sim-vs-robot replay test** | Jumper BUNDLE.md | S | Replay recorded controller frames and require identical outputs, catching mismatches between sim and robot code. |
| 11 | **Claude Code project skills** | Jumper [.claude/skills](https://github.com/KingKongRobotics/jumper/tree/main/.claude/skills) | S | e.g. new-emote, calibrate, deploy-to-robot. |
| 12 | **Swappable magnetic shells / hats** | Sesame [top covers](https://github.com/dorianborian/sesame-robot/blob/main/hardware/printing/stl/top-covers/README.md), Jumper skins | S | Our dome already sits on 4 magnets, so alternative domes and hats are just new prints. |

Later (L): emotes that balance themselves, from an RL policy that adds small corrections on top of a
keyframed motion (Jumper does this for its jump).

## What doesn't fit (and why)

- **Jumper's hardware scale:** jumping, 1 kg grip, 25 V pack, depth-sensor array, four mics, two TFT eyes. These don't fit an 18–20 cm crab on an 850 mAh pack.
- **Jumper's NPU / DDS / 1 kHz Rust controller:** overkill for 18 servos at 50 Hz, and unproven on its own hardware.
- **A phone as the brain:** too big and heavy.
- **Foot micro-switches:** more weight and wires in the feet. We can estimate foot contact from the STS3032 load readings instead (to be tested).
- **Talking:** Sesame's speech and lip-sync faces break the crab's never-speaks rule. If it listens, recognition runs locally.
- **ESP32 captive portal / Arduino firmware:** the ideas carry over, the code doesn't (we run Linux).
