# Decision log: Pinch

Decisions made while finishing Pinch in software, newest last. Each has the reason, so any of them can
be revisited. **Review** marks the ones worth a second look from you.

## 2026-11-18: parts and electronics

1. **In-stock redesign.** The HD-1910 and SCS0009 servos and the Pi Zero 2 W were sold out everywhere.
   The new parts are STS3032 bus servos for the legs, GOTECK GS-4308MG 4.3 g PWM servos for the arms,
   and a Radxa ZERO 3W. The heavy arm servos had put the front thighs at 82% of rated torque just
   standing; the tiny ones bring it to 33%.
2. **Arm servos on the Radxa's own hardware PWM pins** (7, 16, 18, 19, 21, 24). This needs no PCA9685
   board, and one wouldn't fit in the body anyway. The six outputs don't clash with the I2S audio pins.
3. **By-eye arm trim with a travel check.** The arm servos have no feedback, and their listing doesn't
   say clearly whether the 1000–2000 µs pulse range covers 90° or 180°. The software assumes the
   option where a wrong guess makes the arm move *less*. Trim then measures the real travel with one
   question: did the arm move 15°, 30° or 60°?
4. **Shin limit (thigh + shin ≥ −122°).** The CAD sweep found the shin brushing the hip link in the
   sideways crawl. Trimming the hip link instead would have opened its wall at the servo corner.
5. **Shopping is online only** (your instruction): Evelta, robu.in and Amazon.in. Nothing is marked
   "buy locally".

## 2026-11-19: Pinch the pet

6. **Face screen: 2.42" 128×64 white OLED (SSD1309), wide 2:1.** You asked for a long screen for the
   eyes.
   - **Review: why not a 128×32 bar?** A really long 4:1 bar (2.23" 128×32) is only 32 pixels tall, so
     eyes on it look like dashes and can't squint, blink softly or look up and down. 2:1 is the shape
     Vector's face uses.
   - **Why not colour?** Colour bar LCDs need SPI, and the Radxa's only SPI pins are already used by
     the arm servos' PWM.
   - The OLED runs on its own I2C bus (I2C4-M0, pins 27/28), so redrawing the eyes never delays the
     IMU. Its 128×64 frame takes about 23 ms to send at 400 kHz, so about 30 fps is possible.
   - It replaces the eye stalks: a raised "forehead" on the front of the dome holds it above the arms.
   - Evelta's was sold out, so it comes from Amazon.in.
7. **Touch: one MPR121 12-channel capacitive chip** (I2C 0x5A) with copper-tape pads inside the shell,
   instead of two TTP223 buttons. Several zones let Pinch tell a stroke and its direction from a tap, a
   poke or a hold. Being lifted, shaken or bumped comes from the IMU. A leg being held or pushed comes
   from the STS3032 load readings, so it costs no hardware.
8. **Two I2S microphones** (left/right on one data line): stereo, so Pinch can turn toward your voice.
   - **Listening stays local:** sound level, voice activity and prosody (energy, pitch, how fast you
     talk) give your mood. Local speech-to-text gives the words, and the local LLM gives intent.
   - **Privacy:** no audio leaves the robot or the laptop, and recordings are never stored, only what
     was understood.
9. **Situational behaviour is general, not one scripted trick.** Pinch keeps a small world model:
   - you: present, busy (talking, in a meeting, typing), stressed or happy
   - objects on the desk and where each usually lives
   - its own needs

   Detectors turn that into situations (for example "owner in a meeting", "pen out of place",
   "something near the table edge", "owner sounds sad"). A utility-based chooser picks an activity:
   go quiet, tidy, guard the edge, comfort, play, nap, seek attention or show off a trick. The local
   LLM (llama3.2:3b) turns speech into structured intents and can suggest an activity from the
   allowed list; it never moves motors directly.
10. **Review: "put the pen back in the pen stand" is scaled to Pinch's size.**
    - **What can't work:** Pinch is about 9 cm tall, and its claws reach only a few cm above the desk.
      It can't lift a 14 cm pen into a 10 cm cup. (Jumper, at 40 cm, could.)
    - **What it does instead:** it pushes and drags pens and small things (erasers, caps, a ball) back
      to a **low desk tray** it can reach: a pen groove at desk height, with a ramp lip. The tray is
      printed with the robot and doubles as Pinch's "home" spot.
    - **How it learns:** it remembers where each object usually lives, from what it sees or from you
      telling it.

## 2026-11-19 (later): manipulation goes to RL

11. **Walking v1 learned to stand still.** Pinch's commanded speeds are only a few cm/s, and the speed
    reward was so forgiving that standing earned about 60 % of it with none of walking's costs.
    - **v2 changes:** speed is measured on a smoothed (per-gait-cycle) velocity with a tight
      tolerance; the "turn rate" term (which gave standing still a free reward) is down-weighted;
      effort penalties are lighter; commanded speeds are either clearly on or off.
    - **Check before training:** in the training environment, a scripted trot now scores 14.7 against
      5.8 for standing, where v1 scored 9.8 against 9.2.
12. **Manipulation by reinforcement learning, as you suggested.** Scripted grasping of a pen failed in
    the twin:
    - The jaw's usable gap is about 14 mm, which leaves ±1–2.5 mm of tolerance.
    - Pinched above its middle, a round pen wedges out of the jaws as they rise.
    - Walking alone can't place the claw that precisely.

    The literature on legged robots pushing and handling things uses RL with heavy domain
    randomization, usually hierarchically (a locomotion policy underneath), and it transfers to real
    hardware. So:
    - **Grasp skill (RL):** Pinch arrives roughly at the object; the policy moves the arm (swing,
      lift, claw) plus small leg-driven body shifts to get the object between the jaws and lift it.
    - **Push-to-goal skill (RL)** for long things like pens.
    - **Scripted skills stay as fallbacks**, and as the approach phase before an RL skill takes over.
13. **Review: the claw may need widening.** If the RL grasp's success rate stays low, the jaw
    opening is the limit (about 14 mm usable), and the CAD claw gets longer fingers and a wider
    opening angle.
14. **Pushing scripted for now:** a claw held flat as a paddle, with detours around objects and a
    straight final push into the tray. Verified with a cap pushed 30 cm into the tray, no falls. Raking
    (pulling things in) is disabled after it flung a pen off the desk; objects near the edge make
    Pinch raise the alarm instead.
15. **The LLM is gemma4:e4b, not llama3.2:3b.** Only gemma4:e4b was installed in Ollama, and I didn't
    pull a 2 GB model unasked. Set `PINCH_LLM=llama3.2:3b` after `ollama pull llama3.2:3b` to switch;
    it will be faster on the CPU. Instant keyword rules handle the common phrases either way.
16. **Wide-angle camera (your suggestion): 160° OV5647 fisheye with a Pi Zero connector** (Amazon.in,
    ₹3,346). From 4.5 cm above the desk, one camera has to see three things: the claw's workspace right
    in front, the desk around Pinch, and your face about 60 cm up. A narrow lens can't do all three. The
    twin's eye camera is now 80° vertical (96° horizontal, roughly what's left after correcting the
    fisheye distortion).
    - **Review:** the fisheye module is bigger than the slim camera the chassis front-wall pocket was
      sized for. Measure it when it arrives and widen the pocket (`CAM_HEAD` in cad/parts.py).
17. **Where am I on the desk: an ArUco marker on the desk tray's tab.** OpenCV finds it, and solvePnP
    turns the corners into Pinch's position and heading. In the twin, the vision code on rendered
    camera images averages 8.6 mm and 0.9° of error. Between sightings, Pinch dead-reckons from its
    walking commands. Desk things are found by colour (blue pen, pink eraser, red cap, yellow ball)
    and placed on the desk plane, within 0.3–16 mm. Fiducial localization is the standard robust choice
    for a known workspace.
    - **Review:** colour works for Pinch's own toys. Arbitrary objects on your desk would need a
      learned detector (the RK3566 has a small NPU), which is not done.
18. **The face visor replaces the eye stalks** (CAD). A forehead housing on the dome holds the 2.42"
    OLED, tilted back 25° so it looks up at you.
    - Its lower edge is about 16 mm up, above the shoulder brackets.
    - The module slides in from inside the dome and two M2 screws hold it.
    - Pinch is now about 10 cm tall, was 9; the mass is unchanged (the stalks went), and the standing
      load is the same, 33 % of rated.
    - Every emote still passes.
19. **Jaw lips (CAD and twin)**: 1.2 mm ledges along the bottom of each finger's last 10 mm; they just
    meet when the claw closes.
20. **Review: picking up still needs a claw redesign.** Measured in the twin:
    - With the pen placed perfectly, the jaws close on it but can't lift it.
    - The fingers point 40° down along the pen, so near the hinge they sit about 4 mm higher than at
      the tips. They land on the pen's shoulders and pinch above its widest point, and it wedges out.
    - The lips end up beside the pen, not under it.

    The RL grasp run (reach → grasp → lift rewards, gated, with a success metric) diverged to NaN at
    5.6 M steps with 0 % success, and I stopped it (about $0.5). The fix is mechanical: tongs-style jaws
    that come down near vertical (fingers angled down from the arm, the hinge tilted to match), then
    retrain. Until then, tidying pushes, and fetch/pick-up is marked not ready.
21. **Walking: the learned policy for free walking, the scripted gait for precise moves.**
    - v2 inside its trained envelope (normal conditions): forward, sideways and turning track 91–98 %
      of the command; it never fell in 40 harsh episodes. But it stands still for commands below
      ~35 % of its top speed (the training's minimum), and on the harsh bench it tracked 44 % against
      69 % for the scripted gait.
    - v3 (training now): randomization made as harsh as the bench (friction to 0.3, motors 35 % weaker,
      desk tilt to 5°, bigger shoves) and slow commands down to 20 %.
    - On the robot, the body raises too-small commands to the policy's minimum, and precise moves
      (lining up the claw, pushing) keep the scripted gait. That's the usual hierarchical split.
22. **Tidying by pushing: scripted, but brittle → RL push next.** In the twin, pushes work:
    - the cap into the tray when asked (94 s)
    - the eraser, noticed and tidied without asking (109 s)
    - "I can't reach it, help me" when a thing sits too close to the back edge

    But every object needed a different hand-tuned threshold, and changing one broke another. Following
    the loco-manipulation literature, the next step is a learned push skill: a high-level policy that
    sends walking commands to the walking policy and moves the arm, rewarded for moving the object
    toward its goal. Not done yet.
23. **Firmware**:
    - robot/pinch_main.py runs the same brain and body as the twin.
    - Senses run in threads: touch (MPR121), cliff (VL53L0X), the camera, the stereo microphones, and
      the face OLED on its own I2C bus.
    - A local HTTP API (`/say`, `/touch`, `/status`).
    - A systemd service and an install script (deploy/).

    robot/firmware_test.py runs the deployable firmware on emulated hardware: the real I2C drivers talk
    to fake chips, the camera is the twin's render, and the mics play synthetic voices. 8/8 checks pass.
    - **Review:** the VL53L0X driver is a minimal init without ST's calibration. It's enough to tell
      the desk from a drop, but check it on the real sensor.

## 2026-11-19 (night): a claw that can pick things up

Decisions 13, 19 and 20 are replaced by 24; 21 is corrected by 27; 22 is updated by 28; 18 is corrected by 29.

24. **New claw jaws (CAD and twin).**
    - **Research first:** I looked at how small grippers pick thin things off a flat surface. The main
      approaches are:
      - fingernail or scooping tips that slide under the thing: [SPARK Hand](https://arxiv.org/html/2504.20506v1),
        DenseTact-Mini (ICRA 2024), [MIT's tactile scooping gripper](https://arxiv.org/html/2408.00610v1)
      - compliant, underactuated fingers
      - contact-triggered mode switching ([arXiv 2106.09957](https://arxiv.org/abs/2106.09957))
    - **Then I measured the old claw in the twin.** It failed for three reasons:
      - At the reach-down pose, the palm's front-bottom corner was 5.7 mm above the desk, lower than the
        top of a pen (9 mm). So the palm hit any pen lying under it.
      - The jaws met the pen above its widest point (4.5 mm up), so it squeezed out as they rose.
      - A single pivot closes like scissors. On a long pen, the jaws touched only at the narrow end of the
        V, near the hinge, and the pen twisted out when lifted.
    - **The new jaws:**
      - Blades hang about 11 mm below the palm, so the palm clears a pen lying under it by about 3 mm.
      - The blades' lower edges lie flat on the desk when the arm points 20° down.
      - TPU grip pads (printed, superglued) sit on their inner faces.
      - The moving jaw is set at an angle, so the jaws are parallel when 9 mm apart (claw at 19°). A pen is
        pinched along the whole jaw, at its widest point.
      - The tips meet at claw 0, so "claw 0 = closed" kept its meaning in every emote.
      - The maximum opening is about 18 mm mid-jaw.
    - **Why not fingernail tips:** scooping needs compliant tips pressed onto the desk and about ±1 mm
      placement. That's not realistic with by-eye-trimmed PWM servos that have no position feedback.
      Flat pads at the widest point tolerate about ±4 mm.
    - **Verified in the twin (a scripted lower, close, lift, with the thing placed between the open
      jaws):**
      - A pen was held 13/13 times, including ±4 mm off-centre and ±9° skewed.
      - Pencils and pens (6–9 mm), erasers and small blocks were lifted.
      - Things 13 mm or wider, and balls (two point contacts, so they roll out of flat jaws), were not.
    - **In the app:** "pinch, put the pen away" now picks the pen up and puts it in the tray (22 s in the
      twin). If the pick fails, it pushes the pen there instead.
    - **Review:** from random spots, the scripted pick gets hold of the pen about half the time (5/10 in
      the bench), and fewer reach the tray. That's the job of the learned grab (25).
    - **Review:** the pad grip. The twin assumes TPU against plastic at μ ≈ 1.2. If printed TPU is too
      slippery, use a strip of rubber tape.
25. **One learned grab skill for any small thing, not just a pen** (your question). During training:
    - Every environment gets its own random stick (6–11 mm thick, 5–15 cm long) or block
      (12–24 × 7–12 × 6–11 mm).
    - The policy is told the thing's kind and size, which is what the camera measures on the robot. This
      is the usual way to make a grasp policy carry over to things it never saw (e.g. UniDexGrasp).
    - What Pinch does after the grab is the brain's choice: tidy it into the tray, bring it to you, or
      play with it.

    **Fixes found on the way:**
    - **Episode bookkeeping:** Brax's auto-reset restores only the physics, so my environments' own
      values carried over from one episode to the next: the step counter (the "time" input kept
      growing), the last actions and the "held" flag. They are now reset inside the environment when a
      new episode starts.
    - **NaN guard:** a physics blow-up now ends the episode, and training stops on a NaN evaluation.
      push-v1 ran NaN for 25 minutes (about $0.45).
    - **Shove penalty:** the per-step "shove" penalty dominated and taught grasp-v2 to stay away from the
      pen. Now, pushing a thing more than 4 cm away ends the episode instead.
    - **Arm speed:** arm commands are eased like a servo (≤ 2 rad/s, claw 4 rad/s), so early training
      doesn't fling things.
    - **Friction:** the grasp environment had borrowed the walking one's randomization, which overwrites
      every surface's friction with one value (the grippy pads down to 0.3). It now has its own, which
      scales friction instead.
26. **Contact physics for small things.** MuJoCo's default contact behaves like a 20 ms soft spring, so a
    dropped 9 g pen sank about 10 mm. That's past its own radius, and it fell through the desk.
    - **In the twin:** small things and the claw pads now use 4 ms contacts (2 timesteps) with
      priority, so they win against the desk.
    - **The legs keep the default soft contact** the gaits were tuned on. A stiff foot-on-pen contact
      squeezed pens out from under the feet and shot them off the desk.
    - **In the MJX training environment:** 10 ms, since its 4 ms timestep can't go lower.
27. **Walking: the scripted gait stays Pinch's walking** (corrects 21). On the robustness bench's harsh
    world, the learned walkers tracked commanded speed less well than the scripted gait (all three never
    fell):

    | Controller | Speed tracking |
    |---|---|
    | scripted gait | 69 % |
    | walk-v2 | 44 % |
    | walk-v3 | 39 % |

    walk-v3 was also worse than v2 in a mild world. Harsher randomization alone didn't close the gap
    between the MJX training model and the full twin. walk-v2 stays in policies/ for sim-to-real
    experiments and as the walking layer of future hierarchical skills.
    - **Next step, if wanted:** train on the twin's own contact model.
28. **No learned push skill for now** (updates 22).
    - push-v1 diverged to NaN at once; this was before the NaN guard.
    - The claw now picks up sticks and blocks, so pushing is needed only for things too big for it (the
      cap, the ball).
    - The scripted pusher stays, as the fallback.
29. **The face module now actually fits** (corrects 18).
    - **The fault:** the visor pocket was closed at the back, so the module couldn't go in, and its two
      screws ran along the board instead of through its holes.
    - **Now:** the pocket opens into the dome, and the glass sits against the bezel. Four M2 screws go
      through the module's corner holes into short standoffs behind the bezel.
    - **Review (estimates):** the hole positions (±28.25, ±17.25 mm) and the glass thickness (1.5 mm).
      Measure the module before printing.
    - **Also:** the desk tray's 1 mm floor and the ramp's tapering edge are thin on purpose (printed flat,
      and robu's own minimum is 1 mm). The printability check now gives the tray its own limit.

## 2026-11-19 (afternoon): the firmware fixes from the software audit (LIFE_PLAN A7, A8)

The audit found that `robot/pinch_main.py` and its drivers would fail on day one on the real robot.
Decision 30 is the five-beam cliff sensor redesign (a separate change); these are 31 to 38. Every fix
has a check in `robot/firmware_test.py` (now 30 checks) or `robot/selftest.py` (now 38 checks).

31. **Cliff sensors and their reflex.**
    - **libgpiod:** Debian 12 ships python3-libgpiod 1.6, the v1 API, but the code called v2's
      `gpiod.request_lines`, so the cliff sensors never came up. `robot/hw_real.py gpio_output()` now
      speaks both APIs; the test brings the two sensors up through fakes of each.
    - **A failed sensor means "don't know", never "cliff".** The old code read a failure as "no desk"
      and backed off forever. Now a beam that doesn't answer, or a reading older than 0.3 s, blocks only
      forward walking (the direction the sensors guard). The driver now returns 9999 for "nothing in
      range" (a real reading) and None for "no answer" (a failure).
    - **One corner is enough:** the reflex fires when *either* beam sees no desk (max over the beams,
      not min).
    - **It keys on motion, not task names:** it fires for any non-zero walking command
      (`body.cmd_vel`), whatever the task is called.
    - **It latches:** after one back-off it doesn't fire again until both beams see desk for 0.2 s, so
      its own backing-off can't re-trigger it. While latched, forward walking stays blocked.
    - The reflex is one class, `pinch/touch.py CliffReflex`, shared by the firmware and the twin
      (`pinch/sim_runner.py`).
32. **Calibration lives in `/var/lib/pinch/calibration.json`, and the firmware refuses to start
    without it.** Before, it lived in the code folder, which every reinstall overwrote, and a missing
    file silently meant "all offsets zero". Run `calibrate.py` with the firmware's own Python, as root,
    with the firmware stopped (README, ASSEMBLY stage 6, `install.sh`). The save is atomic and fsync'd.
    `PINCH_CALIBRATION` overrides the path.
33. **The 50 Hz loop now has room to spare.** I measured each part's cost.
    - **The biggest cost was one the audit missed:** the arm-collision guard (`crab/kinematics.py
      arms_clear`, added with the new claw) took about 25 ms per tick on the laptop. That alone is more
      than the whole 20 ms budget, and the RK3566 is ~5x slower. Two fixes: a bounding-sphere test
      rules out every box pair that is far apart in one numpy operation, and the separating-axis test
      is vectorised. It now takes 0.1 ms and gives the same answer on 3,969 arm poses and 20,000
      random box pairs.
    - **Gait:** the gait still integrates in 10 substeps per tick, but solves the leg IK only after the
      last one (`Gait.step(solve=False)`). The joint angles are identical; the gait costs 0.8 ms per
      tick instead of 2.0 ms.
    - **Servos:** one STS SYNC_READ packet reads all 12 leg servos (`FeetechBus.states`) instead of 12
      round trips. `Crab.state()` reads once per tick, however many parts of the program ask.
    - **Vision is out of the loop:** the camera thread does the marker, ground flow, colour blobs and
      face, and the loop only receives the answers.
    - **Measured:** the tick's own work (`fw.timing()`, in `/status`) is 0.4 ms mean and 0.8 ms p95 on
      the laptop, against 11.6 ms before. The test requires p95 < 4 ms, a 5x margin for the RK3566. The
      firmware logs a warning if p95 goes over 15 ms on the robot.
    - **Review on the robot:** read `tick_ms` from `/status` once it runs.
34. **Hiccups don't kill it; real faults stop it safely.**
    - **I2C and bus errors** are counted per source and ridden out with the last good value: touch,
      cliff, feet, battery, IMU, and the servo reads and writes. `/status` shows the counts.
    - **A servo that stays silent** for 10 reads in a row (~1 s) becomes a safety stop (torque off), not
      a crash.
    - **No arm pulses after a safety stop.** The arm servos' PWM outputs are only driven while torque is
      on; the shutdown after a safety stop doesn't sit down (sitting down would have re-enabled them).
      The arm trim tool enables arm pulses on their own (`enable_arms`).
    - **SIGTERM** (`systemctl stop`): sit down, torque and arm pulses off, memory saved.
    - **Memory is saved every minute, fsync'd** (the file and the rename), with the write in a thread.
      Also fixed: on the robot, `PinchFirmware(memory_path=None)` meant *no memory file at all*. It now
      means the default (`PINCH_MEMORY`).
    - **The service** gives up after 5 failed starts in 5 minutes (`StartLimitBurst`), so a missing
      calibration or a dead bus can't loop forever.
35. **I2C between threads: one handle per device.** smbus2 sets the device address and then talks in
    two separate system calls. The cliff thread and the touch reads shared one handle on I2C3, so one
    device's bytes could go to the other. Touch and the cliff pair now each have their own handle; the
    kernel serialises whole transfers between handles.
36. **The fisheye camera gets calibrated, and undistorted before any geometry.**
    - `robot/calibrate_camera.py` uses a printed checkerboard (9x6 corners, 25 mm), 25 views and
      `cv2.fisheye.calibrate`. It saves to `/var/lib/pinch/camera_intrinsics.json`. Its self-test
      recovers a known lens to 0.1 px.
    - Marker corners and blob centres are undistorted first; solvePnP then sees an ideal pinhole.
    - **Without the file the camera is "uncalibrated":** it still finds things and faces, but never
      gives Pinch's position, because the edge guard trusts that position. A missing camera is now
      detected with `isOpened()`.
37. **A8: a stuck Pinch is judged on what the camera measured, not on the walking command.**
    - **The two estimates:** `Localizer` keeps two. `pose()` is what Pinch steers by: integrated, and
      pulled to each marker fix. `measured_xy()` is only what the camera measured: the last marker fix,
      carried forward by ground optical flow (`robot/vision.py GroundFlow`). Flow answers only when the
      desk has texture; on a plain desk it returns nothing rather than a wrong number.
    - **The stuck check** (`StuckSense`) uses `measured_xy()` and makes no judgement without a
      measurement.
    - **While stuck,** the walking command is not integrated at all.
    - **The edge guard** keeps an extra distance `sigma()` from the edge: how far the integrated pose
      may have drifted since the last fix (+30 % of the distance walked on commands, +10 % on flow,
      capped at 6 cm). An unsure Pinch stays further from the edge.
    - **Tested:** held in place while walking, it is called stuck from 40 marker fixes, and its
      position stays within 9 mm of the truth. Dead reckoning alone would have put it 256 mm away.
      Walking freely is not called stuck.
    - **Review:** the stuck check needs the marker or a textured desk in view. Facing away from the
      tray on a plain desk, only the tilt and feet checks work.
38. **Small fixes:**
    - The voice direction used 2048 samples per mic, but each audio chunk has 1600, so it never ran. It
      now uses 1024.
    - Sounds play on the crab audio card (`aplay -D plughw:CARD=crabaudio`).
    - The servo port is found by its stable `/dev/serial/by-id` name (`PINCH_SERVO_PORT` overrides).
    - OpenCV is pinned to `>=4.8,<5` (A3).
    - `arecord` is restarted if it dies.
    - The firmware test now starts Pinch turned 0.4 rad toward the tray. Since the claw redesign, the
      left claw hid the marker from the old start pose, so the localisation check had been failing.

## 2026-11-19 (late): sensors, body shape, training physics, budget

39. **Five cliff beams instead of two** (sensor review, LIFE_PLAN §8).
    - **Layout:** a front pair (40° down, splayed 20° outward), one at the rear (40° down) and one on each side (30° down, mounted high).
    - **Why:** Pinch walks backwards (pushing) and sideways, where nothing looked before. The CAD's two sideways windows sat so low that a side drop showed up only after the feet were over it.
    - **Shared definition:** the five beams live in `crab/params.py` `CLIFF_BEAMS`, used by the CAD windows (drilled along each beam; each verified to cut real wall), the twin's rangefinders and the firmware.
    - **Flat-desk readings in the twin:** 51 mm front and rear, 107 mm on the sides.
    - **Wiring:** all five on their own bus, I2C5-M0 (pins 29/31). XSHUT on pins 11/13/15/22/36.
    - **Cost:** 6 sensors bought (one spare); the BOM rises by ₹1,200 (estimate).
40. **The twin's body is its real size.** It was a box ±30 mm wide with a 24 mm-high "dome" and no visor. Now:
    - the chassis box is full width (±56 mm)
    - the dome is the same ellipsoid the shell is cut from
    - the face visor is a tilted box

    Flips, getting high-centred and pushing with the body now use the real shape. Standing and all 21 emotes still pass `sim/check.py`.
41. **Walking is trained in 2 ms physics (walk-v4).** A free CPU diagnosis found the gap wasn't feet-only contacts or solver iterations. It was the 4 ms timestep:

    | Timestep | Scripted gait tracking | Learned walk-v2 tracking |
    |---|---|---|
    | 2 ms (the twin's) | 67 % | 43 % |
    | 4 ms (training's) | 32 % | 36 %, even in its own training model |

    So the old policies learned in physics unlike the twin.
    - **walk-v4:** trains at 2 ms and logs the bench's own measure (the fraction of commanded speed actually walked), so reward numbers can't mislead again.
    - **Cost:** about 2× the GPU time per run.
42. **Modal budget: $100** (the owner, 2026-11-19: "not to waste, but to actually train my robot properly"). About $6.5 was spent before this.
    - Spend goes first on closing the gap between training and the twin (decision 41), then rough terrain and recovery, then grasping. Every run keeps a hard timeout (`PINCH_TIMEOUT_MIN`).
    - The owner's RTX 4050 laptop (over SSH, WSL2) will take the cheap trial runs once it's set up.
43. **Hardware audit fixes.** All are reflected in `hardware/bom.py`, `hardware/electronics.md`, `hardware/ASSEMBLY.md` (new "Checks from the hardware audit" section) and the CAD.
    - **Arm servos:** Evelta sells GS-4308MG in pairs, so the order is 4 packs, not 8 (−₹3,546).
    - **Bus adapter:** the FE-URT-1 needs a USB-C to Mini-USB OTG cable on the host port, replacing the OTG adapter.
    - **New parts:** a 2S protection board, a 0.01 Ω INA219 shunt (the 0.1 Ω one clips at 3.2 A), and a heatsink for the RK3566.
    - **Servo voltage:** the STS3032 is rated 4.8–6 V, not 7.4 V. The max-voltage register is now 6.5 V, and the bus extensions are cut short.
    - **OLED wiring:** RES gets a pull-up with a capacitor for the power-on reset pulse; DC and CS go to GND.
    - **Battery leads:** a slot through the deck over each end of the battery bay.
    - **Print files:** the STLs were regenerated after the CAD changes: 14 files, 31 pieces, TPU for the feet and jaw pads.
    - **BOM:** ₹88,566 (estimates marked).
    - **Still open:** the power switch type (does it latch?), vents if the CPU runs hot, and the real arm-servo plug and lead length.
44. **Self-preservation in the brain** (`pinch/safety.py`, `HeadHome` and `LowPower` in `activities.py`, hooks in `mind.py`).
    - **A learned danger map of the desk:**
      - A foot over the edge, getting stuck, or falling teaches fear of that spot (Rescorla–Wagner): one near-fall gives 0.48, two give 0.67, a no-go.
      - It is remembered across restarts and halves in 8 awake hours.
      - Safe passes unlearn it, and fear partly comes back an hour later (extinction is learning on top, not erasure).
      - A cap stops it fearing the whole desk.
      - Exploring and tidying avoid feared spots; tidying asks for help instead.
    - **Battery hunger (1−charge)²:**
      - Play and exploring fade as the battery drops.
      - From ~25 % Pinch walks home and pleads for charging, at 90 s intervals.
      - At 10 % it saves its memory, yawns and sleeps.
      - Charging brings relief, larger the hungrier it was.
    - **Fear of heights:** held up in the air, fear grows with time ("scared" by 10 s). A stroking hand keeps it calm, and putting it down brings relief.
    - **Verified:**
      - A unit test of the danger map.
      - A twin run draining the battery from 60 % to 5 % (HeadHome at 26 %, LowPower at 10 %, charging detected, then back to tidying; no falls).
      - Held 20 s alone vs stroked: scared vs calm.
    - **Not yet:** height from the cliff sensors (the fear scales with time, not height), a charging dock, and servo heat (needs the servos' temperature in the brain).
45. **Self-directed play** (`pinch/play.py`, the Play activity).
    - **What a game is:** verb × toy × target, built from skills Pinch already has: push a toy to a spot, carry it, or lift it up proudly.
    - **How it chooses:**
      - learning progress (is its prediction of the outcome improving: Oudeyer and Kaplan)
      - novelty (new games and new toys)
      - about 50 % wins being most fun
      - your praise: saying "good crab" right after a game teaches it that game
      - boredom with mastered games
      - 15 % of the time, a random allowed game
    - **Safety rules in code:**
      - toys at least 10 cm from any edge and not in a feared spot
      - goals at least 8 cm in
      - a game that leaves a toy within 6 cm of an edge is banned for a week
    - **Remembered** across days in memory.
    - **Verified:** a twin run of 8 self-chosen games varied them as designed, no toy left the desk and no falls. In an earlier run Pinch waved a held pen in an "excited" emote, which squeezed the pen out and flung it off the desk, so "show" is now a still, proud lift.
    - **Honest limit:** most games failed in the run, because the pick and push skills underneath are still unreliable.
46. **People and relationships** (`pinch/people.py`).
    - **Per person:** familiarity, warmth and fear, updated by who did what: petting and play raise warmth; shaking, scolding, or a stranger lifting it raise fear.
    - **Fading:** familiarity has a 90-day half-life, warmth 30 days, and fear 3 days (twice as fast with someone trusted near).
    - **Behaviour:**
      - favourites get a warmer greeting the longer they were away (love eyes and a dance after 8 h)
      - strangers get Shy: stay home, peek, warm up
      - someone feared (fear > 0.45) gets Hide: back off home, crouch, watch until they leave
      - "Forget me" deletes a person.
    - **Blame:** goes to the person seen most recently (who just walked up), at reduced confidence when several are present. A tie had blamed the owner for a visitor's shaking.
    - **Verified in the brain:** the owner liked +0.75; a rough visitor feared after shaking, and Pinch hides when he returns; four days later only shy; the owner greeted warmly after 8 h; forgetting works.
    - **Recognition:** simulated in the twin and app. On the robot it needs the face models (YuNet + SFace, ~39 MB): **your decision to download.**
47. **The firmware drives the five-beam cliff ring** (code side of 39, and of 43's firmware items).
    - **Bring-up:** `robot/drivers/i2c_devices.py cliff_ring()` holds all five in reset and wakes them
      one at a time on I2C5, giving each its own address (front-left 0x30, rear 0x31, side-left 0x32,
      side-right 0x33; front-right keeps 0x29 and wakes last). Wiring is in `robot/hw_real.py`
      `CLIFF_WIRING`, as `gpiochip3` lines. **Check them with `gpioinfo gpiochip3` on the first boot.**
    - **A sensor that doesn't come up:** it is held back in reset, so it can't answer at 0x29 for
      another one. It is reported as missing, and the other four work.
    - **Reading:** `read_ring()` starts all five measurements at once, then collects them. Done one
      after another, five 30 ms measurements would be 150 ms stale.
    - **Each beam guards its own side.**
      - Each beam guards the horizontal direction it looks in (from `CLIFF_BEAMS`). Its threshold is
        1.6x its flat-desk reading, computed from the same table: 81 mm front and rear, 172 mm on the
        sides.
      - **Reflex:** a beam that sees no desk *while Pinch walks toward its side* makes Pinch back off
        directly away from it, once. It is latched until those beams see desk again.
      - **Blocking:** a beam that sees no desk, or that is missing or stale, blocks every motion toward
        its side, never "cliff". `pinch/touch.py limit_velocity` keeps the closest allowed velocity,
        so walking along an edge carries on and only motion into it is removed.
      - **Shared:** the twin (`pinch/sim_runner.py`) and the firmware use the same class.
      - **Tested:** the firmware test checks the front-right, rear and left-side beams, walking along
        an edge, a dead rear beam, and the ring's bring-up with both libgpiod APIs (with and without a
        dead sensor).
    - **With no cliff sensors at all,** Pinch can only turn on the spot. That is the strict reading of
      "missing means stop".
    - **Ground flow fix (A8):** with Pinch held in place, the claws' motion fooled the flow into
      reporting 12 cm of travel. Now a marker fix less than 0.6 s old outranks flow, and flow faster
      than 0.1 m/s is rejected. "Stuck" is now noticed after 3.9 s (it was 6.6 s).
    - **Hardware audit, firmware side:**
      - The servos' max input is 6.5 V (`setup_servo.py`).
      - `INA219.SHUNT_OHM = 0.01`, with a new `current()`. The pack voltage never depended on the shunt
        value, so that was already right.
      - **An empty pack** (a battery safety stop) plays a `low_battery` beep, saves memory and powers
        the Radxa off (`systemctl poweroff`). The 2S protection board stays the hard floor.
      - The CPU temperature is in `/status` and logged every minute, as a warning above 80 °C.
    - **Seen in the twin:** the side beams sometimes read 10 mm, because a leg swings through the beam.
      A short reading means "desk", so it can't cause a false stop. But a leg in the beam could hide a
      real drop for that instant.

## 2026-11-19 (afternoon): tidying that works, and why it didn't

48. **Six fixes found by running the tidy scenarios, each traced to its cause.** A new bench,
    `sim/benches/tidy.py`, runs both tidy scenarios: the cap put away when asked, and the eraser noticed
    and tidied unasked. Seed 0 is the scenario layout; every other seed moves the desk things ±2 cm.
    One layout is effectively one trajectory, and the twin's contacts are chaotic, so I judge on pass
    rates. **HEAD tidied 0/4 of each and knocked 12 things off the desk; now cap 5/8, eraser 5/8, 0
    falls.**
    - **The side cliff beams looked too far out.**
      - **Problem:** at 30° down they hit the desk 148 mm from the centre, 5.4 cm beyond the feet. Any
        sideways walk the planners allow (feet ≥2 cm from the edge) tripped the reflex and aborted the
        task. 4 false edge stops in one 4-minute scenario.
      - **First fix, 45° down:** it fixed the twin, but the wall check then found 0.07 mm slivers. The
        side wall ends at z ≈ 2 mm (the hip plates swing above it), and the windows were centred at
        z = +5: the sensors sat *above* the wall, and even the old 30° window only clipped its top.
      - **Final:** the side sensors move **down into the wall, under the servo deck (z = −16)**, still
        **30° down**. The desk is hit 111 mm out, 1.7 cm beyond the feet (flat-desk reading 65 mm,
        threshold 105 mm), so they fire only on a real map error.
        - Both windows sit in solid wall and their beams leave the body unobstructed (checked: 0 mm³
          of chassis in the first 25 mm of each beam).
        - The chassis passes the wall check (no thin faces).
        - Front and rear stay at 40° (3.5 cm past the feet): walking forward is faster and needs the
          warning.
      - **Affects:** `crab/params.py`. CAD windows, twin and firmware all read that table. The assembly
        guide's side-sensor height changes with it.
    - **A reflex's success was taken for the interrupted step's.**
      - **Problem:** the edge reflexes did `cancel()` then `start(back_off)`. When the back-off ended,
        `ok` was True, and the mind read that as "the pick worked". It then tried to drop a pen it had
        never picked up.
      - **Change:** new `Body.reflex(step)`. The step it cuts short counts as failed. Used in the twin
        and in `robot/pinch_main.py`.
    - **Objects in view never counted as still.**
      - **Problem:** `see()` reset an object's "still since" clock whenever a fix differed from the
        last fix by more than 1 cm. With camera jitter (σ 4 mm) that happened in ~20 % of sightings,
        so an object Pinch was looking at was never "out of place".
      - **Change:** "moved" now means 2 cm from where it came to rest, and fixes average while it is
        still.
    - **The tray was invisible to planning.**
      - **Problem:** pick stands put a foot in the tray, whose 12 mm walls snag it, and Pinch got stuck.
      - **Change:** new `Desk.feet_clear` counts the tray as off-limits for feet, allowing 1.5 cm more
        than the desk edge. It is used by the planners: goto fitting, push line-ups, pick stands and the
        drop heading. The reflexes and the walking guard keep `feet_margin` (desk edge only).
      - **Why the split:** putting the tray into the reflexes' margin fired "foot over the edge" in
        mid-desk.
      - **Related:** `fit_feet` now searches 16 directions for the nearest clear spot. Sliding toward
        the desk's middle dragged a spot beside the tray 18 cm away. The drop tries headings until the
        stand is clear.
    - **Pushes:**
      - Near an edge, the line-up stands 3, 2, 1 or 0 cm behind the thing before declaring it
        unreachable.
      - The push eases from 4.2 to 1.5 cm/s over its last 8 cm, so a shoved cap no longer slides past
        the tray's mouth into its side wall.
    - **The twin's pen rolled.**
      - **Problem:** it was a smooth capsule with no rolling resistance. Nudged once, it rolled at
        5 cm/s off the desk.
      - **Change:** it now has hex-pen rolling resistance (condim 6, 3 mm).
    - **Tried and dropped: walking round every known desk thing.**
      - Correct in principle: a sideways crab is 19 cm wide and sweeps things in its path. Restricting
        it to things near an edge didn't help either.
      - It changed every approach, and in this bench it cost far more tidies than the knocks it saved
        (tidies 7/8 → 2/8, ball knocks 4 → 2). Reverted.
      - **Still open:** the toy ball starts 7.5 cm from the back edge and gets brushed off in about
        half the eraser runs.
49. **Modal runs are launched with `spawn`, not waited on.**
    - **Problem:** walk-v4 (at 27M steps) and grasp-v4 (still compiling) stopped when this laptop's
      client process died. `--detach` only keeps the *last triggered* function alive, and `main` was
      blocked inside `train.remote()`.
    - **Change:** `modal_train.py::launch` spawns the run and returns at once. The run lives on Modal
      under its hard timeout, and `fetch` gets the results.
    - **walk-v4 is dropped:** learning to walk from scratch tracked ~0 % at 27M steps, and the
      residual gait (decision 47) replaces it.
    - **msi-1** runs its training from a Windows scheduled task (`PinchTrain`, S4U, 4 h limit), with
      wsl.exe in the foreground. A `setsid nohup` process inside WSL died when the SSH session closed,
      because WSL shut the distro down.
50. **Walking becomes the scripted trot plus learned corrections** (`training/gait_env.py`).
    - **Problem:** walkers learned from scratch (v1–v4) tracked the commanded speed worse than the
      scripted gait in the full twin. The emote policy, which is residual on keyframes, passed the
      robustness bench.
    - **The env:** the scripted trot is the reference, recorded on the 2 ms CPU twin for 17 commands.
      - The policy adds ±0.15 rad to the leg targets. With zero output it *is* the scripted gait, so it
        can't do much worse.
      - **Rewards:** following the gait, keeping the body level and at height, and walking at the
        commanded body speed (low-pass filtered; the new `track_vel` term).
      - **Observation:** the emote one, the reference now and 0.1 s ahead plus progress, so the robot
        uses the same code.
    - **Two variants run in parallel:**
      - **gait** on Modal: velocity filter α 0.15, σ 1.5 cm/s.
      - **gait2** on msi-1, the research audit's suggestion: a filter over about a step cycle
        (α 0.06), and a kinder σ of 3 cm/s, so early policies get a gradient.
    - **On the robot (and the bench's `residual` controller):** the live scripted gait runs 0.1 s ahead
      of what goes to the servos. That gives "ahead" for free, for 0.1 s of command latency. Progress
      is the time since the command changed, over 6 s, as in training.
51. **The grasp env now matches the twin's physics** (grasp-v4, relaunched as grasp-v5).
    - **Problem:** grasp-v3 trained at 4 ms with 10 ms contacts and fell apart in the twin, which runs
      2 ms physics with 4 ms contacts.
    - **Change:**
      - **Physics:** the env is built at `sim_dt=0.002`, with the twin's stiff contacts (`STIFF`).
      - **Reset:** the drop-in settles by *time* (0.5 s), not by step count. 150 steps had been 0.6 s
        at 4 ms but only 0.3 s at 2 ms, so the thing was still bouncing when the episode began.
      - **Sizes:** each training environment draws its own sizes (`grasp_randomize`).
      - **Budget:** 30M steps on an A10G, 150 min hard timeout.
52. **Getting over things: a high-step "climb" crawl** (audit A10; `crab/gait.py` `GAITS["climb"]`,
    `Body._climb`).
    - **Measured** in the 2 ms twin, walking forward into fixed things:
      - The trot crosses none of: a 5 mm cable, a pen, an 8 mm step, a 12 mm step.
      - The crawl with its feet lifted 20 mm crosses all four.
      - Sideways, the trot already crosses all but the 12 mm step.
    - **Change:** `recover()` (run by Unstick when the stuck sense fires) now *climbs on* the way Pinch
      was walking, for 6 s, when all feet are on the desk, i.e. caught on something. A foot over the
      edge still shuffles back toward the middle.
    - **Tested** by `sim/benches/obstacles.py` through the robot's own Body: forward and sideways into a
      cable, a pen and a book edge. All six cases get over, with 0 falls. The cable is walked over
      directly; the forward pen and book cases need one climb.
    - **Robustness worlds:** `sim/benches/robustness.py --world rough|stuck` add desk clutter (1–4 mm) and a
      foot caught for 1.5 s. Over 40 episodes each, the residual gait has 0 falls and tracks 86 % and
      81 % of the commanded speed. The scripted gait does 68 % and 65 %.
53. **The residual walking policy drives Pinch** (`policies/gait.npz` = gait-v1, 60M steps).
    - **Bench, 40 randomised robots and worlds:** 0 falls; 86 % speed tracking on the flat desk (the
      scripted gait does 69 %) and a steadier body (tilt RMS 1.6° vs 2.0°). It passes on the rough
      desk and with a caught foot too.
    - **In Body:**
      - The live gait drives the servos.
      - The policy's 0.1 s look-ahead comes from a copy of the gait, run on in five 20 ms steps:
        0.8 ms, within 0.02 rad of fine steps.
      - **Rejected, a 0.1 s delay line:** it cost the tidy planners most of their precision (cap 1/8
        with the policy zeroed).
    - **Where it runs:** plain walking only, and only once Pinch has actually risen from a push
      crouch (the audit found it switching on mid-crouch, and with that the bench had 8 falls). Pushes
      keep the pure scripted gait.
    - **Other fixes from the same audit:**
      - An object's position is its latest sighting, averaged only once it has been still for 0.5 s.
        On the robot the push steers on it, and averaging lagged a pushed cap by up to 14 mm.
      - Line-up and pick-stand checks use the same margins `_goto` applies.
      - An IMU hiccup falls back to the plain gait for that tick.
    - **Not adopted:** keeping the shell, not only the feet, off the tray. The chassis clears the
      tray's 12 mm walls by ~7 mm, and the bench lost tidies with it (13/32 vs 16/32 over 16 layouts).
    - **Asking for help:** when a thing is within 8 cm of an edge and the first push fails, Pinch asks
      straight away instead of waiting for a second failure. Before, the 30 s retry cooldown let other
      tidying take over.
    - **Where it stands:**
      - **Tidy bench, 16 layouts:** cap 9/16, eraser 7/16, 0 falls. HEAD this morning: 0 of 8 for each.
      - **Still open:** the ball starts 7.5 cm from the back edge and gets brushed off in about half
        the runs.
54. **Things stay in the claw: no friction creep in the twin, and letting go properly**
    (`sim/desk.py`, `Body._let_go`, `Body._goto`, new `sim/benches/pick.py`).
    - **Pick bench** (Body picks the pen or eraser from 12 random spots each, then drops it in the
      tray): 3/12 pens and 2/12 erasers were picked, and **none reached the tray**.
    - **Why nothing reached the tray:**
      - **Creep:** with MuJoCo's default contacts (pyramidal cones, no noslip pass), a squeezed pen
        slid out of the claw under its own weight in 3.6 s, even standing still. Body kept believing
        it held it, walked to the tray and "dropped" nothing.
      - **The release:** both release routines closed the claw while lifting the arm away, which
        grabbed the pen again, so it rode back up.
    - **Changes:**
      - The twin runs 3 noslip iterations. An elliptic cone at impratio 10 also held the pen, but it
        picked worse (1/12 pens).
      - Letting go keeps the claw wide open until the arm is up.
      - While carrying, Pinch turns at 40 % speed, and checks every 0.2 s that the thing is still
        between its fingers. If not, the drop fails and the mind re-plans.
    - **Now:** pens 4/12 picked and **4/12 in the tray** (every pick lands); erasers 2/12 picked, but
      still slip out in transit. Erasers usually get pushed instead.
    - **Side effects of noslip (more exact Coulomb friction everywhere):**
      - **Tidy bench, 16 layouts:** eraser 12/16 (was 7/16: picking works now), cap 5/16 (was 9/16:
        pushing a light cap is harder without the creep that used to help it along).
      - **Behaviour scenarios: 12/12**, for the first time.
      - With firmer foot grip, the obstacle test's six cases are all walked over without climbing.
    - **Still open:** picking up is still the weakest skill. The scripted grab finds a good stand for
      a pen only from some spots (it must stand ~15 cm along the pen's axis, which the 45 cm-deep desk often can't fit), and erasers slip.
55. **Training results kept, and not adopted.**
    - **gait2-v1** (msi-1, 60M steps, the research variant: longer velocity filter, kinder tolerance)
      tracks 76 % on the 40-episode bench, against gait-v1's 86 % (both 0 falls). gait-v1 stays.
      gait3-v1 (gait-v1's rewards, bench-strength shoves, 100M steps) is training on msi-1.
    - **grasp-v5** (21M steps, stopped early: success had plateaued at ~31 %).
      - **Pick bench, 24 spots each:** 19 of 48 into the tray, against the scripted grab's 14, but it
        flicked 2 pens off the desk.
      - **In the whole pet:** eraser tidies 6/16 (scripted 12/16), cap 4/16 (5/16), scenarios 10/12
        (12/12). It is right-claw only, which changes how Pinch approaches.
      - **So** the scripted grab stays the default, and grasp-v5 is kept as `policies/grasp_v5.npz`.
        The robot couldn't use it yet anyway: it needs the thing's pose relative to the claw from a
        calibrated camera.
56. **Pick stands are searched, not assumed.**
    - **Problem:** the pick planner stood wherever facing the thing *from Pinch's current spot* put
      it. For things toward the desk's sides, that put the feet off the desk or in the tray. 17 of 32
      bench spots, all at least 9 cm from every edge, were "unreachable" before any attempt.
    - **Change:** the planner now scores every approach and takes the cheapest stand whose feet fit,
      with some room to spare, preferring less walking and turning:
      - a pen or an eraser: along its axis, either way;
      - anything else: from 24 directions;
      - either claw;
      - the arm's 0–20° outward swing.
    - **Result:** unreachable 17 → 11 of 32, and picks 8 → 10 of 32. The rest are mostly things lying
      pointed at a nearby edge (tidying pushes them instead), and grabs that close on nothing.
57. **Pinch knows faces** (your OK, 2026-11-19: YuNet + SFace from the OpenCV model zoo, 39 MB,
    SHA-256 pinned).
    - **Pipeline** (`pinch/faces.py`):
      - YuNet finds faces; SFace turns each into a 128-number fingerprint.
      - A fingerprint within cosine 0.363 (SFace's published threshold) of someone known is that
        person.
      - An unknown face seen 3 times within 10 s becomes a new person, "face-N". A stranger who keeps
        coming back becomes familiar, through `pinch/people.py`.
      - "I'm Sam", said while exactly one face is in view, names that person and marks them as the
        owner.
    - **On the robot:**
      - It runs in the vision thread at 5 Hz.
      - The gallery is saved with the memory (`faces.json`).
      - `deploy/install.sh` downloads and checks the models; they are not in git.
      - Without the models, Pinch falls back to "someone is there".
    - **One thread for OpenCV:** its DNN worker pool kept spinning after each frame and stole CPU
      from the 50 Hz control loop (tick p95 4.6 ms). With one thread it is 2.2 ms.
    - **Tested** (`robot/firmware_test.py`, 42/42):
      - On a real photo (matplotlib's public-domain portrait of Grace Hopper), a face seen a few times
        becomes a person.
      - Saying the name names them, and they are recognised again in dim light.
      - Separately: darker, brighter, mirrored, half-size, rotated 15° and noisy versions all match
        (cosine 0.94–0.98), at 10 ms a frame on the laptop.
    - **Not tested:** telling two *different* people apart. There is only one face photo here, so this
      relies on SFace's published accuracy (99.4 % on LFW). Check it on the robot with two people.
58. **Hurdles and ramps: measured, then trained** (the owner asked for "proper hurdle training like ramps").
    - **Measurement first** (`sim/benches/hurdles.py`, capability table in `docs/HURDLES.md`): one bar or
      ramp straight across the path, 3 robots per case.
      - The scripted trot gets over almost no bars.
      - gait-v1 crosses bars up to 6 mm forward and 8 mm sideways, and ramps up to ~15° forward and
        20° sideways.
    - **Robustness worlds** `terrain` and `terrain_hard` (`crab/terrain.py`): two hurdles and a ramp
      onto a plateau, in a ring round the start. gait-v1 already passes both with 0 falls (83 % / 79 %
      tracking). Average tracking hides the obstacle moments, which is why the capability test exists.
    - **Training** (`training/terrain_env.py`, run terrain-v1 on Modal, ≤3 h):
      - Every training robot gets its own course (domain randomisation of the obstacle boxes'
        positions; heights by how deep each box sits in the floor).
      - Shins collide with obstacles.
      - Larger corrections (0.25 rad; the scale travels with the exported policy as `action_scale`).
      - Following the flat reference counts less.
    - **Adopted only if** it beats gait-v1 on the capability test without losing the robustness worlds.
59. **Pinch rights itself** (audit A11; `crab/righting.py`, `Body._right_self`, the RightSelf activity).
    - **How the motion was found:** by search, not design. 3000 random two-pose swings were tried from
      on its back; 51 turned it over. Those were re-tried from 8 different starts (rolled ±15°, any
      heading, friction ×0.4–1.2, masses ±15 %). The chosen one got upright in all 8, after 0.55 s on
      average.
    - **In the pet:** being on its back breaks through any activity. Pinch squeaks, swings until it
      has been upright for 0.2 s, then stands. After two failed tries it pleads for help.
    - **One lesson per fall:** BodySense reports "upside down" on every tick while it lasts. The danger
      map now learns once per fall; before, it re-learned many times a second.
    - **Scenario "falls on its back: rolls itself back over":** upright again after 0.8 s, still on the
      desk.
    - The shell redesign (A11's alternative) isn't needed.
60. **The owner's PRD is the primary feature reference.** It is private: `docs/PRD.md` and the matrix that
    quotes it are kept locally and out of git (`.gitignore`).
    - `docs/TRACEABILITY.md` maps all of its IDs to code, tests and an honest status (first pass:
      62 Done, 109 implemented but untested, 139 partial, 103 missing, 1 hardware-only).
    - Behaviour stays explicit, testable code. RL is for motor skills where contact physics is hard to
      hand-tune: walking corrections, terrain, emote balance, grasping, possibly pushing.
61. **Exact replay, a decision log, and a test suite** (the foundation for every later behaviour change).
    - **Clock:** `Mind(clock=…)` now really drives the world model and memory. Before, the argument was
      ignored and everything was dated with wall time, so two identical runs differed. The twin passes
      simulation time on a fixed calendar (`SIM_EPOCH`, 2026-11-19 09:00 IST); the robot and the app keep
      real time.
    - **Decision log:** every activity switch is recorded. Each record holds the activity it switched
      from and to, why (first, the previous one finished, urgent, clearly better, after an interrupt),
      the top four scores, what Pinch had noticed and its feelings and needs.
      - It is kept in memory (`Mind.decisions`).
      - The twin can write it as JSON lines (`PinchSim(decision_log=…)`).
      - On the robot it goes to `decisions.jsonl` next to the memory, rotated at 5 MB.
    - **`tests/`:** plain test functions with `tests/run.py` (pytest-compatible; nothing to install).
      - **Exact replay:** same seed and inputs give the same run to the last digit; another seed gives a
        different run.
      - **Decision-log consistency.**
      - **A golden trace:** a fixed 2-minute session must reproduce every decision, motion, sound and the
        final pose. It is the safety net for refactors; re-record it only for intended behaviour changes.
    - **First refactor under that net:** `pinch/body.py` (810 lines) became the package `pinch/body/`:
      - `core`: stepping, reflexes, basic motions;
      - `locomotion`;
      - `manipulation`;
      - `expression`;
      - `common`: constants and helpers.
      The golden trace was identical.
62. **The walking tick fits the RK3566 again.**
    - **Problem:** the firmware's timing check measured walking ticks at p95 4.5–5.5 ms on the laptop,
      about 22 ms on the robot against a 20 ms budget. The cause was `Gait.step`, which runs 15 times a
      tick (10 substeps plus the policy's 5-step look-ahead) and paid numpy's per-call overhead on
      2-vectors (about 200 `norm` calls a tick).
    - **Change:** the same algorithm with scalar maths, in the same operation order.
      - **Speed:** 2.2× faster per step.
      - **Equivalence:** the trot is bit-for-bit identical to before (crawl and climb within 1e-15 rad),
        and the golden trace still matches exactly.
      - **Firmware timing:** p95 3.1 ms. The margin is thin, so the leg IK and the look-ahead are the
        next things to speed up if needed.
    - **Also:** the face detector now runs single-threaded (decision 57). On the robot it stays in the
      vision thread, outside the control loop.
63. **An explicit state layer under utility selection** (the research plan's recommended shape; decision 60).
    - **Structure:** `pinch/activities.py` (723 lines) became the package `pinch/activities/` by family:
      - `rest`, `social`, `defensive`, `recovery`, `tasks`, `play`;
      - `base`: the Activity contract.
      LIBRARY keeps its order, since ties go to the first. The golden trace was identical.
    - **The Activity contract now says, for every activity:**
      - its top-level **state**: idle, explore, social, play, alert, defensive, rest, recover, or task.
        "task" is desk work you asked for, an addition to the usual animal set;
      - **max_s**, a time limit;
      - its **phases**: steps can say `("phase", name)`. Tidy has pick / carry / push / ask_help;
        righting has roll; unstick has work_free / ask_help; play announces its game; explore has go_look.
    - **The mind:**
      - reports `state` and `phase`;
      - logs both state change and phase in the decision log (`phases` holds the sub-state history);
      - ends any activity that overruns its `max_s` (reason "timeout", then a fresh choice);
      - gives an activity that something *urgent* cut short a small preference to come back (+0.1 for
        60 s). Its decision is then labelled "(resumed)".
    - **Tests** (`tests/test_states.py`): every activity has a valid state and limits; an overrunning
      activity times out; in the twin, a tidy interrupted by a stroke resumes and logs its phases.
    - **Golden trace:** re-recorded only because the reason labels gained "(resumed)". Every decision,
      motion, sound and the final pose were otherwise identical.
64. **A behaviour-quality observer and video evidence.**
    - **Observer** (`pinch/observer.py`): reads the decision log, phases and log, and reports patterns
      that look unnatural. It never changes behaviour. The patterns:
      - oscillation (A↔B flips);
      - failure loops (the same activity restarting);
      - unexplained switches;
      - state lock (one state for over 90 % of a long run);
      - repeated activity sequences;
      - timeouts.
      Each finding has an id, an interval, evidence, a severity and a suggestion, ready to become a
      regression test.
    - **Tests:** it catches each problem when planted. Ten minutes of pet life alone have no errors.
    - **First real finding:** spontaneous tidying alternates with idling when it keeps failing. That is
      the tidy regression being bisected now.
    - **Clips** (`sim/clips.py`): GIFs from the twin, written to `media/` (not in git):
      - self-righting;
      - walking sideways over an 8 mm bar with the trained walking policy;
      - picking up a pen and putting it in the tray;
      - a minute of pet life with state, activity and feeling shown.
65. **Walking v2 and terrain results.**
    - **terrain-v1** (86M steps) against gait-v1 on the hurdle capability test:
      - bars: 8 mm forward 2/3 (was 0/3); sideways it sometimes clears 10–12 mm (was never);
      - ramps: 15° forward 3/3 (was 2/3), 20° sideways 3/3 (was 2/3);
      - one small loss: a 6 mm bar sideways, 2/3 (was 3/3).
      On the robustness bench it stays at 0 falls and 87 / 86 / 81 % tracking (flat / rough / caught
      foot), but sways more (tilt RMS 2.6° vs 1.6°). It is adopted only if the 48-layout tidy bench
      doesn't suffer from the sway.
    - **Walking v2 env** (`gait_v2`, training on Modal): 48 commands (curves, diagonals, backwards,
      speeds in between) plus 18 pushing references (paddle arm on either claw, the 11 mm crouch easing
      in). A policy whose config says `covers: [walk, push]` also steadies Body's pushes.
    - **Tidy bench, 48 layouts, for honest numbers.**
      - **Problem:** 16 layouts couldn't tell a regression from chaos. The bisect pinned the cap's
        4/16 → 1/16 drop on a refactor that is bit-identical on plain walking (decision 62); only rounding
        in rare paths changed.
      - **Reference with gait-v1:** cap 5/48 (10 %), eraser 23/48 (48 %), 0 falls. The ball and other
        things are brushed off the desk in many runs.
      - **Verdict:** pushing a light cap accurately is the weakest important skill.
66. **The gait in JAX** (`crab/gait_jax.py`).
    - **What:** the scripted gait and leg IK as pure JAX functions, vectorised over the four legs.
    - **Equivalence:** it matches the robot's numpy gait to 2e-8 rad over random commands, trot and crawl
      (`tests/test_gait_jax.py`, run with .venv-rl).
    - **Why:** training environments can now run Pinch's real walking controller inside the simulation
      batch. That is needed for a learned push skill built on the real walking stack (the old push env
      sat on the retired from-scratch walker), and for walking trained on continuous commands, not
      recorded references.
67. **Learned push steering on the real walking controller** (`training/push2_env.py`).
    - **What is learned:** only the brittle part of pushing, the steering. A 10 Hz policy chooses the
      walking command.
    - **What runs underneath:** the JAX gait, identical to the robot's, with the right claw as the
      paddle and the 11 mm crouch.
    - **Episodes start where Body's scripted line-up ends:** the thing at the paddle, the goal 8–25 cm
      ahead within ±35°.
    - **Randomised:** the thing's size (16–32 mm), mass (3–10 g) and friction, the usual robot physics,
      and shoves.
    - **Reward:** progress to the goal, the paddle staying behind the thing, an arrival bonus, and
      penalties for losing it, tilting, jerky commands and falls.
68. **Robot geometry lives in `crab/`; the layers are enforced.**
    - **What went wrong:** walking v2's and the push env's first Modal launches failed. They imported
      the paddle pose from the pet's brain package (`pinch/`), which isn't shipped to the GPU.
    - **Change:** `crab/arm.py` now holds `arm_pose`, `paddle_pose` and the push constants. Body and
      training both use it, and Body's two duplicated paddle expressions became `paddle_pose(side)`.
    - **Guard:** `tests/test_architecture.py` fails if `crab/` imports the pet, firmware, training or sim,
      or `training/` imports the pet or firmware.
69. **The supervisor, hardened; an emergency stop you can trigger.**
    - **The servo layer** (`Crab.tick`) already clamped every command: joint limits, the speed limit,
      shin vs hip, arm vs arm, and temperature or voltage triggering a safety stop. Two additions:
      - **Never send a non-number:** a NaN, infinite or wrong-length target is held at the last safe
        command and counted (`bad_targets`).
      - **A user emergency stop that can be undone:** `pause()` turns torque off at once and sends nothing
        until `resume()`, which starts again from where the joints actually are. Fault stops (heat,
        battery, bus) still end the program.
    - **Triggers:**
      - HTTP `POST /stop` and `POST /resume`;
      - saying the exact words "emergency stop". "Stop" and "freeze" stay ordinary "stop what you're
        doing" requests, so normal speech can't trigger it.
      While it is on, the mind issues no motion and the current body step is cancelled.
    - **Tests:**
      - **`tests/test_supervisor.py`:** 3000 fuzzed targets (wild, NaN, infinite, wrong length); every
        command sent was finite, within the limits, within the speed limit and kept the shin clear of
        the hip. Nothing is sent during a stop.
      - **Firmware test:** stopping mid-walk goes limp at once and the mind can't move it; resuming
        stands it back up; the spoken stop works (45/45).
70. **terrain-v1 is Pinch's walking policy** (`policies/gait.npz`; previously gait-v1).
    - **Tidying unchanged:** the 48-layout bench (run on msi-1's spare CPU) gives cap 3/48 and eraser
      26/48, against gait-v1's 5/48 and 23/48. Overall 29/96 vs 28/96: the same.
    - **Robustness, 40 episodes each:** 0 falls everywhere.

      | World | terrain-v1 | Scripted gait |
      |---|---|---|
      | Flat | 87 % | 69 % |
      | Rough | 86 % | 68 % |
      | Caught foot | 81 % | 65 % |
      | Terrain | 85 % | 66 % |
      | Hard terrain | 83 % | 65 % |

    - **Obstacles (`docs/HURDLES.md`):** bars to 8 mm forward (2/3) and to 12 mm sideways (1/3); ramps to
      15° forward and 20° sideways (3/3).
    - **Cost:** more body sway (tilt RMS 2.6–3.0° vs gait-v1's 1.6°).
    - **Golden trace:** re-recorded (an intended behaviour change).
    - **Scenarios:** 12/13. The cap tidy fails, as expected at a ~10 % cap rate.
71. **Fault injection in the twin** (`PinchSim.inject`).
    - **Faults:**
      - vision lost (no things, no faces);
      - flaky detection (half the sightings dropped, the rest 3× noisier);
      - audio lost;
      - a dead cliff beam;
      - a weak leg (servos at a fraction of their strength).
    - **Tests** (`tests/test_faults.py`): two minutes of pet life with you there under each fault. No
      falls, Pinch stays on the desk, the mind keeps choosing and the observer finds no errors. With dead
      microphones it obeys nothing it couldn't hear; a dead side beam blocks walking toward its side.
    - **Unchanged without faults:** with no fault injected nothing changes (the golden trace is
      identical).
    - **Ordered to-do list:** `docs/TODO.md`.
72. **Attention** (`pinch/attention.py`, the Orient activity).
    - **Stimuli:** a voice (with the direction the two microphones give), a touch, a person appearing or
      returning after a minute, and a loud noise each get a salience: base strength by kind × how strong
      this one is.
    - **The research plan's mechanisms:**
      - salience fades with a 3 s half-life;
      - **habituation:** the same stimulus from the same place counts for less each time, recovering with
        a 60 s half-life;
      - **inhibition of return:** what it just looked at is damped ×0.35 for 20 s;
      - **hysteresis:** focus switches only to something 25 % more salient.
    - **Gaze:** the eyes follow the focus whenever no activity has aimed them in the last 3 s.
    - **Orient** (alert state): turns toward a fresh, off-axis focus, looks for 1.5 s, then marks it
      attended. Its utility only beats idling, so a voice doesn't interrupt a job you asked for.
    - **Tests** (`tests/test_attention.py`): habituation and recovery; inhibition of return and no
      flicker; a voice from the left makes Pinch look and turn toward it and then move on; a voice during
      a requested tidy doesn't interrupt it. The golden trace is unchanged.
    - **Not yet:** seeing motion and new things in the camera feed as stimuli. That needs the vision
      pipeline to report them.
73. **Candidates benched and not adopted.**
    - **terrain-hard-v1** (92M steps; half the robots on bars to 15 mm and ramps to 21°):
      - better than terrain-v1 over sideways bars (6 mm 3/3, 10 mm 2/3, 15 mm 1/3);
      - slightly worse up a 15° ramp forward (2/3).
      Mixed and marginal, so terrain-v1 stays.
    - **emote2-v1** (msi-1, 42M steps, emotes in 2 ms physics): 0/40 falls with the same tilt as
      emote-v1 (1.4° RMS, 7.4° worst). Its actions differ from v1's, but there is no measurable gain, so
      emote-v1 stays.
    - **Bench:** `sim/benches/robustness.py` now takes `PINCH_EMOTE_POLICY` to bench an emote candidate without
      overwriting the report.
74. **The idle life layer** (`pinch/body/life.py`; the research plan's idle aliveness).
    - **When:** whenever no motion step runs, small procedural motion is added on top of the held pose.
    - **What:**
      - breathing, as before;
      - a slow, wandering weight shift between the legs;
      - now and then (random Poisson timing) a micro-action: a claw flex, a shoulder twitch, one leg
        resettling, or a held still moment. Never the same one twice running.
    - **Liveliness** is the mind's arousal: livelier means more often and bigger; after five quiet
      minutes it settles further.
    - **Small and repeatable:** legs move at most 0.03 rad, arms 0.15 rad. It is seeded from the run's
      seed in the twin, so replay holds, and fresh on every boot on the robot.
    - **Tests** (`tests/test_life.py`):
      - at least 5 kinds of micro-action in 10 minutes, none repeated back-to-back;
      - excited makes 1.3× more than calm;
      - 3 minutes idle in the twin with 0 falls;
      - nothing starts during a motion step.
    - **Golden trace:** re-recorded (an intended change).
75. **The firmware split by responsibility** (`robot/pinch_main.py` 580 lines → parts).
    - **Parts:**
      - `robot/firmware/`: core (start-up, the 50 Hz loop, emergency stop, saving, status) and senses
        (the face, cliff, vision and audio threads, the emulated poll, applying results);
      - `robot/localizer.py`: the pose estimator, with no firmware dependencies;
      - `robot/http_api.py`: the HTTP API.
    - **Entry point:** `robot/pinch_main.py` stays what systemd runs and re-exports the old names. Nothing
      outside changed.
    - **Checks:** no code lost (line check), firmware test 45/45, selftest 40/40, the emulated entry point
      runs, test suite 26/26.
76. **A learned self-righting policy, trained to compare with the searched swing** (`training/righting_env.py`;
    the research plan's training item 3).
    - **Start:** on its back at a random orientation (rolled 180 ± 30°, pitched ± 25°, any heading), joints
      scattered ± 0.2 rad around the stand pose, the usual physics randomisation.
    - **Policy:** 50 Hz targets for all 18 joints over each joint's full range; no reference motion.
    - **Contact:** the whole robot (shell, legs, claws) touches the floor, never itself.
    - **Reward:** being upright, then standing at the stand height (and staying there); light costs for
      jerky and wasteful motion. 4 s episodes; success is upright and at height for a full second.
    - **Check:** the searched swing played into the env turns it upright (z 0.99) within 1.2 s, so the env
      allows what we know works.
    - **Runs:** righting-v1 on Modal (A10G, 60M steps); righting2-v1 on msi-1 without the energy cost, to
      see whether the penalty holds back getting up.
    - **Adoption:** only if it beats the swing over many starts in the twin; then it replaces the swing in
      `Body._right_self`.
77. **Retired code removed** (the architecture TODO).
    - **Removed:** the from-scratch walker (`policies/walk.npz`, its env, `training/policy_controller.py`,
      `sim/walk.py --policy`, `robot/run.py --policy` and `Crab.run_policy`, the robustness `--legacy`
      option, its observation helpers in `crab/policy_io.py`), the first push env (`training/push_env.py`,
      superseded by push2), the disabled rake, and the Colab path (notebook and bundle; training runs on
      Modal and msi-1).
    - **Renamed:** `training/crab_env.py` → `training/robot_model.py`. It now holds only what every env
      shares: the model loader and the physics randomisation. The righting env uses it too.
    - **Layering:** `pinch/` and `robot/` may not import `training/` (added to the architecture test).
      `Crab.run_policy` did, inside a function.
    - **Checks:** tests 26/26 (golden trace identical), firmware 45/45, selftest 40/40, the quick robustness
      bench passes for walk and emote, every env builds, `sim/walk.py` runs.
78. **The twin split by responsibility** (`pinch/sim_runner.py` 319 lines → `pinch/twin/`).
    - **Parts:**
      - `core.py`: building the world, robot, body and mind; the 50 Hz loop.
      - `world.py`: where things are (pose, desk things, claw tip, feet, moving things by hand).
      - `sensors.py`: the camera's sightings, the cliff beams, the grasp view, the grip.
      - `stimuli.py`: talking, touching, sounds.
      - `faults.py`: injected failures.
    - **Imports:** every user now imports `PinchSim` from `pinch.twin`; the old module is gone rather than
      kept as a shim.
    - **Checks:** a line check finds only one line gone (a stray class attribute `faults = {}`, which every
      run overwrote), tests 26/26 with the golden trace identical, and the app and bench tools import.
79. **Continuous integration** (`.github/workflows/tests.yml`): `tests/run.py` on every push and pull
    request.
    - **Status:** written and checked against Linux, but not yet pushed. GitHub only accepts a workflow file
      from a login with the `workflow` permission, and the owner has to grant that.
    - **Runner:** GitHub's Ubuntu runner, Python 3.13, CPU only. It installs mujoco 4.0.1, numpy, pillow and
      headless OpenCV. The JAX gait test skips itself there.
    - **Golden trace:** exact only on the platform that recorded it. On msi-1's Linux the suite was 25/26:
      the golden session drifted 1.3 s at its tenth decision, from MuJoCo's floating point differing between
      Windows and Linux. The trace now records its platform and is skipped elsewhere. `test_replay` still
      checks exact replay on every platform.
80. **The learned get-up replaces the searched swing** (`policies/righting.npz` = righting-v1 at 8.8M steps).
    - **Bench** (`sim/benches/righting.py`): 60 random starts on its back (rolled 180 ± 30°, pitched ± 25°,
      any heading; friction ×0.4–1.2, masses ± 15 %, servo strength ×0.7–1.1), in the CPU twin's physics.
      Both controllers are handed over to the stand pose after 0.2 s upright, as Body does.
      - swing: 34/60 upright and standing, 0.86 s mean, 0.90-percentile 1.01 s. It was searched over ± 15°
        rolls only.
      - policy: **60/60**, 0.66 s mean, 0.90-percentile 0.72 s.
      - **Robot sensing:** the bench gives the policy what the firmware can read. The six PWM arm servos
        report nothing, so their angle is the last command and their speed 0. The result held (60/60,
        p90 0.76 s).
    - **In the twin's scenario** (on its back on the desk): upright again after 0.7 s (the swing: 0.8 s).
    - **Wiring:**
      - `Body._right_self` uses the policy when it is exported and falls back to the swing otherwise.
      - Observation and action mapping live in `crab/policy_io.py` (`righting_observation`,
        `righting_targets`), shared by the env, the bench and Body.
      - The twin and the firmware load it with `crab.policy_np.load_policy`.
    - **Next:** the run continues to 60M steps; its final checkpoint replaces this one if it does at least as
      well on the bench. Real-robot risk: servo speed and gear play on a violent motion. It is only used
      while on its back, and the swing remains the fallback.
    - **Checks:** tests 26/26, firmware 45/45, selftest 40/40.
81. **The benches in one place** (`sim/benches/`).
    - **Moved:** robustness, hurdles (was `hurdle_test`), obstacles, tidy (was `tidy_bench`), pick, righting,
      soak and scenarios (was `pinch_scenarios`). `sim/` keeps the model building and viewing tools.
    - **Shared:** `common.run_parallel` replaces eight copies of the process-pool code. `common.save_json`
      gives every bench `--json`; soak's old `--out` still works.
    - **References updated** in code and docs.
    - **Checks:** righting, tidy, soak, pick, quick robustness and obstacles run from their new places;
      hurdles and scenarios import; tests 26/26.
82. **The daily routine and quiet hours** (`pinch/routine.py`).
    - **Night (22:00–06:00):** sleepiness builds three times faster. The existing Rest activity then ends
      the evening on its own; there is no separate sleep schedule to keep in step with it.
    - **Quiet hours,** set by the owner:
      - by voice: "quiet hours from 10 pm to 7 am", and "no quiet hours" to clear them. They are parsed
        from the raw text, because the rule parser drops digits and speech recognisers write numbers as
        digits;
      - over HTTP: `POST /quiet_hours {"start": 22, "end": 7}` or `{"off": true}`.
      - Inside them Pinch is quiet exactly as in a meeting: no sounds, quiet activities only.
      - They are stored in the memory file's `settings`, so they survive restarts; `/status` shows them.
    - **Local time:** the robot uses its own clock and time zone. The twin fixes India's +5.5, so a run
      replays the same on any machine; `memory.mark_routine` used to read the host's time zone.
    - **Tests** (`tests/test_routine.py`):
      - parsing, and windows that run over midnight;
      - at 23:00 with quiet hours set, two minutes of stroking and talking give no sounds, while the same at
        noon gives some;
      - sleepiness builds about three times faster at night.
    - **Firmware test:** quiet hours set over HTTP make it quiet now, a bad hour gets a 400, and clearing
      them works (46/46).
    - **Golden trace:** unchanged, since the session is at 09:00.
83. **The owner's controls over what Pinch keeps** (`pinch/mind/privacy.py`, HTTP API).
    - **See:** `GET /memory` shows what it remembers: the owner's name and facts, the people it knows
      (names and how it feels about them), homes, likes, how many episodes, danger spots and games, routine
      counts, and settings. Never face fingerprints or episode text.
    - **Forget:** `POST /forget {"what": ...}`: people, owner, facts, episodes, homes, places (learned
      fears), games, routines or everything; `person:<id>` for one person.
      - Forgetting people also deletes their fingerprints from the firmware's face gallery, saved at once.
      - "everything" keeps the owner's settings (quiet hours, recognition switch).
      - Unknown kinds get a 400.
    - **Face recognition off:** `POST /settings {"face_recognition": false}` stops the firmware
      fingerprinting faces (it still notices that a face is there). Stored with the settings. Existing
      fingerprints stay until people are forgotten.
    - **Tests:** `tests/test_privacy.py` (summary, each kind, everything keeps settings, the switch), and a
      firmware check over HTTP (47/47).
84. **Walking v2 (gait-v2) becomes the walking policy** (`policies/gait.npz`; terrain-v1 stays stored).
    - **What it is:** the residual walker trained on 48 commands (curves, diagonals) plus crouched pushing
      with a paddle arm, with bench-strength shoves; 150M steps on Modal.
    - **Tidying** (same 24 layouts, today's code):

      | Walking policy | Cap | Eraser |
      |---|---|---|
      | terrain-v1 | 3/24 | 12/24 |
      | terrain-v1 walking, gait-v2 pushing | 7/24 | 10/24 |
      | gait-v2 for both | **11/24** | 11/24 |

      Its plain walking lines up better for the push too, which is why the split loses.
    - **Robustness:** level with terrain-v1, 0 falls, tracking 86 % desk, 83 % terrain, 81 % stuck.
    - **Desk obstacles** (cable, pen, book; forward and sideways): all crossed. The book head-on now needs a
      climb (49 s instead of 6 s).
    - **Cost:** lower synthetic hurdles. Bars forward to 6 mm (was 8), sideways to 8 mm partial (was 12),
      ramps forward to 10° (15° 1/3), sideways 20° unchanged. `terrain-push-v1` (training on Modal: the
      hurdle worlds plus the same commands and pushing) is meant to get both back.
    - **Framework:**
      - `load_policies()` returns every learned skill by name (`SKILLS`); `PINCH_<FILE>_POLICY` benches a
        candidate in the whole pet.
      - Body chooses per tick between the walking policy and a separate push policy, if one is exported.
    - **Golden trace:** re-recorded (an intended change).
85. **The get-up from 40M steps replaces the 8.8M one.**
    - **Bench, 200 random starts:**
      - righting-v1 at 40M: **200/200**, 0.54 s mean, 0.90-percentile 0.56 s;
      - the 8.8M one: 198/200, 0.66 s;
      - the swing: 107/200, 0.84 s.
    - **Real-robot risk, measured:** its targets change much faster than the swing's (95th percentile
      17 rad/s against 3; median 1.6 against 0.5), close to bang-bang. The twin's servo model limits force
      and damping, so the motion is physically bounded, but expect current spikes during a get-up.
      Get-ups are rare and short. If the servo bench shows trouble, retrain with a stronger action-rate
      cost (righting2 explores the other direction: no energy cost).
86. **Standing clear of the other things on the desk** (`Body._bystander_cost`; `bystanders_fn` is the mind's
    `things_on_desk()`, on the robot as in the twin).
    - **Diagnosis:** the toy ball goes off the desk while Pinch is at the tray, not on the way there. It is
      hit 10–14 cm from Pinch's centre: by the feet when it stands to pick or drop, and more often by the
      arms (the claw swinging in to grab, the paddle while pushing).
    - **Done:** pick and drop stances now prefer spots where every foot stays 3.5 cm from the other things.
    - **Measured on the same 24 layouts:** tidying unchanged (cap 11/24, eraser 11/24). Knock-offs in
      eraser runs fell from 15 to 12, unchanged in cap runs. On 8 traced runs: balls off 6/8 instead of 8/8.
    - **Not done:** the arms' sweep. The same cost over the claw's path is the next step.
87. **What the public repository holds.**
    - **Local only now** (gitignored, kept on the owner's machine):
      - the working notes (morning report, TODO, the life plan with audit notes) and the old pet-mind
        write-up;
      - the raw scraped vendor price pages;
      - the archived first design (a 16 MB zip) and its servo notes;
      - the benched policy candidates. Published policies are only the ones Pinch runs (emote, gait,
        righting) plus terrain-v1, the documented alternative walker.
    - **Removed:**
      - the `brain/` prototype, superseded by `pinch/`. Its sound synthesiser moved to `pinch/sounds.py`.
      - the two 5 MB scripted-walk GIFs.
    - **Added:** four compressed simulation clips in the README (pet life, pen to tray, get-up, a 6 mm bar).
      They come from `sim/clips.py`; the hurdle clip records the hurdle bench's own episode and shows its
      verdict.
    - **History:** git history still has the removed files. None of them are secret (the private PRD
      was already rewritten out of history).
88. **Learned push steering (push2-v1) failed; not used.**
    - **Setup:** 30M steps on Modal. A 10 Hz command policy over the JAX gait was to steer a pushed thing to
      a goal (decision 67).
    - **Result:** reward rose from −21 to +7, but it learned only to stay near the thing (episodes are no
      longer "lost"). Progress toward the goal stayed at 0 and no episode succeeded.
    - **Instead:** pushing improved from the other side. gait-v2 learned crouched pushing in the walking
      policy itself (decision 84: cap 3/24 → 11/24) under the scripted push planner.
    - **If revisited:** the progress reward needs shaping that pays for moving the thing, not for being near
      it (the "near" term dominated), and a curriculum from short pushes.
    - **msi-1 queue:** after grasp-v6 come righting-smooth-v1 (5× action-rate cost, for decision 85's
      current spikes) and a second terrain_push run (a seed replicate of terrain-push-v1).
89. **The arms in the stance cost, and a second arm for pushing.**
    - **Change:** the bystander cost (86) now also covers the working arm, shoulder to claw tip, for pick
      and drop stances.
    - **Tried, then narrowed:** choosing the push arm to spare bystanders cost more than it saved: cap 7/24
      instead of 11/24, and no fewer knock-offs.
    - **Kept for pushing:** the near arm as before, and the far arm only when the near one can't line up from
      the desk. That case used to be an outright "unreachable".
    - **Measured, same 24 layouts:** cap 11/24, eraser 12/24. Runs with something knocked off: 14 and 14
      (13 and 12 before), within noise. Golden trace identical.
    - **Conclusion:** where Pinch stands can't keep the ball on the desk. It rests where Pinch has to work at
      the tray. The fix is behavioural: clear the work area first (move a rollable thing away from the tray
      and the edge before pushing or dropping there), or let `guard_edge` catch it sooner.
90. **An industrial project setup: one operations layer for checks, training jobs on any GPU, and parallel
    work.**
    - **Why:** training ran through three ad-hoc paths (a Modal script, a queue file on msi-1, local commands);
      runs didn't record their commit; environments were listed twice; there was no lint, no hooks, and no way
      for several agents to work and train at once without colliding.
    - **Change:** `docs/ARCHITECTURE.md` is the reference. `pyproject.toml` holds dependency sets and tool
      settings; ruff (350 findings fixed, including a missing type import in `robot/http_api.py`); pre-commit
      (ruff, TOML/YAML, a private-file and attribution guard, layering tests); CI runs the guard, ruff and the
      suite. Training: one environment registry (`training/registry.py`), presets as data
      (`configs/train/presets.toml`), experiments as data (`configs/experiments/`), and a `run.json` per run
      (commit, host, resolved config, state, last metrics). Operations: `python -m ops` submits, queues,
      dispatches, monitors, fetches, exports and evaluates jobs on four backends (local; SSH over Tailscale,
      Linux or Windows/WSL; Modal; SkyPilot), keeps a ledger, checks hosts and creates worktrees. Parallel work:
      one worktree and branch per task, claimed areas (`ops/areas.toml`), shared files merged by the
      integrator. Policies: `policies/registry.toml` records each adopted policy's run, evidence and decision.
      Agents: AGENTS.md and CLAUDE.md; Pinch workflow skills; Google DeepMind's official MuJoCo skills vendored
      from the pinned mujoco package; official Anthropic plugins enabled.
    - **Evidence:** every tool chosen is established and active (stars and last commit checked): ruff 51k, uv 92k,
      pre-commit 16k, pytest 15k, Tailscale 39k, SkyPilot 10.8k (NSDI '23), anthropics/skills 198k,
      google-deepmind/mujoco 18k. The table with alternatives is in ARCHITECTURE.md section 7.
    - **Custom code, and why:** the SSH backend exists because SkyPilot's SSH Node Pools install k3s with
      NVIDIA's GPU Operator, which doesn't list WSL2, and msi-1 is a Windows laptop. On Windows a job runs under a
      one-shot S4U Scheduled Task, because an OpenSSH session kills its children on disconnect.
    - **Measured:** golden trace identical after the lint fixes; tests pass (35, with the new ops tests); a smoke
      run through the registry and experiment path wrote a complete `run.json` (state done, commit recorded).
    - **Rejected:** Hydra for configs (a dependency and indirection the configs don't need yet), a tracking
      server (workers would need to reach it; `run.json` can be imported into MLflow later), an editable install
      (it would make worktrees import each other's code).
91. **Long-duration runs: a whole day of full autonomy and a week of memory, in minutes.**
    - **Why:** the soak bench covered hours. A full simulated day in physics would take ~7 h on CPU (the twin
      runs at 3.5-4x real time; the mind alone at 1000-1500x).
    - **Evidence:** fast-forward plus warm-started sampled windows, as in SMARTS (Wunderlich et al., ISCA 2003);
      accelerated testing that skips uneventful stretches (Feng et al., Nature 615, 2023).
    - **Change:** `sim/benches/longrun.py day|week`. The day: the real mind on a no-physics desk for 24 h
      (stand-up, meeting, lunch, a visitor, night), plus five 10-minute twin windows warm-started from its state.
      The week: 7 days with a reboot each night through the memory file. Checks: observer, sleep at night, quiet
      hours, meetings, variety, physics safety; memory, people, fear, danger map, routines, likes. Known mind bugs
      are strict expected failures (fixing one turns its test red until it is removed from the list).
      `tests/test_long_day.py`, `tests/test_long_week.py` run shortened versions (~90 s together).
    - **Measured:** it found four real bugs (night sleep, attention-seeking loops, fear restored on reboot, things
      kicked off while turning); three are fixed in decision 94. Week: guest familiarity decay ratio 0.9549 vs
      0.9548 expected, guest fear ratio 0.250 (3-day half-life), stand-up learned as a routine on day 2.
    - **Rejected:** 24 h of physics (7 h per day); a faster twin clock or coarser mind tick (they change behaviour).
92. **Emote variation and touch nuance.**
    - **Why:** emotes were identical every time, touch reactions ignored where and how often Pinch was touched,
      and any touch woke it.
    - **Evidence:** layered motion with mood as a time/size pass (Snibbe, Scheeff & Rahardja 1999; Knight & Simmons,
      HRI 2015); an animation engine tracked by a learned policy (Grandia et al., RSS 2024, Disney's bipedal
      character); habituation characteristics (Rankin et al. 2009, updating Thompson & Spencer 1966); Stanley's
      habituation model on a robot (Marsland, Nehmzow & Shapiro 2000); dual-process sensitisation (Groves &
      Thompson 1970).
    - **Change:** each emote play gets a seeded style (tempo, size, arm asymmetry and lag, mirror, body turn) from
      the whole mood (arousal, valence, sleepiness; the twin and firmware pass it), within joint, shin and speed
      limits; legs get half the size change and no lag; back-to-back repeats are pushed apart. Touch gestures carry
      their pads: ReactTouch reacts by side and region (look and turn toward the hand, the touched claw answers, a
      back stroke settles), habituates (60 s recovery, 15 % floor; rough touch dishabituates and sensitises), and
      follows wake rules (`any` / `gentle` default / `deep`). The robustness bench plays variants.
    - **Measured:** emote robustness, 40 episodes: 0 -> 0 falls, tilt RMS 1.4 -> 1.2 deg, worst 7.4 -> 7.0 deg;
      88 extreme variants pass `sim/check.py`'s servo criteria (worst 96 % of the speed limit, as the plain clips).
    - **Rejected:** side-specific new emotes (would change the balance policy's training set); lag on the legs.
93. **grasp-v6 adopted; the gentler get-up at 20M not.**
    - **grasp-v6** (30.6M steps on Modal, stopped at its 3 h cap): pick bench, 24 layouts each: eraser into the
      tray 11/24 (scripted grab 2/24), pen 10/24 (9/24), 0 falls, pen knock-offs 2 (1). Now `policies/grasp.npz`.
    - **righting-smooth-v1** (20.6M of 40M on msi-1, its 2 h cap): 200/200, 0.54 s, but commanded joint speed
      p95 20.2 rad/s vs 18.4 for righting-v1 - not gentler. Kept local; the full 40M run (righting-smooth-v2) is
      training on msi-1. The righting bench now reports commanded joint speed (p95 and peak).
94. **Three mind bugs the long runs found, fixed; nested interruptions resume.**
    - **Night sleep:** Rest ended once sleepiness fell below 0.05 and each 10 s nap took 0.08 off, so a night was
      ~1 minute of nap per 16 of idling. At night Rest now sleeps on until morning (touch, voice and the owner still
      preempt it) and pays sleepiness off 4x more slowly; one rest covers a night (watchdog 9 h).
    - **Attention-seeking loop:** SeekAttention won every ~75 s for hours. Unanswered bids now back off
      exponentially (90 s, 3, 6, 12 ... up to 20 min), reset when the owner pays attention - the
      escalate-then-give-up pattern of a needy pet, not a nag. Shy stays one shy spell while a stranger is here
      instead of restarting every 10 s.
    - **Fear restored on reboot:** the danger map's decay was never saved; it is now written back every minute.
    - **Resume stack:** an interrupted activity that was waiting to resume was forgotten when the next interrupt
      came; interrupted activities now stack (3 deep), and a waiting one gets a fresh window when what interrupted
      it restarts.
95. **Curiosity about unknown things and motion.**
    - **Why:** vision knew only 4 named things and faces; nothing noticed new objects or motion.
    - **Evidence:** approach-retreat novelty management (Gordon, Fonio & Ahissar 2014, J. Neurosci.); novelty as a
      threat prediction unlearned by safe encounters (Akiti et al. 2022, Neuron); the Wundt curve as a difference
      of two sigmoids (Berlyne; Saunders & Gero 2001); releasing stimuli plus motivation (Arkin, Fujita et al., AIBO).
    - **Change:** `pinch/novelty.py`: a sensor-agnostic `Observation(kind, xy, velocity, size, confidence,
      signature)`; novelty (x0.55 per safe close look, remembered per signature, half recovers in 3 days) and
      wariness (0.3 + 0.6 x timidity, lowered by safe looks); distance kept 9 cm + 16 cm x wariness; startle
      proportional to closeness x speed x looming. `pinch/activities/curiosity.py`: Watch (gaze and turn follow
      motion; allowed when quiet) and Investigate (freeze, stop-go slow approach, pause, 2+ sides, retreat, revisit
      until familiar; edge, tray and danger-map aware; resumes what it interrupted). Twin props (box, tube, toy)
      that can appear, vanish and move, perceived with noise and field of view; the firmware passes observations
      through. The new activities draw their tie-break noise from their own stream.
    - **Measured:** 5 seeds: 3 looks, closest 18 -> 15 -> 13 cm, 2-3 sides per look, familiar after ~125 s, 0 falls,
      0 edge recoveries; gaze tracking error 0.03; a toy rushing to 12 cm at 0.45 m/s startles, slow motion at 33 cm
      does not. The robot's camera pipeline must provide these observations (unknown blobs with desk position,
      size, tracked velocity, score and an appearance signature); not built yet.
    - **Rejected:** a pure novelty drive (no caution); learning-progress curiosity (fits skill learning, not
      encounters); putting unknown things into the world model's named objects (Play and GuardEdge would act on them).
96. **terrain-push-v2 not adopted; gait-v2 stays the walker.**
    - **Why:** one walker for hurdles, ramps and crouched pushing (decision 84's terrain_push env) was meant to get
      terrain-v1's obstacle skill back without losing gait-v2's tidying.
    - **Evidence:** the adoption gate (docs/dev/TRAINING.md): the experiment's own benches, same seeds and layouts.
    - **Measured** (144M steps, A10G; `ops eval configs/experiments/terrain-push.toml gait_tp2`): terrain world 0
      falls, 86 % speed tracking; desk 0 falls, 87 %; forward ramps 10 deg 3/3 (scripted 1/3); tidy, 24 layouts:
      eraser 16/24 (gait-v2 12/24) but cap 2/24 (gait-v2 9-11/24).
    - **Rejected:** adopting it for the eraser gain - the cap, pushed rather than picked, is the case the crouched
      push references exist for, and it collapsed. Kept local as `gait_tp2` for a later mixed-curriculum attempt.
97. **GPU choice by measured throughput; Runpod RTX 4090 (Secure) for the next big run.**
    - **Why:** pick the GPU that gives the most simulation steps per dollar on this workload (MJX + Brax PPO),
      not by spec sheet; and use the owner's $15 Runpod credit without a run dying when it runs out.
    - **Measured** (`modal_train.py::throughput`: a short real training run, steps/s after XLA compilation,
      gait_v2): A10G 13.9k (2048 envs) / 14.9k (8192); A100-40GB 18.6k / 21.5k; L40S 27.2k / 25.3k. Ada (L40S,
      RTX 4090 family) is ~1.8x the A10G; the A100 is ~1.4x at ~2x the price; more than 2048 envs barely helps.
      The L4, H100 and grasp cases didn't run: the Modal workspace paused at 15:39 (spend limit), which also
      stopped grasp-v7b (checkpoint on the volume).
    - **Choice:** Runpod RTX 4090 Secure Cloud, $0.89/h (catalog read 2026-11-20: stock MEDIUM in EU-RO-1 and
      US-IL-1), ~2x Modal's steps per dollar; to be confirmed with a 5-minute run before a long one. Community
      Cloud (RTX 3090 at $0.22/h) rejected: hosts can disappear mid-run and stock was low. A100/H100 rejected on
      value.
    - **Never dying mid-run:** at a $0 balance Runpod stops pods and loses anything outside a volume. So a rented
      host carries `price_per_h` and `budget_usd`; `ops` caps each job at 85 % of the budget, jobs run in
      `/workspace` (kept when a pod stops), and the pod stops itself when its job ends.
    - **When:** the motion-tracking policy, once its environment exists; until then msi-1 trains.
98. **The claw stays as it is; grabbing works within it.**
    - **Why:** the owner saw a pen pick fail because the claw closes sideways with a 15-23 mm opening, so a pen is
      graspable only lying along the arm.
    - **Studied:** a "crab tongs" redesign (two hooked fingers geared together, closing front-to-back). In a scratch
      rig it held a pen lying across the claw 9/9 (today's claw 0/9), the eraser 27/27 (1/27) and the cap 9/9 (0/9).
      Sources behind it: Dogar & Srinivasa 2010 (push-grasp funnels), Rodriguez, Mason & Ferry 2012 (caging to
      grasping), the Petoi Bittle claw (a proven gear-coupled micro-servo claw). Rejected options: wider sideways
      jaws (no pen across) and a scoop (it pushed things instead of sliding under them).
    - **Decision:** the owner chose to keep today's claw; the tongs add printed or bought gears and a more delicate
      build. Instead, the pick planner checks fit before grabbing (approach a pen along its length), pushes or
      rotates a thing into a pickable pose, or pushes it to the tray; the cap and ball stay push-only. The tongs
      study is the fallback if grabbing stays the bottleneck.

99. **The project was audited against research and open-source practice; the setup defects it exposed are fixed.**
    - **Why:** every result so far came from one simulator that has never been compared with anything real. The
      audit's seven findings: the simulated servo is mostly guesses and training has no delay; the twin hands the
      mind true positions; the world is one desk; the mind has no declared structure; the training tools moved
      upstream; most comparisons used too few trials; the robot's software is not safe to leave running.
    - **Evidence:** [research/PROJECT_AUDIT.md](research/PROJECT_AUDIT.md) and nine reports (97 papers and 86
      repositories checked against the live sources).
    - **Change:** CI renders through a virtual display and runs the robot's two self-tests (master had been red
      since the recorder merge: a renderer probe aborted the test process on a machine with no display);
      `python -m ops check` runs the robot self-tests; `sim/benches/stats.py` gives every success count its 95 %
      range and says whether two results differ; the robustness bench exits non-zero on FAIL; a training seed per
      experiment or `--seed`; the installer no longer copies a removed folder; local jobs from a task worktree find
      the shared training environment.
    - **Measured:** local `ops check`: guard, ruff, 102 tests, robot self-test 40/40, firmware test 44/44 (face
      recognition skipped without its models). CI on Linux green on the pull request (two runs, 8 and 11 minutes).
      A CPU smoke run from a task worktree: submit, train, fetch, export (difference from the trained network
      3.6e-07), seed recorded.
    - **Rejected:** SciPy as a bench dependency (not installed in CI or on the robot); rliable (archived).

100. **Leg servo: the Feetech STS3215 (7.4 V) replaces the STS3032 as the working choice; hand size is a
     preference, not a rule.**
    - **Why:** the STS3032 has no published measured model, so every learned skill rests on guessed servo numbers
      (decision 99, finding 1). The owner asked for parts whose measured data is public and said size is not a
      hard limit: "we need a good robot, that is the hard rule".
    - **Evidence:** [research/SERVO_CHOICE.md](research/SERVO_CHOICE.md). Rhoban/bam (629 stars, ICRA 2025,
      arXiv 2410.08650) publishes fitted models `feetech_sts3215_7_4V` (m1-m6), usable in MuJoCo and mjlab; Open
      Duck Mini (4.8k stars) walks on that servo and model. The Dynamixel XL330-M288 also has a model, but it was
      fitted near 7.5 V (above its rated 6 V), has plastic gears and no Indian seller.
    - **Change:** legs = 12 x STS3215 7.4 V. Consequences being worked out on branches: the robot grows (a sizing
      study in simulation precedes any CAD), the servo rail and battery are redesigned, the parts list is rebuilt.
      The six arm servos are unchanged for now.
    - **Measured:** nothing yet on the new servo. Price for 14: INR 31,140 against INR 46,586 (Evelta, 2026-11-21).
    - **Risk:** the version the published fit most likely describes (C001, 1/345 gear) is sold out in India today;
      other gear versions are good servos without a public model. Command delay, gear play, voltage sag and
      unit-to-unit spread still need a short check on real units.
    - **Rejected:** staying with the STS3032 and measuring it ourselves first (possible, but training would wait
      for a bench and the model would be ours alone to maintain); the XL330-M288 (reasons above).

101. **The twin's electronics are data with a stated source, and can be switched from perfect to realistic.**
    - **Why:** the twin had perfect electronics (no command delay, an exact IMU, stepless joint angles, a battery
      that never changed, beams that never missed), so nothing learned or tested in it had met the errors a real
      robot has (decision 99, finding 2).
    - **Evidence:** domain randomisation of delay, noise and supply voltage is standard practice in the templates
      this project follows: pollen-robotics/microduck_rl (2.9k stars) randomises command delay and battery voltage
      6.5-8.2 V; Rhoban/bam (629 stars, ICRA 2025, arXiv 2410.08650) measures the servo's command delay. Part
      numbers come from the makers' datasheets (BNO055 rev 1.8, INA219 SBOS448G, MPR121, Feetech and Pololu pages).
    - **Change:** `configs/electronics.toml` holds every electronic value with a unit, a source (measured,
      datasheet or estimate) and a reference; `crab/electronics.py` loads it and `tests/test_electronics.py`
      refuses a value without a source. `realism="ideal"` (default, the golden trace) or `"datasheet"` in
      `PinchSim` and `SimBackend`: servo command delay, encoder steps, IMU rate, lag, noise and bias, distance-beam
      noise and dropouts, battery drain and sag, the 5 V regulator's limit, INA219 steps. Seeded and replayable.
      Described in [dev/ELECTRONICS.md](dev/ELECTRONICS.md).
    - **Measured:** falls, ideal against datasheet, 40 runs each: scripted trot 0/40 and 0/40, trained walk 0/40
      and 0/40, emotes 0/40 and 0/40 (95 % range 0-9 % each). Get-up: trained policy 100/100 in both (96-100 %);
      scripted swing 53/100 against 49/100 (not distinguishable).
    - **Honest limit:** 24 values are measured, 20 come from datasheets and 50 are estimates (wire resistance, pack
      internal resistance, computer power, sensor rates in our configuration). An estimate says why in the file and
      is replaced when a real number exists. Camera, microphone and touch signals are not yet modelled.
    - **Rejected:** hard-coding the numbers in the twin (no way to see which are guesses); making the realistic
      mode the default now (the golden trace pins today's behaviour; the switch flips when training has moved).

102. **Parts list rebuilt around the STS3215: servos run straight from a 2S Li-ion pack, three separate orders,
     charging dock parts included.**
    - **Why:** decision 100 changed the leg servo. The old power design asked one 5 V 5 A regulator to feed twelve
      leg servos, which it cannot (audit report `ext_runtime_power.md`).
    - **Evidence:** Open Duck Mini v2 (apirrone/Open_Duck_Mini, 4.8k stars) runs 14 of the same servos straight
      from two 18650 cells with a BMS, and a 5 V regulator for its computer; its parts sheet and wiring drawing were
      read for this. TheRobotStudio/SO-ARM100 (8.2k stars) uses the Waveshare bus driver board chosen here.
      Details and every seller page opened are in [research/BOM_CHANGE.md](research/BOM_CHANGE.md).
    - **Change:** `hardware/bom.py`, `hardware/BOM.md`, `hardware/BOM.csv`, `hardware/electronics.md`: 12 + 2
      STS3215 7.4 V; a prebuilt Samsung 25R 2S pack (2500 mAh, BMS) through a 15 A fuse and a 20 A switch to the
      servos; two 5 V 5 A regulators (computer and sensors; arm servos); Waveshare servo driver board; a 0.005 ohm
      shunt on the INA219; one VL53L5CX; the dock's charger module, spring contacts and protection parts.
    - **Measured:** whole list INR 77,412 (was 88,566). Order 1, servo bench: INR 10,859. Order 2, bring-up:
      INR 13,214. Order 3, full build: INR 50,424 (53,339 with the recommended extras). In stock on 2026-11-21:
      ThinkRobotics, Waveshare ST3215 7.4 V, SKU WVSH0593, INR 2,399.99. Nothing has been bought.
    - **Risk:** a full pack is 8.4 V on a servo sold as 7.4 V (sellers disagree on the range; test one servo on the
      bench first, or add a 7.6 V regulator); the 5 V regulator board is sold as 7 V minimum input, which a nearly
      empty pack falls below; the pack's BMS cut-off current is not on its listing; the in-stock servo is probably,
      not provably, the gear version the published model describes.
    - **Rejected:** a 7.4 V regulator for the legs (10 A or more: large and hot, and the published model already
      contains supply voltage); LiPo packs (out of stock, no built-in protection); building a pack from loose cells
      (soldering cells is not a beginner's job).

103. **A second design in the twin, v2: the robot resized for the STS3215, driven by the servo's published
     measured model.**
    - **Why:** decision 100 needs a body that fits the larger servo, and the point of that servo is that its model
      is measured. No CAD or hardware exists for it, so the model is built from primitives.
    - **Evidence:** the sizing study, [research/SIZING_STS3215.md](research/SIZING_STS3215.md) (leg scale 1.6,
      trot slowed 1.5 times); Rhoban/bam `bam/params/feetech_sts3215_7_4V/m6.json` at commit b9e1111 and its
      `MujocoController`, the same model Open Duck Mini trains on.
    - **Change:** `CRAB_DESIGN=v2` (read once in `crab/params.py`; v1 stays the default, bit for bit).
      `sim/build_model.py --design v2` writes `sim/models/crab_v2.xml`: links 35 / 48 / 83 mm, 1.571 kg, each
      servo a box of its real size and mass. Two servo levels in `crab/servo_model.py`: the full BAM model
      (voltage, back-EMF, load-dependent friction) and a first-order form for places the full model cannot run.
      `SimBackend(servo_model="first_order"|"bam")`. Described in [dev/V2_MODEL.md](dev/V2_MODEL.md).
    - **Measured:** stands with the worst joint at 9 % of the maker's stall torque (23 % of rated). Scripted trot:
      10.0 cm/s forward, 10.2 cm/s sideways, tilt under 6 degrees, no falls (today's robot: about 12 cm/s).
      Sideways at the cap runs the thigh at 71-78 % of stall at peak and 81-83 % of rated on average: heat, not
      speed, is the limit. Full model against first order on the same walk: speed within 0.7 %, peak torque 1-7
      points higher in the full model, power 4-6 % higher.
    - **Honest limit:** the fit is from a one-servo pendulum rig at 7.4 V and firmware gain 32; the gear version
      of the fitted servo is inferred; no temperature or current limit is modelled. The arms are 1.5 times longer
      on the same micro servos: fine alone, marginal with a held load (75 % of rated with a 10 g pen). Emotes and
      desk routines are still tuned for v1.
    - **Rejected:** replacing v1 in place (every adopted policy and the golden trace are v1's; they stay until v2
      policies beat them); using BAM's deprecated first-order export alone (it cannot follow the supply voltage).

104. **Every electronic part placed in the resized body, wired, and simulated as a circuit.**
    - **Why:** the owner asked for all electronics fitted and wired, and an exact simulation; a lumped 171 g "electronics"
      mass and a simple battery model could hide parts that do not fit and power that does not hold.
    - **Change:** `configs/layout_v2.toml` (outline, mass, pose, connector side, keep-out of every part) with
      `sim/fit_check.py`; `configs/wiring_v2.toml` with `crab/wiring.py` checks (pins, I2C addresses incl. the VL53
      reset-pin readdressing, the BNO055 on its own bus, ratings, regulator load); a circuit solved every control tick
      in the twin (pack, fuse, BMS, wire drops, both 5 V regulators, INA219 at its shunt, brown-out restarts the mind
      from memory); IMU and cliff beams at their mounts; the visor face on v2. A later audit replaced 18 estimates
      with datasheet values and tagged the rest needs-bench, design-choice or unverified.
    - **Measured:** fits after raising the chassis 4.2 mm; 1.75 kg (was 1.571); worst pack draw 16.2 A peak during a
      get-up swing, no fuse blow or brown-out at full charge; a mixed session lasts about 92 minutes to 80 % used.
    - **Needs the bench:** pack resistance and curve, BMS limit, regulator dropout, harness resistance, most outlines.
    - **Parts list changes found** (for phase 2): a 0.002 ohm shunt, one cable per servo, a larger hub board, four pogo pins.

105. **A second trainer on mjlab (MuJoCo Warp + RSL-RL), next to Brax/MJX.**
    - **Why:** audit finding 5: Brax maintains only its training code, MuJoCo Playground moved to this stack, and the
      measured servo models (BAM) plug into mjlab, not MJX.
    - **Evidence:** mujocolab/mjlab (Isaac Lab's manager-based API on MuJoCo Warp); pollen-robotics/microduck_rl as the
      template.
    - **Change:** `training/mjlab_pinch/` (velocity task with position or BAM actuators, randomised delay, bias, IMU
      misalignment, pushes, friction, mass, motor strength), ONNX and numpy export, a sim-to-sim replay, `ops/mjlab_run.py`.
    - **Measured:** msi-1 (RTX 4050): 57,600 steps/s at 16,384 envs in 3.9 GB. The first proof walker does not move
      in the CPU twin (0 cm/s, 0 falls in 20 rollouts): open, to be solved before it trains anything adopted.
    - **Also:** experiments may set `design = "v2"`; the trainer, run records and `ops eval` follow it.

106. **A locked 15-phase plan; phase 1 started.**
    - **Why:** the project kept changing parts and retraining with no measurable path to the finished pet.
    - **Evidence:** how BD-X and Olaf (Disney), Cozmo and Vector (Anki), AIBO (Sony), Open Duck Mini and ToddlerBot
      were developed: character first; body, motion and simulated training shaped together, then frozen; servos
      measured before training; a minimal real body early. Outline in [ROADMAP.md](ROADMAP.md).
    - **Change:** the stale roadmap replaced by the outline; the plan, the character brief, the move vocabulary and
      the hardware requirements are kept locally by the owner (guard and .gitignore). Three task branches that belong
      to later phases were archived with tags instead of merged: `archive/navigation` (phase 10), `archive/grab-fit`
      (phase 8/11), `archive/activity-mix` (phase 4/11).

107. **Phase 4: the mind's architecture as data, body parts with owners, ambient life always on.**
    - **Why:** interruption rules were score thresholds plus two hard-coded activity names; states were names only;
      one motion program drove all 18 joints and the idle-life layer stopped whenever anything ran, so Pinch went
      still during activities (ambient life in 52-65 % of awake seconds, still gaps up to a minute); the eyes kept the
      direction a stimulus had when first noticed, so they drifted off it while walking (gaze error 0.29-0.34).
    - **Evidence:** Anki Vector's engine (digital-dream-labs/vector, 119 stars, frozen 2022): an AnimationStreamer
      with locked animation tracks and procedural layers under canned animation (KeepFaceAlive blinks and eye darts);
      the Cozmo SDK (anki/cozmo-python-sdk, 728 stars): actions that want a locked track fail with TRACKS_LOCKED.
      Recording: the existing session recorder already holds one timeline; MCAP (`mcap`, 1.15k stars) noted as a
      viewer export, not adopted yet. Observer thresholds partly from HRI literature (Leite et al. 2013 long-term
      HRI survey), partly calibrated on Pinch's own sessions.
    - **Change:** `configs/mind/behaviour.toml` (the nine priority categories, every activity placed; only the top
      three cut short a non-interruptible activity), `configs/mind/states.toml` (sub-states, allowed motions,
      intent for every body part, interruption rank, timeout rule; a sleep state; disallowed motions refused and
      logged), `pinch/body/arbiter.py` + `configs/mind/channels.toml` (one owner per body part, the higher layer
      wins, claims all or nothing), ambient life on every part no one uses (`configs/mind/ambient.toml`), a fade over
      0.24 s when the joint targets would snap between motions, eyes re-aimed from the current pose every tick,
      fatigue, facts about its person kept only after their yes, the recording naming body-part owners and every
      behaviour file's version and hash, an aliveness check in the observer.
    - **Measured:** ambient life in 96.3 % of awake seconds on two 10-minute sessions (seeds 5, 11; body alone ~71 %;
      longest still gap 2-3 s); gaze error while walking 0.11-0.14 on six seeds (limit 0.15); 22/22 moves within the
      torque limits; full test suite green.
    - **Golden trace re-recorded:** behaviour changed on purpose: ambient life during activities, the handover fade
      and each state's interrupt rank change which activity follows a touch reaction at 77 s of the seed-7 session.
    - **Rejected:** a behaviour-tree library (py_trees, 655 stars) - the arbiter is about 100 lines and the selection
      already works; ambient motion on the legs under a walk (it would fight the gait); a recording-format switch
      to MCAP or Rerun now (the existing recorder covers the timeline; an exporter can come later).

108. **Phase 4, parts 2-3: one feeling drives everything Pinch shows, it learns from its person, and its states have real sub-states.**
    - **Why:** each activity picked its own face and kept it, so the face hid the feeling 31 % of awake time in the
      review scene; sounds, walking and posture ignored the mood; nothing Pinch did changed with feedback; the
      states' sub-states and timeout rules were declared but not used; social exchanges stopped dead; woken for a
      hello it waved with a sleepy face; after a bang the eyes stayed on where the bang had been (a body-relative
      angle) while it worked.
    - **Evidence:**
      - One affect state shown through all behaviour: Kismet (Breazeal 2003), the Oz project's believable agents
        (Loyall 1997), Disney's "illusion of life".
      - Display rules (Ekman & Friesen 1969).
      - Habituation of harmless startles (Thompson & Spencer 1966).
      - Expectation and appraisal (surprise = outcome - expectation).
      - Learning from a person's praise and scolding: TAMER (Knox & Stone 2009), bounded as a gradient bandit
        (Sutton & Barto).
      - Forethought, the eyes before the body (Takayama et al. 2011).
      - Proxemics by relationship (Mumm & Mutlu 2011).
      - Engagement has an explicit close (Sidner et al. 2005; Bohus & Horvitz 2009).
      - Top-down over bottom-up attention (Desimone & Duncan 1995); motion onset still captures it (Abrams & Christ
        2003).
      - Long-term HRI novelty and variety (Leite et al. 2013); an activity rhythm settled but not random (Blumberg's
        level of interest).
    - **Change:**
      - **Expression:** `pinch/mind/expression.py` and `configs/mind/expression.toml` drive face, sounds, emote style,
        walking tempo and posture from the feeling: an activity's face shows only if it fits the feeling, and a face
        that doesn't fit the state is swapped for one from the same feeling's list.
      - **Learning:** `pinch/mind/learning.py`, a per-person, per-context bandit that fades over 14 days, is cleared
        when a person is forgotten, and gives an acquired taste.
      - **Expectations:** per activity, so outcomes better or worse than expected raise delight or disappointment.
      - **Pure scores:** activity scores are pure (`prepare` / `utility`), with score stages and a learnable decision
        log (scores, parts, affect, context).
      - **States:** `configs/mind/states.toml`:
        - `phases` maps every activity step to a sub-state (`Mind.substate`, checked for every activity);
        - `timeout_do` carries out each state's timeout rule;
        - `close` gives social exchanges a disengage sub-state;
        - `gaze_hold` keeps the eyes on the task's target against weaker pulls;
        - woken for something calm, it passes through the sleep state's wake sub-state (a stretch, a sleepy face)
          before it greets.
      - **Attention:** heard and felt stimuli are placed in the room (`pinch/attention.py` `bearing`), and turns
        start with the eyes.
      - **Shared runtime:** `pinch/runtime.py` holds the mind-body step that the twin and the firmware share.
      - **Observer:** new checks for no_closing, late_reaction, dragging_step, face against state and gaze while
        walking; a step the body queues behind a reflex is now recorded when it starts.
      - **Benches:** `sim/benches/` gains congruence, liveliness, learning and rhythm.
    - **Measured** (on the v1 body, before decision 111 made v2.1 the only body; re-measured there under 111):
      - **Congruence, review scene:** the face hid the feeling 31 % -> 0 % of awake time, and showed the opposite of
        it 0.3 % -> 0.0 %. All congruence checks are 0 % on three scenarios.
      - **Liveliness:** 6/7.
        - Eyes reach a stimulus in a median 0.05 s, and lead the body 100 % of the time.
        - Forethought 100 % of 12 goal moves; outcome reaction 100 %.
        - Meaningful gaze is 88 % on target, but only about 20 points above chance (67-69 %) in the denser scenes,
          against the 25 required. **Open.**
      - **Rhythm, 20 seeds x 14 days:** flicker 0 %, median dwell 28 s, 1.68 switches per minute, repeat4 37 %,
        day14/day1 0.99. Next-activity entropy is 1.48 bits over its own choices (2.005 over all switches).
      - **Learning:** after 10 days of praise for exploring and scolding for playing, it explored 23 % more (target
        +25 %, **open**) and played 59 % less (41 % kept: its personality stays).
      - **Control tick:** p95 2.35 ms on an idle machine (limit 4 ms).
      - **Tests:** the full suite is green.
    - **Golden trace re-recorded:** behaviour changed on purpose. Social exchanges close; turns wait 0.4 s for the
      eyes. Together these shift the seed-7 run after 67 s, where it now gets briefly stuck while tidying and the
      stroke at 70 s waits behind the recovery.
    - **Rejected:**
      - A learned (deep RL) activity chooser now: there is no data yet. The decision log and the pure scores are the
        seam where one can be added in phase 11.
      - A language model choosing activities: it adds latency and needs the laptop; it stays optional, for phase 11.
      - A behaviour-tree library (py_trees): utility selection with states as data already covers what it would add.
      - A stricter liveliness rule passed by changing the bench: the uniform-random baseline is kept, and a
        time-shifted control is only reported next to it.

109. **The v2.1 body: a boxed body with a nose and chin, real parts and routed cables; four findings of a papers audit
     adopted.**
    - **Why:** the owner's audit of v2: jagged notches where the legs swept through the pill, no room or mounts for the
      real parts, and a short list of training and hardware lessons from recent small-robot papers not yet applied.
    - **Evidence:** SPINDER (Sci. Rep. 16:26147, 2026) on printed-link stiffness; Tan et al. (RSS 2018): policies
      trained without command latency failed on hardware; Chen et al. (Bioinspir. Biomim. 17 046001, 2022): sideways
      walking costs 40 % less on a 3-joint-leg robot; LeRobot's SO-101 gripper (huggingface/lerobot, 29k stars) for
      the claw's overload setting; Pollen's rustypot STS3215 register table.
    - **Change:**
      - **Body** (`cad/v2`, `configs/layout_v2.toml` chassis.plan): the pill cut to a rounded box between the leg
        bays, a nose for the face plate and a chin under the arms; six M2.5 heat-set cover screws; battery bay and
        door; a printed electronics deck; vents; makers' STEP models where they exist (`cad/vendor/fetch.py`);
        cables path-found (`cad/v2/routing.py`); print, closure and fit checks.
      - **Stance:** thighs level and hips 12.5 deg forward (ground clearance 34.7 -> 39.6 mm).
      - **Training:** Brax environments apply each command 0-20 ms late (mjlab already did).
      - **Servos:** claw servos get Overload_Torque 25 at setup.
      - **Print:** legs and arms at 60 % infill, 4 walls (SPINDER: 40 % less link flex than 40 % infill, +3 g a leg).
      - **Gait:** sideways stride cap 0.5 -> 0.6.
      - **Fixes:** the fuse moved clear of the limb liners (closure 2.1 -> 1.0 %); 29 estimated part sizes checked
        against the products bought (IMU board 21 mm wide; room for the wider ToF clones).
    - **Measured:** 2,275 g; fit and wiring checks pass; closure 1.0 % (limit 2 %); 22/22 moves within the torque
      limits; sideways 6.1 cm/s with the hottest joint at 67-69 % of rated RMS (sideways 17 % less energy per metre
      than forward); mixed hour 57 % of rated (limit 70 %); stand margin 76.9 mm inside the support polygon.
    - **Rejected:** keeping the pill with notches for the leg sweeps (the owner's "jagged cuts"); a full cover over
      the servos (the owner's DIY look: servos may show).

110. **Heat-aware rest: every bus servo has a thermal model, and a hot leg makes Pinch sit down.**
    - **Why:** an hour of walking can heat a leg servo past its protection; nothing in the twin or the mind knew.
    - **Evidence:** a one-node thermal model (copper loss of the motor torque, gear friction, idle draw) with the
      STS3215's 10 K/W and 240 s from SO-ARM100 hold tests (TheRobotStudio/SO-ARM100, 8.2k stars); a smaller servo
      scaled by case area.
    - **Change:**
      - **Model:** `crab/servo_model.py` (heat), with `fit_heat` to fit it to a bench log.
      - **Mind:** the hottest leg servo feeds a heat need that floors tiredness, a cool-down drive (sit down where it
        is from about 45 C, urgent near 55 C) and a walking cost on activities ranked human and below.
      - **Body:** walking slows from 55 C and stops at 60 C; the firmware cuts torque at 65 C.
      - **Benches:** `sim/benches/heat.py` (an hour of walking on demand, three heat conditions); `sim/mixed_hour.py`
        reports peak and final temperature per joint.
    - **Measured:** the twin's mixed hour on v2.1 stays far below the 43 C at which cool-down ends; the bench's
      worst case (40 C room, pessimistic thermal resistance) is what exercises the drive. The heat bench ran on v1
      (the desk was too small for v2): it is to be re-run on v2.1 with the 1.20 m desk (decision 111).
    - **Rejected:** a lumped "fatigue = heat" shortcut (heat and tiredness recover at different rates).

111. **v2.1 is the only body: the v1 robot, its design switch, CAD, models and policies are gone, and everything
     that was sized for v1 is fitted to v2.1.**
    - **Why:** phases 2-4 were built side by side. The twin always loaded the v1 model (`sim/desk.py` hard-coded
      `crab.xml`), so the pet mind, its benches and the golden trace had only ever run the old body, and the owner
      asked for one cohesive system. On the real v2.1 model the robot hardly moved: it spent 177 of 240 s trying to
      get unstuck.
    - **Evidence:** measured on the v2.1 model: standing body height 92 mm (v1: 49), feet 15-17 cm from the body
      centre (v1: 10), side cliff beams looking 21-23 cm out; the owner's desk is 100-140 cm wide.
    - **Change:**
      - **One body:** `crab/params.py` has no `CRAB_DESIGN`; 165 v1 files deleted (model, meshes, CAD, sizing study,
        policies); the model builder's v1 branches removed (the generated model is byte-identical).
      - **Robot code:** the PWM arm path removed (every servo is on the bus).
      - **Desk:** the twin's desk is 1.20 x 0.60 m, with its places and objects moved clear of the v2.1 legs.
      - **Cliff ring:** reads from the v2.1 standing height, with at least 25 mm of slack over a flat desk.
      - **Personal space:** distances the mind keeps from things scale with the body (`BODY_SCALE` 1.73, from the
        foot reach): curiosity standoffs, the startle radius, the step back, in-view habituation.
      - **Startle:** flinches back first and cowers after (v2.1's scared move lasts 3.6 s).
      - **Habituation:** a fright arriving in steps within 2 s is one episode (one rushing toy used to dull itself
        below the startle line).
      - **Firmware:** servo voltage limits per servo, legs 5.5-8.6 V and arms 4.5-6.0 V (the old 6.5 V cap would
        have refused the 7.4-8.4 V leg rail; Feetech register table and product pages); each servo's supply checked
        against its own rail.
    - **Measured:** on v2.1 Pinch walks about 1.4 m in 4 minutes, with no edge stops and no getting stuck (was 0.34 m,
      177 s unsticking); the generated model is byte-identical after the builder cleanup; electronics, circuit and
      mjlab tests pass.
    - **Rejected:** keeping v1 behind the switch (noise, and tests passing on the wrong body); scaling only the tests
      (the mind would keep things inside its own legs).

112. **The v2.1 get-up policy adopted (righting-v21-lag).**
    - **Why:** the v1 policy stands the v2.1 body up from 38 % of falls (45 % with realistic servos); the scripted
      swing manages 24 %.
    - **Evidence:** trained on the v2.1 body with 0-20 ms command latency (Tan et al. 2018); the 200-start righting
      bench.
    - **Change:** `policies/righting.npz` and the registry (v1 slots retired).
    - **Measured:** 200 random falls on its back: 175/200 = 88 % (82-91) ideal, mean 1.92 s; 191/200 = 96 % (92-98)
      with datasheet realism, mean 1.42 s. Commanded joint speed p95 11.6-13 rad/s against the 5.1 rad/s servo limit
      (the servos clip it; the numbers include the clipping).
    - **Rejected:** the older righting-v21e (about 80 %).

113. **The v2.1 walking policy adopted for straight walking (terrain-v21-lag); turns use the scripted trot.**
    - **Why:** the scripted trot clears only 8 mm bars and 10 deg ramps walking forward.
    - **Evidence:** 60 M steps on msi-1, corrections on the scripted trot, trained on the hard-terrain curriculum
      with command latency.
    - **Change:** `policies/gait.npz`. In `pinch/body/locomotion.py` the policy fades out (over 0.25 s) above
      0.08 rad/s of turning: it holds the body nearly still in yaw (0.14 of 1.6 rad asked in 4 s; the trot 1.19).
    - **Measured:**
      - **Robustness** (24 episodes a world, paired): falls 0/0/0/0 on desk, rough, stuck and terrain (scripted
        0/0/0/0); speed tracking 81/80/79/77 % (scripted 73/70/71/62 %).
      - **Hurdles forward:** 18 mm bars 3/3, 20 mm 2/3 (scripted: 8 mm 3/3, 10 mm 1/3).
      - **Ramps forward:** 15 deg 3/3, 20 deg 2/3 (scripted 15 deg 2/3, 20 deg 0/3).
      - **One loss:** sideways up a 20 deg ramp, 0/3 (scripted 3/3).
    - **Rejected:** the policy on every walk (turning suppressed); waiting for a retrain with turning commands
      weighted up (a later improvement).

114. **Firmware on the v2.1 body: gated camera fixes, a bigger tray marker, a fixed test clock.**
    - **Why:** on v2.1 the firmware self-test failed 7 checks. A small marker seen from across the 1.2 m desk has a
      second, mirrored pose solution, and one such fix jumped the position estimate 30 cm, so the robot guarded the
      wrong edge. Position error was 24-34 mm with the 40 mm marker. The camera assumed a 45 mm standing height. The
      self-test's results depended on the hour it ran (asleep at night). "Stuck" used a fixed 8 mm, inside a far fix's
      jitter.
    - **Change:**
      - **Fix gating** (`robot/localizer.py`): a fix further than max(5 cm, 3 sigma) or 0.5 rad from the estimate is
        believed only when 3 in a row agree within 3 cm.
      - **Marker:** 60 mm.
      - **Camera height:** `robot/vision.py` projects from the v2.1 standing height.
      - **Test clock:** `PinchFirmware(clock=)`, and the self-test runs at 13:00.
      - **Stuck:** a walk covering under a fifth of its expected distance.
      - **Purr while busy:** a stroke during an activity it won't drop gets a purr at once.
    - **Measured:** position error 34 -> 13 mm; self-test 37/44 -> 43/44 (the one left is the camera seeing desk
      things, decision 115); no fall off the desk in three full runs after the gating.
    - **Rejected:** a filter on the marker's reprojection error alone (both mirrored solutions reproject well).

115. **A notch in the chin in front of the camera (owner's choice).**
    - **Why:** standing, the chin hid the desk 15-45 cm ahead from the camera: 33 of 63 sight lines to the desk
      15-60 cm ahead (+-40 deg) were blocked, and the firmware self-test's "sees desk things" check failed on v2.1.
    - **Evidence:** measured on the model (`mj_ray` from `eye_cam` against the visible geometry, standing); the
      owner chose a notch over moving the camera or lowering the chin.
    - **Change:** `configs/layout_v2.toml` chassis.plan `notch` (|y| < 28 mm, from the face plate forward, above a
      plane falling at 35 deg from the lens's lower edge); `crab/shapes.py` BodyPlan subtracts it; `cad/v2/body.py`
      cuts it from the body (the wall follows it); `tests/test_v2.py` checks the sight lines. The CAD export also got
      fixes found on the way: an empty bay-liner half is skipped (the arm liners lie wholly in the tray), the claw's
      jaw-meeting search treats no contact as no contact, and `export_parallel.py --reuse` re-runs the export step
      from parts already built.
    - **Measured:** blocked sight lines 33/63 -> 0/63; body 1308 -> 1306 g, robot 2.275 -> 2.273 kg.
    - **Rejected:** the other options put to the owner (moving the camera, reshaping the whole chin): each changes
      more of the body than a local cut.

116. **Fixes found fitting the mind to v2.1: no false cliff back-offs, waiting for help after asking, one call at a
     time; the golden trace re-recorded.**
    - **Why:** on the notched v2.1 body three things showed in the tests. A startled flinch back carried a rounding
      0.4 mm/s sideways; one rolled side-beam reading then counted as "walking toward a cliff" 27 cm from the edge,
      and the reflex cancelled the flinch. Pet life spent 47 % of its awake time tidying: it retried things it
      couldn't reach every 30 s and asked for help each time (the owner's rule: a pet, not a tidy robot). One failed
      tidy said "uh oh" five times in 1.7 s.
    - **Evidence:** the twin traces (the cliff ring's states during the flinch; the body's pick and push traces); the
      observer's findings on the recorded 3-minute pet life.
    - **Change:**
      - **Cliff ring** (`pinch/touch.py`): a beam's back-off needs at least 5 mm/s toward its side (the gait's own
        "going" speed); a beam reading cliff still blocks every motion toward its side.
      - **Asking for help** (`pinch/mind/core.py` `wait_for_help`, `pinch/activities/tasks.py`): once Pinch asks its
        person for help with a thing, it leaves it alone until someone moves it (3 cm) or 10 minutes pass; "right by
        an edge, hopeless" scales with the body (0.08 m x BODY_SCALE).
      - **Sounds** (`Mind.sound`): the same call is not repeated within 3 s.
      - **Known problem:** the observer test lists failed_grasp as known: the v2.1 claw holds thin things poorly
        (`sim/grip_check.py`: pen 38 %, eraser 52 %, 5 mm rod 2 %; balls and cubes 100 %) until the tips-first claw is
        designed; the test fails when grasps work, so the entry comes off.
      - **Golden trace** re-recorded for decisions 111-116 (v2.1 body, notch, these fixes).
    - **Measured:** the startle test backs away again; pet life tidying 47 % -> 30 % of awake time, "uh oh" 9 -> 4 in
      3 minutes, warnings 7 -> 4 (not counting the known claw); full suite 273/273; firmware self-test 44/44 (the
      camera now sees desk things, decision 115).
    - **Rejected:** a longer flinch (the reflex, not the distance, stopped it); fewer tidy retries without waiting for
      the person (Pinch would just give up silently).
