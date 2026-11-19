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
    `sim/tidy_bench.py`, runs both tidy scenarios: the cap put away when asked, and the eraser noticed
    and tidied unasked. Seed 0 is the scenario layout; every other seed moves the desk things ±2 cm.
    One layout is effectively one trajectory, and the twin's contacts are chaotic, so I judge on pass
    rates. **HEAD tidied 0/4 of each and knocked 12 things off the desk; now cap 5/8, eraser 5/8, 0
    falls.**
    - **The side cliff beams looked too far out.**
      - **Problem:** at 30° down they hit the desk 148 mm from the centre, 5.4 cm beyond the feet. Any
        sideways walk the planners allow (feet ≥2 cm from the edge) tripped the reflex and aborted the
        task. 4 false edge stops in one 4-minute scenario.
      - **Change:** the side beams now look **45° down**, hitting 1.5 cm beyond the feet (flat-desk
        reading 76 mm, threshold 121 mm). They now fire only on a real map error. Front and rear stay
        at 40° (3.5 cm past the feet): walking forward is faster and needs the warning.
      - **Affects:** `crab/params.py`. CAD windows, twin and firmware all read that table.
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
