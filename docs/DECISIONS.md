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
