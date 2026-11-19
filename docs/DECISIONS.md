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
