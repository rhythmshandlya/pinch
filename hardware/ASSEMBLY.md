# Building the crab: beginner's guide (hand-sized version)

You need:
- the order ([BOM.md](BOM.md)): parts and tools from Evelta, the battery and a few items from robu.in,
  magnets / tape / glue / a USB-C adapter from Amazon.in, and the printed parts from robu.in's print
  service (26 pieces from
  [cad/out/print/](../cad/out/print), quantities in [print_sheet.csv](../cad/out/print_sheet.csv))
- a laptop and a table you can solder on
- about an evening per stage

Read the [electronics safety rules](electronics.md#safety-rules-read-before-first-power-up) first.

> Every step has been rehearsed in the simulator. When this guide says "run X", that script exists
> in this project and already passes in simulation (`python robot/selftest.py`: 32/32).

What you'll end up with: a crab about 20 cm across the feet and 9 cm tall (eye stalks included),
weighing about 610 g. Every wire is inside the body except the short loops that cross each joint.

---

## Stage 0: order, and check the servos when they arrive

1. Order everything marked **required** in [BOM.md](BOM.md). The 3D prints go to robu.in's print service:
   - upload the 12 STL files from `cad/out/print/`
   - choose **PETG** (and **TPU** for `foot_tip`)
   - set the quantities from `print_sheet.csv`:
     - chassis 1, top_shell 1, belly_door 1, pi_tray 1
     - coxa_link 4, femur_link 4, tibia 4, foot_tip 4
     - shoulder 2, upper_arm 2
     - dactyl_L 1, dactyl_R 1
2. **When the servos arrive, measure one of each with the calliper** and compare it with `cad/servo.py`.

   | | STS3032 (legs) | GS-4308MG (arms) |
   |---|---|---|
   | Case | 23 × 12 mm, 27.5 mm tall without the horn | about 15.8 × 8.5 mm, about 17.6 mm tall (**estimated**: measure it) |
   | Ears | 4 holes: 28 mm apart along the case, 6.5 mm across | 2 holes about 18 mm apart; 20.2 mm across the ears (listing) |

   The STS3032 numbers come from Feetech's drawing. The GS-4308MG numbers are estimated from its
   photos and the listing, so **measure it carefully**: case length and width, the height from the case
   bottom to the underside of the ears, the ear hole spacing, and how far the output shaft sits from
   the case end. If any number is off by more than 0.3 mm:
   1. change it in `cad/servo.py`
   2. run `python cad/build.py`, then `python cad/check.py`
   3. reprint only the parts that changed (they're small and cheap)

## Stage 1: set up the Radxa ZERO 3W (no robot needed)

1. Download **Radxa OS** for the ZERO 3W from Radxa's docs (https://docs.radxa.com/en/zero/zero3, the
   Debian image with the vendor kernel: it has the camera driver) and flash it to the microSD with
   balenaEtcher or Raspberry Pi Imager ("use custom image"). Both are free.
2. First boot: follow Radxa's ZERO 3 "first boot / headless setup" page to join your Wi-Fi and enable
   SSH (or plug in a micro-HDMI screen and a keyboard once). Power it from a 5 V USB-C phone charger.
   Then SSH in from your laptop.
3. Copy this project folder to the Radxa, for example `scp -r robotic-crab <user>@<radxa-ip>:~`.
4. Turn on the hardware it needs: `sudo rsetup` → **Overlays** → enable
   - **I2C3-M0** (pins 3 / 5: IMU, battery monitor, distance sensors)
   - **PWM8-M0, PWM9-M0, PWM12-M1, PWM13-M1, PWM14-M0, PWM15-M1** (the six arm servos)
   - the **Raspberry Pi Camera v1.3** overlay (our OV5647 camera)
   - and install our audio overlay: **Install overlay from source** → `robotic-crab/hardware/radxa/crab-audio.dts`,
     then enable it

   Reboot.
5. Install the software:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-opencv git
   python3 -m venv --system-site-packages ~/crab-venv && source ~/crab-venv/bin/activate
   pip install numpy pyserial smbus2
   ```
6. Check that everything imports: `python robot/selftest.py` (it needs `pip install mujoco`, which works on
   the Radxa but is optional).
7. Check the sound once the amp and mic are wired (stage 5): `aplay -l` and `arecord -l` should list
   **crab-audio**. Our audio overlay is the one piece not yet tried on a real ZERO 3W. If it doesn't
   show up, the speaker still works with Radxa's own MAX98357A overlay (made for the ROCK 3A, same pins).
   Ask for help adapting it.

The robot scripts talk to I2C, PWM and USB devices, so run them with `sudo ~/crab-venv/bin/python ...`
(or add your user to the `i2c` and `dialout` groups).

## Stage 2: the power chain (soldering, before any servo)

Follow the [wiring table](electronics.md#wiring-table) and [wiring.svg](wiring.svg).
1. Solder the battery lead → **fuse holder** → **INA219** (VIN+ / VIN−) → **main switch** → both regulators.
   Red to +, black to −. Put heat-shrink on every joint. Leave the wires long enough (about 8 cm) to fit
   the switch into the belly later.
2. Plug in a charged battery (fuse in), switch on, and **measure**:
   - **regulator #1** (servos) must read **4.9–5.2 V**
   - **regulator #2** (Radxa) must read **4.9–5.2 V**

   If not, stop: nothing else gets connected until these are right.
3. Solder a small **servo hub** on perfboard. Every header is 3 pins in the order **GND / V+ / Signal**,
   the same order as on the servo plugs:
   - **5 headers for the leg bus:** 4 legs + the FE-URT-1. V+ and GND come from regulator #1; the
     Signal pins are all joined and go to the FE-URT-1's signal.
   - **6 headers for the arm servos:** V+ and GND from regulator #1, but each Signal pin stays on its
     own and gets a jumper wire to its Radxa pin (16, 18, 21, 24, 7, 19: see the
     [arm servo table](electronics.md#arm-servo-signals-radxa-hardware-pwm)). Label them 0–5.
   - Join the hub's GND to a Radxa GND pin, so the arm signals share a ground with the servos.

## Stage 3: prepare every leg servo (one at a time)

New bus servos all have ID 1, so they must be set up **one at a time, before assembly**.
1. Connect **one** STS3032 to the hub. Connect the FE-URT-1 to the hub (signal + GND), and its USB to the
   Radxa's USB-C port through the OTG adapter.
2. Run `python robot/setup_servo.py --id N`, with N from the
   [ID map](electronics.md#servo-id-map-matches-crabparamspy-joint_names-order) (1–12). The servo moves to
   its centre position.
3. **Stick a label with the ID on it**, and leave it centred. You'll fit its horn in this position.
4. Repeat for all 12. Keep the 2 spares at ID 1, in their bags.

The arm servos (GS-4308MG) need no setup: they have no ID. Just label them L-swing, L-lift, L-claw,
R-swing, R-lift, R-claw as you build the arms.

> Try it first without hardware: `python robot/setup_servo.py --id 2 --rehearse`.

## Stage 4: mechanics

**How servos mount**
- **Leg servos (STS3032)** drop into an *ear ledge*: the yaw servos into the deck, the thigh servos into
  the coxa link's cradle, the knee servos into the pocket at the end of the thigh plate. The ears rest
  on the ledge, and the screws that come with the servo go through the ears into the small pilot holes.
- **Arm servos (GS-4308MG)** are tiny. Each drops into a snug pocket: the swing servo in the front of the
  deck, the lift servo in the shoulder bracket, the claw servo in the palm (its shaft is the claw's
  hinge). Drive the two little screws from its bag through the ears straight into the plastic. They cut
  their own thread, so go slowly and stop when snug.

**How links attach to horns**
1. Screw the horn (it comes in the servo's box: a metal disc for the STS3032, a small plastic horn for the
   GS-4308MG) onto the printed link with the small screws from the box. The horn sits centred on the
   link's hub.
2. Press horn + link onto the **centred** servo, with the link in its *straight* pose (see
   `robot/calibrate.py`). Arm servos: switch them to centre first with `--trim-arms` (stage 6), or
   just fit the link roughly straight. The trim fixes the rest.
3. Fix it with the horn's centre screw.

The spline only lets you fit a horn in steps of a few degrees. Calibration measures the rest.

**Each leg (4×):**
1. **Yaw servo** into its ear ledge in the chassis deck, output end outward.
2. **Coxa link + thigh servo:** the thigh servo into the coxa link's cradle, then mount the coxa on the
   yaw servo's horn with the leg pointing straight out.
3. **Femur link + knee servo:**
   1. The knee servo drops into the pocket at the end of the thigh plate.
   2. Fit its ear screws.
   3. Mount the plate on the thigh servo's horn with the thigh horizontal.
4. **Tibia + foot:**
   1. Mount the tibia on the knee horn with the shin straight in line with the thigh.
   2. Slide the TPU foot tip over the end and fix it with an **M2×10 + nut**.
5. **Leads:** run the thigh and knee leads along the coxa link and into the body through the slot above
   the yaw servo. Leave a loop, so the leg can swing ±55° without pulling.

**Each arm (2×):**
1. **Swing servo** into its pocket at the front of the deck (screws through the ears).
2. **Shoulder bracket + lift servo:** the lift servo into the bracket's pocket first, then the bracket onto
   the swing servo's horn, pointing straight forward-out along the arrow on the deck.
3. **Upper arm + claw servo:** the claw servo into the palm, shaft at the front (it's the claw's hinge).
   Then mount the upper arm on the lift servo's horn, arm horizontal.
4. **Dactyl** (the moving claw finger) on the claw horn, with the claw closed: `dactyl_L` on the left arm,
   `dactyl_R` on the right.
5. **Leads:** each arm servo has its own 150 mm lead. Run the three along the arm and into the body
   through the front slot, and plug them into the hub's arm headers: left swing 0, left lift 1, left claw
   2, right swing 3, right lift 4, right claw 5.

Turn every joint gently by hand over its range (`crab/params.py`). The CAD check
(`python cad/check.py`) shows nothing collides, but print tolerances can.

## Stage 5: electronics into the body

> Board positions in this stage are the plan from the design's weight estimate; they haven't been
> fitted in the CAD yet (see docs/ROADMAP.md). If a board doesn't fit where it says, find it another
> spot inside the body before taping anything.

1. **Deck:** the FE-URT-1, servo hub and fuse holder go on the deck between the yaw servos, held with
   double-sided foam tape and zip ties. The **BNO055** goes flat on foam tape near the middle, with its
   **x arrow pointing forward**. The two regulators and the INA219 go on the *underside* of the brain
   tray (foam tape), so the deck isn't crowded.
2. **Switch:** it snaps into the rectangular cutout in the belly, behind the battery door.
3. **Distance sensors:** one VL53L0X module behind each side-wall window, lying along the wall below the
   deck, glued so it looks out through the window tilted about 25° down.
4. **All servo leads** drop through the big hole in the deck and plug into the hub (one branch per leg,
   one plug per arm servo).
5. **Battery:** it goes into the bay from below, lying across the body, leads toward the middle. Close the
   belly door with 2 × M2×10.
6. **Brain tray:**
   1. Screw the Radxa ZERO 3W onto its 4 stand-offs (M2.5×8).
   2. Zip-tie the MAX98357A amp to the slots.
   3. Tape the microphone board at the front-right corner, under the small mic hole in the dome.
   4. Solder the Radxa's wires as in the wiring table (direct to the pads; no header needed), including
      the six arm-servo signal wires from the hub.
   5. Screw the tray onto its 4 posts (M2×8).
7. **Camera:** push the camera head into the pocket behind the lens hole in the chassis front wall (between
   the arms). Its flex goes up through the slot to the Radxa's camera connector.
8. **Speaker and touch pads (inside the dome):**
   - Glue the 28 mm speaker face-up into its ring under the grille.
   - Tape the two TTP223 boards inside the dome, one at the head and one at the back.
9. **Magnets:** glue 4 magnets into the chassis bosses and 4 into the shell posts. **Check that each pair
   attracts before the glue sets** (mark the faces with a pen). The shell then clicks on and pulls off by hand.

## Stage 6: first power-up

1. Shell off. Battery in, switch on. Nothing should move yet (torque is off).
2. On the Radxa, check that every servo answers and every sensor is found:
   `python robot/run.py --check`. It lists the 12 leg servos, the 6 arm PWM outputs, the IMU and the
   battery, and says if anything is missing. Torque stays off.
3. **Calibrate** in three short steps (rehearse each first with `python robot/calibrate.py --sim`):
   1. `python robot/calibrate.py`: hold the crab in the straight pose when asked (legs).
   2. `python robot/calibrate.py --check-directions`: bend each leg joint a little when asked. Any servo
      that turns the other way round from what the software expects is fixed.
   3. `python robot/calibrate.py --trim-arms`: rest the crab on its belly. Each arm servo moves a little;
      say whether it went the right way, then nudge it with + / − until it looks straight. Once, the left
      arm lifts to about 30° and you say if it went 15°, 30° or 60°: that sets how far these servos turn
      per pulse (their listing is unclear on it).
4. Run `python robot/run.py --emote wave`. The left claw should wave.

## Stage 7: on its own feet

1. `python robot/run.py` runs the demo: wave, happy, walk forward, crab-walk left and right, turn, angry, sleep.
2. Low battery: the software warns at 6.8 V and stops cleanly at 6.4 V. Recharge then (SkyRC E3, in the
   LiPo bag).

## Stage 8: smarter

* **Learned walking:**
  1. Train on Colab with [training/colab_train.ipynb](../training/colab_train.ipynb).
  2. Copy the result into `training/checkpoints/` and test it with `python sim/walk.py --policy ...`.
  3. Run it on the crab with `python robot/run.py --policy ...`.
* **Pet mind:** see [docs/PET_MIND.md](../docs/PET_MIND.md). Gemma 4 runs on your PC (`ollama pull gemma4:e4b`).
  The heart and mind already run in the simulator. On the robot, the next step is the senses layer
  (camera, mic, touch → events), listed in the roadmap.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| A servo doesn't answer | Wrong ID, or two servos share an ID. Unplug the others and run `setup_servo.py` on it alone |
| An arm servo doesn't move | Its PWM overlay isn't enabled, or its signal wire is on the wrong Radxa pin: `run.py --check` names the missing pins |
| An arm moves twice (or half) as far as it should | Redo `calibrate.py --trim-arms` and answer the 15 / 30 / 60° question carefully |
| Joint jerks or vibrates | Loose horn screw, or that servo's P gain isn't 32 |
| Robot sags or drifts | Calibration is off. Redo `calibrate.py` carefully |
| Radxa reboots when walking | The Radxa is on the servo regulator, or the 5 V regulator is too weak. It needs its own 5 V regulator |
| Battery runs out fast | Normal: about 27 min of non-stop walking, or about 35 min of mostly standing and emotes. Use the spare pack |
| Shell falls off | Magnet glued with the wrong pole up. Pull it out and flip it |
