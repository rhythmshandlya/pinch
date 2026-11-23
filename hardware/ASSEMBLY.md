# Building Pinch

The step-by-step assembly guide for the v2.1 body is written together with the build (phase 13), from the real
parts, so every step is one that was actually done. Until then there is no guide here, on purpose: the old one
described an earlier, smaller robot.

What exists today:

- **Parts and prices:** [BOM.md](BOM.md), in three orders (servo bench, bring-up, full build).
- **Wiring:** [electronics.md](electronics.md), including the
  [safety rules](electronics.md#safety-rules-read-before-first-power-up): read them before the first power-up.
- **The body CAD:** [cad/v2/](../cad/v2/). Printable parts are written to `cad/out/v2/print/` (see "3D printed
  parts" in [BOM.md](BOM.md)).
- **Software on the Radxa:** `sudo bash deploy/install.sh`, then calibrate the servos with
  `robot/calibrate.py` (see "On the robot" in the [README](../README.md)).
