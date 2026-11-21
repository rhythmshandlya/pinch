# Sizing study: a Pinch crab built on the STS3215 (simulation, 2026-11-21)

Question: if the 12 leg joints use the Feetech STS3215 (7.4 V, 55 g) instead of the STS3032, how big and heavy
must the crab be, and will it move well? This is simulation only, before any CAD. Every number below comes from
the scripts in `sim/sizing/` (rerun instructions in its README) or from a source named here.
Raw results: `sim/sizing/out/*.json`, `tables.md`, `variants.md`, `recommend.md`.

## 1. Short answer

- Build it at **leg scale s = 1.6** (coxa 35 mm, femur 48 mm, tibia 83 mm; today 22 / 30 / 52 mm).
  Mass about **1.55 kg** by the safe upper bound (printed parts scaled by volume), **1.43 kg** if the printed
  shells keep thin walls. Span 23 x 34 cm, 12 cm tall. Today: 0.61 kg, 17 x 21 cm, 7.5 cm tall.
- The servo has plenty of torque for **standing**: the worst joint needs 8 % of the datasheet stall and 23 % of
  the rated torque (today 11 % and 34 %).
- It does **not** have a free pass for **walking**. The STS3215's rotor inertia is large (BAM measured 0.026 kg.m^2
  at the joint, about 50 times what the repo guesses for the STS3032). With the repo's current trot, leg motors run
  at or above their continuous rating. Slowing the trot period by 1.5x fixes most of it (RMS 53-91 % of rated) and
  costs top speed (about 10 cm/s instead of 12 cm/s today).
- Battery: **2S 1500 mAh, about 81 g**. The model says 77 min of mixed activity; it still reaches 40 min if every
  leg servo burns an extra 70 mA while idle (not known, see section 8).
- What gets worse: weight (2.6x), fall energy from 75 cm (11.5 J against 4.5 J), desk footprint (2.3x the area),
  top speed, walking efficiency, and the claws no longer reach the desk unless the arms grow too.
- What gets better: standing margin, metal gears, a servo with a measured public model (BAM), and a
  working reference walker on the same servo (Open Duck Mini, per SERVO_CHOICE.md).

## 2. How today's robot is built (read from the repo)

- Legs: 4 x 3 joints, links coxa 22, femur 30, tibia 52 mm (`crab/params.py`), hips at x = +-22, y = +-46 mm.
- Stance (`crab/emotes.py` STAND): femur +0.35 rad, tibia -1.92 rad, shins near vertical. Hip axis 4.9 cm above the floor.
- Masses (`sim/models/mass_props.json`, total 605 g): body 318 g, each leg 61 g (three links, two servos inside
  them), arms 42 g. The coxa servo is in the body, the femur servo in the coxa link, the knee servo in the femur link.
- Servos: 12 x STS3032 (20 g) and 6 x GS4308 (5.8 g) in the arms. The STS3032 model is a datasheet guess
  (`crab/params.py`: stall 0.37 N.m at 5 V, 11.6 rad/s, armature 0.0005 kg.m^2).
- Checks reproduced as a sanity test of my scripts: the femur holds 0.042 N.m standing = 34 % of rated (the
  repo's own claim is 33 %), and the V x I power bound gives about 2.9 W standing (the repo's 0.56 A at 5 V).
  Battery life with that bound is about 33 min on 850 mAh (the repo's estimate: 35 min).

## 3. The model (what I built and what it is not)

`sim/sizing/model.py` builds the crab with MjSpec: 4 legs x 3 joints plus today's two arms, position actuators,
real servo boxes with real masses. Scaled by s: link lengths, hip positions, foot radius. Servo placement follows
today's CAD. Printed parts: today's printed mass x s^3 (by volume, the upper bound; s^2 is shown as a sensitivity).
Body: today's chassis and dome scaled by max(s, what the hip servos need); the height grows only as far as servo
height + battery thickness need. Battery mass from energy at 137 Wh/kg (the BOM's 850 mAh pack: 6.29 Wh, 46 g).
Electronics 80 g (my estimate for today's parts), +10 g for a 7.4 V servo rail. Arms are exactly today's.

Servo model, STS3215 (read with `gh api` from Rhoban/bam, files saved in `sim/sizing/`):
`bam/to_mujoco.py` formulas, with the constants of `bam/feetech/actuator.py` (vin 7.4, firmware kp 32,
error_gain 0.166, max_pwm 0.97) and the `m1` JSON (kt 1.1776, R 2.4787, armature 0.02608, friction base 0.0515,
viscous 0.0599, max_velocity 5.29, delay 1.05 ms):
forcerange = vin*kt/R = 3.52 N.m, kp = 18.1 N.m/rad, damping = viscous + kt^2/R = 0.619, frictionloss 0.0515.
`m6` (kt 1.2753, R 2.7534, armature 0.0216, max_velocity 5.10, delay 5 ms) was also run.
**This is BAM's own first-order approximation, not the full model.** BAM's file says the export is deprecated and
cannot reproduce the load-dependent friction or the torque-dependent damping. I add the firmware target slew
limit (max_velocity) and the command delay in the loop, as BAM's actuator class does. The extra m6 friction terms
are not modelled. The STS3032 keeps the repo's guessed model. Both go through the same code.

Gait: the existing `crab/gait.py` is reused unchanged. Before it is imported, the leg geometry in `crab.params` is
overwritten with the scaled values, so its own rules scale with the robot (lengths x S, periods x sqrt(S),
S = tibia / 100 mm: today 0.52, s=1.6 gives 0.83). Statics are measured with joint friction removed so the
actuator force is exactly the torque the pose needs.

Torque references for the STS3215: "model stall" 3.52 N.m (BAM's electrical number, probably not reachable in the
real firmware), "datasheet stall" 1.91 N.m (19.5 kg.cm), "rated" 0.637 N.m (6.5 kg.cm). The datasheet figure is the
conservative one and is used for all % below unless stated. For the STS3032 (5 V): stall 0.37, rated 0.12 N.m.

## 4. The sweep (scale s, default gait)

Size and mass (battery sized in the loop so the mixed activity lasts 40 min with the default gait):

| design | gait scale S | span L x W x H (cm) | mass (g) | 12 leg servos (g) | printed (g) | electronics (g) | battery (g) | hip height (cm) |
|---|---|---|---|---|---|---|---|---|
| today STS3032 | 0.52 | 16.9 x 21.1 x 7.5 | 606 | 240 | 205 | 80 | 46 (850 mAh) | 4.9 |
| s=1.3 STS3215 | 0.68 | 20.4 x 28.6 x 9.6 | 1279 | 660 | 424 | 90 | 70 (1300 mAh) | 6.3 |
| s=1.4 STS3215 | 0.73 | 21.2 x 30.5 x 10.3 | 1349 | 660 | 500 | 90 | 65 (1200 mAh) | 6.7 |
| s=1.5 STS3215 | 0.78 | 22.1 x 32.3 x 11.0 | 1434 | 660 | 590 | 90 | 59 (1100 mAh) | 7.2 |
| s=1.6 STS3215 | 0.83 | 23.0 x 34.1 x 11.8 | 1536 | 660 | 692 | 90 | 59 (1100 mAh) | 7.7 |
| s=1.7 STS3215 | 0.88 | 23.9 x 35.9 x 12.5 | 1643 | 660 | 805 | 90 | 54 (1000 mAh) | 8.1 |
| s=1.8 STS3215 | 0.94 | 24.8 x 37.7 x 13.2 | 1768 | 660 | 929 | 90 | 54 (1000 mAh) | 8.6 |
| s=2.0 STS3215 | 1.04 | 26.7 x 41.6 x 14.7 | 2052 | 660 | 1213 | 90 | 54 (1000 mAh) | 9.5 |

Standing torque per joint (only the femur carries load; the shins are near vertical):

| design | coxa N.cm | femur N.cm | tibia N.cm | femur / datasheet stall | femur / model stall | femur / rated |
|---|---|---|---|---|---|---|
| today STS3032 | 0.0 | 4.2 | 0.2 | 11% | 11% | 34% |
| s=1.3 STS3215 | 0.0 | 10.0 | 0.1 | 5% | 3% | 16% |
| s=1.4 STS3215 | 0.0 | 11.1 | 0.2 | 6% | 3% | 17% |
| s=1.5 STS3215 | 0.0 | 12.5 | 0.2 | 7% | 4% | 20% |
| s=1.6 STS3215 | 0.0 | 14.2 | 0.3 | 7% | 4% | 22% |
| s=1.7 STS3215 | 0.0 | 15.9 | 0.3 | 8% | 5% | 25% |
| s=1.8 STS3215 | 0.0 | 18.0 | 0.4 | 9% | 5% | 28% |
| s=2.0 STS3215 | 0.0 | 22.7 | 0.7 | 12% | 6% | 36% |

Slow trot (existing gait, default period) at 5 and 10 cm/s, forward and sideways. Each cell: speed reached,
peak motor torque as % of datasheet stall, RMS torque as % of rated, peak joint speed:

| design | forward 5 | sideways 5 | forward 10 | sideways 10 |
|---|---|---|---|---|
| today STS3032 | 4.3 cm/s; 26 %, 41 %; 3.6 rad/s | 4.6; 39 %, 48 %; 4.4 | 9.0; 27 %, 43 %; 6.2 | 9.8; 46 %, 58 %; 6.2 |
| s=1.3 | 4.5; 93 %, 121 %; 3.5 | 5.0; 103 %, 133 %; 4.0 | 9.7; 97 %, 128 %; 4.7 | 10.4; 134 %, 166 %; 4.9 |
| s=1.5 | 4.2; 82 %, 108 %; 3.2 | 4.9; 90 %, 120 %; 3.7 | 9.3; 86 %, 113 %; 4.0 | 10.1; 114 %, 151 %; 4.6 |
| s=1.6 | 4.3; 77 %, 102 %; 3.1 | 4.7; 85 %, 112 %; 3.6 | 9.3; 83 %, 108 %; 3.7 | 9.9; 106 %, 139 %; 4.5 |
| s=1.8 | 4.1; 69 %, 96 %; 2.8 | 4.8; 75 %, 102 %; 3.3 | 9.0; 78 %, 100 %; 3.2 | 9.9; 93 %, 127 %; 4.1 |
| s=2.0 | 4.1; 62 %, 89 %; 2.7 | 4.6; 67 %, 95 %; 3.1 | 9.0; 72 %, 92 %; 2.9 | 9.6; 81 %, 117 %; 3.8 |

(Rows 1.4, 1.7 are in `sim/sizing/out/tables.md`. The commanded speed is 5 or 10 cm/s; the gait reaches 85-100 % of it.)

Joint speed needed against the servo limit (peak actual joint speed, commanded peak in brackets; firmware limit
5.29 rad/s for the STS3215, 7.9 for today's STS3032 model):

| design | limit rad/s | fwd 5 | side 5 | fwd 10 | side 10 | gait speed cap fwd / side (cm/s) | peak rad/s at cap |
|---|---|---|---|---|---|---|---|
| today STS3032 | 7.9 | 3.6 (3.3) | 4.4 (4.2) | 6.2 (6.0) | 6.2 (6.1) | 13.1 / 13.1 | 7.8 |
| s=1.3 | 5.3 | 3.5 (2.9) | 4.0 (3.4) | 4.7 (4.4) | 4.9 (4.6) | 14.9 / 14.9 | 5.5 |
| s=1.5 | 5.3 | 3.2 (2.7) | 3.7 (3.1) | 4.0 (3.8) | 4.6 (4.1) | 16.1 / 16.1 | 5.4 |
| s=1.6 | 5.3 | 3.1 (2.6) | 3.6 (3.0) | 3.7 (3.5) | 4.5 (3.9) | 16.6 / 16.6 | 5.4 |
| s=2.0 | 5.3 | 2.7 (2.3) | 3.1 (2.6) | 2.9 (2.8) | 3.8 (3.2) | 18.5 / 18.5 | 5.3 |

Speed is not the limit: 5 and 10 cm/s need 2.7-4.9 rad/s, under 5.29. Only the gait's own top speed gets close to
the servo limit, and the firmware slew limit then cuts it a little (achieved 14-17 cm/s against a 15-18 cap).

Electrical power, servos only (model: copper loss I^2 R + positive mechanical power + 6 mA idle per servo, with
I = motor torque / kt; "V x I bound" is the repo's older method, rail voltage x motor current, which ignores
the duty cycle and is an upper bound). Mix = 35 % standing, 30 % idle emote, 25 % trot 5 cm/s, 10 % trot 10 cm/s
(my assumption). Battery power adds 4 W for the computer (repo's figure) at 90 % regulator efficiency:

| design | stand W | idle emote W | trot 5 W | trot 10 W | mix servo W | mix W (V x I bound) | mix battery W | mAh for 40 min |
|---|---|---|---|---|---|---|---|---|
| today STS3032 | 0.7 | 0.8 | 1.5 | 2.2 | 1.1 | 4.2 | 5.7 | 638 |
| s=1.3 | 0.8 | 0.9 | 12.3 | 20.7 | 5.7 | 15.3 | 10.8 | 1212 |
| s=1.5 | 0.8 | 0.9 | 9.9 | 16.0 | 4.6 | 14.3 | 9.6 | 1079 |
| s=1.6 | 0.8 | 0.9 | 9.0 | 14.2 | 4.3 | 14.0 | 9.2 | 1033 |
| s=1.8 | 0.9 | 1.0 | 7.6 | 11.7 | 3.7 | 13.8 | 8.6 | 965 |
| s=2.0 | 1.0 | 1.2 | 6.7 | 9.9 | 3.4 | 14.1 | 8.2 | 922 |

Fall energy from a desk edge (m g h, h = 0.75 m, impact speed 3.84 m/s for every design): today 4.5 J; s=1.3 9.4 J;
s=1.5 10.5 J; s=1.6 11.3 J; s=1.8 13.0 J; s=2.0 15.1 J (2.1x to 3.4x today). Energy per kg is the same; what
rises is the total energy the legs, servos and shell must absorb.

## 5. Why walking is hard on this servo, and the fix

Rotor inertia. With the STS3215's measured armature (0.026 kg.m^2) the motor torque during a trot is mostly
spent accelerating the rotor. Test at s = 1.5: armature set to 0 gives peak 24-39 % stall and 31-46 % rated;
the measured value gives 82-114 % and 108-151 %. The slower the leg moves, the less it costs (acceleration falls
with the square of the period). Slowing the trot period by 1.5x or 2x, from the gait's own value:

| design | fwd 5 | side 5 | fwd 10 | side 10 | gait cap cm/s | mix servo W | mAh for 40 min |
|---|---|---|---|---|---|---|---|
| today STS3032 | 4.3 cm/s: 26% / 41% / 3.6 / 1.5 W | 4.6: 39% / 48% / 4.4 / 1.5 W | 9.0: 27% / 43% / 6.2 / 2.1 W | 9.8: 46% / 58% / 6.2 / 2.2 W | 13.1 | 1.1 | 638 |
| s=1.5, period x1 | 4.2: 82% / 108% / 3.2 / 9.7 W | 4.9: 90% / 120% / 3.7 / 10.1 W | 9.3: 86% / 113% / 4.0 / 15.0 W | 10.1: 114% / 151% / 4.6 / 16.9 W | 16.1 | 4.6 | 1079 |
| s=1.5, period x1.5 | 4.6: 41% / 54% / 2.1 / 3.9 W | 5.0: 57% / 65% / 2.7 / 4.1 W | 9.1: 54% / 72% / 4.0 / 7.2 W | 9.5: 88% / 97% / 4.3 / 9.4 W | 10.7 | 2.4 | 799 |
| s=1.5, period x2 | 4.5: 28% / 39% / 2.0 / 2.7 W | 4.9: 41% / 46% / 2.3 / 2.8 W | 7.2: 39% / 49% / 3.3 / 4.1 W | 7.3: 57% / 65% / 3.3 / 5.2 W | 8.0 | 1.7 | 715 |
| s=1.6, period x1.5 | 4.4: 39% / 53% / 2.1 / 3.7 W | 4.9: 56% / 62% / 2.6 / 3.8 W | 9.0: 52% / 68% / 3.8 / 6.6 W | 9.7: 81% / 91% / 4.0 / 8.4 W | 11.1 | 2.3 | 783 |
| s=2.0, period x1.5 | 4.6: 35% / 51% / 1.8 / 3.1 W | 4.9: 54% / 58% / 2.2 / 3.2 W | 9.1: 46% / 60% / 3.0 / 5.1 W | 10.0: 70% / 78% / 3.4 / 6.0 W | 12.4 | 2.0 | 757 |

Cells: speed reached: peak torque % datasheet stall / RMS % rated / peak rad/s / servo watts. All scales and the
x2 rows for every s are in `sim/sizing/out/variants.md`.

BAM m6 instead of m1 (s=1.6, x1.5): peak 32-70 % stall, RMS 45-75 % rated, mix servo power 1.8 W (m1: 2.3 W).
The m1 numbers are the pessimistic ones.

Caveat on the baseline. Today's armature (0.0005) is a guess, not a measurement. To show how much hangs on it, I
reran today's robot with armature x52 (the STS3215's class of value): RMS torque becomes 255-271 % of rated and
power 40 W. That is not a prediction (a small servo's rotor inertia is smaller), but it means **today's walking
margins in the repo are only as good as that unmeasured number**, and the comparison "STS3215 walks with less
margin than the STS3032" holds only under today's guess. The STS3215 numbers rest on measured data.

## 6. Recommendation: one size

| | value |
|---|---|
| Leg scale | s = 1.6: coxa 35.2, femur 48.0, tibia 83.2 mm, foot radius 8 mm |
| Hips | x = +-35 mm, y = +-74 mm; body chassis about 135 x 180 mm footprint, 53 mm tall (servo 35 + pack 18) |
| Standing | hip 7.7 cm above the floor; span 23.0 x 34.1 x 11.8 cm |
| Mass budget | **1.55 kg limit** (this model, printed parts by volume); aim for **1.45 kg** with thin-wall prints (model: 1.43 kg at ~s^2) |
| Itemised (g) | 12 STS3215 660, 6 arm servos 35, printed legs 349, printed arms 7, printed body 335, electronics 90, battery 81 |
| Battery | 2S (7.4 V) **1500 mAh, about 81 g**, 18 mm thick, in the belly bay |
| Gait | existing trot, period x1.5 (top speed about 11 cm/s; 5 and 10 cm/s both reached) |
| Servo rail | 7.4 V regulated; walking peaks are far below the 12 x 3 A stall (BAM: 2.99 A each), so size it for the walking peak plus fall margin, with a fuse |

Why 1.6 and not smaller: the STS3215 is about 2x the STS3032 in each dimension (45 x 25 x 35 against 23 x 12 x 27.5
mm). Today's femur link is 1.3 servo lengths. At s = 1.6 it is 1.06; at 1.3 it is 0.86 and at s = 1.97 it equals
today's proportions. Below about 1.5 the hips, thigh and knee servos will probably not fit without a new layout;
I cannot judge that without CAD. Why not larger: each +0.1 in s adds about 100 g and 0.8 J of fall energy and
improves torque margin only a few percent (peak 77 % at 1.6, 69 % at 1.8, default gait).

Results for this design (`sim/sizing/recommend.md`, m1 model, period x1.5):

| | today (STS3032) | recommended (STS3215) |
|---|---|---|
| Mass | 0.61 kg | 1.56 kg (1.43 kg at ~s^2 prints) |
| Standing femur torque | 11 % of stall, 34 % of rated | 8 % of datasheet stall (4 % of model stall), 23 % of rated |
| Trot 5 cm/s, worst joint peak / RMS | 26-39 % / 41-48 % | 40-56 % / 53-62 % |
| Trot 10 cm/s, worst joint peak / RMS | 27-46 % / 43-58 % | 52-82 % / 68-91 % |
| Peak joint speed at 10 cm/s | 6.2 rad/s of 7.9 | 3.8-4.0 rad/s of 5.29 |
| Top trot speed reached | 11.9 cm/s | about 10 cm/s (cap 11.1) |
| Servo power: stand / idle / trot 5 / trot 10 | 0.7 / 0.8 / 1.5 / 2.2 W | 0.85 / 0.94 / 3.7 / 7.5 W |
| Battery for mixed 40 min | 638 mAh (has 850) | 783 mAh needed, 1500 fitted |
| Runtime, model / extreme idle (150 mA per servo) | 53 / 20 min | 77 / 25 min |
| Span | 16.9 x 21.1 cm | 23.0 x 34.1 cm (area 2.3x) |
| Fall energy from 75 cm | 4.5 J | 11.5 J (2.5x) |

Better: standing margin (about 1.5x), measured model, metal gears, joint speed headroom (about 25 % against the
limit). Worse: everything that scales with mass or size, plus walking torque margin (peak torque in the walk is
about 2x today's fraction) and walking power (about 2.5x today's at 5 cm/s).

## 7. Risks

1. **Walking heat.** Even at period x1.5 the sideways 10 cm/s gait runs RMS 91 % of rated, the cap speed 99 %.
   Long continuous walking at that speed would overheat the knee and femur servos. Keep fast sideways walking
   short, or lengthen the period (x2 gives 57 % peak, 65 % RMS, top speed 8 cm/s). Needs temperature feedback
   (the servo reports it).
2. **Claws.** The arms are today's (reach 5.4 cm from the pitch axis). The pitch axis is now 8.1 cm above the
   floor against 5.3 cm today, so the claws cannot reach the desk in the normal stance. Either crouch, or scale
   the arms by about 1.5x (a bigger arm servo; not studied here).
3. **Drop.** 11.5 J at 3.84 m/s onto a hard floor, with 12 x 55 g servos in the legs. I did not simulate the impact,
   so I cannot say whether the gears or the 3D-printed horn mounts survive. This is the largest unknown.
4. **Voltage rail.** BAM's model holds at 7.4 V with firmware kp 32. A 2S pack is 8.4 V full and under 7.4 V when
   half empty, so the servos need a regulated 7.4 V rail (buck-boost or a 3S pack with a buck). The extra mass
   (10 g) is a guess. At 12 x 3 A stall the rail must survive falls and stalls: fuse and regulator sizing is open.
5. **Supply.** SERVO_CHOICE.md found the C001 variant that BAM most likely fitted sold out in India.
6. **Printed mass.** The s^3 rule may overshoot or undershoot; the budget is a limit to design against.

## 8. What I could not determine

- The real armature of the STS3032, so today's walking margins are uncertain (section 5).
- Idle current of an STS3215 with torque on. The datasheet idle current is 6 mA (Feetech datasheet via DFRobot,
  fetched 2026-11-21), but no-load current is 150 mA and holding current is not given. The runtime shown for
  150 mA per servo (25 min on 1500 mAh) is an extreme case; the pack tolerates about 70 mA extra per servo.
- Stall figures disagree between sources: BAM 3.52 N.m model (7.4 V x kt / R), datasheet 19.5 kg.cm = 1.91 N.m
  (one retailer says at 6 V, the datasheet via DFRobot says at 7.4 V; I took the lower), stall current 2.5 A
  (datasheet) against 2.99 A (BAM).
- Whether the leg layout fits at s < 1.6, and the real mass of the printed parts and of the 80 g electronics.
- Whether the firmware slew limit (5.29 rad/s) is the only rate limit; BAM models only that.
- The gait is the existing one with its own Froude scaling, not retuned; a tuned or learned gait could do better
  than "period x1.5". The mixed-activity profile and the 4 W computer load are assumptions (repo figure).
- Contact is simulated with a flat floor and today's foot friction, no desk edge or fall.

## 9. Sources

- Rhoban/bam, `bam/to_mujoco.py`, `bam/feetech/actuator.py`, `bam/params/feetech_sts3215_7_4V/m1.json`, `m6.json`
  (read with `gh api`, 2026-11-21).
- docs/research/SERVO_CHOICE.md (this repo): servo sizes, masses, Evelta torque figures.
- Feetech STS3215 datasheet (translated copy and DFRobot listing, via web search 2026-11-21): idle 6 mA, no-load 150 mA,
  stall 2.5 A, rated 0.65 A at 7.4 V; 19.5 kg.cm.
- `sim/power.py`, `hardware/BOM.csv` (pack mass and capacity), `sim/models/mass_props.json` (today's masses).
