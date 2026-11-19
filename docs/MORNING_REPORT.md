# Morning report: Pinch

**TL;DR** (2026-11-19, evening)
- **Walking is now learned and robust.** Pinch walks with the scripted trot plus small corrections from a trained policy (gait-v1).
  - Across 40 randomised robots and worlds: **0 falls**, and **86 %** of the commanded speed (the scripted gait alone: 69 %).
  - It also passes a cluttered desk and a foot that gets caught.
  - It gets over a 5 mm cable, a pen and a book's edge. A high-step "climb" crawl is the fallback when it's caught.
- **All 12 behaviour scenarios pass** for the first time: tidying when asked, tidying unasked, asking for help, meetings, comfort, neediness, touch, sulking, 10 minutes alone.
- **Tidying was traced to its causes and fixed.** That came out of a new bench of 16 desk layouts per case: this morning it tidied 0 of 8; now the eraser 12/16, the cap 5/16, with 0 falls.
  - Side cliff sensors aimed too far out, so they kept aborting tasks.
  - A reflex's success was mistaken for a pick's.
  - Objects in view never counted as "still".
  - Planning ignored the tray.
  - Things slid out of the claw (a simulator setting), and letting go re-grabbed them.
- **The CAD was re-verified with the print files.** The chassis's side-sensor windows were cutting paper-thin slivers. The sensors moved down into solid wall, and the chassis now passes the wall check. Results are below.
- **msi-1 is a working training node** (WSL2 + JAX on the RTX 4050, runs from a Windows scheduled task, so it survives SSH disconnects).
  - It trained the gait2 variant (76 %, so gait-v1 stays).
  - It is now training gait3 (100M steps, shoves as hard as the bench's), at no Modal cost.
- **Modal spend: about $11–12 of $100** (estimate). Nothing is left running except grasp-v5 (hard timeout).
- **Needs you:** OK to download the face models (~39 MB), and measure the OLED module before printing. See the end.

## Try it (2 minutes)

```powershell
.venv\Scripts\python -m pinch.app
```

Talk to it ("hi, I'm …", "dance", "play dead", "I'm sad today", "I'm in a meeting", "put the cap away"),
stroke it, tickle it, mess up its desk, send in a visitor, drain its battery, or tick **Listen** so it hears you
through your microphone.

## "Done means": where each item stands

| Goal | Status | Verified by |
|---|---|---|
| Full robot designed (CAD/STL) | ✅ 14 unique printed parts (31 pieces). Every part is one valid solid. Chassis walls now pass: the side cliff windows were cutting 0.07 mm slivers, fixed. Thin on purpose: the tray's flat ramp and the 1.2 mm TPU jaw pads. ⚠ The visor's 4 screw posts have 1.1 mm walls (robu's minimum is 1 mm); their size waits on measuring the OLED module | `cad/build.py`, `cad/walls.py`, `cad/check.py`: all 14 parts pass the print rules, every servo fits its bracket, **241/241 poses collision-free** (stand, every emote keyframe, trot, crawl and climb samples, each joint at its limits) |
| Digital twin | ✅ MuJoCo desk, the tray and its marker, desk things, **five** cliff beams, wide camera, live face, the body at full size, no friction creep in the claw | `sim/desk.py`, `pinch/sim_runner.py` |
| Locomotion trained + robust | ✅ **Residual gait (gait-v1)**: flat desk 0/40 falls, 86 % tracking; rough desk 0/40, 86 %; caught foot 0/40, 81 % (scripted: 69 / 68 / 65 %). Gets over a cable, a pen and a book edge; a climb gait is the fallback | `sim/robustness.py` → [ROBUSTNESS.md](ROBUSTNESS.md), [rough](ROBUSTNESS_rough.md), [caught foot](ROBUSTNESS_stuck.md); `sim/obstacle_test.py` |
| Expressive motions trained + robust | ✅ 21 emotes, one trained balance policy, **0 falls in 40 harsh episodes** | `sim/robustness.py`, `sim/check.py` |
| Pet brain in simulation | ✅ **12/12 scenarios**; self-preservation (danger map, battery hunger, height fear), self-directed play, people and relationships | `sim/pinch_scenarios.py` → [BEHAVIOUR_TESTS.md](BEHAVIOUR_TESTS.md) |
| Picks things up, arranges the desk | ⚠ **Partial.** Tidy bench (16 layouts): eraser 12/16, cap 5/16. Pick bench: pens 4/12 picked, and every one lands in the tray; erasers 2/12 (they slip in transit and get pushed instead). Learned grab: GRASP_RESULT | `sim/tidy_bench.py`, `sim/pick_bench.py` |
| Virtual pet on the laptop | ✅ `python -m pinch.app` | ran headless + windowed |
| Firmware ready to deploy | ✅ `robot/pinch_main.py` + drivers + HTTP API + systemd + install. **39/39** firmware checks, **40/40** selftest. Same Body, reflexes and walking policy as the twin | `robot/firmware_test.py`, `robot/selftest.py` |
| Budget ≈ ₹1L, parts in India | ✅ ₹88,566 + prints (~₹2–3k, an **estimate**) | `hardware/bom.py` → [BOM.md](../hardware/BOM.md) |

## What changed today (details: [DECISIONS.md](DECISIONS.md) 48–54)

- **48. Tidying, at its causes.**
  - **Side cliff sensors:** they looked 5.4 cm past the feet and stopped every sideways walk. They now sit low in the wall, 30° down, and see 1.7 cm past the feet.
  - **Reflexes:** a reflex that cuts a step short now makes that step fail.
  - **Stillness:** "still" survives camera jitter.
  - **Tray:** the planners keep feet out of the tray, and searches for a clear spot look in all directions.
  - **Pushes:** they ease off near the goal and stand closer to things near an edge.
  - **Pen:** the twin's pen no longer rolls like a marble.
- **49. Training that survives a closed laptop.** Modal runs are spawned, not waited on: walk-v4 and grasp-v4 had died with this laptop's client. msi-1 trains from a scheduled task.
- **50, 53. Walking = scripted trot + learned corrections.**
  - **On the robot:** the policy's look-ahead comes from a copy of the gait run 0.1 s on. That costs 0.8 ms; the RK3566 is the limit.
  - **Rejected:** a 0.1 s delay line, which cost precision.
  - **Audit fixes:** no corrections mid-crouch (without that, 8 falls); positions don't lag; margins agree.
- **51. The grasp env matches the twin's physics** (2 ms, stiff contacts, settle by time).
- **52. Getting over things:** the "climb" crawl, plus rough-desk and caught-foot robustness worlds.
- **54. Things stay in the claw.** 3 noslip iterations; the claw stays open while lifting away; turns are gentle while carrying, and the grip is checked.

## Training runs (Modal estimate: GPU time × list price; A10G about $1.10/h; check the dashboard)

| Run | Where | Time | ≈ Cost | Result |
|---|---|---|---|---|
| earlier runs (walk-v1–v3, emote-v1, grasp-v1–v3, push-v1) | Modal | | $6.5 | emote-v1 used; the rest superseded |
| walk-v4 (from scratch, 2 ms) | Modal A10G | ~28 min | ~$0.5 | stopped at 27M steps when this laptop's client died; ~0 % tracking, so dropped for the residual gait |
| grasp-v4 | Modal A10G | ≤50 min | ≤$0.9 | died while compiling (same cause); relaunched as v5 |
| **gait-v1** (residual) | Modal A10G | 57 min | ~$1.05 | ✅ **used** (`policies/gait.npz`) |
| grasp-v5 | Modal A10G | ≤150 min | ≤$2.75 | GRASP_RESULT |
| gait2-v1 (residual, research variant) | **msi-1** | ~100 min | free | 76 % tracking: worse than gait-v1, kept as `policies/gait2_v1.npz` |
| gait3-v1 (residual, bench-strength shoves, 100M) | **msi-1** | ~3 h | free | running; bench it with `PINCH_GAIT_POLICY=gait3_v1 python sim/robustness.py --what walk` |
| **Modal total** | | | **≈ $11–12 of $100** | |

## Needs you

1. **Face models** (people recognition): OK to download OpenCV's YuNet + SFace (~39 MB, from the OpenCV model zoo on GitHub)? Until then, people are recognised in the twin from simulated sightings only.
2. **Measure the 2.42" OLED module** before printing (hole positions and glass thickness are estimates). The visor's screw posts get thicker once we know there's room.
3. **Decisions to review** (DECISIONS.md):
   - side cliff sensors low in the wall (48)
   - no friction creep in the twin (54): it makes cap pushing harder than before, likely closer to reality
   - keeping gait-v1 over gait2 (50/53)
4. **Pinch can't right itself from its back** (A11): accept "beetle asks for help", or reshape the shell before printing.

## What's left (in order)

1. **Picking up:** finish grasp-v5's evaluation (below); the eraser slipping in transit; pens from awkward spots (the stand must be ~15 cm along the pen's axis).
2. **The toy ball** starts 7.5 cm from the back edge and gets brushed off in about half the tidy runs. Walking round every desk thing was tried and cost more tidies than it saved (decision 48).
3. **Cap pushing** under the more exact friction: 5/16.
4. **gait3 from msi-1:** bench it; adopt it if it beats gait-v1.
5. **Hardware checks when parts arrive:** GS-4308MG and wide-camera sizes, the OLED module, the arm servo travel question (the trim tool asks it), the audio overlay, `tick_ms` on the real RK3566.
