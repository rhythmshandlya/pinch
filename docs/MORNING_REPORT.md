# Morning report: Pinch

*Updated 2026-11-19, evening. Details are in [DECISIONS.md](DECISIONS.md) (decisions 48–65). The owner's
private PRD and its traceability matrix are kept locally and out of git.*

## TL;DR

- **Direction set by the owner.** Three priorities, in order: (1) clean code and architecture, (2) a
  framework that can meet every PRD requirement, (3) the important features done and trained. msi-1 is never
  idle, and Modal runs alongside it.
- **Architecture:**
  - The three biggest files were split by responsibility. `pinch/body/` holds core, locomotion,
    manipulation and expression; `pinch/activities/` holds rest, social, defensive, recovery, tasks and play;
    `pinch/mind/` holds core, senses, selection, people and selfcare.
  - A **golden trace** proved each split changed nothing.
  - A **test suite** (`python tests/run.py`, 10 tests) covers exact replay, the decision log, the state
    layer, the observer and the golden trace.
- **Framework for the PRD:**
  - one simulation clock, giving exact replay;
  - a **decision log** (every switch, its reason, scores, what was noticed and how it felt);
  - an explicit **state layer**: each activity has a state, a time limit and phases; overruns end with a
    timeout; interrupted activities resume;
  - a **behaviour-quality observer** (oscillation, slow alternation, failure loops, state lock, repetition,
    unexplained switches);
  - **video clips** as evidence;
  - a research-backed **approach plan** for every feature area ([APPROACH_PLAN.md](APPROACH_PLAN.md),
    75 sources). Its verdict: *train the body, write the mind.*
- **Trained:**
  - **gait-v1** walking corrections: 0 falls and 86 % tracking on the harsh bench, against 69 % for the
    scripted gait. It also handles rough desks, a caught foot, bars up to 6–8 mm and ramps up to ~15–20°.
  - The **emote** balance policy: 0 falls.
  - **Running now:** walking v2 (curves, diagonals and pushing; 150M steps), two terrain runs (hurdles and
    ramps), emote2 (2 ms physics) on msi-1, and grasp-v6 queued overnight.
- **New abilities:**
  - **Self-righting** off its back in 0.8 s. The motion came from a search, robust from 8 different starts.
  - **Face recognition** (YuNet + SFace): people become familiar, and "I'm Sam" names the face in view.
  - **Pens go into the tray**, now that nothing slips out of the claw.
- **Honest problems:**
  - **Tidying is weak.** In the 16-layout bench the cap goes in 1–5/16 times and the eraser 7–12/16.
    16 layouts can't separate a real regression from chaos, so a 48-layout benchmark is being set up.
  - The toy ball still gets brushed off the desk in about half the tidy runs.
- **Modal spend: ≈ $22–26 of $100** (estimate; check the dashboard). **Firmware: 42/42** emulated checks,
  **40/40** selftest.

## Try it

```powershell
.venv\Scripts\python -m pinch.app                       # the pet on your laptop
.venv\Scripts\python tests\run.py                       # the test suite (about 6 minutes)
.venv\Scripts\python sim\clips.py                       # GIFs of Pinch in the twin -> media\
```

## "Done means": where each item stands

| Goal | Status | Evidence |
|---|---|---|
| Robot designed (CAD/STL) | ✅ 14 parts (31 pieces). All pass print rules and the wall check (the visor posts and bezel were thickened). Servos fit, and **241/241 poses are collision-free** | `cad/check.py`, `cad/walls.py` |
| Digital twin | ✅ desk, tray, desk things, five cliff beams (the side ones now low in solid wall), camera, face, noslip contacts | `sim/desk.py`, `pinch/sim_runner.py` |
| Locomotion trained + robust | ✅ residual gait-v1. It passes the flat, rough, caught-foot and terrain worlds. Capability table in `docs/HURDLES.md`. ⏳ terrain and walking-v2 training | `sim/robustness.py`, `sim/hurdle_test.py`, `sim/obstacle_test.py` |
| Expressive motions trained + robust | ✅ 21 emotes, 0/40 falls. ⏳ emote2 at 2 ms physics | `sim/robustness.py` |
| Pet brain in simulation | ✅ 12 behaviour scenarios (11–12/12, run to run), the state layer, the decision log and the observer. ⚠ Many PRD items are still partial or missing (see the private matrix) | `sim/pinch_scenarios.py`, `tests/` |
| Picks things up / arranges the desk | ⚠ partial: pens 4/12 picked, and all land in the tray; erasers 2/12; tidy bench as above | `sim/pick_bench.py`, `sim/tidy_bench.py` |
| Virtual pet on the laptop | ✅ `python -m pinch.app` | headless run |
| Firmware ready to deploy | ✅ in emulation: 42/42 checks, 40/40 selftest. The walking tick fits the RK3566 budget after a 2.2× gait speed-up. ⚠ Real-hardware items are unverified until the parts arrive | `robot/firmware_test.py`, `robot/selftest.py` |
| Budget ≈ ₹1L | ✅ ₹88,566 + prints (estimate) | `hardware/bom.py` |

## Compute

| Run | Where | Status | Notes |
|---|---|---|---|
| gait-v1 | Modal | ✅ used | `policies/gait.npz` |
| gait2-v1, gait3-v1 | msi-1 | done | gait2 76 %, worse than gait-v1; gait3 (bench-strength shoves) being benched |
| terrain-v1, terrain-hard-v1 | Modal | running | hurdles and ramps; bench against gait-v1 on `sim/hurdle_test.py` |
| gait-v2 | Modal | running | 48 commands + pushing references, 150M steps |
| emote2-v1, then grasp-v6 | msi-1 queue | running / queued | `C:/Users/armaa/pinch_queue.txt` on msi-1 |
| grasp-v5 | Modal | kept, not used | better isolated picks, worse whole-pet tidying |

## Needs you

1. Measure the OLED module before printing: its hole positions and glass thickness are estimates.
2. Decisions to review: the side cliff sensors low in the wall (48), noslip contacts (54), self-righting by
   searched motion (59), the state layer (63).
3. Test face recognition with two different people once there's a camera. Only same-person matching is
   tested.

## Next

1. A 48-layout tidy benchmark, then fix tidying at its causes (pick success, push accuracy, the ball).
2. Bench the terrain, gait3 and walking-v2 policies, and adopt the best.
3. Keep closing PRD gaps in dependency order: attention, the idle variation engine, play state machines,
   daily routine and user controls (see APPROACH_PLAN.md §4).
