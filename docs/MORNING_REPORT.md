# Morning report: Pinch

**TL;DR**
- Pinch exists as a full digital twin, a working virtual pet on your laptop, and deployable firmware
  that passes its tests on emulated hardware.
- The brain is done: emotions, neediness, touch, voice, meetings, memory, tricks.
- Learned emote balancing passes the robustness bench.
- Learned walking works well within its trained range; a harder-trained v3 was finishing as I wrote.
- **What isn't done:** reliable picking up (the claw needs a redesign: decision 20) and robust tidying
  (scripted pushing works but is brittle; a learned push skill is training).
- Hardware: about ₹91k plus prints, under the ₹1 lakh budget.
- Modal spend: about $4.3 of the $10 budget.

## Try it (2 minutes)

```powershell
.venv\Scripts\python -m pinch.app
```

Talk to it ("hi, I'm …", "dance", "play dead", "I'm sad today", "I'm in a meeting"), stroke it, tickle
it, mess up its desk, start a meeting, or tick **Listen** so it hears you through your microphone.

## "Done means": where each item stands

| Goal | Status | Verified by |
|---|---|---|
| Full robot designed (CAD/STL) | ✅ 13 printed parts, now with the **face visor** and the **desk tray**; ⚠ camera pocket to re-measure for the wide camera | `cad/build.py` (all solids valid, walls ≥ 1.6 mm); collision sweep: see "CAD sweep" below |
| Simulated as a digital twin | ✅ MuJoCo desk with edges, the tray and its ArUco marker, desk things, cliff sensors, a wide eye camera, the live face on the robot | `sim/desk.py`, `pinch/sim_runner.py`; camera localization tested on its own renders (8.6 mm / 0.9°) |
| Locomotion trained + robust | ✅ trained (PPO on Modal); robust: **0 falls in 40 harsh episodes**; tracks 91–98 % of commands inside its range; ⚠ under harsh conditions 44 % (scripted gait 69 %) → **v3** with harsher training (result below) | `sim/robustness.py` → [ROBUSTNESS.md](ROBUSTNESS.md) |
| Expressive motions trained + robust | ✅ 21 emotes; one trained policy balances any emote; **0 falls in 40 harsh episodes**, same attitude error as open-loop | `sim/robustness.py`, `sim/check.py` (all 21 within servo limits) |
| Pet brain: emotions, needs, neediness | ✅ | `sim/pinch_scenarios.py` → [BEHAVIOUR_TESTS.md](BEHAVIOUR_TESTS.md) |
| Touch | ✅ strokes / tickles / boops / holds / lifts → purr, giggle, startle, love eyes | scenarios + firmware test (real MPR121 driver on a fake chip) |
| Voice understanding | ✅ instant rules (17/17 test phrases) + local LLM (gemma4:e4b) for names, facts and mood | language tests; app |
| Situational smarts | ✅ meetings detected from sound alone (~65–100 s; instantly from "you're on mute"/"I'm in a meeting") → shh + silent fidgeting → celebrates after; comforts you when you're sad; seeks attention when ignored; raises the alarm about things at the desk edge; asks for help when it can't reach something | scenarios, firmware test |
| Activities and tricks | ✅ 17 activities, 21 emotes, tricks on command, shows off the tricks you praise | scenarios |
| Picks things up / tidies the desk | ⚠ **partial.** Tidying by pushing works in the twin (a cap into the tray on request, ~90 s; an eraser on its own once), but it's brittle across objects. **Picking up does not work yet** with the current claw (decision 20) | scenarios; contact traces |
| Memory | ✅ your name, facts, where things live, likes (learns praised tricks), routines, episodes; survives restarts | scenario "remembers after a restart" |
| Virtual pet on the laptop | ✅ `python -m pinch.app` | ran headless + windowed (screenshot verified) |
| Firmware ready to deploy | ✅ `robot/pinch_main.py` + drivers + HTTP API + systemd + install script; **8/8** on emulated hardware | `robot/firmware_test.py`, `robot/selftest.py` (32/32) |
| Budget ≈ ₹1L, parts in India | ✅ ₹90,672 + prints (~₹2–3k, an **estimate**: robu prices after upload) | `hardware/bom.py` → [BOM.md](../hardware/BOM.md) |

## Modal spend (estimate)

GPU time × list price (L4 about $0.80/h, A10G about $1.10/h). The exact figure is in your Modal
dashboard.

| Run | GPU | Time | ≈ Cost | Result |
|---|---|---|---|---|
| walk-v1 | L4 | ~58 min | $0.77 | learned to stand still (reward bug), replaced |
| emote-v1 | L4 | ~28 min | $0.37 | ✅ used (`policies/emote.npz`) |
| walk-v2 | A10G | ~49 min | $0.90 | ✅ walks; see robustness |
| grasp-v1 (two tries) | A10G | ~30 min | $0.55 | diverged (NaN), stopped: the claw can't lift a pen even with perfect placement |
| walk-v3 | A10G | ~50 min | ~$0.92 | harsher randomization, see below |
| push-v1 | A10G | ~35 min | ~$0.65 | hierarchical push skill, see below |
| image builds, downloads | CPU | — | ~$0.1 | |
| **Total** | | | **≈ $4.3** | nothing is left running |

## Decisions to review (details in [DECISIONS.md](DECISIONS.md))

1. **Face: a 2.42" 128×64 OLED (2:1), not a 4:1 bar** (#6). Eyes on a 32-pixel-tall bar can't squint
   or look up and down. It runs on its own I2C bus.
2. **Claw redesign needed for picking up** (#20): tongs-style vertical jaws, then retrain the grasp.
   This is the biggest open item.
3. **Wide camera module size** (#16): measure it and widen the front-wall pocket.
4. **"Put the pen in the pen stand" is scaled to the desk tray** (#10): Pinch is 10 cm tall.
5. **LLM is gemma4:e4b** (#15): llama3.2:3b wasn't installed. Run `ollama pull llama3.2:3b` and set
   `PINCH_LLM=llama3.2:3b` for faster replies.
6. **Pinch is 10 cm tall** with the face visor (was 9).

## What's left (in order)

1. **Claw redesign** (vertical, tongs-style jaws), then grasp RL again. Reliable picking up.
2. **Learned push skill**: evaluate push-v1 in the twin and wire it in as the default tidy method.
   The scripted pusher stays as the fallback.
3. **Learned walking v3**: if it beats the scripted gait on the harsh bench, make it the default for
   free walking.
4. **Hardware checks when parts arrive**:
   - measure the GS-4308MG and the wide camera
   - the arm servo travel question (the trim tool asks it)
   - the audio overlay
   - the minimal VL53L0X driver
5. Object recognition beyond Pinch's coloured toys (a small detector on the RK3566's NPU).
