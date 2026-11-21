# Code audit: the mind and behaviour code (pillar D)

Audited: `master` at commit 3bbc1a9 as checked out, read-only. One unmerged branch is labelled **(unmerged)** wherever it
appears: `worktree-agent-ac628a12174226b32` (4 commits ahead of master: behaviour mix, tidy need, `docs/PET_BENCHMARK.md`).
Nothing was run (no tests, benches or training). Counts come from `grep` and from a short read-only Python script that
parses the files (`ast`), so they are exact for what they measure. Plain-language terms are defined on first use.

Short verdict: the mind is a **single-process utility chooser** (every activity scores itself, the highest score wins, with a
few rules for interrupting) sitting on a good *shell* (decision log, observer, recorder, golden trace, timeouts). The
weakness is the middle: no layers, no declared priorities, all tuning numbers inside code, a personality that is five fixed
numbers, and a world model hard-wired to one desk, one tray and four coloured objects. On the real robot two input paths the
mind depends on are not connected at all (see 8, 10 and 12).

---------------------------------------------------------------------------------------------------------------

## 1. Decision architecture (piece D12)

### 1.1 How the next activity is chosen

It is a **utility chooser with a thin state label and a timeout/resume layer on top**. It is not a behaviour tree and not a
state machine.

* Every activity is an object with `utility(m)` (a number: "how much do I want this now", 0 = not at all) and `run(m)` (a
  Python generator that yields steps). Contract and defaults: `pinch/activities/base.py:16-30` (`state`, `interruptible`,
  `min_s`, `max_s`, `quiet_ok`, `own_jitter`).
* `LIBRARY` is a hard-coded list of 26 instances in a fixed order; ties go to the first:
  `pinch/activities/__init__.py:36-38`.
* Selection, twice a second (`next_choice = t + 0.5`, `pinch/mind/core.py:147`), in `Selection._choose`
  (`pinch/mind/selection.py:55-85`):
  1. score = `utility(m)`; forced to 0 if the mind is "quiet" and the activity is not `quiet_ok` (selection.py:58-60);
  2. +0.1 if it is the activity that was cut short in the last 60 s (`RESUME_S, RESUME_BONUS`, selection.py:53,61-62);
  3. + 0.02 x random jitter (selection.py:63), from the shared RNG, or a separate RNG for activities with `own_jitter`;
  4. the running activity keeps at least 90 % of the score it started with ("commitment", selection.py:67-68);
  5. best wins. A switch happens only if the current activity is `interruptible` AND (`urgent`: best >= 1.5 and > current +
     0.1; or `better`: > current + 0.15 and the current one has run longer than its `min_s`), selection.py:77-79. A
     non-interruptible activity can only be pre-empted by `startle` or `right_self`, and only when urgent (selection.py:79).
* The `state` (`idle, explore, social, play, alert, defensive, rest, recover, task`, base.py:13) is only a label per
  activity: it is logged and used by the observer and realism score, and has **no behavioural rules attached**
  (decision 63, `docs/DECISIONS.md:763`). `Mind.state` returns the label of the running activity (selection.py:98-101).

### 1.2 One tick of the mind, in order (twin; the robot firmware has the same shape)

Host loops: `PinchSim.step` (`pinch/twin/core.py`, the mind call is at line 134; the mind runs every 5th 20 ms tick = 10 Hz,
`pinch/twin/common.py:26`) and `PinchFirmware.step` (`robot/firmware/core.py:157-230`):

1. Sensors write into the mind. Twin: `TwinSensors.perceive()` (`pinch/twin/sensors.py:48`) sets `world.pose`, calls
   `world.see(...)` per object, sets `owner.present`. Robot: `robot/firmware/senses.py:123-133` does `world.see(name, xy)` for
   coloured blobs, `m.observe(obs)`, `m.see_person(...)`. Speech, touch and hearing arrive by direct calls `hear_text`,
   `hear_summary`, `hear_event`, `touch` (`pinch/mind/senses.py:11,102,120,131`).
2. `MindCore.tick(dt, body_busy, body_ok)` (`pinch/mind/core.py:118-152`):
   a. `llm.poll()` then `_apply_llm` (core.py:123-126);
   b. owner presence edge then `affect.feel("owner_returned"/"owner_left")` (core.py:127-136);
   c. `affect.quiet` = meeting, `quiet_until`, or quiet hours (core.py:138);
   d. `Affect.tick` grows needs and relaxes mood (`pinch/affect.py:117-139`);
   e. `_self_preservation` (danger-map decay, fear of heights, charging relief; `pinch/mind/selfcare.py`);
   f. `_timeout` ends an activity past `max_s` (selection.py:87-96);
   g. every 0.5 s: `situations(world, affect, memory)` (`pinch/world.py:160`), then `_choose` (selection.py:55), then, if
      the winner is new, `_start` (selection.py:114-130: writes the decision record, creates the generator `a.run(self)`);
   h. `_advance(body_busy, body_ok)` (selection.py:132-185) pulls steps from the generator. `face`, `sound`, `look_at`,
      `phase`, `quiet`, `remember` are applied at once (up to 20 per call). The first *motion* step (`MOTION_STEPS`,
      `pinch/mind/common.py:19`: emote, walk, goto, turn_to, pose, push, pick, drop, release, recover, right_self) is returned,
      and the mind then waits until the body is no longer busy (`waiting_motion`).
3. Host: if `cancel`, `body.cancel()`; then `body.start(motion)` (`pinch/body/core.py:55-63`) builds a generator `_<kind>(...)`.
   Sounds: `mind.outputs()[2]` (core.py:160-170) are played (`hw.play`).
4. Host reflexes run after the mind in the same tick and can take over the body: cliff beams (`CliffReflex`,
   `pinch/touch.py:327`), foot-over-edge `body.reflex(("recover",))` (`robot/firmware/core.py:206-211`); `StuckSense` and
   `BodySense` feed `mind.touch("stuck"|"lifted"|...)` (core.py:200-217).
5. Every 20 ms: `BodyCore.tick` (`pinch/body/core.py:80-100`) takes one pose vector from the active task generator, or the
   held pose plus idle-life offsets when no task runs (`pinch/body/life.py:42`), then `crab.tick(q)`.
6. Every 3rd tick the face renders `mind.face_expr` and `mind.gaze` (`robot/firmware/core.py:228-230`).

### 1.3 Are the layers separate?

**No. Perception, decision, behaviour planning and execution are interleaved in one object.**

* `Mind` is seven mixins glued into one class sharing one `self`:
  `class Mind(Senses, Selection, PeopleAware, Privacy, SelfCare, Curious, MindCore)` (`pinch/mind/__init__.py:28`).
* Activities receive the whole mind `m` and read and write any attribute. Examples: `Tidy` writes `m.tidy_request`,
  `m.last_trick`; `Investigate` calls the private `m._suspend` and reads `m.decisions[-1]["prev"]`
  (`pinch/activities/curiosity.py:125-127`); `SeekAttention` creates `m.seek_tries` / `m.seek_next` with `getattr`
  (social.py:138-153); `ReactTouch` creates `m.touch_state` lazily (social.py:256-265).
* Perception output (`situations()`, `world.see`, `attention.notice`) and the decision (`utility`) share one loop and one data
  object; there is no typed "percept". The one real interface is the **motion step vocabulary** (tuples whose first item is a
  kind) between mind and body. That boundary is clean and is the best part: it is why the robot runs the same mind.
* Behaviour planning (where to stand, which side to circle) lives inside the activity generators, e.g. `Investigate`'s
  `_spot` does desk-edge, tray and fear checks itself (`curiosity.py:30-48`).

### 1.4 Declared priority order (safety above everything ... idle at the bottom)?

**No declared order.** It exists only as the *magnitudes* of utilities scattered over the activity files:

| Utility | Activity (file:line of the return) |
|---|---|
| 6.0 | right_self when flipped (`recovery.py:15`) |
| 5.0 | low_power at battery <= 10 % (`recovery.py:112`) |
| 4.0 | unstick (`recovery.py:46`) |
| 3.0 | hide from a feared person (`defensive.py:83`) |
| 2.2 | startle (`defensive.py:15`) |
| 2.0 | go_quiet in a meeting (`social.py:208`) |
| 1.4-2.2 | greet (`social.py:92`) |
| 1.7 | react_touch (`social.py:279`) |
| 1.6 | obey (`social.py:16`) |
| 1.5 | tidy when asked (`tasks.py:18`) |
| 1.35 | guard_edge (`tasks.py:97`) |
| 1.3 | after_meeting (`social.py:237`) |
| 0.25 and below | idle (`rest.py:14`), orient (0.15 + ...), most spontaneous ones |

The safety reflexes that must come first live **outside** the mind, in the host (cliff reflex, edge recover, e-stop): the right
place. But the spoken e-stop reaches them through the mind's rules (`pinch/language.py:98` `pick("estop", ...)`, then
`mind.estop_request`, `senses.py:72-73`). The only written rule about order is that "urgent" means score >= 1.5
(selection.py:77). Consequences: `hide` (3.0) outranks a meeting; nothing states that `low_power` (5.0) must beat `unstick`
(4.0); a new activity returning 2.5 silently outranks `startle`. `QUIET_OK` in `base.py:7` is a stale, unused set (the live
flag is the class attribute `quiet_ok`; only `sim/benches/longrun.py:70` rebuilds a dict from the classes).

### 1.5 Interruption and resumption

* Interruption: `_choose` as above. `_interrupt()` (core.py:82-84) only forces an immediate re-score; it is called from touch,
  requests, scolds, stuck, loud scares (many places in `senses.py`, `people.py`, `curiosity.py`).
* Hard stop on overrun: `_timeout` (selection.py:87-96), reason "timeout".
* Resumption: only after an *urgent* pre-emption (selection.py:80-81 `_suspend`); the pre-empted activity gets +0.1 for 60 s,
  with a stack of 3 (`RESUME_DEPTH`, selection.py:105-122). After a merely "clearly better" switch nothing is remembered.
  `Investigate` works around this by suspending its predecessor itself (`curiosity.py:125-127`, `NO_RESUME` at curiosity.py:24).
* A resumed activity restarts its generator from the top (`_start` always calls `a.run(self)`); there is no mid-activity
  checkpoint. A resumed `Tidy` copes because it re-derives its to-do list from the world.
* The request slot is a single value (`self.request`, core.py:49): two commands in a row overwrite each other
  (`senses.py:60-80`).

### 1.6 Output channels: who commands, who resolves conflicts

| Channel | Commanded by | Conflict rule | Evidence |
|---|---|---|---|
| Legs | `walk, goto, turn_to, push, recover, right_self, pose` steps, one at a time | The body has **one task slot**; `start()` replaces it. Reflexes pre-empt (`reflex()` marks the cut step failed). The mind waits while the body is busy (selection.py:135-139). Idle-life offsets only when no task runs. No blending of two requests | `pinch/body/core.py:55-71,93-94` |
| Claws / arms | `emote`, `pick`, `drop`, `release`, `push`; a held object overrides the arm pose | same single slot; the only layering is "carried object forces the arm" and emote style | `body/core.py:95-98` |
| Gaze / body orientation | `look_at` sets `mind.gaze` (eyes, selection.py:155-157); `turn_to` turns the body; `goto/push/pick/drop` force gaze (0, 0.3) when they start (selection.py:181-183); otherwise attention drives gaze if no activity aimed it in the last 3 s | time-based, last writer wins. Gaze is **eyes only**: no head or body layered on top of walking. The only body-turn is the emote `gaze` style (`crab/emotes.py:344-346`) | `core.py:160-170` |
| Face | `face` steps set `mind.face_expr` (selection.py:150-151); `say_mood` | last writer wins and the face **never reverts to the feeling by itself**: the `outputs()` docstring says "the feeling unless an activity set one recently" but the code just returns `self.face_expr` (core.py:160-170). A long `tidy` keeps "focused" whatever the mood does. Only `Idle` re-reads `affect.feeling()` (rest.py:20) | `core.py:160` |
| Sound | `sound` steps append to a list drained every tick (selection.py:152-154, core.py:163) | the only filter is `affect.quiet`. No priority, no "not two at once", no cool-down; a sound followed by a motion plays on the first tick of the motion (loose sync) | `selection.py:152` |

There is **no per-state rule for any channel** ("in `alert` the face may not be overwritten", "in `rest` no legs"): this is
exactly what piece D12 asks for and it is absent.

### 1.7 What is good here (confirmed)

Decision log with reason and top-4 candidates for every switch (selection.py:18-42); `max_s` on everything; commitment and
minimum-time hysteresis; the `own_jitter` separate RNG so adding an activity does not reshuffle the others (selection.py:44-51);
exact replay under a simulated clock. A utility chooser is a legitimate, simple choice for ~25 behaviours. The fault is the
missing structure around it, not the choice.

---------------------------------------------------------------------------------------------------------------

## 2. Inventory

### 2.1 Activities: 26 (`pinch/activities/__init__.py:36-38`)

Typical duration is read from the step script (not measured). "u" is the utility.

| # | name (class, file:line) | state | trigger / utility | what it does | max_s; typical |
|---|---|---|---|---|---|
| 1 | idle (`Idle`, rest.py:9) | idle | u = 0.25 always | face = feeling; 15 % look_around, 10 % stretch/yawn/fidget/sneeze, else wait 1.5-4 s; loops | 120 s; until pre-empted |
| 2 | obey (`Obey`, social.py:10, non-interruptible) | social | a request pending: 1.6 | trick (emote or spin), come, go_away, stop, sleep, wake, play, tidy/put_away/fetch (sets `tidy_request`), quiet (900 s), how_are_you, greet, goodbye, teach_home | 120 s; 3-15 s |
| 3 | go_quiet (`GoQuiet`, social.py:202) | social | meeting seen or `quiet_until`: 2.0 | shh emote, then silent fidget / look_around / wait loop | 7200 s; the whole meeting |
| 4 | after_meeting (`AfterMeeting`, social.py:231) | social | meeting ended < 15 s ago: 1.3 | happy trill, excited, walk to owner, beg | 60 s; ~15 s |
| 5 | seek_attention (`SeekAttention`, social.py:133) | social | 0.9 x (social need - 0.45) x neediness + 0.25 if ignored > 180 s; back-off 90 s doubling to 1200 s | turn to owner, chirp; escalates with ignored time to walking over, tricks, whimper | 120 s; 10-40 s |
| 6 | comfort (`Comfort`, social.py:107) | social | owner sad/stressed/tired: 1.25 x strength, 300 s gap | walk to owner, purr, handshake, low pose | 120 s; ~15 s |
| 7 | celebrate (`Celebrate`, social.py:171) | social | owner happy: 0.95 x strength, 120 s gap | joy face, fanfare, dance/applause/excited | 60 s; ~5 s |
| 8 | react_touch (`ReactTouch`, social.py:248, non-interruptible) | social | touch < 1 s ago: 1.7 (0 if asleep and the wake rule says no) | by gesture and place: purr, giggle, boop, sneeze, handshake, wave, low pose, scaled by habituation | 60 s; 2-8 s |
| 9 | startle (`Startle`, defensive.py:10, non-interruptible) | alert | safety need > 0.6: 2.2 | scared face, squeak, scared emote, step back 4 cm | 30 s; ~5 s |
| 10 | orient (`Orient`, defensive.py:25) | alert | attention focus off to the side (> 0.4 rad) and < 2 s old: 0.15 + 0.9 x salience | look, turn toward it, wait 1.5 s, attention moves on | 15 s; ~4 s |
| 11 | right_self (`RightSelf`, recovery.py:8, non-interruptible) | recover | flipped: 6.0 | squeak, up to 2 righting tries, then plead | 60 s; 10-25 s |
| 12 | unstick (`Unstick`, recovery.py:38) | recover | StuckSense says stuck: 4.0 | grumble, 2 x `recover` shuffle, up to 4 pleas 20 s apart | 180 s; 10-100 s |
| 13 | hide (`Hide`, defensive.py:76) | defensive | a person with fear R > 0.45 present: 3.0 | scared, whimper, go home, low pose, wait while they stay | 600 s |
| 14 | shy (`Shy`, defensive.py:53) | defensive | a stranger present (F < 0.3): 0.9 | go home, peek at the owner seat every 4-6 s (<= 12 peeks) | 120 s |
| 15 | head_home (`HeadHome`, recovery.py:75) | recover | 3 x (hunger - 0.3), hunger = (1 - battery)^2 | go home, beg for a charge every 90 s | 300 s |
| 16 | low_power (`LowPower`, recovery.py:104, non-interruptible) | rest | battery <= 10 % and not charging: 5.0 | save memory, sleep emote, doze | 7200 s |
| 17 | tidy (`Tidy`, tasks.py:10) | task | asked: 1.5 for 120 s; unasked: 0.75 x tidiness x (0.6 + 0.6 x tidy need) when a taught-home object is > 6 cm from home | grasp (pick + drop) or push the object to its home/tray; ask for help after 2 failures | 400 s; 30-200 s |
| 18 | guard_edge (`GuardEdge`, tasks.py:89) | task | an object within 3.5 cm of an edge: 1.35, 120 s gap | hurry over, point, look at owner, beg | 120 s |
| 19 | play (`Play`, play.py:10) | play | 0.8 x playfulness x (play need - 0.35) x (1 - sleep) x (1 - hunger) | choose a PlayBook game (push/carry/show a toy to tray, middle or owner), learn from the result | 300 s |
| 20 | show_off (`ShowOff`, play.py:71) | play | owner attending (30 s) and mood positive: 0.5 x valence, 90 s gap | turn to owner, a liked trick, wait 6 s for praise | 120 s |
| 21 | explore (`Explore`, play.py:94) | explore | 0.65 x (curiosity need - 0.35) x (1 - sleep) x (1 - hunger) | go to a random safe desk spot, look around | 300 s |
| 22 | rest (`Rest`, rest.py:33) | rest | 1.5 if asked + 1.2 x sleep^2 + 1.4 x max(0, energy - 0.7) + 0.5 x owner-away | go home, yawn, nap loop; a whole night is one rest | 9 h |
| 23 | sulk (`Sulk`, social.py:186) | social | just scolded (< 4 s): 1.1 | sad face, whimper, sad emote | 120 s; ~8 s |
| 24 | greet (`Greet`, social.py:87) | social | owner returned < 8 s ago, not quiet: 1.4 + 0.8 x warmth | beep, turn to owner, wave, excited, dance if warmth >= 0.6 | 60 s; ~8 s |
| 25 | watch (`Watch`, curiosity.py:60) | alert | an unnamed thing is moving: 0.6 + 0.3 x confidence | eyes follow it, turn to keep it in view, up to 15 s | 20 s |
| 26 | investigate (`Investigate`, curiosity.py:93) | explore | an unfamiliar, still, unnamed thing: (0.5, or 0.3 + 0.4 x curiosity need, + 0.3 x interest) x curiosity trait x damping | approach-retreat excursion: notice, approach, pause, circle, retreat | 120 s; 30-90 s |

Unmerged additions: an `ActivityMix` wrapper around every utility (see 2.5) and `pinch/activities/pet_draft.py` with 7 more
activities that are **not in LIBRARY** and imported by nothing: BatBall, Sniff, Groom, Doze, CheckIn, ShowTrick, EyeMess
(unmerged).

Of the 26, the ones that exist to look alive when nothing happens: idle, explore, watch, investigate, play, show_off, rest
(7). Everything else reacts to an event or serves a task.

### 2.2 Emotes (body motions): 22 keyframed clips, `crab/emotes.py:83-235` (`E["..."]` in `_build`)

idle, wave, angry, happy, shy, sleep, shh, fidget, dance, beg, play_dead, stretch, excited, sad, curious, scared, point,
handshake, yawn, sneeze, look_around, applause. Durations are roughly 2-6 s (sum of each `_seq`). The mind never names
`idle` (a loop clip); `angry, happy, curious, applause, play_dead, shy, sneeze` are reached only through spoken tricks, touch
reactions or the idle dice. Poses `stand, low, nap` (`pinch/body/common.py:18`) and the `spin` trick (a walk) are not emotes.

### 2.3 Faces: 25 eye expressions, `pinch/face.py:40-66` (`EXPRESSIONS`)

neutral, happy, joy, excited, love, content, curious, surprised, sad, lonely, angry, grumpy, scared, shy, sleepy, asleep, bored,
focused, quiet, dizzy, dead, proud, squeeze, thinking, music. `dead`, `thinking` and `music` are never set and never returned
by `feeling()`. Overlays: zzz, !, ?, ..., shh, note, tear, sweat, anger (`_overlay`, face.py:199); 8 glyph eyes (`_glyph`, face.py:171).

### 2.4 Sounds: 17 procedural tones, `pinch/sounds.py:18-36` (`SOUNDS`)

greeting_beep, happy_trill, curious_chirp, startled_squeak, sad_tone, sleepy_hum, angry_buzz, purr, giggle, whimper,
attention_chirp, fanfare, yawn, sneeze, uh_oh, boop, low_battery. Never used by the mind: `sneeze`, `low_battery`.

### 2.5 Drives, needs, emotions and other state variables, with update rules

| Variable | Where | Update rule |
|---|---|---|
| need `social` | `pinch/affect.py:32-39`, rates :27 | +1/600 per s x neediness x (1 + 1.5 if ignored > 120 s) (:126); -0.10..-0.45 on petting, speech, praise (table :50-82) |
| need `play` | same | +1/900 per s x playfulness; -0.5 on played_with |
| need `curiosity` | same | +1/700 per s x curiosity trait; -0.3 on novelty |
| need `sleep` | same | +1/7200 per s x sleep drive (3 at night 22:00-06:00, `routine.py:16-17`); the "x2 if awake > 10 h" branch (affect.py:130) is dead because `awake_hours` is never passed (`core.py:139`); Rest lowers it by 0.08 (day) or 0.02 (night) per 10 s nap loop |
| need `energy` | same | = 1 - battery fraction each tick (:132-133) |
| need `tidy` | same | master: +0.02 per out-of-place object per 0.5 s (core.py:145-146), -0.6 on tidied. **(unmerged)**: grows 1/60 per s for fresh mess, 1/420 for old mess, fades 1/300 (`_notice_mess`) |
| need `safety` (fear) | same | -0.05 per s; +0.3 to +0.8 on lifted, shaken, loud noise, sudden motion, fell, stuck |
| valence, arousal | affect.py:89-90 | pushed by 30 events (`EVENT_EFFECTS`), relax to a resting point shaped by needs, time constants 30 s and 12 s (:135-138) |
| named feeling | `Affect.feeling` affect.py:146-187 | if-chain over safety, last event (< 4 s), sleep, valence, arousal, needs |
| `ignored_s` | affect.py:97,121-124 | seconds the owner is present but not attending |
| attention stimuli | `pinch/attention.py:17-20` | salience = base x strength / (1 + 0.6 x familiarity); half-life 3 s; habituation half-life 60 s; inhibition of return 20 s x 0.35; focus switches only if 1.25x better |
| per-thing novelty and wariness | `pinch/novelty.py:33-49,111-244` | novelty x 0.55 per safe look; wariness by Rescorla-Wagner toward 0 (alpha 0.4) or 1 (0.5); familiarity remembered per signature, half-life 3 days |
| touch habituation (5 channels) and sensitisation | `pinch/touch.py:81-86` | h += alpha (1 - h) per touch, decays with tau 60 s; an intense touch halves all h and adds 0.3 to s (tau 20 s) |
| wake pressure | touch.py:128-164 | per-gesture weights, decay tau 30 s, wakes at 1.0 |
| person F (familiarity), W (warmth), R (fear) | `pinch/people.py:21-33,61-95` | F grows with presence, half-life 90 d; W moves with petting/praise/scolding, pulled to 0.2F, half-life 30 d; R from rough handling, half-life 3 d (twice as fast with a trusted person near) |
| danger-map spots | `pinch/safety.py:17-24,27-77` | Rescorla-Wagner on near-falls / stuck / fell; fear half-life 8 awake hours; safe passes unlearn; at most 40 spots |
| play book, per game | `pinch/play.py:71-84,96-130` | n, last 10 wins, last 10 prediction errors, running mean toy displacement, boredom (decay 1200 s), a week's ban if a toy ends near an edge |
| likes per trick/game | `pinch/memory.py:106` | +0.25 on praise within 12 s of a trick, -0.3 on a scold (senses.py:56-65) |
| owner mood, meeting probability | `pinch/hearing.py:126-199` | prosody to arousal and a label; meeting evidence +0.02 per s of talk, +0.35 on call phrases, ends after 60-90 s quiet |
| battery hunger | `pinch/safety.py:79` | (1 - charge)^2 |

Unmerged: `ActivityMix` (`pinch/mind/mix.py`) adds per-activity and per-class time-share accounting (30-minute window, activity cap
30 %, "task" class cap 12 %) and `sated_s` (tidy 30-75 min), multiplying every non-exempt utility.

---------------------------------------------------------------------------------------------------------------

## 3. Tunable numbers (piece D15)

### 3.1 Counts

Method: parse each file with `ast`, count numeric literals outside docstrings. "Non-trivial" leaves out 0, 1, -1 and 2.
Both numbers over-count a little (array indices, 24 hours, loop bounds) and miss numbers inside strings. Treat as +-15 %.

| Group | Files (non-trivial count per file) | Non-trivial numbers |
|---|---|---|
| Activities | `pinch/activities/*.py`: base 2, curiosity 60, defensive 22, play 27, recovery 23, rest 31, social 85, tasks 19 | **269** (all literals 430) |
| Mind core | `pinch/mind/*.py`: core 26, selection 16, senses 18, selfcare 13, curiosity 5, people 7, privacy 1 | **86** (all 194) |
| Behaviour models | affect 148, novelty 61, people 51, play 36, safety 18, routine 21, attention 18, memory 20 | **373** |
| Perception, touch, hearing | touch 74, hearing 86, world 51, language 9, body/life 21 | 241 |
| **What shapes behaviour** (activities + mind + models) | | **about 730** |
| Content tables (not tunables as such) | emotes 269, face drawing 220, sounds 129 | 618 |
| Everything counted (31 files) | | 1,602 (all literals 2,718) |

### 3.2 Thirty examples

| # | path:line | What |
|---|---|---|
| 1 | `pinch/mind/selection.py:53` | RESUME_S 60, RESUME_BONUS 0.1 |
| 2 | `selection.py:63` | tie-break jitter 0.02 |
| 3 | `selection.py:68` | commitment 0.9 |
| 4 | `selection.py:77` | urgent threshold 1.5, margin 0.1 |
| 5 | `selection.py:78` | "clearly better" margin 0.15 |
| 6 | `pinch/mind/core.py:147` | re-score every 0.5 s |
| 7 | `core.py:127` | owner "present" within 120 s |
| 8 | `core.py:98` | tidy retry window 600 s, 30 s gap, 3 failures |
| 9 | `pinch/affect.py:27` | need growth rates 1/600, 1/900, 1/700, 1/7200 |
| 10 | `affect.py:51` | "petted": valence +0.35, arousal +0.05, social -0.35, safety -0.3 (30 such rows, :50-82) |
| 11 | `affect.py:126` | ignored > 120 s: loneliness x 2.5 |
| 12 | `affect.py:137-138` | mood relaxation 30 s / 12 s |
| 13 | `affect.py:151-186` | about 25 thresholds that decide the named feeling (0.55, 0.8, 0.45 ...) |
| 14 | `pinch/activities/defensive.py:15` | startle 2.2 at safety > 0.6 |
| 15 | `pinch/activities/recovery.py:15,46,85,112` | 6.0 / 4.0 / 3.0 x (hunger - 0.3) / 5.0 at battery 0.10 |
| 16 | `pinch/activities/social.py:92` | greet 1.4 + 0.8 x warmth within 8 s |
| 17 | `social.py:147` | seek attention 0.9 x (need - 0.45) |
| 18 | `social.py:149` | back-off 90 s doubling to 1200 s |
| 19 | `pinch/activities/play.py:20` | play 0.8 x ... x (need - 0.35) |
| 20 | `play.py:101` | explore 0.65 x (curiosity - 0.35) |
| 21 | `pinch/activities/rest.py:41` | rest 1.5, 1.2, 1.4, 0.7, 0.5 |
| 22 | `pinch/activities/tasks.py:22` | tidy 0.75 x tidiness x (0.6 + 0.6 x tidy need) |
| 23 | `pinch/attention.py:17-20` | base saliences 0.9/0.85/0.65/0.7/0.5/0.4, fade 3 s, inhibition 20 s |
| 24 | `pinch/novelty.py:37,41,45,47` | MOVE_V 0.03, per-look novelty 0.55, threat alpha 0.4, standoff 0.09-0.25 m |
| 25 | `pinch/play.py:25-26,84,90` | edge limits 0.10/0.08/0.06 m, ban 7 days, score weights 0.6/0.4/0.8/0.25, 15 % random game |
| 26 | `pinch/people.py:20` | half-lives 90 d / 30 d / 3 d |
| 27 | `pinch/touch.py:81,86,137` | habituation alphas, floor 0.15, wake tau 30 s |
| 28 | `crab/emotes.py:381-387` | emote style: tempo/amplitude per arousal 0.30/0.25, ranges 0.78-1.2 and 0.6-1.15, jitter 0.06 |
| 29 | `pinch/body/life.py:46,67` | breathing 0.015 rad, micro-action gap exponential(8 s / (0.4 + liveliness)) |
| 30 | `pinch/world.py:178,181,182` | near-edge 3.5 cm, settled after 20 s, out-of-place at 6 cm |

### 3.3 Is any behaviour setting stored as data?

**No for behaviour. Yes only for learned memory and four owner settings.**

* No TOML/JSON is read for utilities, rates, thresholds, cool-downs or probabilities: a `grep` for
  `tomllib|toml|json.load` in `pinch/mind`, `pinch/activities`, `affect.py`, `attention.py`, `novelty.py`, `play.py`, `people.py`,
  `crab/emotes.py` finds nothing. (`configs/` holds RL experiment and training files; `policies/registry.toml` is for trained
  policies.)
* Owner settings live in the memory JSON: `quiet_hours`, `utc_offset_h`, `face_recognition`, `wake_on_touch`
  (`pinch/routine.py:20`, `touch.py:137`).
* Learned values are JSON in `~/.pinch/memory.json` (see 6).
* Emotes, faces, sounds, personality, quiet-mode flags and every utility are Python source. a local design note (not in the repository) says "every number
  lives in `brain/pet_mind.py`": that directory does not exist any more.

---------------------------------------------------------------------------------------------------------------

## 4. Character (piece D13)

### 4.1 Personality

There is a personality object, but it is **five fixed floats that never change and are never saved**:
`Personality(neediness=1.0, playfulness=0.8, timidity=0.5, tidiness=0.7, curiosity=0.8)` (`pinch/affect.py:17-23`).

* Used as multipliers in only a handful of places: neediness (`affect.py:126`, `social.py:147`), playfulness (`affect.py:128,136`,
  `play.py:20`), curiosity (`affect.py:129`, `curiosity.py:107`), tidiness (`tasks.py:22`), timidity (`affect.py:105-108`,
  `mind/curiosity.py:15,38`, novelty `threat0`, `novelty.py:115`).
* Created once per Mind with the defaults (`pinch/mind/core.py:24`). No caller outside tests passes another one (`grep
  "Personality("` finds only `affect.py` and `core.py`).
* Not in the memory file: `Memory._blank` (`memory.py:23-27`) has no personality, needs or emotion keys, so after every reboot
  needs and mood return to the `Needs` defaults (affect.py:32-39).
* **It does not persist or develop.** What drifts with experience is separate and small: `likes` per trick/game, `PlayBook`,
  per-person F/W/R, `familiar` things. Those are learned preferences, not traits.
* No "character bible" exists in code or docs (no file says who Pinch is, what it never does, how it moves). The unmerged
  `docs/PET_BENCHMARK.md` compares companion robots (Vector cool-downs, Moflin's four traits that grow with treatment, aibo's
  raised-by-owner personality) but changes no trait code.

### 4.2 Emotion model

* Dimensions: **valence** (-1..1) and **arousal** (0..1) as a point (`affect.py:89-90`), plus seven **needs** (0..1). A named
  feeling is read off by an if-chain (`feeling()`, affect.py:146-187).
* Change over time: events push (30-row `EVENT_EFFECTS`, :50-82); needs grow per second (:27); valence and arousal relax toward a
  resting point that depends on the needs (:135-138). There is no separate slow "mood" layer, no per-trait baseline, no
  confidence or trust dimension (Vector has them, per the unmerged benchmark note).
* What it affects: (a) the **face**, only through `Idle` and `say_mood`; (b) emote **tempo and amplitude**
  (`body/expression.py:39-44` `mood_fn`, then `EmoteStyler.mean`, `crab/emotes.py:395-400`); (c) idle-life liveliness
  (`twin/core.py` sets `life.liveliness = arousal`); (d) utilities through the needs; (e) the `quiet` flag, fear and hunger. It does
  **not** change walking speed, gait or posture outside emotes.
* Hard cut-off: `needs.safety > 0.55` always returns "scared" before anything else (affect.py:151); the "recent event" window is 4 s.

---------------------------------------------------------------------------------------------------------------

## 5. Expression (piece D14)

* **Emotes are hand-authored keyframes in Python.** `Emote(name, keyframes=[(t, pose_vector)])` with smoothstep interpolation
  (`crab/emotes.py:52-71`); poses are `pose(STAND, overrides)` with `fnmatch` joint patterns (:28-41) and `crouch()`. 22 clips.
* **Variation is a procedural layer on top.** `EmoteStyler` (`crab/emotes.py:377-470`) draws a `Style` per play from a seeded
  RNG: tempo, amplitude, arm asymmetry, arm lag, mirror, a body-turn "gaze". Arousal makes it faster and bigger, sleepiness slower
  and smaller, valence a little bigger, `gain` (habituation) smaller; the same emote twice running is pushed apart (`MIN_APART`
  0.06); every variant is clipped to joint limits, the shin limit and a servo-speed budget (`make_safe`, `allowed_speed`), and
  arms cannot collide (a test samples them). This is a real strength, and it cites Laban and Knight & Simmons.
* **Learned balance on top of keyframes:** with `policies/emote.npz` the legs get small live corrections
  (`pinch/body/expression.py:46-75`).
* **Layers that exist:** (1) base keyframes, (2) style, (3) emote-policy corrections, (4) idle-life offsets only when nothing runs
  (`body/life.py`, additive, <= 0.03 rad legs and 0.15 rad arms), (5) carried-object arm override. **There is no gaze on top of
  walking and no emote while walking**: one task slot. The observer even has a `conflicting_movement` detector for it
  (`pinch/observer.py`, docstring).
* **Face:** `Face.update` (`face.py:100-125` region) blends the 25 presets exponentially (~80 ms) and adds saccade jitter and
  blinks from its own RNG (blink every 2-6 s, 25 % double). Expressions are parametric presets (lid angle, lid cover, scale, glyph,
  overlay) in a dict, not animations.
* **Sound:** 17 tone recipes (`sounds.py:18-36`): lists of (start Hz, end Hz, seconds, gap).
* **Timing and synchronisation: none designed.** Face, sound and motion are separate steps of a Python generator. A
  `("sound", x)` followed by `("emote", y)` start in the same tick; nothing aligns a sound to a keyframe, a face change to a motion
  beat, or the eyes to a body turn. The face does not follow the emote's phase.
* **Mood to motion:** only emote tempo and amplitude. Locomotion style is not mood-dependent.
* **Authoring tool: none.** `sim/view.py` plays the existing emotes in the MuJoCo viewer (`--emote NAME`, arrow keys). There is no
  editor, no timeline, no way to preview face + sound + motion together, no export format. A new emote means writing Python and
  re-running `sim/check.py` for the contact sheet (it regenerates `docs/images/emotes.png`, which shows as modified in `git status`).

---------------------------------------------------------------------------------------------------------------

## 6. Memory and learning (piece D16)

### 6.1 What is remembered, where, in what format

One JSON file, `~/.pinch/memory.json` (env `PINCH_MEMORY`), written whole every 60 s and at shutdown (`pinch/memory.py:19`,
`robot/firmware/common.py:34`), atomically with fsync (`memory.py:58-72`).

| Key | Content | Written by | Used by |
|---|---|---|---|
| `owner` | name, up to 40 facts of 120 chars | `set_owner_name`, `add_fact` (LLM only) | LLM prompt context only |
| `homes` | object -> (xy, taught) | **`teach_home` only** (see 6.3) | `situations()` out-of-place, `Tidy` goal |
| `sightings` | last 50 resting positions per object | `see_object`, **never called** | nothing |
| `routines` | meeting/away/active counts per hour | `mark_routine("meeting")` only (`GoQuiet`) | `meeting_likely` |
| `stats` | pets, praise, scolds, tricks, tidies, last_play, last_seen, born | `count` | display only; `last_play` and `born` never read |
| `episodes` | last 400 notable moments `{t, kind, text}` | `remember` | LLM context (last 6) |
| `likes` | per trick/game, -1..1 | praise/scold within 12 s | `ShowOff`, `PlayBook` |
| `known_words` | `{}` | nothing (dead field, `memory.py:27`) | nothing |
| `settings` | quiet_hours, utc_offset_h, face_recognition, wake_on_touch | owner by speech or API | routine, touch |
| `people` | per person F, W, R, name, owner, timestamps | `see_person`, `_blame` | greeting, Hide, Shy |
| `danger` | spots {x, y, v, s, what} | falls, stuck, near falls | planners, Tidy, Explore |
| `play` | games and toys records | `Play` | game choice |
| `familiar` | per-signature novelty, threat, time | `NoveltyTracker._remember` | Investigate |

Face fingerprints (128-number SFace vectors) are **not** in this file: `FaceID.save` (`pinch/faces.py:105`) writes a separate
gallery JSON; photos are never kept. `Privacy.forget` clears per kind (`pinch/mind/privacy.py:20-59`) and returns face ids for the
firmware to drop.

### 6.2 What is learned

Names (owner name from "I'm Sam" or the LLM; attached to the one face in view), faces (SFace gallery; a new face becomes a person
after 3 sightings in 10 s, `faces.py` `NEW_AFTER`), preferences (likes), games (PlayBook, learning-progress scoring), places to fear
(danger map), familiarity with unnamed things, who is kind or rough (F/W/R), the daily meeting rhythm (one array). The learning
rules are well sourced (Rescorla-Wagner, habituation, learning progress).

### 6.3 Gaps and bugs

* **Object homes are not learned.** `Memory.see_object` ("its home is the place it rests most", memory.py:112-123) is never
  called (`grep -rn see_object` over the repo finds only the definition and copies in other worktrees). `homes` come only from "this
  goes here" (`teach_home`, `senses.py`) or from tests. `situations()` flags `object_out_of_place` only when a home exists
  (`world.py:180-183`), so **spontaneous tidying can only happen to things the owner taught**, while the module docstring
  (`memory.py:5`) says it learns "from where it usually sits".
* **Forgetting** is only by decay for F/W/R, danger and familiarity, by caps (400 episodes, 40 facts, 50 sightings, 40 danger
  spots) and by the owner's `forget`. No consolidation, no importance weighting (episodes are first-in first-out), no handling of
  contradicting facts.
* **No schema or version.** The file has no version number; the loader only adds missing top-level keys (`memory.py:37-40`);
  nested changes, such as a new field inside a person record, are not migrated. A corrupt file is silently ignored and then
  overwritten (`memory.py:41-42`).
* **Not persisted:** personality, needs, valence/arousal, `tidy_tries`, attention, a suspended activity.
* One monolithic file rewritten whole (`indent=1`, `memory.py:56`): fine at this size.

---------------------------------------------------------------------------------------------------------------

## 7. People and language

### 7.1 How a person is represented

Two disconnected representations:

* **The owner** is `OwnerBelief` (`pinch/world.py:100-111`): present, last seen/voice/touch/attention times, `busy` (meeting),
  `mood` (a word), `voice_dir`. There is **no position**: the owner is assumed to sit at the constant `Desk.owner_xy = (0.0, -0.30)`
  (`world.py:17`) and every "go to the owner" walk targets it.
* **Known people** are `People` (`pinch/people.py:34-124`): id -> dict(name, owner flag, F, W, R, last_seen, t_update), fed by
  `see_person(pid, confidence)` (`pinch/mind/people.py:7-24`). Blame for events goes to the most recently seen person
  (`people.py:82-95`). No position, no gaze direction, no per-person mood, and no link to `OwnerBelief` beyond the `owner` flag set
  when a name is told with one face in view.
* Attention to a person is a one-line stimulus (`attention.notice("person", ...)`, `mind/people.py:13`).

### 7.2 How speech is understood

* Input is **text** (`Mind.hear_text`, `pinch/mind/senses.py:11`). **No speech-to-text engine exists in the repository** (`grep -i
  vosk|whisper|stt` finds only comments). Text arrives from the app box and the robot's inbox (`robot/firmware/senses.py:88`). The
  microphones produce only voice activity, pitch, a meeting estimate and a direction (`pinch/hearing.py`; `AudioDirection`,
  GCC-PHAT, :202-217, constructed on the robot at `robot/firmware/core.py:64`).
* **Layer 1, rules, offline and instant:** `understand_rules` (`pinch/language.py:68-130`): regexes for 25 intents (`INTENTS`,
  :24), a 9-word object list (`OBJECTS`, :27), about 25 trick words (`TRICKS`, :29), 7 feeling word lists, meeting phrases. English
  only, keyword based.
* **Layer 2, local LLM (Ollama, default `gemma4:e4b`):** only for sentences the rules call chat / none / owner_feeling /
  tell_name / teach_home (`senses.py:14-15`). It fills a JSON schema with enumerated intent, trick, object and emotion plus free-text
  name and fact (`language.py:381-407` region); `sanitize` clamps everything to the lists (`language.py:219`). Asynchronous:
  `submit` in a thread, `poll` each tick (`core.py:123-126`).
* **No-laptop fallback:** the rules alone. But availability is checked **once at construction** (`core.py:29-31`: `if not
  self.llm.check(): self.llm = None`). If the laptop appears later the robot never uses it; if it vanishes mid-run every call
  silently becomes an error result. No re-check, no back-off, no indicator. Decision 9 says the LLM "can suggest an activity from the
  allowed list" (`docs/DECISIONS.md:45-56`); the code has no such path (the LLM only parses sentences).
* LLM-derived facts are stored raw (`fact[:120]`) and fed back into the next prompt (`memory.summary`): a free-text memory loop
  (low risk here, noted).

### 7.3 What the robot can do in response

Per intent (`senses.py` `_apply`, `social.py` `Obey`): greet / goodbye emotes, praise and scold (learn likes, blame the nearest
person), about 24 tricks, come (walk to the fixed owner spot), go_away, stop, sleep / wake, play, quiet (15 min) and quiet hours,
tidy / put_away / fetch (push or carry the named thing), teach_home, how_are_you (face + sound + emote by feeling,
`common.py:21`), and "emergency stop". It cannot answer questions, ask for clarification, refuse, or show that it did not
understand (no "confused" behaviour on a failed parse). An unknown noun gives `object=""`, and `Tidy` then picks the first
out-of-place thing or does a look-around.

---------------------------------------------------------------------------------------------------------------

## 8. Objects

* Representation: `ObjectBelief(name, xy, yaw, seen_t, still_since, graspable, on_desk, rest_xy)` (`pinch/world.py:88-98`), keyed
  by a **name string**. No class-versus-instance split, no size, shape, mass or material, no affordance list. The only property is
  one boolean `graspable`; the **robot sets it to False for every coloured object** (`robot/firmware/senses.py:125`) and the twin
  takes it from its object table (`pinch/twin/sensors.py:65`).
* The name list lives in four places: the voice list `OBJECTS` (`language.py:27`), the twin's `OBJECTS` (`sim/desk.py:29`), the
  vision colour table (`robot/vision.py` header: blue pen, pink eraser, red cap, yellow ball) and `CARRYABLE = {"pen", "eraser"}`
  (`play.py:24`). The mind knows an object only if they all line up.
* Affordances (pushable, graspable, rollable) do not exist as data. `Tidy` picks grasp-or-push by `ob.graspable` (`tasks.py:51`);
  `PlayBook` allows carry/show only for `CARRYABLE`.
* **Unknown objects:** the mind has a good *emotional* model of a thing it cannot name (`pinch/novelty.py`: novelty, wariness,
  Wundt-curve interest, standoff distance, approach-retreat `Investigate`, `Watch` for motion, startle) but **no way to act on one**.
  Unnamed things never enter `world.objects`, so `Play`, `Tidy`, `GuardEdge` and `situations()` all ignore them; the body only keeps
  its feet off them (`unknown_things()`, `mind/curiosity.py:45`). Once familiar (novelty < 0.2) a thing stops being interesting and
  nothing else happens. Sensible but passive.
* **On the real robot this path is not connected.** `robot/firmware/senses.py:129` reads `data.get("observations", ())`, but
  nothing under `robot/` produces `observations` (`grep -rn observations robot` finds only that line, and no `Observation(` is
  constructed under `robot/`). So on hardware the whole curiosity stack (Investigate, Watch, motion startle, novelty) receives no input.

---------------------------------------------------------------------------------------------------------------

## 9. Generality check

### 9.1 Hard-wired scene

| Assumption | path:line |
|---|---|
| Desk 0.70 x 0.45 m centred at the origin; you sit at y = -0.30; tray at (0.18, 0.08); home spot (-0.05, 0.08); edge margin | `pinch/world.py:14-21` (`Desk`) |
| The tray is the "tidy target" and appears in foot planning and spot choice | `world.py:71-80`, `activities/curiosity.py` (`_spot`: `d.tray_gap(p) < 0.04`), `tasks.py:39` |
| "Owner" = a fixed point, plus 0.29 / 0.32 m offsets ("a spot on the desk near you") | `activities/base.py:35-36`, `tasks.py:41`, `play.py:48` |
| Named objects: voice list, carry list, the app's mess-up button | `language.py:27`, `play.py:24`, `pinch/app.py` `mess_up` ("pen", "eraser", "cap", "ball") |
| Danger map is in the desk frame and capped by desk extent | `safety.py` `cap(desk)`, `core.py:42` |
| Rooms, floors, stairs, a map | absent |
| Robot localisation needs the tray's ArUco marker | `robot/vision.py:1-6,132-160` |

Per-file counts of lines naming desk geometry or the fixed seats (`owner_xy|home_xy|tray_xy|tray_gap|tray_half|half_x|half_y|desk.`):
activities 37 (social 13, tasks 9, play 5, defensive 3, curiosity 2, recovery 2, base 2, rest 1), mind 2 (selfcare 1, senses 1),
`world.py` 17, `play.py` 5, `safety.py` 3. Lines naming tidy, tray, the four objects, put_away, fetch or out-of-place
(case-insensitive): `tasks.py` 19, `world.py` 19, `language.py` 12, `mind/core.py` 10, `affect.py` 5 (the `tidy` need and `tidiness`
trait), `play.py` 4, `curiosity.py` 4, `senses.py` 3, `social.py` 2.

### 9.2 Fraction of activities tied to the tidy scenario

* **Chore-specific:** tidy and guard_edge = 2 of 26 (8 %). `Obey` has four chore branches (tidy, put_away, fetch, teach_home).
* Adding `play` (tray target, pen/eraser carry): 3 of 26 (12 %).
* **Tied to the one desk** (fixed seat, home spot, tray or desk edges): obey, greet, comfort, seek_attention, go_quiet,
  after_meeting, shy, hide, head_home, rest, tidy, guard_edge, show_off, explore, play, investigate = **16 of 26 (62 %)**. The 10
  that are not: idle, celebrate, sulk, react_touch, startle, orient, right_self, unstick, low_power, watch.
* Weight in the *day*: the observer says tidy-type work must stay under 30 % of awake time (`pinch/observer.py:182`
  `TASK_MAX_SHARE`), because before the (unmerged) fix it took 46-69 % in review sessions (unmerged `mix.py` docstring). The
  unmerged mix adds satiation and a 12 % class cap but leaves the 16 desk-bound activities as they are.

---------------------------------------------------------------------------------------------------------------

## 10. Privileged information (things a real robot could not sense)

| # | Where | What is cheated |
|---|---|---|
| 1 | `pinch/twin/sensors.py:50-52` | pose comes from simulator truth; the robot uses the tray ArUco marker and optical flow (`robot/vision.py`), which fails when the marker is out of view |
| 2 | `twin/sensors.py:54-65` | named objects with perfect identity, yaw and `graspable`, noise 4 mm, gated only by a field-of-view test (no occlusion, no lighting, no mis-detection except the `flaky` fault) |
| 3 | `twin/sensors.py:63` | `or held or not on_desk`: an object that has left the desk is seen from anywhere, whatever the field of view |
| 4 | `twin/sensors.py:68-70` | owner presence is a script flag (`user_active`), not a detection; `pinch/app.py:129-136` `presence()` sets it |
| 5 | `twin/sensors.py:98-101` | an unnamed-thing sighting = true position + 4 mm, true velocity + 1 cm/s, true size +- 3 mm, and a `signature` that is the exact shape and size string (`"box:40"`); the robot has no producer (see 8) |
| 6 | `pinch/world.py:14-21` | desk size, owner seat, tray and home spot are constants known in advance. The comment at `world.py:15` says the robot "learns its own with the cliff sensors"; I found no code that updates `Desk` from sensors (UNVERIFIED beyond the `hw.desk` lookup at `robot/firmware/core.py:39`) |
| 7 | `pinch/mind/senses.py:11` | speech arrives as perfectly transcribed text: no recogniser, no errors, no wake word, no speaker direction |
| 8 | `pinch/app.py:94-105` | visitors are announced with id and name and fixed confidence 0.9; on the robot identity comes from SFace and is uncertain |
| 9 | `pinch/mind/senses.py:129` | `TOUCH_DIR` is 0 for every gesture; the touched side is used only when the gesture carries pads (`touch_place`), so the twin's button touches carry no place |
| 10 | `pinch/mind/core.py:26-27` | `clock` is simulated time in the twin (exact replay) and wall time on the robot, where an unset timezone (`utc_offset_h` None) changes when "night" is |

The owner's position and attention are also "known" because they are fixed. That is a modelling choice rather than a leak, but it
makes "come to me" meaningless when the person is anywhere else.

---------------------------------------------------------------------------------------------------------------

## 11. Tests

### 11.1 What the mind tests assert (13 files read)

| File | Asserts | Kind |
|---|---|---|
| `tests/test_states.py` | every activity has a valid state and 0 < min_s < max_s, names unique; an overrunning activity logs a timeout; a tidy interrupted by a stroke resumes and logs phases | contract + one twin scenario |
| `tests/test_attention.py` | habituation falls and recovers; inhibition of return; hysteresis; a voice on the left turns the crab > 0.6 rad with the eyes left; a voice does not interrupt a tidy | properties |
| `tests/test_life.py` | >= 5 micro-actions in 10 min, never the same twice, amplitude <= 0.16 rad, livelier when aroused, none during steps | statistical properties |
| `tests/test_routine.py` | quiet hours parse, are kept in memory, silence by night, sleepier at night (> 2.5x) | properties |
| `tests/test_curiosity.py` | familiar in 3-4 looks, closer each time, Wundt curve, proportionate startle, phase order notice-approach-pause-inspect-retreat, creep speed < 0.045 m/s, respects edges and feared spots, quiet hours keep it in place | properties, strong |
| `tests/test_touch_nuance.py` | pads to place, left/right gaze, claw answers, habituation characteristics, wake rules | properties |
| `tests/test_variation.py` | same emote twice differs; within joint, speed and arm limits; mood scales tempo and amplitude; mirror; seeded replay | properties |
| `tests/test_decisions.py` | every switch logged with a reason; top candidate matches; the JSON-lines file equals memory | invariants |
| `tests/test_observer.py`, `test_recorder.py` | planted problems are detected, a calm session is clean, recording changes nothing | tests of the tooling |
| `tests/test_long_day.py`, `test_long_week.py` | 24 h fast-forward: sleeps at night, quiet hours silent, variety, no lock or oscillation, replay exactness; 7-day memory survives reboots, familiarity grows, stand-up routine learned; `KNOWN_ISSUES` as strict xfail (good practice) | aggregate checks |
| `tests/test_golden.py` | exact decisions, motions, sounds and final pose of one seeded 2-minute session | **pinned trace**, recorded on Windows, skipped elsewhere (`:54-56`) |

So the suite is mostly behavioural properties. The one pinned trace is a single 120 s scenario and **is itself a tidy scenario**
(`SCRIPT`, test_golden.py:25-26 includes "put the cap away"). "Behaviour is pinned" means that scenario is.

### 11.2 Important behaviour not covered in `tests/`

(Names searched under `tests/`, `sim/benches`, `robot/firmware_test.py`.) Not referenced anywhere under `tests/`: `PlayBook`, the
`People` model (only `see_person` in `test_privacy`), `LLMUnderstanding` and `sanitize`, `MeetingDetector`, `OwnerVoiceMood`,
`Affect` (the feeling if-chain), and the activities head_home, low_power, greet, show_off, unstick, seek_attention, sulk,
celebrate (benches `scenarios.py` and `longrun.py` mention some, but those are not the suite). Not tested at all: the `_choose`
rules (margins 1.5 / 0.1 / 0.15, commitment 0.9, resume stack of 3); priority among emergencies (flipped while stuck during a
meeting); `understand_rules` outside quiet hours (25 intents, 9 objects, negations such as "not bad"); the single-slot request
overwrite; `forget` per kind; loading a memory file from an older version; the laptop LLM appearing or vanishing. The one known
failing behaviour (`physics_nothing_knocked_off`, `sim/benches/longrun.py:447-452`) is a body bug carried as a strict xfail.
`docs/BEHAVIOUR_TESTS.md` (generated) reports **12/13** with "tidies: pushes the cap into the desk tray when asked" **FAIL**.

---------------------------------------------------------------------------------------------------------------

## 12. Weak points, ranked (what an experienced game-AI or social-robot engineer would flag)

1. **No layered architecture and no declared priority order.** One object, 26 self-scoring activities, priority only as hidden
   magnitudes (selection.py:77-79; recovery.py:15,46,112; defensive.py:15,83). Consequence: any new activity can silently outrank
   safety or a command, and "what wins" can only be learned by reading 26 files.
2. **Output channels are not arbitrated.** One body task slot, face = last writer, sound = unfiltered list, gaze = a 3-second timer
   (selection.py:150-157, core.py:160-170, body/core.py:55-63). Consequence: it cannot look while walking, hold a face against the
   feeling, or layer an emote onto locomotion; the observer has to catch conflicts after the fact.
3. **The "pet" inputs do not reach the real robot.** Nothing under `robot/` produces `observations` (robot/firmware/senses.py:129)
   and no speech recogniser exists anywhere. Consequence: on hardware, curiosity, motion startle, watching and understanding speech
   are unwired; what is tested in the twin is not what will run.
4. **The world model is one desk, one tray, four coloured objects and one fixed seat** (world.py:14-21; language.py:27; play.py:24;
   robot/vision.py header; 16 of 26 activities use desk geometry or the fixed seat). Consequence: a floor, a room, a moved owner or
   a new object breaks or silences most of the mind; this is the "stuck near a tray" the owner saw.
5. **Personality is five constants that never change, are not saved, and are not a character** (affect.py:17-23, memory.py:23-27).
   Consequence: no long-term identity or growth, nothing that makes this crab differ from another, nowhere for a character bible to attach.
6. **All ~730 behaviour numbers are inside Python** (3.1, 3.3). Consequence: behaviour cannot be tuned, compared, versioned or rolled
   back without code edits; the golden trace is re-recorded on every retune; the owner cannot adjust anything.
7. **Utility scales are not normalised and the chooser compares unlike quantities** (rest.py:41, tasks.py:22, social.py:147: each
   activity invents its own formula; the 1.5 / 0.1 / 0.15 thresholds are global). Consequence: tuning one activity shifts others; the 0.02
   jitter is tiny next to 0.15 margins, so variety has to come from elsewhere (the unmerged mix is a patch for exactly this).
8. **Learned memory is partly dead or fragile.** `see_object` never called (memory.py:112), `known_words` dead, `awake_hours` never
   supplied (affect.py:117,130), no schema version (memory.py:23-40), needs and mood not saved. Consequence: "learns where things belong"
   is false today; an upgrade can silently corrupt old files.
9. **Language is a keyword list and the LLM is checked once, with no health handling** (language.py:68-130; core.py:29-31).
   Consequence: a laptop that joins later is never used; failures are silent; unknown words give "chat", never "I did not understand";
   commands are English only and overwrite each other in one request slot (core.py:49).
10. **Unknown objects can be feared and studied but never used** (novelty.py versus `world.objects`; play.py:24). Consequence: the
    "play with anything" and "pick up a pen and inspect it" goals need a general object model with affordances that does not exist;
    acting needs a name and a hard-coded list.
11. **Docs and code disagree.** a local design note (not in the repository) describes `brain/pet_mind.py` with Heart / Mind / Reflexes and 6 drives; `brain/` does not
    exist and the real mind has 7 needs and 26 activities. The `outputs()` docstring (face follows the feeling) is not what the code does;
    `QUIET_OK` (base.py:7) is unused; decision 9 claims an LLM activity suggestion that is not implemented. Consequence: a newcomer or an
    agent reads wrong designs.
12. **Test coverage is lopsided.** Strong on curiosity, touch, variation, routine; none on the chooser rules, People, PlayBook, language
    rules, the LLM sanitiser, Affect, or 8 of 26 activities (11.2); the pinned golden trace works only on Windows. Consequence: refactoring
    the selector (which this audit recommends) has no safety net except the golden trace and the benches.

---------------------------------------------------------------------------------------------------------------

## Unmerged branch notes (label: unmerged)

Branch `worktree-agent-ac628a12174226b32`, 4 commits (`6bb0db3`, `7e0160b`, `e44fd3b`, `80a96d1`), 13 files, +796 / -14.

* `pinch/mind/mix.py` (91 lines, unmerged): `ActivityMix` with `WINDOW_S` 1800, `ACTIVITY_MAX` 0.30, `CLASS_MAX {"task": 0.12}`, `URGENT`
  1.5; `factor()` multiplies each utility by the remaining share room and by zero when sated; `ended()` sets a random `sated_s` rest.
  Selection multiplies every utility by it; activities get `mix = "pet" | "task" | None` and `sated_s` in `base.py`. Reflexes, commands
  and self-care are exempt (`mix = None`). It is a standard ethology-style time-sharing and satiation device (Tinbergen, Ludlow,
  Blumberg, per its docstring). It fixes tidy hogging, not the architecture, and adds about 10 more constants (30 %, 12 %, 1800 s,
  30-75 min), all in code.
* The `tidy` need is rewritten (unmerged): fresh mess bugs it in about 60 s, old mess in 4-6 min, fades at 1/300 per s; `Tidy` gets
  `sated_s = (1800, 4500)` and `SESSION = 2` (`tasks.py` diff).
* `pinch/activities/pet_draft.py` (375 lines, unmerged, **not in LIBRARY, imported by nothing**): 7 draft activities with their own ad-hoc
  cool-downs that the file itself says to replace with `sated_s`.
* `docs/PET_BENCHMARK.md` (82 lines, unmerged): a matrix of 10 companion robots, a ranked table of 11 behaviours with the mechanic and which
  robot cites it, and a "fit for a desk crab" cut. Its own caveats: owner-review evidence is thin and partly vendor-written; no source
  gives a time share per behaviour for any robot. Useful as a shopping list for D13; it names no decision architecture and changes no
  personality code.
* `tests/test_pet_mix.py` (156 lines, unmerged): hog tests, exemption tests, "put the cap away" works at once, a working day is not chores.
* Also in the branch: `Idle` never repeats a motion twice running (`rest.py`), `ObjectBelief.first_seen` (`world.py`).

---------------------------------------------------------------------------------------------------------------

## Could not determine

* Measured run durations of each activity: I gave step-script estimates. Measured shares exist in the recorder and bench output
  (`sim/benches/longrun.py`), which I was told not to run.
* Whether `Desk` dimensions are updated from sensors on the robot (`world.py:15` says so): no such code found outside the `hw.desk`
  lookup (`robot/firmware/core.py:39`); `robot/hw_real.py` not traced.
* Whether the stale-face issue (face not following the feeling) shows in practice on the robot: code reading only.
* Whether `robot/firmware_test.py` exercises Hide and Shy through `see_person` (it mentions hide); not read.
* Exact line numbers for some inner `run()` bodies in `social.py` and `tasks.py` (cited at class level instead).
* Whether the golden trace passes on this checkout (not run; platform-specific).
