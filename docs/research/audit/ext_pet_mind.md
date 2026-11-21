# External audit: how the mind of Pinch should be built (pieces D12-D16)

Audit date 2026-11-21. Scope: topics (a)-(h) of the brief. Code claims cite `path:line`. Papers were checked against
arXiv's API, Crossref or the page itself. GitHub numbers come from `gh api` today. Anything I could not check
myself is marked **UNVERIFIED**.

## 0. The short version (read this if nothing else)

1. **Pinch's mechanism is not wrong.** Utility scores, drives, a mood point, resume stack, hysteresis and a decision
   log are the same family as Vector, aibo and Cozmo. Keep it. What is off is everything *around* the selector.
2. **The real gap is the missing "performance layer".** In Disney's BD-X characters and in Reachy Mini, what the
   robot does (an "activity") is separate from how it shows it (layers of body, gaze, face, sound that blend).
   In Pinch, an activity is a Python generator that writes face, gaze, sound and motion itself
   (`pinch/mind/selection.py:150-179`). So there is no place to say "who wins the gaze channel right now".
3. **Do not adopt a behaviour-tree library.** The mind is ~25 activities. A tree buys nothing for that size and
   would be a rewrite. Add a small, declared **channel arbiter** instead (section 4).
4. **Behaviour settings in code are the biggest tuning problem**, and there is no maintained utility-AI library to
   adopt (the best GitHub hit has 19 stars). Write a ~100-line "response curve + TOML" module and validate it with
   pydantic (29.6k stars). Version it in git; compare versions on the benches Pinch already has.
5. **Pinch is ahead of most projects on measurement.** `pinch/realism.py` already computes action and activity
   entropy, repeat rate, gaze/expression agreement and jerk. Add responsiveness latency and a *multi-day* novelty
   curve. Keep the owner's video review as the judge.
6. **The language model should stay a slow "director" on the laptop** that picks among authored behaviours and
   writes a nightly diary. This is what Reachy Mini's app and the 2025 MiRo work do. Pinch's current plan already
   points this way. Never in the reflex loop, and always a local fallback.

## 1. What the code does today (so the comparison is fair)

| Fact | Evidence |
|---|---|
| 26 activities in one flat `LIBRARY`, each with `utility(m)`, `run(m)` generator, `min_s`, `max_s`, `interruptible`, `quiet_ok` | `pinch/activities/__init__.py` (LIBRARY list at the end); `pinch/activities/base.py:22-36` |
| Choice = max over `utility + 0.02*jitter`, a 0.9 x start-score "commitment", +0.1 resume bonus, switch only if `>0.15` better after `min_s`, or "urgent" (`>=1.5`) | `pinch/mind/selection.py:55-85`, constants at `:53` |
| The only arbitration of output channels is that an activity writes `face_expr`, `gaze`, `sounds`, motion directly while it runs | `pinch/mind/selection.py:150-179` |
| One shared "quiet" flag zeroes every activity not on a hard-coded list | `pinch/activities/base.py:7`; `selection.py:59-60` |
| A decision log with the reason for every switch (a real strength) | `selection.py:18-42` |
| Mood = valence + arousal + 7 needs (social, play, curiosity, sleep, energy, tidy, safety); 5 fixed personality floats; ~30 event rows in one table | `pinch/affect.py:17-27, 30-47, 50-85` |
| Personality never changes (no code writes to `Personality` outside the constructor; it is read in `affect.py` and `mind/curiosity.py:15,38`) | grep over `pinch/` |
| Utilities are magic numbers inside each class, e.g. play `0.8*playfulness*max(0,play-0.35)...`, explore `0.65*...`, social `1.4+0.8*warmth`, `1.25*strength`, `0.9*(n-0.45)` | `pinch/activities/play.py:20,79,101`; `social.py:16,92,114,147` |
| About 270 decimal literals sit in the activity and affect files (rough count of lines with a decimal) | `grep -c` over `pinch/activities/*.py`, `pinch/affect.py` |
| Memory is one JSON file: owner facts, object homes, routine counts, stats, last 400 episodes, per-activity "likes" | `pinch/memory.py:1-28`, `MAX_EPISODES = 400` |
| Emotes: ~22 keyframed, played through a `Style` (tempo, amplitude, asymmetry, lag, mirror, gaze) | `crab/emotes.py:294-330, 377` |
| Realism metrics already exist | `pinch/realism.py:1-25` |
| A "behaviour mix" (satiation and time-sharing) exists in an **unmerged worktree**, not on master | `.claude/worktrees/agent-ac628a12174226b32/pinch/mind/mix.py` |
| a local design note (not in the repository) is stale: it points at `brain/pet_mind.py` (no such folder) and describes an older Heart/Mind design | the local design note; `ls brain` fails |

## 2. Topic (a): what production pet robots do

Source quality note. For aibo, Lovot, Moflin, EMO, Eilik, Loona the only detail I could use is what the project's own
draft `docs/PET_BENCHMARK.md` (in the unmerged worktree) already collected from Sony's help guide, press and reviews.
I did **not** re-fetch those pages. Vendor text and thin reviews are weak evidence for *how* they are built. What I
verified myself this pass is below.

| Robot | Verified this pass | How behaviour is built (published?) |
|---|---|---|
| **Anki Vector / Cozmo** | Community reverse engineering: a behaviour starts from an external stimulus or internal event, then shows affect (face, head, eyes, sound) and may change emotional state; each behaviour has a cool-down; Vector does **not** habituate automatically (cool-downs are set case by case) ([character study](https://randym32.github.io/Anki.Vector.Documentation/guides/Vector%20character%20study.html)). Behaviour taxonomy: self-maintenance and reflexes, social, play, with Kismet-style cross-cutting attitudes ([taxonomy](https://randym32.github.io/Anki.Vector.Documentation/guides/behavior%20taxonomy.html)). A behaviour tree with a mode selector exists in the shipped config; "engine tells anim what to do, anim performs it" (search snippets of [startup behaviour](https://randym32.github.io/Anki.Vector.Documentation/software-design/startup%20behavior.html) and a community architecture page; the tree page itself has only a figure) | Authored behaviour tree plus emotion model plus animation groups. Animators used Maya (Kickstarter article, snippet only). The 543-page technical reference manual has an "emotion model, behaviour engine, actions and animation engine" chapter, but I could not extract its text: **UNVERIFIED** details |
| **Disney BD-X droids** | Verified in the paper ([arXiv 2501.05204](https://arxiv.org/abs/2501.05204), Grandia et al., RSS 2024). Animation engine has three layers: a looping **background** animation (blinks, antennas) when nothing else runs; **triggered** artist clips blended over it (0.1 s ramp for show functions, 0.35 s for body); a **joystick** layer. Antennas and eye size are driven procedurally by path speed. Show functions (eyes, antennas, lamp) are open loop and synced by animation signals. The audio engine plays synchronised with clips. Clips made in Maya. A "motion stop" button forces standing and freezes joints. | Authored by animators, played by an engine, executed by an RL policy; **high-level choice is a human puppeteer** |
| **Reachy Mini** (Pollen) | The conversation app gives the language model a `play_emotion` tool that plays a **recorded** clip from the open dataset `pollen-robotics/reachy-mini-emotions-library`, and a `stop_emotion` tool. Moves go into a **layered motion system**: primary moves (dances, emotions, poses, breathing) queue while speech-reactive wobble blends on top ([app README](https://github.com/pollen-robotics/reachy_mini_conversation_app)). The repo is real: 1,678 stars (robot repo), 364 stars (app) | Recorded clip library, LLM picks by name. How the LLM chooses is **UNVERIFIED** |
| **MiRo-e** | "Hey MiRo!" (Htet et al., HRI 2025, pp. 1779-1781, DOI 10.1109/HRI61500.2025.10973974) joins an LLM to MiRo-e's layered control architecture, with deep-network perception. I only read the metadata, not the body. A related 2025 paper uses LLM prompts on MiRo-e for parent-child sessions ([arXiv 2507.10427](https://arxiv.org/abs/2507.10427), Li, Schijve et al.) | Brain-inspired layers (reflex, action selection) under the LLM |
| aibo, Lovot, Moflin, EMO, Eilik, Loona | Moflin: Casio's own page describes an emotion map with named regions (calm, anxious, joy, lonely), touch and voice as inputs, and personality shaped by care; voice is processed locally (snippets, no internals). Everything else: see `PET_BENCHMARK.md` | Internals unpublished: **UNVERIFIED** |
| **Apple ELEGNT lamp** | [arXiv 2501.12493](https://arxiv.org/abs/2501.12493), Hu, Huang, Sivapurapu, Zhang, 2025: a movement framework that scores *functional* and *expressive* utility together (intention, attention, emotion); six-scenario study. No LLM in the abstract | Design framework, not a runtime |

### Pet behaviours by category (and what Pinch lacks)

"Pinch today" is my reading of the activity list and of the (unmerged) draft benchmark. "Lacks" means no activity or
mechanism exists on master.

| Category | What successful pets do (source) | Pinch today | Lacks? |
|---|---|---|---|
| Idle life | Never fully still: breathing, blinks, glances, tiny movements even asleep (Vector, Moflin, Lovot per PET_BENCHMARK; BD-X background layer, verified; Reachy Mini breathing move, verified) | `Idle` is an *activity*; fidgets stop when another activity runs | **Yes: no always-on ambient layer** |
| Attention seeking | Escalating bids with a ceiling (Cozmo, Vector, aibo per PET_BENCHMARK) | `SeekAttention` with a need threshold (`social.py:147`) | Partly (back-off is in the draft only) |
| Reactions to touch | Squint and purr when stroked, startle at loud sound, protest when held (Vector character study, verified) | `ReactTouch`, `Startle`, 8 pads with zones (`pinch/touch.py`) | No (good) |
| Reactions to voice and faces | Name response, greeting after absence, learns faces | `Greet`, `AfterMeeting`, `people.py`, `hearing.py` | No (needs the real robot) |
| Play | Own toy, turn taking games (Vector taxonomy, verified) | `Play`, `ShowOff`, PlayBook idea | Partly; one toy-less pet look |
| Exploration | Searching/patrolling territory (Vector taxonomy) | `Explore`, `Investigate`, `Watch` | No |
| Sleep and charging ritual | Snoring on the charger, wake-up animation group, visible day rhythm (Vector startup page, snippet) | `Rest`, `routine.py`, `LowPower`, `HeadHome` | **Yes: no dock ritual, no authored wake-up/goodnight sequence** (dock is piece A1) |
| Moods | One "stimulation" level sets liveliness (Vector); mood visible in posture | valence/arousal and named feeling; face and `EmoteStyler` | Mood does not yet style *walking* (the draft idea of gait presets by mood is not built) |
| Growth over time | Moflin milestones, aibo matures over years (PET_BENCHMARK, vendor text); owners value changes | Memory "likes" only; personality fixed | **Yes: no growth, no new behaviours unlocking** |
| Chores | None of these pets are sold on chores | `Tidy` quirk | Pinch is the outlier (the owner's verdict) |

## 3. Topics (b)-(h): the table

| Topic | What Pinch does today | Best current paper (verified) | Best maintained code (gh stats today) | Verdict | What to adopt |
|---|---|---|---|---|---|
| **(b) Decision architecture** | Flat utility selector, generators for steps, commitment and resume stack (`selection.py:55-130`) | Ghzouli, Berger, Johnsen, Wasowski, "Behavior Trees and State Machines in Robotics Applications", IEEE TSE 49(9) 2023, [arXiv 2208.04211](https://arxiv.org/abs/2208.04211). It compares the two languages and mines open-source ROS projects. Its abstract does not say trees beat state machines. Older anchor still inside current tools: Colledanchise and Ogren, IEEE T-RO 2017 (title/venue seen in a search result, not read) | BehaviorTree.CPP 4,305 stars, pushed 2026-11-13, MIT (C++). py_trees 655 stars, pushed 2026-11-04, licence field "NOASSERTION" (it is BSD by my memory: **UNVERIFIED**). transitions (state machines) 6,607 stars, pushed 2025-09-11, MIT | **Confirmed** (selector); **missing** (channel arbitration) | Keep custom code. Add the channel arbiter in section 4. No library |
| **(c) Language models in the loop** | Gemma on the laptop: 18-37 s per decision, used for understanding speech and suggesting (a local design note, measured by the project) | Park et al., "Generative Agents", UIST 2023, [arXiv 2304.03442](https://arxiv.org/abs/2304.03442): memory stream, retrieval = recency (0.995/hour decay) + importance (LLM rated 1-10) + relevance (embedding), reflection when summed importance passes 150 (about 2-3 times a day), plans from day to hour to 5-15 min. GenEM: Mahadevan et al., [arXiv 2401.14673](https://arxiv.org/abs/2401.14673), 2024, LLM writes expressive control code from a robot's skills; users found it competent and easy to understand | Reachy Mini conversation app (LLM calls `play_emotion`) 364 stars, pushed 2026-11-19, Apache-2.0. Ollama 187,801 stars, MIT. llama.cpp 135,898 stars, MIT. Outlines (constrained JSON) 16,016 stars, Apache-2.0 | **Improve** | LLM as slow director on the laptop only (section 5) |
| **(d) Emotion and personality** | Circumplex (valence, arousal) plus homeostatic needs plus 5 fixed traits (`affect.py:17-85`) | Moerland, Broekens, Jonker, "Emotion in RL agents and robots: a survey", Machine Learning 2017, [arXiv 1705.05172](https://arxiv.org/abs/1705.05172): emotion is elicited by homeostasis, appraisal or value; valence is the most used dimension, arousal second. Ojha, Vitale, Williams, Int. J. Social Robotics 2021 (found as a title in a search): no appraisal model implements every property | No maintained library worth adopting: models are hand-coded in every product I could check | **Confirmed**, with a **missing** slow layer | Keep valence-arousal + needs. Add a slow *mood* tier and *drifting* traits (section 6). Skip dominance and OCC appraisal |
| **(e) Expression engine** | One emote = one keyframe clip played with a `Style`; face, sound and gaze are set by the activity; no ambient layer (`crab/emotes.py:294-377`) | Grandia et al. BD-X, [arXiv 2501.05204](https://arxiv.org/abs/2501.05204) (layers, above). Hielscher, Bulling, Arras, "Interactive Expressive Motion Generation Using Dynamic Movement Primitives", [arXiv 2504.06735](https://arxiv.org/abs/2504.06735), 2025. Ribeiro and Paiva, "Nutty-based Robot Animation: Principles and Practices", [arXiv 1904.02898](https://arxiv.org/abs/1904.02898), 2019, an animation-principles pipeline (older, kept because it is the only one I found that is built as a tool, not a talk) | Blender 22,289 stars, pushed 2026-11-21, authoring tool. Open Duck Mini 4,763 stars, pushed 2026-01-31, Apache-2.0 (open BD-X-style walker; its docs say the imitation reward follows the Disney paper). Sesame robot 5,114 stars, pushed 2026-11-14, Apache-2.0 (already studied in INSPIRATION.md). I found **no** maintained, general "Blender to servo robot" pipeline: GitHub searches for it returned nothing | **Replace the structure, keep the content** | Layered engine: ambient background + triggered clip + procedural modulation + mood style; keep Pinch's `Style` idea (it is the right one) |
| **(f) Memory and learning** | 164-line JSON store; 400 episodes; likes; danger map; no summaries (`pinch/memory.py`) | Generative Agents (above); Packer et al., "MemGPT: Towards LLMs as Operating Systems", [arXiv 2310.08560](https://arxiv.org/abs/2310.08560), 2023; Chhikara et al., "Mem0", [arXiv 2504.19413](https://arxiv.org/abs/2504.19413), 2025 | Letta 25,506 stars, pushed 2026-10-18, Apache-2.0. mem0 69,294 stars, pushed 2026-11-18, Apache-2.0. graphiti 32,972 stars, pushed 2026-11-20, Apache-2.0. cognee 33,442 stars. sqlite-vec 8,273 stars, pushed 2026-05-18, Apache-2.0. generative_agents (reference code) 22,399 stars, **last push 2024-08-05** | **Improve** | Keep a local store, move to SQLite, add nightly laptop summaries (section 7). Do **not** run Letta, mem0 or graphiti: they are servers that call an LLM on every write (my reading of their descriptions; resource use not measured: **UNVERIFIED**) |
| **(g) Settings as data** | About 270 numeric literals in code; personality, `NEED_RATES`, `EVENT_EFFECTS` are module tables | Response curves in utility AI are the standard way to turn a raw input into a score. The only source I could open is the [gameai.com response-curve glossary entry](https://www.gameai.com/manuals/index.php/Response_Curve); Dave Mark's own talk/book (Infinite Axis Utility System) I could **not** reach: **UNVERIFIED** as a primary source | pydantic 29,605 stars, pushed 2026-11-20, MIT. Hydra 10,760 stars, pushed 2026-11-21, MIT. Godot beehave (behaviour trees as editable data) 3,330 stars, pushed 2026-09-20, MIT. **Utility-AI libraries: none with traction** (top GitHub results for "utility ai" have 19 stars or fewer) | **Missing** | TOML files + pydantic models + a small curves module; section 8 |
| **(h) Measuring pet-likeness** | `realism.py`: action/activity entropy, bigram entropy, repetition distance, repeat rate, gaze and expression agreement, sound fit, jerk, idle motion, activity shares; `observer.py` repetition detector | Godspeed: Bartneck, Kulic, Croft, Zoghbi, Int. J. Social Robotics (Crossref lists 2008 online; print 2009). RoSAS: Carpinella et al., HRI 2017, DOI 10.1145/2909824.3020208 (title and DOI checked; authors not retrieved). Novelty: Abendschein, Edwards, Edwards, Frontiers in Robotics and AI 9 (2022): 6 weeks with a robot kitten, interest peaks then blends into routine, novelty is "renewable"; Zhao and McEwen, Frontiers in Robotics and AI 2025: a 6-month home robot, 19 families re-interviewed 4 years later, use fell sharply after year one and stayed low (recall, not logs) | None needed: the metrics are small functions | **Confirmed** (strong), with two **missing** pieces | Add responsiveness latency and a multi-day "fade" curve (section 9) |

## 4. One recommendation for (b): keep custom code, add a declared channel arbiter

**Decision: keep the utility selector and the generator-style activities. Do not adopt BehaviorTree.CPP or py_trees.**

Why:
- BehaviorTree.CPP is C++. The robot runtime is plain Python on an RK3566, and the project avoids new stacks.
- py_trees would work in Python, but Pinch's activities already *are* small sequences (`run()` generators) under a
  selector with resume and hysteresis. A tree would re-express the same thing. The cost is a rewrite of 26
  activities, and the golden trace is pinned.
- Nothing I found shows trees beat utility selection for a pet. The one verified study (Ghzouli et al.) reports
  that trees and state machines are used in similar ways. Vector uses a tree plus a mode selector, but Vector's
  problem was hundreds of behaviours.

What to add instead (this is the missing piece D12 asks for). Borrow the *shape* of the BD-X engine and Reachy
Mini's layered motion:

1. **Five layers, top wins per channel** (highest priority first):
   `reflex/safety` (cliff, fall, e-stop, over-heat) > `commanded task` (something the owner asked) >
   `reaction` (touch, startle, greeting) > `activity` (what the selector chose) > `ambient life`
   (breathing, blinks, small glances, idle fidgets; always running).
2. **Five channels:** legs, claws, gaze, face, sound.
3. **A table, in data, not code.** For each (layer, channel): `owns`, `shares` or `ignores`, plus a blend time.
   Example: a reflex owns legs and claws; a touch reaction owns face and sound but only *shares* legs (the walk
   continues, damped); ambient life owns gaze and face only when nothing above claims them; sound is muted by
   quiet hours at the arbiter, not inside each activity.
4. **Blend ramps** like BD-X: shorter for face/sound (0.1 s), longer for body (0.35 s).
5. **The selector keeps choosing activities.** Activities stop writing channels directly. They *request* a
   channel from the arbiter (`face("happy")`, `gaze(target)`), which grants or refuses by the table.
6. The `quiet` flag and `QUIET_OK` list (`base.py:7`) become one row in that table.

Where it runs: all on the robot, at the existing mind rate. No laptop needed. It also fixes the "stuck at tray"
class of problem structurally: a reflex layer that can always override legs is visible and testable.

## 5. Topic (c): language models around a pet that never speaks

- **Pattern 1, director over authored clips** (Reachy Mini: the model calls `play_emotion(name)`). Fits Pinch best.
  Pinch needs a tagged clip library (name, mood, energy, duration, safe-while-walking) so the model picks a name,
  never joint angles.
- **Pattern 2, generation of expression** (GenEM, 2024): the model writes control code from skills. Good *offline*
  to draft new emotes; a person reviews them; they are then replayed as ordinary clips. Not at runtime.
- **Pattern 3, memory, reflection, planning** (Generative Agents, 2023): run *at night on the laptop* while the
  robot charges. Produces a diary line, a few "reflections" and a rough plan for the next day.
- **Small-robot systems 2024-2026 that I could find:** Hey MiRo (HRI 2025, MiRo-e, LLM over a layered controller);
  MiRo-e co-regulation study (arXiv 2507.10427, 2025); Reachy Mini conversation app (open source, 2025-26);
  a cat robot "PEPE" with LLM hierarchical emotional gestures (Applied Sciences 15(23):12759, Dec 2025; I saw only a
  snippet; evaluated with a questionnaire, not long-term); Nadine, a *humanoid* with LLM and memory
  ([arXiv 2405.20189](https://arxiv.org/abs/2405.20189), Kang, Ben Moussa, Magnenat-Thalmann, 2024). I found **no
  long-term (weeks) deployment of an LLM-driven small pet robot**: the evidence base is thin.
- **Where it runs:** on the laptop only. The project already measured 18-37 s per decision on a laptop CPU
  (a local design note); I did not benchmark anything on the RK3566 (2 GB, 0.8 TOPS), but nothing I found
  suggests a useful LLM runs there at pet speed.
- **Safety rules:** the model returns constrained JSON (Outlines or Ollama's own structured output); code
  validates names against the clip library; its result only *nudges* utilities for a while (this is already the
  design in the local design note). Daily call budget, as the project's local planning note already says.
- **Laptop away:** nothing changes except the bias and the diary stop. The mind runs on drives alone. This is the
  correct current design (**confirmed**).

## 6. Topic (d): ONE emotion model to adopt

**Adopt Pinch's existing model, in three tiers, and let personality drift.**

- *Emotion* (seconds to minutes): the valence-arousal point already in `affect.py`. Keep it.
- *Need* (minutes to hours): the homeostatic drives already there. Keep them. This matches Moerland et al.'s
  homeostatic and dimensional families, and Casio's Moflin map (2D with named regions).
- *Mood* (hours to days): **missing.** One slow-moving average of recent valence and arousal that biases the
  resting point (`resting_valence` is a constant today, `affect.py` near the `Affect` fields). It is what makes a
  pet seem to have "a good day". This three-tier shape is ALMA's (Gebhard, AAMAS 2005; Crossref confirms the title
  and venue; I did not read it).
- *Personality*: five floats (`affect.py:17-23`) become **slowly changing traits**: each day, a small step toward
  what treatment taught (a lot of petting raises neediness a little, rough handling raises timidity), with a
  hard clamp and a visible log. This is the cheapest honest version of "growth", and Moflin and aibo are sold on
  it (vendor claims only).
- *Skip:* dominance (a third axis), OCC appraisal, and neural emotion models. Ojha et al. 2021 says appraisal
  models are unsettled, and a desk pet does not need 22 labelled emotions.

## 7. Topic (f): what to remember and how

| Remember | Why it pays | Representation | Forget how |
|---|---|---|---|
| People: names, faces, warmth, fear | Core of "knows me" | Per-person row (already designed in the project's local planning note) | Strangers expire in 14 days (the plan's own proposal); "forget me" deletes at once |
| Places: object homes, danger spots | Safe desk behaviour | Grid or list of cells with a score | Danger score decays over awake hours (already designed) |
| Routines: when owner is in or away | Quiet hours, greetings | 24 hourly counts (already in `memory.py`) | Slow exponential decay of counts |
| Preferences: which activities you praise | Personal feel | `likes` per activity (exists) | Drift toward 0.5 when unused |
| Episodes with a one-line summary | Callbacks, diary, LLM context | SQLite table: time, kind, text, importance 1-10, last-used | Keep top-N by `importance x recency`; summarise the rest |

Recommendation: **keep a local store; change the file to SQLite (standard library) and add the Generative Agents
scoring (recency x importance x relevance).** Importance can be a fixed number per event type (no model needed), and the
laptop model rewrites old episodes into summaries at night. Add `sqlite-vec` (8,273 stars, pushed 2026-05-18)
**only if** you later want meaning-based recall; relevance can start as simple keyword or kind matching. Letta,
mem0, graphiti and cognee are agent-memory servers; for a robot with a 2 GB board and a laptop that is often
off, they add moving parts without adding a pet behaviour.

## 8. Topic (g): behaviour settings as data

What the industry does (from the sources I could open): a utility score is `curve(input)` per consideration, the
curves live in data so designers can change them; behaviour trees are saved as files (BehaviorTree.CPP uses XML;
Godot's beehave is a data-driven tree editor).

For Pinch (one person, Python):
1. `configs/mind/*.toml`: one file per activity (curves, thresholds, `min_s`, `max_s`, cool-downs) and one for
   personality, one for `NEED_RATES`, one for `EVENT_EFFECTS`.
2. A pydantic model validates each file at load (range checks catch typos; 29.6k stars, MIT, already common).
3. A tiny `curve(kind, x, **params)` helper (linear, logistic, power, step), about 100 lines, tested.
4. A **settings hash** written into every session log and every decision log line, so any video can be traced to the
   exact numbers.
5. **Version comparison** = run the existing benches on two git commits of the config and diff
   `realism.py` metrics. No new tool. Hydra (10.8k stars) would add sweeps, but the project already has its own
   experiment files and ops launcher; only adopt Hydra if you start sweeping behaviour settings in bulk.
6. Golden trace: freeze it once with the *current* numbers moved verbatim into TOML. The first refactor must be
   a no-change move.

## 9. Topic (h): measuring pet-likeness

Already built (confirmed): entropy of actions and activities, bigram predictability, repeat rate, repetition
distance, gaze agreement, expression agreement, sound fit, jerk, idle motion, activity and class shares. This is
better than most HRI papers do. The owner's video review stays the judge.

Missing, and computable in the twin:
- **Responsiveness:** seconds from a touch, voice, or face event to the first visible change on *any* channel
  (face, gaze, sound, body). Pets answer within about a second; report median and 95th percentile.
- **Ambient liveliness:** share of awake time in which at least one channel is changing, and the longest fully
  still interval. (Needs the ambient layer from section 4 to be meaningful.)
- **Fade curve:** run the twin over 14 to 30 compressed days; plot daily `activity_entropy`, repeat rate and
  the share of spontaneous behaviours. The field studies say novelty fades and can be renewed (Abendschein et
  al. 2022; a 6-month home study dipped then recovered per its abstract, de Graaf et al.). A mind that keeps its daily
  entropy flat has done its job.
- **Legibility:** `gaze_agreement` already exists. Add "state-to-expression agreement over each *layer*", not just
  the face.
- **Human scales:** Godspeed (animacy, likeability, perceived intelligence, safety) and RoSAS (warmth, competence,
  discomfort) for the owner's own periodic review; both are short questionnaires. One owner is a pilot, not a study.

## 10. The six most important differences from how successful pet robots are built

1. **No always-on ambient layer.** Pets are never still. In BD-X and Reachy Mini a background layer runs under
   everything; in Pinch "idle life" is just one activity that stops when another starts (`selection.py:150-179`).
2. **Activities own the output channels.** No declared priority, no per-channel rules, no blending
   (compare BD-X's three blended layers). A reflex or reaction cannot cleanly sit on top of a walk.
3. **Tuning is buried in code.** About 270 numeric literals, utilities such as `0.9*(n-0.45)`
   (`social.py:147`), module-level event tables. No versioning of "the character", no comparison between versions.
4. **No slow time-scales.** Personality is fixed, there is no mood tier and no growth. Moflin, aibo and Cozmo are
   all described as changing with how they are treated.
5. **Memory records but does not reflect.** Last 400 episodes and some counters; nothing turns them into a
   story ("you were away three days"), a preference, or a daily plan (Generative Agents pattern).
6. **The language model is a chat add-on, not a director over authored behaviours.** Reachy Mini and the 2025 MiRo
   work give the model a *named clip library* to call. Pinch has the clips and the validation idea but not the
   tagged library or the nightly reflection job. And the quirk that makes it "a tidy-up robot" (chores as the
   main thing it does, none of the compared pets do chores) is a tuning symptom of the same cause as item 3.

## 11. Recommended target architecture (layered)

1. **Reflex and safety layer (robot, 50 Hz):** cliff, fall, heat, e-stop, stuck. Owns legs and claws when active.
   No dependence on anything above. (Piece C11.)
2. **Perception and appraisal (robot, ~5-10 Hz):** touch, sound, faces, objects become *events* and update
   emotion and needs (existing `affect.py` table, moved to data).
3. **Affect (robot):** emotion (fast) + mood (slow, new) + needs, with drifting personality traits.
4. **Behaviour selection (robot, ~2 Hz):** the current utility selector with response curves from data,
   hysteresis, resume stack, behaviour mix (satiation and time-sharing) from the worktree after review, and the
   decision log. No tree library.
5. **Channel arbiter and expression engine (robot, 50 Hz):** layers ambient > activity > reaction > task >
   reflex over legs, claws, gaze, face, sound, with blend ramps and mood style; plays tagged clips from the
   emote library.
6. **Director and reflection (laptop, minutes to nightly, optional):** LLM suggests among *named* behaviours and
   clips as validated JSON, writes the diary and daily plan from the episode store, drafts new emotes for human
   review. Mind runs unchanged when it is away.
7. **Data and measurement:** `configs/mind/*.toml` + pydantic, SQLite memory, settings hash in every log, benches
   plus `realism.py` (with responsiveness, ambient liveliness and the multi-day fade curve) compared across
   config versions.

## 12. What I could not verify, and what I did not do

- The Vector technical reference manual (543 pages) came back as a binary PDF; its emotion-model chapter is
  **UNVERIFIED** here. Same for the body text of "Hey MiRo!" (only DOI and venue read).
- aibo, Lovot, Moflin, EMO, Eilik, Loona internals: published detail is vendor marketing; used only as reported in
  `PET_BENCHMARK.md`.
- Dave Mark's Infinite Axis Utility System primary source; Colledanchise and Ogren (title and venue only); Ojha et
  al. 2021 (title only); ALMA (title only); de Graaf et al. long-term home study (abstract-level via a portal
  page); the PEPE and AwaR(e)obot papers (snippets only).
- py_trees licence; any resource measurement of Letta, mem0 or graphiti on a small device.
- Web calls used: about 22 searches or page fetches plus about 10 `gh`, arXiv-API and Crossref checks, under the
  35-50 budget. Nothing outside this report file was edited.
