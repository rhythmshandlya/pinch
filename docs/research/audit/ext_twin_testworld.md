# External audit: digital twin, test world and evaluation method (pieces B4, B5, B6)

*Audit date 2026-11-21. Scope: how Pinch's twin and tests compare with current papers and maintained projects.
Evidence rules: code claims cite `path:line`; papers were fetched (arXiv abstract page) unless marked UNVERIFIED;
GitHub numbers come from `gh api` on the audit date. Plain-language terms are defined where first used.*

## 1. What Pinch has today (read from the code)

- **One desk, four named objects.** `sim/desk.py:22-28` defines a 70 x 45 cm desk, one tray and four objects (pen,
  eraser, cap, ball); `sim/desk.py:31-36` adds three "never seen" props (box, tube, toy). Layout variety comes only
  from random positions of these few objects.
- **The twin's camera is an oracle, not a picture.** `pinch/twin/sensors.py:48-65` ("perceive") does not look at
  pixels. For each named object it computes bearing and distance from ground truth and reports it if it is inside
  the field of view, adding 4 mm Gaussian noise. So no perception code is exercised in the loop. The rendered
  camera (`mujoco.Renderer`) is used only for stills and clips (`sim/check.py:78-139`, `sim/clips.py:26`).
- **The camera model is a pinhole of 80 degrees vertical** (`sim/desk.py:175`, applied at `:166-168`). The real lens
  is a 160 degree fisheye. MuJoCo cannot render a fisheye directly (section 4), so this gap is not just unmodelled,
  it is currently unrepresentable with the stock renderer.
- **No latency model** anywhere in `pinch/twin/*.py` or `sim/desk.py` (a search for latency/delay finds only
  real-robot code in `robot/`).
- **People are not bodies.** `pinch/twin/stimuli.py:6-23` offers `say`, `touch`, `hear_audio`. A person is a flag
  (`user_active`, `sensors.py:68-70`) and a few calls. There is no hand, no face, no legs in the physics, so nothing
  tests "a hand reaches in", "a face at desk height" or "someone walks past". Props can move along waypoints
  (`pinch/twin/world.py:98-105`), which is the only moving thing.
- **Tests are scripted scenarios with hard-coded pass conditions.** `sim/benches/scenarios.py:27-39` (decorator),
  13 scenarios, results in `docs/BEHAVIOUR_TESTS.md` (12/13 pass, tidy-cap fails). Benches run 24-48 seeded layouts
  (`docs/DECISIONS.md:814-815`, `:1100`: "cap 11/24, eraser 11/24"). The benches print raw counts
  (`sim/benches/pick.py:22-26,60`; `sim/benches/common.py` is only a process pool and a JSON writer). I found no
  confidence interval, bootstrap or Wilson interval anywhere in the repository code.
- **A realism score exists and is sensible.** `pinch/realism.py:1-30` defines entropy, repetition, gaze agreement, jerk
  and LDLJ; LDLJ is computed per step from jerk and peak speed (`pinch/recorder.py:315-317`).

## 2. Master table

| topic | what Pinch does today | best current paper (verified) | best maintained code (gh stats) | verdict | what to adopt |
|---|---|---|---|---|---|
| (a) Scene generation | One hand-written desk; 4 objects; random positions (`sim/desk.py:22-36`) | RoboCasa, Nasiriany et al., RSS 2024, arXiv 2406.02523 (120 kitchen scenes, 2,500+ objects, built on robosuite/MuJoCo); RoboCasa365, ICLR 2026, arXiv 2603.04356 (2,500 kitchens; from search result). Infinigen-Articulated, arXiv 2505.10755 v3 (exports URDF/USD/MJCF; from search result) | robocasa/robocasa 1,880 stars, pushed 2026-11-18, code MIT / assets CC BY 4.0 (README). princeton-vl/infinigen 7,464 stars, pushed 2026-11-11, BSD-3. mujoco_menagerie 4,394 stars, pushed 2026-11-10 | **missing** (generator), **improve** (assets) | Write a thin generator on MuJoCo MjSpec; take object assets (not scenes) from RoboCasa's Objaverse set; see section 3 |
| (b) Simulated people | A flag plus say/touch calls (`stimuli.py:6-23`) | Habitat 3.0, Puig et al., arXiv 2310.13724 (humanoid avatars driven by policies or real people; automated humanoid evaluation ranked policies like human-in-the-loop tests). HuNavSim 2.0, Escudero-Jimenez et al., arXiv 2507.17317 (behaviour trees for human behaviours) | robotics-upo/hunav_sim 168 stars, pushed 2026-09-08, MIT. facebookresearch/habitat-sim 3,847 stars, **archived=true** (README: no official maintenance beyond v0.3.4). MyoHub/myosuite 1,290 stars, pushed 2026-11-20, Apache-2.0 | **missing** | Kinematic (mocap) hand, face and walker bodies scripted by waypoints; see section 3 |
| (c) Camera realism | Pinhole 80 deg, oracle perception, clips only (`desk.py:175`, `sensors.py:48`) | MuJoCo Playground, Zakka et al., arXiv 2502.08844, RSS 2025 (on-device Madrona rendering for vision policies); GaussGym, Escontrela, Abbeel et al., arXiv 2510.15352 (Gaussian splat renderer, over 100k steps/s on consumer GPUs); PolaRiS, arXiv 2512.16881 (real-to-sim evaluation from video scans) | google-deepmind/mujoco_warp 1,646 stars, pushed 2026-11-20, Apache-2.0 (ray-traced batch renderer with per-camera intrinsics and Gaussian splat rendering per README; **no fisheye listed**). madrona_mjx 163 stars, pushed 2026-02-18, MIT. qureshinomaan/SplatSim 163 stars, pushed 2025-09-04, no licence. GaussGym repository not located (UNVERIFIED) | **missing** | Render the real eye camera in the loop through a cubemap fisheye wrapper; see section 4 |
| (d) Validate twin vs robot | A2 planned: fit actuator model from bench data | BAM, Duclusaud, Passault, Padois, Ly, "Extended Friction Models for the Physics Simulation of Servo Actuators", arXiv 2410.08650 (ICRA 2025 per the repository citation). Dao and Fern, arXiv 2604.11090 (April 2026): compare hardware and sim rollouts as distributions of joint observations and actions, under 5 minutes of hardware data | Rhoban/bam 629 stars, pushed 2026-11-10, Apache-2.0; its README lists an identified Feetech STS3215 model and a MuJoCo CPU and MuJoCo Warp API | **improve** (planned, but no metric for "matches") | Use BAM for servos; adopt the proprioceptive-distribution check as the single "twin matches robot" score; add measured latency |
| (e) Tests as data | Python scenarios + golden trace + benches (`scenarios.py`, `BEHAVIOUR_TESTS.md`) | Scenic 3.0, Vin, Kashiwa, Rhea, Fremont, ... Seshia, arXiv 2307.03325 (CAV 2023): 3D scenes, ray-traced visibility, temporal requirements in LTL. STEP, arXiv 2503.10966 (RSS 2025) for the statistics half | BerkeleyLearnVerify/Scenic 389 stars, pushed 2026-11-20, BSD-3, release v3.1.1 (2026-05-08); VerifAI 221 stars. Scenic has no MuJoCo interface (its simulators folder lists carla, gta, isaac, lgsvl, metadrive, newtonian, webots, xplane) | **improve** | Do not adopt Scenic itself; write the timeline as TOML data with Scenic-style distributions and assertions; see section 5 |
| (f) Statistics | Raw counts (11/24); no intervals (`DECISIONS.md:1100`) | Agarwal et al., "Deep RL at the edge of the statistical precipice", NeurIPS 2021, arXiv 2108.13264. STEP, Snyder et al., RSS 2025, arXiv 2503.10966: sequential test for comparing two policies, up to 37% fewer trials (from search-result summary) | google-research/rliable 876 stars, pushed 2024-08-12, **archived=true**, Apache-2.0. scipy 15,231 stars, pushed 2026-11-20 (binomtest, fisher_exact, bootstrap) | **improve** | Wilson intervals + same-layout paired tests; use scipy, not rliable; see section 6 |
| (g) Automatic quality | LDLJ, jerk, entropy, gaze agreement (`realism.py`, `recorder.py:315`) | Balasubramanian, Melendez-Calderon, Burdet et al., "On the analysis of movement smoothness", J NeuroEng Rehabil 12:112, 2015; Melendez-Calderon, Shirota, Balasubramanian, Front Bioeng Biotechnol 2021 (SPARC and LDLJ valid only with velocity; LDLJ-A for acceleration; from search results). VISOR, Saurabh et al., arXiv 2605.10408 (May 2026): VLM test oracle, over 1,000 robot videos | VISOR code not checked (UNVERIFIED). No maintained VLM-judge library found | **confirmed** (smoothness) / **missing** (VLM pre-screen) | Keep the metrics; add SPARC as a second smoothness view; VLM pre-screen only as a filter, never a pass/fail gate |

## 3. (a) and (b): worlds and people

### (a) Procedural scenes: what exists and what fits a 20 cm robot

| Candidate | MJCF / MuJoCo? | Licence | gh stats (2026-11-21) | Fit for Pinch |
|---|---|---|---|---|
| RoboCasa + robosuite | Native MuJoCo (robosuite) | code MIT, assets CC BY 4.0 (RoboCasa README); robosuite licence reads NOASSERTION on GitHub | robocasa 1,880 stars, pushed 2026-11-18; robosuite 2,720 stars, pushed 2026-11-19 | Kitchens only, human-scale furniture, about 10 GB asset download, Python 3.11 plus robosuite master branch (README). Too heavy as a whole; its **object assets** (Objaverse-derived, MJCF) are the useful part |
| MuJoCo Playground | Native, but robot tasks, not scene generation | Apache-2.0 | 2,326 stars, pushed 2026-11-21 | No scene generator. Useful for MJX/Warp training envs only |
| Infinigen Indoors | Paper names Omniverse and Unreal as export targets, **not MuJoCo** (arXiv 2406.11824, CVPR 2024) | BSD-3 | 7,464 stars | Photoreal rooms for rendering, not shown to be physics-ready at desk scale. Infinigen-Articulated has an MJCF exporter (`src/infinigen/core/sim/exporters/mjcf_exporter.py`, found by gh code search) for 18 articulated categories (drawers, doors): an optional later extra |
| ProcTHOR / AI2-THOR | Unity, no MJCF | Apache-2.0 (ai2thor) | procthor 486 stars, last push 2023-04-07; ai2thor 1,831, pushed 2025-11-04 | Rooms, but wrong engine and a stale generator. Reject |
| Holodeck | Unity (AI2-THOR) | Apache-2.0 | 584 stars, last push 2025-04-02 | Reject (same reason) |
| Habitat / Habitat 3.0 | Own engine | MIT | habitat-sim **archived**; README says no official development past v0.3.4 | Reject as a base; keep the paper for how people are modelled |
| ManiSkill3, Genesis, BEHAVIOR-1K/OmniGibson | Other engines (SAPIEN, Genesis, Isaac) | Apache-2.0 / Apache-2.0 / MIT | 3,532 / 30,281 / 1,804 stars | Would mean leaving MuJoCo and the planned mjlab migration. Reject |
| Asset sets already in MJCF | YCB and Google Scanned Objects conversions exist, but the one I found (`kevinzakka/mujoco_scanned_objects`, 298 stars, MIT) was last pushed 2022-07-07; menagerie holds robots, not clutter | n/a | stale | RoboCasa's objects are the maintained route |

**Pick (a): do not adopt a simulator's scene system. Write a generator of roughly 300 lines on `mujoco.MjSpec`**
(the `mujoco-spec-editing` skill covers it). It samples: surface size and type (desk, table, floor, rug via friction
and bumpiness), clutter from RoboCasa's Objaverse MJCF objects plus procedural primitives scaled to 1-10 cm, tray or
no tray, lamp direction and colour, textures. Reasons: Pinch's world is desk and floor at 20 cm scale, which none of
the surveyed systems target; MjSpec is already the project's tool; the objects are the hard-to-make part and RoboCasa
publishes them under CC BY 4.0. Fallback: Infinigen-Articulated for drawers and doors when furniture is wanted.
Effort: M (1-3 weeks) for the generator; the asset import is a few days. Runs on the laptop or any CPU box.

### (b) Simulated people

How the field does it: Habitat 3.0 (arXiv 2310.13724) animates humanoid avatars from learned or scripted policies
and lets real people drive them, and found automated humanoid tests rank policies like human tests (abstract).
Social-navigation simulators script pedestrians: HuNavSim 2.0 (arXiv 2507.17317) combines behaviours as trees of
actions and conditions (the abstract names only behaviour trees; the social-force model and the indifferent /
curious / fearful styles come from search results, not fetched). SEAN 2.0, SocNavBench and Arena-Rosnav are walking-crowd
tools (Unity, ROS); SocNavBench has 58 stars, last pushed 2022-03-09; arena-rosnav 168 stars, last push 2023-09-07;
I could not find the SEAN repository (UNVERIFIED). None of these model a hand over a desk or a face at desk height:
they are built for robots that share a corridor with people.

What is feasible in MuJoCo: SMPLSim (ZhengyiLuo/SMPLSim; builds SMPL/SMPL-H/SMPL-X MJCF humanoids per a search
result; stars not checked) and MyoSuite (1,290 stars) give full-body or muscle models. They are overkill: the crab
never needs a person's physics, only their **geometry moving along a path**, so that the camera, ToF beams, touch
and collisions see it.

**Pick (b): kinematic mocap bodies** (MuJoCo bodies with `mocap="true"`, moved by writing `data.mocap_pos`; they are
placed, not simulated): (1) a hand (palm box plus five capsule fingers) that enters from a desk edge on a Bezier path,
pauses, offers, retreats or sweeps; (2) a face (sphere with eye and mouth textures, so a face detector and gaze logic
have something to see) at 20-40 cm height; (3) a pair of legs (two capsules and a pelvis box) walking along the floor.
Each is a few dozen lines, scripted by waypoint lists in the timeline format of section 5. Borrow HuNavSim's idea
that a behaviour is a small tree of actions with parameters (indifferent, curious, fearful), written as data. For
spoken lines, language-model personas are possible, but evidence is thin: the one robot-specific paper (PACE, arXiv
2607.15579) and several simulator-realism papers came from search snippets only (UNVERIFIED). The project already
exercises speech with fixed lines; keep personas for later.

## 4. (c) Camera realism and (d) validation

### (c) Perception in the loop

- **Today it is not in the loop** (section 1). This is the largest gap in this audit: the pet's whole understanding
  of objects and people is untested against pixels, lighting, blur or lens distortion.
- **Fisheye.** MuJoCo's native renderer is perspective or orthographic only: `mjtProjection` in the installed
  MuJoCo 4.0.1 has just `mjPROJ_ORTHOGRAPHIC` and `mjPROJ_PERSPECTIVE`, and the changelog says the enum "will allow for
  more projection types in the future like fisheye cameras". GitHub issue google-deepmind/mujoco#1859 (fisheye
  request, closed) was answered by a maintainer: render several pinhole cameras and build the fisheye by sampling
  them; a user then reported mismatched exposure between the views. The MuJoCo Warp README lists per-camera field of
  view and intrinsics but not fisheye, and a code search for "fisheye" in mujoco_warp returns 0 hits.
- **So: yes, a 160 degree fisheye can be rendered, by a cubemap.** Render five perspective views (90 degree squares:
  front, left, right, up, down), then remap pixels with the real lens model (equidistant or Kannala-Brandt, from a
  checkerboard calibration with OpenCV `cv2.fisheye`). Avoid exposure mismatch with one light setup and no
  auto-exposure. Cost: five renders per frame; fine for tests at 5-10 frames per second on CPU, not for training.
- **Training vision at scale.** MuJoCo Playground added Madrona rendering for vision policies (arXiv 2502.08844); its
  README now points to the MJWarp batch renderer. madrona_mjx has 163 stars, last pushed 2026-02-18. Gaussian-splat
  real-to-sim is the current research direction (GaussGym arXiv 2510.15352, PolaRiS arXiv 2512.16881; a third-party
  summary says a visual gap remains), but both need a real scene scan and a GPU; not worth it before hardware exists.
- **What others do for visual sim-to-real:** randomise lighting, textures, camera pose and distractors (Tobin 2017,
  Sadeghi and Levine 2017: old, but still the method inside newer systems). Search snippets also cite Garcia 2023
  (arXiv 2307.15320) and VIRAL 2025 for sensor-delay and image-quality randomisation (not fetched: UNVERIFIED). I
  found no small-robot-specific study.
- **Verdict: missing.** Adopt the fisheye wrapper plus lighting, texture, blur and exposure randomisation inside the
  world generator, and run Pinch's real detector code on the rendered frames (on the RK3566 it runs the small model;
  on the laptop, heavier ones). Keep oracle perception as a switch: it is fast and separates mind bugs from vision bugs.

### (d) Validating the twin against the robot

- **Servos:** BAM (arXiv 2410.08650; Rhoban/bam, 629 stars, pushed 2026-11-10, Apache-2.0) fits Stribeck,
  load-dependent and quadratic friction from pendulum trials and plugs into MuJoCo CPU and MuJoCo Warp. Its README
  lists a Feetech STS3215; Pinch's legs use the STS3032 (a related but different servo), so a fresh fit is needed.
  The arm servos have no feedback, so only open-loop response (camera or IMU timing) can be measured.
- **What matters first.** The sources I found (search results, not a formal ranking) agree that the actuator model
  and latency dominate the gap (Masuda and Takahashi, quoted in a Stanford CS224R slide deck; Tan et al. 2018, arXiv
  1804.10332, measured about 3 ms servo and 15-19 ms controller latency with a one-step command spike: old, but still
  the standard method). Suggested order for Pinch, my inference from those sources: (1) command-to-motion latency per
  servo bus and per camera frame, (2) servo step and ramp response, friction and stall load, (3) IMU noise and bias,
  (4) foot friction on two or three real surfaces, (5) mass and centre of mass (weigh the real robot). Contact last.
- **A measure of "the twin matches the robot".** Dao and Fern (arXiv 2604.11090, April 2026) compare hardware and
  simulation rollouts as distributions of joint observations and actions: no motion capture or time alignment, under
  five minutes of hardware data. That is cheap enough for one beginner with one robot. Adopt it as the B4 score: play
  the same five scripted motions (stand, walk, turn, righting, an emote) on both, report the distance between the
  distributions of joint position, servo load and IMU rate, and require it to shrink after each parameter fit.
- **How far to trust an unvalidated twin?** I found no published rule of thumb. Closest evidence: PolaRiS (arXiv
  2512.16881) says existing simulation benchmarks correlate poorly with real performance for strong policies, and that
  its scene-matched twins correlate much better (the abstract page gives no coefficient; an r = 0.9 figure came from
  a search snippet: UNVERIFIED). Practical reading: until B4 passes, trust the twin for **ordering ideas** (does A beat
  B in logic and layout) and for **finding crashes**, not for absolute numbers such as "27-40 min battery" or
  "11 of 24 grabs".

## 5. (e) Scenario tests as data: one format

Findings: Scenic 3.0 (arXiv 2307.03325) is the best scenario language (distributions over scenes and behaviours,
temporal requirements in LTL). It is actively maintained (389 stars, pushed yesterday, BSD-3, release 3.1.1) but has
**no MuJoCo interface**, and writing a driver costs more than it gives at this size. OpenSCENARIO is driving-specific
(not checked further). Robot-testing literature: property-based testing in simulation for tabletop actions (arXiv
2108.08726, 2021) and metamorphic testing of drone obstacle avoidance in simulation (arXiv 2209.11031, 2022) were the
closest items found; I found no 2023-2026 robotics survey (search result only, not fetched). VISOR (arXiv 2605.10408)
is the newest robot test-oracle paper and concerns VLM judges (section 7).

**Pick (e): a TOML "timeline plus assertions" file, Scenic-like in spirit, run by the existing recorder.**

```toml
name = "hand_offers_then_withdraws"
seeds = 24                                  # layouts are sampled, not fixed
[world]  surface = "desk"  clutter = [0, 6]  lighting = "any"  tray = "maybe"
[[events]] at = "5s"   do = "hand.enter"  from = "uniform(left,right)"  path = "bezier"  speed = "0.1..0.25"
[[events]] at = "+3s"  do = "hand.hold"   height = "0.03..0.08"
[[events]] at = "+4s"  do = "hand.leave"
[[assert]] always = "falls == 0"                                   # property (safety)
[[assert]] within = "2s of hand.enter"  expect = "gaze_on(hand)"   # time-bounded reaction
[[assert]] by = "hand.hold+6s"  expect = "activity in {greet, investigate}"
[[assert]] metamorphic = "same outcome, +-1 step, when lighting changes"
[[assert]] rate = ">= 0.8"                                         # share of seeds that must pass
```

Why: it is plain data (piece D15 asks for settings as versioned data); the assertions are of the three kinds the
literature separates (always-properties, time-bounded reactions, metamorphic relations); and the pass rule is a
**rate with an interval**, not one run. The golden trace stays but is demoted: it pins refactors (same seed, same
output), while this file judges behaviour (many seeds, properties). Convert the 13 scenarios in `scenarios.py` first;
keep Scenic's sampling syntax (`uniform(a,b)`) so a later move to Scenic stays possible. Fallback: the same content as
pytest parametrised cases with `hypothesis` (property-based testing), if Python is preferred over TOML.

## 6. (f) Statistics for the benches

Is rliable still the standard? The paper (Agarwal, Schwarzer, Castro, Courville, Bellemare, NeurIPS 2021, arXiv
2108.13264) is still the reference for reporting: stratified bootstrap intervals, interquartile mean and performance
profiles, written for 3-10 runs per task. The code (google-research/rliable) is **archived** (last push 2024-08-12), so
do not build on it. It also targets many tasks with few seeds; Pinch has the opposite shape: one outcome with 24-48
yes/no layouts. For that, simpler tools in scipy are right: Wilson intervals for one rate, Fisher's exact test (or,
for the same layouts, McNemar's test) for two policies, and a sequential test to decide when to stop. STEP (Snyder et
al., RSS 2025, arXiv 2503.10966) is the current robotics reference for that last part: a sequential test comparing two
policies that cuts trials by up to 37% (per search-result summary); I did not check its code. Patterson, Neumann,
White and White, "Empirical Design in Reinforcement Learning", JMLR 25(318), 2024 (arXiv 2304.01315) is the best
practice guide; I did not read a seed count in it.

What this means for numbers in `DECISIONS.md` (I computed these with Python and scipy on 2026-11-21):

| result | 95% Wilson interval | reading |
|---|---|---|
| cap 11 of 24 (`DECISIONS.md:1100`) | 0.28 to 0.65 | the true rate could be anywhere from about a quarter to two thirds |
| 19 of 48 into the tray vs scripted 14 of 48 (`DECISIONS.md:648`) | 0.27-0.54 vs wide overlap | Fisher p = 0.39: **not distinguishable** |
| 11 of 24 vs 16 of 24 (illustrative) | 0.28-0.65 vs 0.47-0.82 | Fisher p = 0.24: a 21-point gain is not significant |
| 5 of 16 vs 9 of 16 (`DECISIONS.md:615,637`, cap) | 0.14-0.56 for 5/16 | the note "16 layouts couldn't tell a regression from chaos" (`:815`) was right |

Sample size: to see 46% vs 67% with 80% power you need about 86 layouts per policy; to see 46% vs 56%, about 390. So
24-48 layouts detect only large gains (30 points or more). Recommended practice: (1) always print rate, interval and
n; (2) run both policies on the **same** layouts and report discordant pairs (McNemar); (3) call a result a win only
if the interval of the difference excludes zero, otherwise write "inconclusive"; (4) use 100 or more layouts for
adoption decisions, since the generator (section 3) makes layouts cheap; (5) use STEP-style stopping to save compute
on clear wins and losses. Verdict: **improve**, low effort (a 40-line `sim/benches/stats.py`).

## 7. (g) Automatic quality measures and learned judges

- **Smoothness: confirmed, with one caveat.** `pinch/recorder.py:315-317` computes LDLJ from step duration, peak speed
  and integrated squared jerk, the velocity-based form. Melendez-Calderon, Shirota and Balasubramanian (Front
  Bioeng Biotechnol 2021; from search results, full text not fetched) report that SPARC and LDLJ-V are valid only
  with velocity, and that LDLJ-V is error-prone on rotation-rate data from gyros. So keep applying it to joint-target
  velocity, not to raw IMU rates. Add SPARC as a second view. The docstring cites "Balasubramanian et al. 2012" for
  LDLJ (`realism.py:18`); the paper I could verify for the smoothness family is the 2015 JNER paper (12:112), so
  check that citation (which 2012 paper is meant: UNVERIFIED).
- **Variety and entropy:** the metrics in `realism.py:1-30` (action and bigram entropy, repetition distance) match
  the owner's complaint (one scripted routine). They are proxies; report them per seed with intervals as in section 6.
- **Legibility:** gaze agreement follows Takayama, Dooley and Ju 2011 (older paper cited in the docstring, not
  re-verified). Fine as is.
- **VLM judges.** Evidence is mixed. VISOR (arXiv 2605.10408, May 2026; Saurabh, Valle, Arrieta, Ali, Arcaini) tested
  two commercial VLMs on four robot tasks, over 1,000 videos: one had higher recall, the other higher precision, and
  **their self-reported uncertainty did not predict correctness** (abstract). Other numbers (91.2% agreement with a
  human majority on 102 clips; a fine-tuned judge at r = 0.89 on 50 videos vs 0.44 zero-shot) come from search
  snippets about arXiv 2610.26811 and 2605.03821 that I did not fetch (UNVERIFIED), and are small samples.
  **Verdict:** a VLM can pre-screen (drop obviously stuck or empty clips, flag falls, write a text description) before
  the owner watches, but must not be a pass/fail gate; the owner's review stays the judge, as the project already
  decided. It would run on the laptop, offline, after a session.

## 8. Recommended test-world design for Pinch, ordered by value for effort

1. **Statistics helper (S, 1-2 days).** Wilson interval, McNemar or Fisher, and an "inconclusive" label in every
   bench printout; 100 or more layouts for adoption (section 6). The cheapest change with the largest effect on the
   quality of decisions.
2. **World generator on MjSpec (M, 1-3 weeks).** Surfaces (desk, table, floor, rug, slope), 0-10 objects from
   RoboCasa's Objaverse MJCF set plus primitives, lighting, textures, tray or none. Seeds recorded in the session
   file. This is B5's core and feeds every bench and the recorder.
3. **Scripted people as mocap bodies (S-M, about 1 week).** Hand, face at desk height, walking legs on the floor,
   driven by the timeline format; behaviours as small trees (indifferent, curious, fearful) after HuNavSim.
4. **Timeline plus assertions format (S-M, about 1 week).** TOML as in section 5; port the 13 scenarios; keep the
   golden trace for refactors only.
5. **Camera in the loop (M, 1-2 weeks).** Cubemap fisheye wrapper calibrated to the real lens, lighting, exposure,
   blur and noise randomisation; run the real detector on rendered frames; keep the oracle as a switch. Without this,
   perception cannot be called tested.
6. **Latency and actuator model in the twin (S now, M after hardware).** Add configurable command and camera latency
   now (defaults from a Tan et al. 2018 style measurement: a few ms servo, tens of ms for vision and control); fit
   servos with BAM and score with the proprioceptive-distribution check (arXiv 2604.11090) once a robot exists.
7. **VLM pre-screen of recorded videos (S, a few days).** A laptop model writes a one-paragraph description and flags
   stuck, empty or fall clips for the owner's viewing queue. Never a gate.
8. **Later, only if needed:** Infinigen-Articulated for drawers and doors; a Gaussian-splat scan of the owner's real
   room (GaussGym or PolaRiS style); language-model personas for spoken lines.

### What in the current approach should change

- **Stop calling the twin's perception tested.** `sensors.py:48-65` hands the mind ground truth plus noise; at least
  a share of the benches must run on rendered fisheye frames.
- **Stop reporting bare counts.** "11/24" cannot separate a real gain from chance (section 6).
- **Demote the golden trace** to a refactor guard; judge behaviour with many-seed property tests.
- **Replace the single desk and four fixed objects** with generated worlds; they teach the pet this desk, not the
  world (the owner's stated goal).
- **Give people a body.** A flag cannot test reach, gaze or avoidance.
- **Keep:** the LDLJ, jerk, entropy and gaze metrics, the session recorder, owner video review, MuJoCo plus the mjlab
  plan. All confirmed.
- **Do not adopt:** Habitat (archived), ProcTHOR and Holodeck (Unity, stale), Isaac-based tools (leave the stack),
  rliable (archived), Scenic itself (no MuJoCo driver), full RoboCasa (kitchens, 10 GB).

## 9. What I could not verify

SEAN 2.0 repository and primary paper; GaussGym repository location and star count; the exact correlation
coefficient in PolaRiS; seed-count advice in Patterson et al.; the figures for the 2605.03821 and 2610.26811 VLM-judge
papers and the PACE persona paper; whether the 2012 LDLJ citation in `realism.py:18` is right; whether any third-party
MuJoCo driver for Scenic exists. VISOR's agreement with human raters is not on its abstract page. Search results for
Habitat 3.0's SMPL-X use were secondhand and are not relied on.
