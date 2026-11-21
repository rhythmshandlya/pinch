# Roadmap

*Replaced on 2026-11-21. The earlier roadmap (one combined order for a hand-sized robot on STS3032 servos) is in
git history. The project now follows a locked 15-phase plan; the detailed version, with each phase's exit checks,
is kept locally by the owner. This page is the public outline.*

The direction comes from how successful pet and character robots were built (Disney's BD-X and Olaf, Anki's Cozmo
and Vector, Sony's AIBO, Open Duck Mini, ToddlerBot): write the character first; shape the body, its motion and
simulated training together, then freeze the body; measure the real servo before training; train once; build a
minimal real body early; layer expression and the mind on top; finish with weeks at home. Sources are in
[DECISIONS.md](DECISIONS.md) and [research/](research/).

| Phase | What it delivers | Judged by |
|---|---|---|
| 1 | Character brief, move vocabulary, hardware requirements; the repository tidied | Every physical need has a numbered requirement; the owner recognises the character |
| 2 | A candidate body shaped together with its moves, in simulation | Torque and heat margins on every move; parts fit; safe gaps |
| 3 | The real servos measured on a bench; the hardware frozen | Simulated vs measured tracking error within 2 degrees |
| 4 | The mind's structure: one owner per body part, always-on idle life, settings as data | Its structural tests; looks while walking |
| 5 | Random test and training worlds: desks, floors, rooms, clutter, people | Held-out worlds; scenarios as data |
| 6 | All movement learned once on the frozen design | Falls, speed tracking, get-up rate on unseen worlds |
| 7 | The backbone robot (legs, body, computer) built and walking | Real against simulated walking |
| 8 | Expression that does not repeat | Measured repetition; owner ratings |
| 9 | Senses on the real robot: faces, sounds, touch, where it is, maps of known places | Recognition and position accuracy |
| 10 | Exploring and finding its way | Reaching goals, no falls, nothing knocked off |
| 11 | People, play, memory, learning from its owner | Its behaviour tests; two owners give two different pets |
| 12 | A whole life in software | Day- and week-long runs; a realism score; a blind human rating |
| 13 | The full robot built | Safety checks on the assembled robot |
| 14 | Real against simulated, and weeks at home | Measured differences; daily logs |
| 15 | Release | Every release gate passed |

Purchases come in three steps, each the owner's call: the servo bench (phase 2), the bring-up kit with the leg
servos and printed body (phase 7), and the rest (phase 13). See [hardware/BOM.md](../hardware/BOM.md).
