# Design variants

## `standard-st3215.zip`: large crab, ST3215 servos (verified, archived 2026-11-18)

The first full design. It's about 38 × 37 cm with the legs spread, weighs 2.06 kg and uses 20 Feetech/Waveshare ST3215-7.4V servos.
It was archived when the requirement became **"hand-sized"**, which these 45 mm servos can't meet.

Everything in it was verified at the time of the snapshot:
- measured motor model (Rhoban BAM) + 5 ms latency + 0.5° backlash; emotes pass under both the measured and datasheet servo models
- walking: trot + crawl, 6 directions, 12/12 pass
- CAD: 10 parts / 26 pieces; print rules incl. 1.6 mm walls; all servos fit; 100/100 poses collision-free
- power budget: 25–47 min on a 2S 3300 mAh pack
- robot software self-test 17/17, including our own Feetech driver against a byte-level emulated bus

To use it: unzip it into an empty folder, then follow its README. ST3215 servos are stocked by
evelta.com (Feetech STS3215-C001) and robu.in.
