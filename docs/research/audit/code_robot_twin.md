# Code audit: robot runtime (pillar A) and digital twin / test world (pillar B)

Audited: `master` at commit 3bbc1a9, read-only, 2026-11-21. Nothing was run (no tests, benches or training).
Every claim has `path:line`. "UNVERIFIED" marks things I could not confirm from the code alone.

Terms used once here: a **sync read / sync write** is one serial packet that asks (or tells) many servos at once.
A **twin** is the MuJoCo copy of the robot. **Ground truth** is the exact number the simulator knows (true position,
true object place) that a real robot never has. **GIL** is Python's one-thread-at-a-time lock.

There are really TWO twins, and the difference matters for question 9:
- **PinchSim** (`pinch/twin/`): fast, physics only, no camera image. Used by every bench, almost every test and the app.
- **Firmware on EmulatedHardware** (`robot/hw_emulated.py` + `robot/firmware/`): the real robot program on fake chips and a
  rendered camera. Used only by `robot/firmware_test.py` (a manual script, not in CI).

---

## A. Robot runtime (piece A3)

### 1. Processes, threads, loop rates, timing

One Python process (`robot/pinch_main.py:63-70`), started by systemd as root (`deploy/pinch.service:10`, `Nice=-5` at line 21).
No separate processes except `arecord` (microphones, `robot/hw_real.py:139`) and one `aplay` per sound (`hw_real.py:147-153`).

| What | Where | Rate | Notes |
|---|---|---|---|
| Control loop (body, servos, supervisor) | main thread, `robot/firmware/core.py:287-294`, `DT = 0.02` (`robot/crab_robot.py:26`) | 50 Hz | one servo write per tick, then `hw.wait(DT)` |
| Mind tick | same thread, `k % 5` (`core.py:166-171`) | 10 Hz | the whole pet mind runs inside the control tick |
| IMU, battery, feet-load, stuck check, cliff reflex | same thread, `k % 5` (`core.py:166-217`) | 10 Hz | |
| Touch pads | same thread, `k % 2` (`core.py:221-227`) | 25 Hz | |
| Face frame build | same thread, `k % 3` (`core.py:228-230`) | 16.7 Hz | |
| Servo health check (temp, volts) | `crab_robot.py:170-190`, every 10th tick | 5 Hz | |
| Servo state read | `Crab.state()` cached once per tick (`crab_robot.py:80-89`) | up to 50 Hz when a policy walks or emotes (`pinch/body/locomotion.py:93,201`, `expression.py:64`), else 5-10 Hz | one sync read of 12 servos, "~2 ms" per the comment |
| OLED thread | `robot/firmware/senses.py:11-20` | ~15 fps target (0.06 s sleep) | own I2C bus 4 |
| Cliff thread | `senses.py:34-37`, `robot/drivers/i2c_devices.py:143-161` | ~10-14 Hz (0.03 s sleep plus up to 50 ms per read) | busy-polls the sensors |
| Vision thread | `senses.py:55-64` | at most 5 Hz (0.2 s period) | ArUco + colour + Haar + YuNet/SFace + ground flow, all in Python |
| Audio thread | `senses.py:66-81`, `hw_real.py:137-145` | 0.1 s chunks, 16 kHz stereo | |
| HTTP server | `robot/http_api.py:74-75` | one thread per request | |
| Memory save | `core.py:249-271` | every 60 s, fsync in a thread | |
| LLM call | `pinch/language.py:221-232` | per sentence | daemon thread, 90 s timeout |

**What is timing critical:** the 50 Hz servo loop only. Timing is kept by sleeping to a drifting schedule
(`core.py:289-294`; `robot/backends.py:272-281` for the bus backend). If the loop falls more than 0.5 s behind it resyncs
instead of bursting (`core.py:293-294`). There is no real-time scheduling (no `SCHED_FIFO`, no CPU pinning, no
`isolcpus`), only `Nice=-5`. The control tick, mind, policies and three sensor threads share one GIL.

**Budget:** there is no stated CPU or memory budget, no `MemoryMax`, no cgroup, nothing about the 2 GB RAM. The only number
is "the tick's own work p95 < 4 ms on the laptop, a 5x margin for the RK3566" (`robot/firmware_test.py:144-145`,
`docs/DECISIONS.md:287,299`). The "5x slower" is an assumption; no measurement on the RK3566 exists
(`docs/DECISIONS.md:301`: "read tick_ms from /status once it runs"). `tick_ms` excludes the sensor threads, which compete
for the same CPU and GIL (decision 57 itself records the 4.6 ms p95 caused by OpenCV worker threads: `pinch/faces.py:34-37`).
The 0.8 TOPS NPU is not used anywhere (grep for rknn/npu finds nothing in code).
Slow loop and hot CPU are only logged (`core.py:298-302`).

### 2. Servo input/output

**Legs (12 x STS3032 on a Feetech bus)**
- Protocol: own pure-Python Feetech STS driver, `robot/feetech_bus.py`. Packet `FF FF ID LEN INSTR PARAMS CHK` (`:6-8, 64-66`).
- Baud: 1,000,000 (`robot/backends.py:500`, `robot/setup_servo.py:285`), pyserial `timeout=0.02` (`RealBackend`, `backends.py:500-513`).
- Port: `/dev/serial/by-id/*` else `/dev/ttyUSB0` (`backends.py:485-497`).
- Write pattern: one SYNC_WRITE of `goal_position` (2 bytes) for all 12 each tick (`feetech_bus.py:152-157, 239-244`). No speed, acceleration or time register is written (the loop speed-limits targets instead, `crab_robot.py:117`).
- Read pattern: one SYNC_READ of registers 56..63 (position, speed, load, voltage, temperature) for all 12 (`feetech_bus.py:185-217`, `backends.py:223-242`). If any reply is missing it falls back to 12 single reads, each with 2 retries at 20 ms (`backends.py:223-236`, `feetech_bus.py:115-125`): one silent servo can cost ~60 ms in one tick.
- Feedback used: position (every tick, joint angle), load (feet-down sense: femur load > 0.04, `hw_real.py:172-178`), voltage and temperature (health check, `crab_robot.py:170-190`). Speed goes to the learned policies (`pinch/body/manipulation.py:244-246`).
- Error handling: a bad checksum, short reply or any servo error-flag byte raises `BusError` (`feetech_bus.py:98-113`). Errors are counted and the last good value reused for up to `MAX_STALE = 10` consecutive reads, then `BusError` -> `Crab.state()` -> safety stop (`backends.py:172, 238-239`; `crab_robot.py:84-88`). The comment says 10 reads is "~1 s", but when a policy reads every tick it is 0.2 s. Because servo error flags (for example overload while a foot is jammed) also raise `BusError` (`feetech_bus.py:111-112`), a stalled leg looks the same as a silent servo and ends in a full torque-off stop. No stall or current limit is configured anywhere (`setup_servo.py:232-238` writes only P/I/D, return delay and voltage limits).
- Write failures: counted; after 25 in a row it re-raises (`backends.py:203-204`), which ends the process (not a `SafetyStop`; systemd restarts it).
- Torque: `BusBackend.torque()` is one SYNC_WRITE of `torque_enable` (`backends.py:260-270`). On, at `Crab.start()` (`crab_robot.py:210-218`) and `resume()` (`:158-167`); off at shutdown, e-stop and safety stop. Torque is never switched off at rest while the robot is alive (no "servos off when resting").
- Torque-on ordering is a bench question: `resume()` and `start()` enable torque before any goal is written, and nothing writes `goal_position = present_position` first (`backends.py:260-270`). If the servo keeps its stale goal register, it can snap to it at full speed for one tick. UNVERIFIED on a real STS3032.
- Calibration: `robot/calibrate.py`. Legs: person holds the robot in the "all joints zero" pose, torque off, offset = average servo angle (10 samples) (`:356-368`); direction check by bending each joint 20 degrees by hand (`:371-393`). Arms: person judges "straight" by eye with +/- nudges and a one-time travel question (15/30/60 degrees) (`:399-456`). Saved atomically to `/var/lib/pinch/calibration.json` (`robot/servos.py:269, 320-329`); the firmware refuses to start without it (`servos.py:332-338`, `hw_real.py:68`). Hand-held accuracy is bounded by how well a person can hold a pose (the 0.2-0.5 degree figures in `robot/selftest.py:199` are simulation only).
- ID setup: `robot/setup_servo.py` sets ID, P=32, I=D=0, return delay 0, voltage limits 4.0-6.5 V (`:232-238`).
- Register map: the driver names the servo model `sts3215` (`feetech_bus.py:48`) and the spec table maps the BOM's STS3032 to it (`crab/params.py:95`). That the STS3032 shares the STS3215 register map and answers SYNC_READ is UNVERIFIED (only emulated).

**Arms (6 x GS-4308MG, PWM, no feedback)**
- Driven by the Radxa's own six hardware PWM outputs through Linux sysfs (`backends.py:370-438`, pins in `BoardPWM.PINS`), 50 Hz, 20 ms period, duty = microseconds x 1000. PCA9685 over I2C is an unused alternative (`:441-475`).
- Mapping: 1500 us centre, 1000 us per 180 degrees (`backends.py:35-40`), clipped 500-2500 us. The real travel is unknown ("could mean either"; `backends.py:31-34`) and is measured once by the trim tool (`calibrate.py:396, 418-432`).
- No feedback: the robot assumes the arms are where they were last sent (`crab_robot.py:91-100`) and at boot assumes the REST pose (`crab_robot.py:210-218`). Arm motion is limited only by the global per-tick step limit.
- PWM stays on after a process crash: the sysfs `enable` file stays 1 when the Python process dies, so the pulses keep going (`backends.py:426-438`; only `all_off()` in `torque(False)` stops them).

### 3. Sensors: driver, fake, or nothing?

| BOM sensor / output | Real driver | Only a fake? | Evidence |
|---|---|---|---|
| 12 x STS3032 (position, load, temp) | yes, own driver | yes: `robot/bus_emulator.py` (register model written from the Feetech SDK) | `feetech_bus.py`, `bus_emulator.py:44-138` |
| 6 x GS-4308MG PWM | yes, sysfs and PCA9685 | yes: fake sysfs tree in the selftest | `backends.py:370-475`, `selftest.py:345-370` |
| FE-URT-1 USB adapter | pyserial | emulated port | `backends.py:500-513` |
| OV5647 camera | `cv2.VideoCapture(0)` at 640x480 (`hw_real.py:92-96`) | the twin render (`hw_emulated.py:62-66`) | whether `/dev/video0` is usable through plain OpenCV on the Radxa's CSI camera is UNVERIFIED (CSI cameras on Rockchip usually need ISP/media-pipeline setup) |
| BNO055 IMU | yes, raw registers, fusion mode (`backends.py:308-336`) | fake smbus in selftest (`selftest.py:290-311`) | no calibration-status read |
| 5 x VL53L0X | minimal driver, "default settings, no fine tuning" (`i2c_devices.py:68-114`) | `FakeRanger` (`drivers/fakes.py:78-96`) | it skips ST's reference/SPAD/tuning init, so accuracy, range and surface sensitivity are unknown; XSHUT GPIO lines are hard-coded and the code says "check with gpioinfo on the first boot" (`hw_real.py:23-31`) |
| MPR121 touch (8 pads) | yes (`i2c_devices.py:45-65`) | `FakeTouch` | thresholds 12/6 never tuned on real pads |
| 2 x I2S mics | `arecord -D plughw:CARD=crabaudio -f S32_LE -r 16000` (`hw_real.py:137-145`) | synthetic stereo (`hw_emulated.py:68-79`) | needs the custom overlay `hardware/radxa/crab-audio.dts` (not read in depth) |
| INA219 | yes (`backends.py:339-368`) | fake smbus | `current()` exists but nothing in the firmware calls it (grep) |
| SSD1309 OLED | yes (`i2c_devices.py:19-40`) | `FakeOLED` | 1024 bytes in 32-byte I2C blocks per frame; at 100 kHz that is ~100 ms per frame, so ~15 fps may be unreachable (bus speed from the overlay: UNVERIFIED) |
| MAX98357A + speaker | `aplay` of a rendered waveform (`hw_real.py:147-153`) | `played` list | |
| CPU temperature | sysfs (`hw_real.py:155-161`) | constant 45 C | log only |

**Camera pipeline** (`robot/vision.py`, `robot/localizer.py`, `robot/calibrate_camera.py`):
- Frame: 640x480, one frame at most every 0.2 s (`senses.py:64`).
- Where am I: one printed ArUco marker, `DICT_4X4_50`, ID 7, 40 mm tab (30 mm black) on the back of the desk tray (`vision.py:27-29, 125, 132-163`), `solvePnP` with `SOLVEPNP_IPPE` after fisheye undistortion. Without the measured lens file the camera is "uncalibrated" and never gives a position (`vision.py:143-144`, `core.py:96-98`).
- How fast am I really moving: phase correlation of the desk patch in front, `cv2.phaseCorrelate` (`vision.py:197-236`).
- What things are there: four HSV colour ranges (blue pen, pink eraser, red cap, yellow ball), biggest blob over 25 px, its centroid projected onto the desk plane (`vision.py:30-36, 165-183`). Anything else is invisible to this code.
- Is a person there: OpenCV Haar frontal-face cascade (`vision.py:129-130, 185-194`).
- Who is it: YuNet detector + SFace 128-number fingerprint from the OpenCV model zoo (`pinch/faces.py:21-50`; models fetched with SHA-256 by `deploy/install.sh:29-30`), one OpenCV thread, runs on the Radxa CPU.
- Lens: checkerboard 9x6, 25 mm squares, `cv2.fisheye.calibrate`, 25 views (`calibrate_camera.py:270-289`), saved to `/var/lib/pinch/camera_intrinsics.json`.
- Where speech understanding runs: nowhere on the robot. There is no speech-to-text in the repository (grep for whisper, vosk, asr finds nothing in `pinch/`, `robot/`, `hardware/`). The microphones feed only loudness, voice activity, pitch, claps, meeting detection and direction (`pinch/hearing.py:1-13`). Words reach the mind only as typed text through `POST /say` (`http_api.py:41-42`). Typed text goes to keyword rules on the robot (`pinch/language.py:68`) and, if available, to an Ollama model on the laptop (`language.py:166-232`).

### 4. What has only run against an emulator, and what is first-time on hardware

Everything on the robot side has only run against emulators. In detail:

| Piece | Emulated by | First real run |
|---|---|---|
| Servo bus bytes, SYNC_READ/WRITE, EEPROM lock, ID change | `robot/bus_emulator.py` | all of it; the emulator was written from the same SDK reading as the driver, so it cannot catch a misreading |
| PWM arms (sysfs, 6 overlays, pulse mapping) | fake sysfs tree | pulse range, direction, travel, centre |
| Calibration (offsets, directions, trim) | `SimPerson` (`calibrate.py:459-483`) | the whole procedure with a human hand |
| IMU, INA219, MPR121, OLED, VL53L0X register code | `robot/drivers/fakes.py`, fake smbus | all of it; VL53L0X is the riskiest (simplified init) |
| XSHUT bring-up of 5 sensors by GPIO | fake libgpiod v1 and v2 | pin numbers and bus addresses |
| Camera: lens model, marker pose, colours | twin render (rectilinear 80 degree pinhole, `sim/desk.py:175-176`) plus a synthetic fisheye self-test (`calibrate_camera.py:337-353`) | real fisheye image, real light, real colours, real exposure |
| Microphones | synthetic voice and 2-sample integer delay (`hw_emulated.py:68-79`) | everything including servo noise in the mics |
| Supervisor limits, e-stop, safety stops | `SimBackend`, `FakeServos` (`tests/test_supervisor.py`), selftest | thresholds against real servo reports |
| Loop timing | laptop CPU | RK3566 |
| systemd service, install script, boot | never | all of it |
| Chain "firmware loop -> byte-level bus -> physics" | never together: `EmulatedHardware` uses `SimBackend` directly (`hw_emulated.py:34`); the byte-level bus is used only in the selftest (`backends.py:545-558`) | |

### 5. Supervisor, safe mode, watchdog, boot and shutdown

Faults detected and the response (all in `robot/crab_robot.py` unless noted):

| Fault | Detection | Response |
|---|---|---|
| NaN, infinite or wrong-size target | `:113-115` | hold still, counted in `bad_targets` |
| Target outside joint limits | `:116` | clipped to the CAD limits |
| Target faster than the servo | `:117` (`MAX_STEP`) | clipped per tick |
| Shin folding into the hip | `:119` | clamped |
| Two arms about to touch | `:122-124` | arms hold, legs carry on |
| Servo temperature over 65 C | `:178-179` | safety stop: torque off, arm pulses off, loop ends |
| Pack below 6.4 V (or bus below 4.6 V) for 1 s | `:180-185` | safety stop (battery: also plays a tone and runs `systemctl poweroff`, `firmware/core.py:324-329`); warning at 6.8 V |
| A servo silent or erroring for 10 reads | `backends.py:238-239` -> `crab_robot.py:84-88` | safety stop |
| 25 failed writes in a row | `backends.py:203-204` | exception, process exits, systemd restarts |
| Transient I2C or servo-reply error | `firmware/core.py:130-142`, `backends.py:194-208` | counted, last good value used |
| IMU dead | `backends.py:251-258` | silently returns "level, no rotation" forever (only an error log at 25 failures) |
| Battery monitor dead | `backends.py:210-211` | last good value used forever |
| Cliff beam missing, failing or stale (0.3 s) | `pinch/touch.py:362-372` | "unknown": no walking toward that side (`:354-360`) |
| Cliff sees no desk while walking toward it | `touch.py:382-387` | stop and back off once (latched) |
| Foot over the desk edge (from marker pose) | `firmware/core.py:206-212` | "recover" reflex |
| User e-stop (`POST /stop`, or "emergency stop" spoken) | `core.py:144-150, 160-162` | torque off, limp, until `/resume` |
| Slow loop, CPU over 80 C | `core.py:298-302` | log only |
| Exception inside the mind or body | none: `step()` has no try/except around `m.tick` (`core.py:166-185`) | the process dies, `finally` sits the robot down (`core.py:312-329`) |

**Boot:** systemd starts `pinch_main.py` after the network is up (`pinch.service:2-3, 25`). Calibration missing -> exit 2 (`pinch_main.py:64-66`); `Restart=on-failure`, 5 tries in 5 minutes (`pinch.service:6-7, 19-20`). `Crab.start()` assumes the arms are in the REST pose, reads the legs, enables torque and eases to stand in 2 s (`crab_robot.py:210-218`), without checking that the robot is upright or on a surface.

**Shutdown:** SIGTERM sets `running = False` (`core.py:273-275`), the robot sits to REST over 1.5 s, torque and arm pulses go off, memory is saved with fsync (`core.py:312-323`, `crab_robot.py:243-252`). After a safety stop it does not move, only releases (`crab_robot.py:243-252`).

**Safe mode:** a safety stop returns exit code 0 (`pinch_main.py:70`), so `Restart=on-failure` does NOT restart it: the robot stays limp until someone starts it (the unit's comment at `pinch.service:15-16` says a restart waits 5 s, which only applies to crashes). A 65 C stop therefore never recovers by itself. There is no degraded mode (for example "sit and watch" with a dead leg).

**Watchdog:** none (no `WatchdogSec`, no heartbeat). A hard kill (OOM kill with 2 GB RAM, power glitch of the computer only) leaves the leg servos energised at their last goal and the PWM outputs running.

**Logging:** Python `logging` to stdout, so journald (`pinch_main.py:51`); optional session recording with `--record` or `PINCH_RECORD` (`pinch_main.py:48`), off by default and not set by the installer (`deploy/install.sh:34-40`); decision log file next to the memory file (`core.py:44-45`). No log size policy beyond journald defaults.

**Update and rollback:** none. `deploy/install.sh:13-18` copies the folders over `/opt/pinch` with `cp -r` and runs an unpinned `pip install`; there is no versioning, atomic swap, health check after update or rollback. In addition the copy line names a folder that does not exist: `cp -r crab pinch robot brain policies sim` (`install.sh:15`) but the repository has no `brain/` (checked with `ls` of the repository root), and the script runs with `set -e` (`install.sh:5`), so it stops at that line. The project also requires Python >= 3.12 (`pyproject.toml`), while the script targets Debian 12 (`install.sh:10`), whose system Python is 3.11 (UNVERIFIED for Radxa OS).

### 6. HTTP API (`robot/http_api.py`)

Listens on `0.0.0.0:8080` (`:74`), plain HTTP, no authentication, no TLS, no rate limit, no origin check, request size taken from `Content-Length` with no cap (`:36-38`).

| Endpoint | Effect |
|---|---|
| `GET /status` | feeling, activity, needs, owner name, pose, battery, diary lines (what it did and heard) (`firmware/core.py:345-356`) |
| `GET /memory` | summary of what it remembers: owner, people names, facts, counts |
| `POST /say {text}` | makes it "hear" any sentence (300 chars): commands, tricks, teaching names, "I'm in a meeting" |
| `POST /touch {gesture}` | fakes a stroke, tickle, boop, hold, claw hold |
| `POST /forget {what}` | erases kinds of memory or a person (and their face fingerprints) |
| `POST /settings {face_recognition}` | turns face recognition on or off |
| `POST /quiet_hours` | sets or clears quiet hours |
| `POST /stop`, `POST /resume` | e-stop (torque off) and undo it (torque on, stands up) |

Any device on the same Wi-Fi (or network segment) can: read the owner's name, facts and the diary; rewrite or erase the robot's memory and turn off face recognition; remove the owner's quiet hours; make it move with `/say` (dance, come here, tidy), which moves a 0.6 kg robot on a desk; stop it dead; and `/resume` after the owner pressed stop (so a user e-stop is not authoritative). A web page open in any browser on the network can also send these POSTs (no token or origin check). `/stop` runs on the HTTP thread and writes to the servo port while the control loop may be mid-write or mid-read (`http_api.py:63-64` -> `firmware/core.py:144-150` -> `backends.py:260-270`); there is no lock on the serial port, and the bus is half-duplex, so a collision can garble one tick. Not measured.

### 7. Laptop dependence

| Function | Needs the laptop? | Without it |
|---|---|---|
| 50 Hz control, reflexes, safety, walking, emotes, face, touch, sounds | no | full |
| Mind, memory, routine, curiosity | no (runs on the Radxa) | full |
| Keyword understanding of typed text | no (`pinch/language.py:68`) | works |
| LLM understanding of free sentences | yes (Ollama over HTTP, `language.py:166-232`) | keyword rules only; the call times out in its thread (90 s) |
| Speech to text | does not exist | n/a |
| Companion app, `/say` | needs some device on the network | no text input channel; touch still works |

Weak spot: the LLM is checked once, when the mind is built (`pinch/mind/core.py:29-31`: `if not self.llm.check(): self.llm = None`). If the laptop is off at boot (3 s timeout, `language.py:178-185`), the LLM stays off until the next reboot, even when the laptop appears later. Pinch also sends the memory summary and typed text to the laptop in clear text (`language.py:186-197`). The robot is safe without it.

---

## B. Power (piece A1)

### 8. `sim/power.py`

**Model:** for each scenario it steps the MuJoCo model and computes servo current as `|motor torque| / kt` per joint plus 6 mA idle per servo (`sim/power.py:1-9, 34-42`), with the motor torque taken as actuator force minus back-EMF x speed. Back-driven phases count as draw. Battery current = `(mean servo amps x 5 V + 4.0 W) / 0.90 / 7.4 V` (`power.py:27-31, 74`). Runtime = `0.80 x 850 mAh / battery amps` (`power.py:30-31, 75`).

**Inputs, all constants:** compute plus camera plus sensors `COMPUTE_W = 4.0` ("~3 W, Wi-Fi on", a guess, `:27`), regulator efficiency 0.90, pack 7.4 V flat, 80 % usable, 850 mAh. Servo constants (kt, back-EMF) come from datasheet-derived specs in `crab/params.py:95-128`, not measurements (the only measured servo, the ST3215 from Rhoban BAM at `params.py:128`, is not the one used).

**How 27-40 minutes arises** (my arithmetic from `hardware/electronics.md:80-86`): standing = 0.56 A at 5 V = 2.8 W of servo, plus 4 W = 6.8 W; divided by 0.9 and 7.4 V = 1.02 A; 0.68 Ah / 1.02 A = 40 min. Trot = 1.2 A x 5 V = 6 W plus 4 W = 10 W; 1.50 A; 27 min.

**Biggest consumers:** while standing, the computer and sensors are 59 % of the draw (4 of 6.8 W) and holding torque 41 %; while trotting, servos are 60 %. Because the 4 W is a constant, a pet that mostly stands gets ~35-40 minutes however it behaves; reaching a whole day needs a dock or swaps, not servo savings alone.

**Not modelled:** voltage sag under load, regulator light-load efficiency, Wi-Fi bursts, the OLED, amplifier and speaker (assumed inside the 4 W), cold or aged cells, the buck converters' quiescent current, the BMS board, PWM-arm holding current (they hold position against gravity with no feedback), or the camera and vision load. The servo current in standing is a physics output of a model never compared with a real servo (piece A2).

**Rest or torque-off logic:** none in the model and none in the runtime. Torque goes off only on shutdown, e-stop or safety stop (`crab_robot.py:148-156, 192-199, 243-252`). The "rest" and "low_power" activities (`pinch/activities/recovery.py:104-125`) keep the legs energised.

**Low-battery behaviour:** the INA219 gives pack volts; `battery_fraction` is a straight line from 6.4 to 8.4 V (`hw_real.py:168-170`), not a LiPo curve and not corrected for load (a pack under walking load reads lower and "recovers" when it stops). The mind gets hungry as `(1 - charge)^2` (`pinch/safety.py:79-81`), goes home and begs at 40/25 %, dozes at 10 % (`recovery.py:76-125`), the supervisor stops at 6.4 V and powers the computer off (`core.py:324-329`), and a BMS board is the hard floor (`BOM.md:53`).

**Defect:** the mind decides "put on charge" whenever the battery fraction rises by 1e-5, which is 20 microvolts (`pinch/mind/selfcare.py:38-45`; fraction = (V-6.4)/2). INA219 resolution is 4 mV (`backends.py:361-363`), and voltage rises every time load drops, so a real robot will repeatedly "feel fed" and set `charging = True`. The twin never shows this because its battery is a constant (`robot/backends.py:89`, `hw_emulated.py:97-98`) and `PinchSim` never sets `world.battery` (grep: no "battery" in `pinch/twin/` or `sim/benches/`).

**Docking or charging:** nothing. Charging is by hand, off the robot (`hardware/BOM.md:46-47`, `electronics.md:171`); there is no dock, no charge contacts, no way to know the charge state, and the INA219 current is never read.

---

## C. Twin (piece B4)

### 9. How the twin produces what the mind senses

Key: (a) raw signal simulated and the SAME perception code runs; (b) the perception RESULT is made from ground truth with noise and limits; (c) ground truth passed straight through.

| Sense | PinchSim (benches, tests, app) | Firmware on EmulatedHardware (`firmware_test.py` only) |
|---|---|---|
| Named objects (pen, eraser, cap, ball) | (b) `perceive()`: true pose, bearing within 0.9 rad, range 0.05-0.6 m, noise sigma 4 mm (12 mm "flaky"), `pinch/twin/sensors.py:48-66`. Things that are off the desk or in the claw bypass the field of view (`:63`). No occlusion, light, colour or identity error. | (a) MuJoCo render 640x480 (`hw_emulated.py:62-66`) -> real `Vision.things` colour thresholds (`vision.py:165-183`) |
| Objects used by the body (grab and push planner) | (c) `Body(..., objects_fn=self.obj)` = exact pose (`pinch/twin/core.py:69`) | belief from the mind with a constant 4.5 mm sigma (`robot/firmware/core.py:104-108`) |
| Unknown props (box, tube, toy) | (b) `perceive_props()`: 4 mm position noise, 1 cm/s velocity noise, size noise, confidence falling with range, `sensors.py:74-101` | not simulated |
| Faces and people | (c) a flag: `user_active` makes the owner present (`sensors.py:68-70`); other people are `mind.see_person(id, confidence, name)` calls (`sim/benches/longrun.py:186-193`) | no person in the 3D scene; the face-recognition check feeds a photo (`firmware_test.py:373-410`, `requirements.txt` comment) |
| Sound | audio chunks can be fed to the real `Hearing` (`stimuli.py:17-23`) = (a) for loudness, meeting and voice detection; no direction is passed (`stimuli.py:20-21`) | (a) synthetic mono voice made stereo with an exact 2-sample delay -> real `Hearing` and `AudioDirection` (`hw_emulated.py:68-79`, `senses.py:66-81`) |
| Speech (words) | (c) text goes straight to the mind: `say()` -> `mind.hear_text` (`stimuli.py:6-11`) | text via `POST /say` (same) |
| Touch | (c) a gesture name straight to the mind (`stimuli.py:13-15`) | (a) fake MPR121 zones -> real driver -> real `TouchGestures` (`core.py:221-227`) |
| IMU | (c) exact gyro and gravity from MuJoCo, no noise, bias or lag (`backends.py:126-128`) | same `SimBackend` |
| Cliff beams | (a)-ish: MuJoCo ray distance from a rangefinder site (`sim/desk.py:141-153`), exact, passed to the reflex as millimetres (`sensors.py:103-115`); skips the VL53L0X driver and its error behaviour | (a) ray distance -> fake VL53L0X registers -> real driver and `read_ring` (`hw_emulated.py:81-86, 49-52`) |
| Battery | none: `world.battery` stays at its default; `SimBackend.battery` constant 7.6 V (`backends.py:89`) | constant fraction 0.8 (`hw_emulated.py:97-98`) |
| Servo feedback | (c) position from the joint plus gear-play joint, speed, load = actuator force / force limit, voltage fixed 5.0 V, temperature fixed 35 C (`backends.py:113-125`); no 12-bit quantisation, no bus errors, no delay | same |
| Feet carrying weight | (c) from MuJoCo contacts (`pinch/twin/world.py:49-64`); the robot uses servo load (`hw_real.py:172-178`) | contacts (`hw_emulated.py:100-110`) |
| Robot's own position | (c) true `qpos` (`twin/world.py:15`, `twin/core.py:69`); no localiser, no drift, no `pose_sigma` | (a) rendered marker -> real localiser (`firmware_test.py:71-73`) |
| Grip (is the pen between the fingers) | (c) contact test (`sensors.py:32-45`, wired at `twin/core.py:72`) | not wired (see below) |
| Learned-grasp view | (b) claw gap vs pen, noise 2 mm (`sensors.py:15-30`, `twin/core.py:74`) | not wired |

**Consequence found while tracing this:** on the robot, `grip_fn` and `grasp_view_fn` are never set (grep of `robot/` finds neither), and `object_belief` returns a constant height 0.0045 (`robot/firmware/core.py:108`). The pick routine decides "held" by `o[1] > oz + 0.004` when no `grip_fn` exists (`pinch/body/manipulation.py:219`), which can never be true with a constant height. So on hardware every pick reports failure, and the learned grasp is never used (`manipulation.py:205`). The twin hides this because it supplies both.

### 10. Latency, noise and fault models

- **Command latency:** none in the twin. `sim/latency.py` is an 18-line delay line used only by `sim/check.py` (and `sim/walk.py`), with the 5-20 ms values in `crab/params.py`. `SimBackend` and `PinchSim` apply commands the same tick. The robustness bench adds one 20 ms tick of delay to the policy's action in half its episodes (`sim/benches/robustness.py:8, 87, 172, 195`).
- **Noise:** only object sightings (4 mm; 12 mm flaky), prop sightings, and the 2 mm grasp view (`sensors.py`). No IMU, servo, cliff, touch, voice or camera-image noise. No servo position quantisation. Microphones have no ego-noise (servos, footsteps).
- **Physics randomisation (robustness bench only):** floor friction 0.35-1.2, masses +-20 %, servo strength -30..+15 %, stiffness +-25 %, desk tilt up to 4 degrees, shoves every 2.5 s (`robustness.py:1-21`). Horn mounting offsets exist for calibration tests (`backends.py:561-564`) but not in `PinchSim` (`twin/core.py:52` uses the default calibration).
- **Faults** (`pinch/twin/faults.py:6-27`): `vision` (sees nothing), `flaky_detection` (half the sightings dropped, noise x3), `audio` (hears nothing), `cliff_dead` (one beam returns None), `weak_leg` (one leg's force range scaled). That is all. Not injectable: bus errors, a silent servo, brown-out, battery sag, IMU freeze, camera blur or glare, marker occluded, lost laptop. Hardware-level faults are injected only inside `firmware_test.py:180-203` (flaky I2C and bus).

---

## D. Test world (pieces B5 / B6)

### 11. Worlds

- **One world:** a flat 70 x 45 cm desk with the floor 75 cm below (`pinch/world.py:15-16`, `sim/desk.py:57` floor at z = -0.75; the docstring at `desk.py:3` says "a metre"). Built by `sim/desk.py:46-171`.
- **Fixed furniture:** one desk tray at (0.18, 0.08), half-size 8 x 3.5 cm, with an upright ArUco marker tab (`pinch/world.py:18-19`, `desk.py:79-101`); the owner's chair spot is a constant (0, -0.30) (`world.py:17`).
- **Objects:** `OBJECTS` pen, eraser, cap, ball with fixed shapes, masses, colours (`desk.py:29-34`); fixed `START` positions (`desk.py:43`); `PROPS` box, tube, toy parked at (3, 3, -0.5) until shown (`desk.py:37-42`). Test-only fixed obstacles through `extras` (cables, bars) (`desk.py:64-73`).
- **Variation:** seeds change only the mind's random streams; benches jitter object starts +-2 cm, +-0.3 rad (`tidy.py:21-30`); pick drops things at random spots (`pick.py:26-32`); terrain worlds for walking: rough clutter, stuck foot, hurdles 3-15 mm, ramps up to 21 degrees (`robustness.py:12-17`, `crab/terrain.py`). The desk size, tray, marker, object set and light never change.
- **Lighting and visuals:** one fixed light plus a headlight (`sim/models/crab.xml:8, 120`), a gradient sky, a checker floor texture. No lighting, texture, colour, exposure or camera variation anywhere.
- **Not present:** a room floor scene, furniture, stairs, doors, walls, a second walkable surface. The floor 75 cm down is only a hazard.
- **People:** no bodies. A person is (1) a boolean `user_active` (`twin/core.py:87`, `sensors.py:68-70`), (2) typed text (`stimuli.py:6-11`), (3) gesture names (`stimuli.py:13-15`), (4) `see_person(id, conf, name, owner)` calls (`longrun.py:186-193`), (5) props moved by waypoints for "something rushes at it" (`twin/world.py:98-129`). Nothing has a position, a voice source, a height or a gaze.

### 12. Tests and benches

**CI** (`.github/workflows/tests.yml`): Linux, Python 3.13, `pip install mujoco==4.0.1 numpy pillow opencv-python-headless ruff`, then `ops.guard tree`, `ruff check .`, `tests/run.py`. 30-minute timeout. JAX is not installed, so JAX tests skip; the golden trace skips on Linux because it was recorded on Windows (`tests/test_golden.py:54-55`). `python -m ops check` runs the same three steps (`ops/cli.py:228-236`).

**Not run by CI or `ops check`:** `robot/selftest.py` (about 40 checks of drivers, calibration, supervisor, emulated bus, PWM) and `robot/firmware_test.py` (about 45 checks of the firmware, HTTP API, camera, face models, e-stop, SIGTERM, real-hardware glue). A grep of `tests/`, `ops/` and `.github/` for either name finds nothing; their pass counts in decisions (45/45, 40/40) are manual runs. Of the `robot/` code, only `crab_robot.py` (`tests/test_supervisor.py`) is imported by the test suite (plus the recorder in `test_recorder.py`).

| File | What it measures | Physics? | Seeds / size |
|---|---|---|---|
| `test_architecture.py` | import layering rules | no | all files |
| `test_supervisor.py` | fuzzed targets never reach servos unsafe; e-stop sends nothing | no (fake servos) | thousands of random targets |
| `test_attention.py` | habituation model; a voice turns Pinch | model + twin | 1 scenario |
| `test_curiosity.py` | novelty model; twin: a prop makes it look, approach, startle, never edge | model + twin | seed 3 plus a few |
| `test_decisions.py` | every switch has a reason | twin | seed 1, 36 s |
| `test_faults.py` | 5 faults x 120 s stay safe | twin | seed 4 each |
| `test_golden.py` | exact 2-minute trace | twin | seed 7; skipped on Linux |
| `test_replay.py` | same seed replays exactly | twin | 2 seeds, 40 s |
| `test_life.py` | idle micro-actions | model + twin | 3 min |
| `test_long_day.py` | 24 h fast-forward plus 2 short physics windows | mostly mind-only | seeds 0-3 |
| `test_long_week.py` | 7-day memory run | mind-only | seed 0 |
| `test_observer.py` | observer catches planted problems | synthetic + 10 min twin | |
| `test_recorder.py` | recorder is read-only and exact | twin | seed 3, 20 s |
| `test_routine.py` | quiet hours, sleep | mind | |
| `test_states.py` | every activity has limits | mind + twin | |
| `test_touch_nuance.py` | touch reactions | mind | |
| `test_variation.py` | emote variants stay in limits | no physics | |
| `test_privacy.py` | forget, summary | mind | |
| `test_ops.py` | job scripts, guard | no | |
| `test_gait_jax.py` | JAX gait equals numpy gait | no | skipped without JAX |

**Benches** (`sim/benches/`, run by hand, in parallel processes; runtimes are not stated in the files; the twin "runs about 3.5x real time on one CPU core" per `longrun.py:6`):

| Bench | Measures | Size |
|---|---|---|
| `robustness.py` | walk and emote falls under random physics, 5 worlds | 40 episodes each (8 with `--quick`), `:240-241` |
| `hurdles.py` | cross a bar or ramp, forward or sideways, scripted and policy | 3 robots per case, `:134-142` |
| `obstacles.py` | walk into a cable, pen, book; recover | 6 cases, 1 run each, `:56` |
| `tidy.py` | put the cap away; tidy the eraser | 8 seeds x 2 cases, `:51-56` |
| `pick.py` | pick a pen or eraser and drop in the tray, mind off, true object pose | `--n 12`, `:55` |
| `righting.py` | get up from the back, swing vs policy | 100 starts default, `:105` |
| `scenarios.py` | 15 behaviour checks (tricks, tidy, meeting, comfort, touch, curiosity...) | 15 scenarios, one seed each |
| `soak.py` | hours of a random day, verdict | `--hours`, `--seeds`, `:111-112` |
| `longrun.py` | 24 h day (5 physics windows of 10 min), 7-day memory | seeds 0-3, one week |

All benches use `PinchSim`, so they run on ground-truth pose and object positions (question 9). `longrun` uses a mind-only fast-forward (`FastPet`, `longrun.py:206`) for long spans and physics for sampled windows; everything else is physics.

**Coverage gaps:** firmware, drivers, vision, HTTP API, calibration and bus code have no CI; the localiser and the edge guard under pose error are not in any bench; no fault tests for the bus, battery or IMU; no test of the install or the service; no hardware-in-the-loop timing; seeds per bench are small and scenarios are single-run; "the twin's contacts are chaotic" (`tidy.py:27`), so one pass is weak evidence.

### 13. Scenarios as data

Mostly code. Scenarios in `sim/benches/scenarios.py` are Python functions with their own asserts (`:27-40`). The only data-like timeline is the generated `DayPlan.events`, a seeded list of `(time, kind, argument)` tuples (`longrun.py:84-197`), applied by `apply_event` to either the fast mind or `PinchSim` (`longrun.py:177-196`). Its pass/fail checks are code (`longrun.py:476-553`). There is no file format, no assertion language, no declarative scenario. Training experiments are TOML (`configs/experiments/*.toml`) but those are RL runs, not scenarios.

---

## E. Generality and risk

### 14. Hard-coded desk-scene assumptions

Counts are lines matching a desk-geometry pattern (`Desk(`, `desk.`, tray, half_x/half_y, edge_margin, `desk_top`, marker) and a named-object pattern (`"pen"|"eraser"|"cap"|"ball"`, `OBJECTS`, `START`, `PROPS`). Roots of the assumption: `pinch/world.py:14-21` (desk size, tray, home spot, owner chair, edge margin as constants) and `sim/desk.py:29-43`.

| File | Desk pattern | Object pattern | Key lines |
|---|---|---|---|
| `sim/desk.py` | 25 | 10 | 3, 29-43, 57, 79-101, 174-175 |
| `robot/vision.py` | 12 | 5 | 27-37 (marker ID 7, four colours, heights), 132-138 (marker is on the tray tab) |
| `robot/firmware/core.py` | 2 | 0 | 39, 49, 90, 108 (z = 0.0045 for everything) |
| `robot/firmware/senses.py` | 1 | 0 | 124 (a thing counts only if inside the desk) |
| `robot/hw_real.py` | 2 | 0 | 67 (`Desk()` default), 70 |
| `robot/hw_emulated.py` | 2 | 0 | 32-33, 102 |
| `robot/localizer.py` | 1 | 0 | docstring only; pose only from the marker |
| `robot/backends.py` | 0 | 2 | |
| `pinch/twin/core.py` | 5 | 10 | 44, 83-86 (pen and eraser homes at the tray) |
| `pinch/twin/world.py` | 4 | 2 | 24-26, 54 |
| `pinch/twin/sensors.py` | 0 | 4 | 15-30 grasp shapes for pen and eraser only; right claw only (`:17`) |
| `sim/benches/scenarios.py` | 14 | 15 | |
| `sim/benches/longrun.py` | 9 | 10 | desk positions, 3 objects nudged |
| `sim/benches/pick.py` | 8 | 2 | |
| `sim/benches/tidy.py` | 3 | 7 | |
| `sim/benches/soak.py` | 2 | 2 | |
| `sim/record.py`, `sim/clips.py` | 1, 5 | 5, 2 | |

For scale, outside the listed folders: `pinch/body/manipulation.py` has 33 and `pinch/body/locomotion.py` 16 lines that mention the desk.

What it means: position on the robot exists only if the camera can see one printed marker on one tray (`vision.py:140-148`); the "edge" is the desk rectangle (`pinch/world.py:23-27, 49-51`); the cliff beams are tuned to a flat surface; the object set and colour thresholds are four named toys.

### 15. Ranked weak points (12)

1. **Grabbing can never succeed on the robot.** No `grip_fn` or `grasp_view_fn` in `robot/`, constant height belief (`robot/firmware/core.py:108`), held test impossible (`pinch/body/manipulation.py:219`). Consequence: every pick fails on hardware and the learned grasp never runs; the pick benches describe the twin's helper functions.
2. **The benches judge the mind and body on ground truth the robot will not have.** Pose and object poses are exact (`pinch/twin/core.py:69`), with no localiser, no drift, and `pose_sigma_fn` unset in the twin. On the robot the position comes from one marker with 30 % drift between fixes and a cap of 6 cm (`robot/localizer.py:276-277`) against a 7 cm edge margin (`pinch/world.py:21`). Consequence: edge safety, tidying and push success rates are optimistic by an unknown amount.
3. **The robot code is not in CI.** `robot/selftest.py` and `robot/firmware_test.py` are never run automatically (grep of `tests/`, `ops/`, `.github/`). Consequence: a refactor can break drivers, vision or the HTTP API without any red build.
4. **The compute budget is a guess.** No RK3566 timing; "5x slower" assumed (`docs/DECISIONS.md:287,299,301`); a Python mind, numpy policies, Haar + ArUco + colour + flow + YuNet + SFace + Hearing FFTs share one GIL on 4 small cores; the NPU is unused; no real-time priority; a slow loop is only logged (`core.py:298-299`). Consequence: the 50 Hz loop may run late exactly when the pet is busiest.
5. **Install and update cannot be trusted.** `deploy/install.sh:15` copies a nonexistent `brain` under `set -e`; unpinned `pip` (`:18`); Python 3.12 requirement versus Debian 12's 3.11; no rollback. Consequence: the first deploy fails at step 2, and a bad update bricks the pet with no way back.
6. **Open, unauthenticated control port.** `0.0.0.0:8080`, no token (`http_api.py:74`); `/resume` undoes the owner's stop; `/memory`, `/forget`, `/say` open to the LAN; e-stop writes the bus from another thread (`http_api.py:63-64`). Consequence: anyone on the Wi-Fi can move, wipe or read the robot, and a stop can collide with a bus read.
7. **Servo bus assumptions are untested on a real STS3032.** SYNC_READ support and register map (`feetech_bus.py:48`, `crab/params.py:95`); servo error flags treated as "silent" and turned into a safety stop after 10 reads (`feetech_bus.py:111-112`, `backends.py:238`); possible snap to a stale goal at torque-on (`crab_robot.py:158-167, 210-218`); a 60 ms stall when a servo is silent (`backends.py:223-236`). Consequence: the first power-up may stop at once, or a leg may jerk.
8. **No speech recognition, and the LLM is decided once at boot.** The mics give only prosody and direction; words arrive only as typed text (`http_api.py:41-42`); `Mind` drops the LLM permanently when the laptop is off at boot (`pinch/mind/core.py:29-31`). Consequence: "understand people" has no spoken input path today, and a late laptop never helps.
9. **Power logic does not match the hardware.** A 20-microvolt rise is read as "put on charge" (`pinch/mind/selfcare.py:38-45`) while the INA219 resolution is 4 mV; battery fraction is a straight line (`hw_real.py:170`); no torque-off at rest; compute fixed at 4 W; current never read (`INA219.current()` unused); no dock. Consequence: spurious "fed" emotions, a 27-40 minute figure resting on guesses, no route to all-day running.
10. **The twin's senses are cleaner than the real ones.** A rectilinear 80 degree render versus a 160 degree OV5647 fisheye (`sim/desk.py:175-176`); colours tuned in the twin (`vision.py:30-35`); one fixed light; exact IMU, no servo quantisation, no latency, no ego-noise in the mics; VL53L0X rays exact and independent of surface. Consequence: vision, direction and cliff reflexes are validated against signals they will never see.
11. **One hard-coded world, scenarios as code, thin statistics.** Desk and marker constants (`pinch/world.py:14-21`), four named toys; tests and scenarios are code; 15 single-seed scenarios, 6 obstacle cases, 16 tidy runs; the golden trace and JAX test skip in CI. Consequence: generality (rooms, other objects, people) cannot be tested or even expressed, and chaotic contact makes few-run passes weak.
12. **Crash and kill behaviour.** Any exception from the mind or body ends the process (no guard around `m.tick`, `core.py:166-185`); a safety stop exits 0 and is never restarted (`pinch_main.py:70`, `pinch.service:19`); no watchdog; a hard kill leaves servo torque and PWM pulses on (`backends.py:426-438`). Consequence: a software bug makes the pet sit down for good, and a hang or OOM leaves live servos with no supervisor.

---

## Could not determine

- Real RK3566 timings, RAM use, CPU load and thermals: not measured anywhere in the repository.
- Whether the STS3032 answers SYNC_READ, uses the STS3215 register map, keeps or resets its goal at torque-on, and what its error flags do under stall.
- Whether the OV5647 shows up as a plain OpenCV device on Radxa OS, and its real exposure behaviour.
- Whether the `crab-audio.dts` overlay builds and gives S32_LE stereo (the hardware folder was only skimmed).
- Whether Radxa OS ships Python 3.12 or newer (Debian 12 base is 3.11, UNVERIFIED), and whether the code needs 3.12 features.
- I2C bus speeds from the overlays and the resulting OLED frame rate.
- Test runtimes: not stated, and not run (the brief forbade it).
- Internals of `pinch/body`, `pinch/mind`, `crab/` beyond the lines cited (outside this audit's file list).
