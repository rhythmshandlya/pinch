"""Phase 3 bench: prove one leg servo (STS3215) and one arm servo (STS3032) behave like the model, step by step.

    python robot/bench.py plan                          # the steps, in order, with the exact commands
    python robot/bench.py heat --id 1 --minutes 60      # hold a load for an hour, log temperature and voltage
    python robot/bench.py delay --id 1                  # command -> first motion, and the bus round trip
    python robot/bench.py regulator                     # type in the 5 V regulator's readings at 6.0 / 6.5 / 7.0 V
    python robot/bench.py track --logdir bench/logs --params bench/sts3215.json   # model vs measured, deg RMS
    python robot/bench.py report                        # every result so far against the phase 3 checks
    add --rehearse to heat / delay to run against an emulated servo (no hardware)

Recording the pendulum logs and fitting the servo model use Rhoban's BAM, the tool our servo model comes from
(github.com/Rhoban/bam, Apache-2.0): `python -m bam.feetech.all_record ...`, `python -m bam.process ...`,
`python -m bam.fit ...` (`plan` prints them with our numbers). This script adds only what BAM does not: the hour at
8.4 V, the delay, the regulator and the pass/fail against the checks. Results go to .ops/bench/phase3.json; the
measured delay is also written to configs/electronics.toml [measured].
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.feetech_bus import MODELS, FeetechBus  # noqa: E402

RESULTS = ROOT / ".ops" / "bench" / "phase3.json"
ELECTRONICS = ROOT / "configs" / "electronics.toml"
PENDULUM = dict(mass=0.5, length=0.10)      # kg, m: PLACEHOLDER: weigh and measure the printed arm + weights on the horn (order 1 row)
CHECKS = dict(track_deg_rms=2.0, surface_c=48.0, rail_v_min=4.75, rail_input_v=6.5)


def load():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


REHEARSE = False


def save(key, value):
    path = RESULTS.with_name("phase3_rehearsal.json") if REHEARSE else RESULTS      # rehearsals never count
    path.parent.mkdir(parents=True, exist_ok=True)
    r = json.loads(path.read_text()) if path.exists() else {}
    r[key] = value
    path.write_text(json.dumps(r, indent=1))


def record_measured(name, value, note):
    """Write `name = value  # note` into configs/electronics.toml's [measured] table (added at the end if missing)."""
    lines = ELECTRONICS.read_text(encoding="utf-8").splitlines()
    entry = f"{name} = {value}  # {note}"
    if "[measured]" not in lines:
        lines += ["", "[measured]", entry]
    else:
        i = lines.index("[measured]") + 1
        j = i
        while j < len(lines) and not lines[j].startswith("["):
            j += 1
        block = [ln for ln in lines[i:j] if not ln.startswith(f"{name} =")]
        lines[i:j] = block[:len(block) - (1 if block and not block[-1].strip() else 0)] + [entry] + ([""] if j < len(lines) else [])
    ELECTRONICS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def open_bus(a):
    model = MODELS["sts3215"]                    # the STS series shares one register map (STS3215 and STS3032)
    if a.rehearse:
        from robot.bus_emulator import EmulatedPort
        return FeetechBus(EmulatedPort(model, [a.id]), model)
    import serial  # pip install pyserial
    return FeetechBus(serial.Serial(a.port, baudrate=1_000_000, timeout=0.05), model)


def plan(_a):
    p = PENDULUM
    print(f"""(pendulum: weigh it and measure the horn-to-weight length, then edit PENDULUM in this file)
Phase 3 bench, in order (the owner does the hands-on steps; each takes 5-60 minutes):
 1. One servo on the FE-URT-1 board, 2S pack (8.4 V full) through the fuse:
      python robot/setup_servo.py --id 1          (leg servo; the arm servo gets --id 13)
 2. Pendulum on the horn ({p['mass']} kg at {p['length'] * 100:.0f} cm, weighed and measured), record and fit with BAM:
      python -m bam.feetech.all_record --mass {p['mass']} --length {p['length']} --motor sts3215 --id 1 --logdir bench/raw --vin 8.4
      python -m bam.process --raw bench/raw --logdir bench/logs
      python -m bam.fit --logdir bench/logs --actuator sts3215 --model m6 --output bench/sts3215.json --workers 4
 3. Model against measurement (check: <= {CHECKS['track_deg_rms']} deg RMS; ToddlerBot reached 1.3):
      python robot/bench.py track --logdir bench/logs --params bench/sts3215.json
 4. One hour holding the pendulum level at 8.4 V (check: below the servo's own limit; surface <= {CHECKS['surface_c']} C):
      python robot/bench.py heat --id 1 --minutes 60
 5. The 5 V regulator on a bench supply at 6.0, 6.5 and 7.0 V in (check: >= {CHECKS['rail_v_min']} V at {CHECKS['rail_input_v']} V):
      python robot/bench.py regulator
 6. Command delay (recorded in configs/electronics.toml):
      python robot/bench.py delay --id 1
 7. Repeat 2-4 for the arm servo (STS3032 at its 40 % torque limit, --id 13), then:
      python robot/bench.py report""")


def heat(a):
    bus = open_bus(a)
    bus.write(a.id, "torque_enable", 1)
    bus.goal_positions({a.id: a.angle})          # the pendulum held level: the worst steady load
    rows, t0 = [], time.time()
    limit = bus.read(a.id, "max_temp")
    print(f"holding {a.angle:.2f} rad for {a.minutes:.0f} min; the servo's own temperature limit is {limit} C")
    while time.time() - t0 < a.minutes * 60:
        s = bus.state(a.id)
        rows.append(dict(t=round(time.time() - t0, 1), **{k: s[k] for k in ("temperature", "voltage", "load", "position")}))
        print(f"\r{rows[-1]['t'] / 60:5.1f} min  {s['temperature']:4.0f} C  {s['voltage']:4.1f} V  load {s['load']:+.2f}", end="")
        time.sleep(a.every)
    print()
    bus.write(a.id, "torque_enable", 0)
    peak = max(r["temperature"] for r in rows)
    surface = float(input("Touch-thermometer reading on the servo case and the nearest printed part (C): ") or "nan") \
        if not a.rehearse else float("nan")
    save(f"heat_{a.id}", dict(minutes=a.minutes, peak_c=peak, limit_c=limit, surface_c=surface, log=rows[-5:]))
    print(f"peak {peak:.0f} C (limit {limit} C), surface {surface} C")


def delay(a):
    bus = open_bus(a)
    bus.write(a.id, "torque_enable", 1)
    rtt, motion = [], []
    for _ in range(a.repeats):
        t = time.perf_counter()
        bus.read(a.id, "present_position")
        rtt.append(time.perf_counter() - t)
    for k in range(a.repeats):
        start = bus.state(a.id)["position"]
        goal = start + (0.2 if k % 2 == 0 else -0.2)
        t = time.perf_counter()
        bus.goal_positions({a.id: goal})
        while time.perf_counter() - t < 0.5:
            if abs(bus.state(a.id)["position"] - start) > 0.009:          # half a degree of motion
                motion.append(time.perf_counter() - t)
                break
        time.sleep(0.3)
    bus.write(a.id, "torque_enable", 0)
    r = dict(read_rtt_ms=1000 * statistics.median(rtt),
             command_to_motion_ms=1000 * statistics.median(motion) if motion else None)
    save(f"delay_{a.id}", r)
    print(r)
    if not a.rehearse and motion:
        record_measured(f"servo{a.id}_command_to_motion_ms", round(r["command_to_motion_ms"], 1),
                        f"robot/bench.py delay, median of {len(motion)}, {time.strftime('%Y-%m-%d')}")


def regulator(_a):
    rows = {}
    for vin in (6.0, 6.5, 7.0):
        v = input(f"Set the bench supply to {vin} V, load the 5 V rail (about 2 A), read its output (V): ")
        rows[vin] = float(v)
    ok = rows[CHECKS["rail_input_v"]] >= CHECKS["rail_v_min"]
    save("regulator", dict(readings=rows, ok=ok))
    print("PASS" if ok else "FAIL: choose another regulator (phase 3 rule)")


def track(a):
    """RMS of (BAM model's simulated position - measured position) over every log, in degrees."""
    import numpy as np
    from bam.logs import Logs
    from bam.model import load_model
    from bam.simulate import Simulator
    model = load_model(a.params)
    errs = []
    for log in Logs(a.logdir).logs:                       # as bam.mae does: replay each log's commands in the model
        q, _, _ = Simulator(model).rollout_log(log, reset_period=None, simulate_control=True)
        errs.append(np.array(q) - np.array([e["position"] for e in log["entries"]]))
    e = np.degrees(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    save("track", dict(deg_rms=float(e), logs=len(errs)))
    print(f"model vs measured: {e:.2f} deg RMS over {len(errs)} logs (check <= {CHECKS['track_deg_rms']})")


def report(_a):
    r = load()
    tr = r.get("track", {}).get("deg_rms")
    heats = {k: v for k, v in r.items() if k.startswith("heat_")}
    lines = [("tracking <= 2 deg RMS", tr is not None and tr <= CHECKS["track_deg_rms"], tr),
             ("an hour below the servo's limit", bool(heats) and all(h["peak_c"] < h["limit_c"] for h in heats.values()),
              {k: h["peak_c"] for k, h in heats.items()}),
             ("touchable surfaces <= 48 C", bool(heats) and all(h["surface_c"] <= CHECKS["surface_c"] for h in heats.values()),
              {k: h["surface_c"] for k, h in heats.items()}),
             ("5 V rail holds at 6.5 V in", r.get("regulator", {}).get("ok", False), r.get("regulator", {}).get("readings")),
             ("delay recorded", any(k.startswith("delay_") for k in r), {k: v for k, v in r.items() if k.startswith("delay_")})]
    for name, ok, val in lines:
        print(f"{'PASS' if ok else 'open':5s} {name:34s} {val}")
    return 0 if all(ok for _, ok, _ in lines) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "regulator", "report"):
        sub.add_parser(name)
    h = sub.add_parser("heat")
    d = sub.add_parser("delay")
    for p in (h, d):
        p.add_argument("--id", type=int, required=True)
        p.add_argument("--port", default="/dev/ttyUSB0")
        p.add_argument("--rehearse", action="store_true")
    h.add_argument("--minutes", type=float, default=60.0)
    h.add_argument("--angle", type=float, default=1.5708, help="rad: the pendulum level (worst steady load)")
    h.add_argument("--every", type=float, default=5.0)
    d.add_argument("--repeats", type=int, default=20)
    t = sub.add_parser("track")
    t.add_argument("--logdir", required=True)
    t.add_argument("--params", required=True)
    a = ap.parse_args()
    global REHEARSE
    REHEARSE = getattr(a, "rehearse", False)
    return {"plan": plan, "heat": heat, "delay": delay, "regulator": regulator, "track": track, "report": report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
