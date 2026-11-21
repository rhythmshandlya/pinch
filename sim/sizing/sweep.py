"""Sweep the leg scale s for the STS3215 crab, next to today's STS3032 robot, and print the tables.

    python sim/sizing/sweep.py                 # runs everything (a few minutes), writes sim/sizing/out/*.json + tables.md
    python sim/sizing/sweep.py --tables-only   # re-print tables from the saved json

Each design runs in its own process (worker.py), at most 4 at once. The battery of each STS3215 design is sized so
that the simulated "mixed activity" lasts 40 minutes (iterated, because the pack's own mass changes the load).
"""

import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PY = sys.executable
SCALES = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
COMPUTE_W = 4.0        # repo's figure (sim/power.py): Radxa ZERO 3W busy + camera + sensors
EFF = 0.90             # regulators (repo's figure)
USABLE = 0.80
MIX = {"stand": 0.35, "idle": 0.30, "trot5": 0.25, "trot10": 0.10}      # "mixed activity" (our assumption)
TARGET_MIN = 40.0
G = 9.81
IDLE_SENS = 0.150      # A per servo: the datasheet no-load current, as an extreme for "electronics on, holding"


def run_designs(cfgs):
    OUT.mkdir(exist_ok=True)

    def one(cfg):
        p = OUT / f"{cfg['name']}.json"
        subprocess.run([PY, str(HERE / "worker.py"), json.dumps(cfg), str(p)], check=True, stdin=subprocess.DEVNULL,
                       capture_output=True)
        return json.loads(p.read_text())

    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, cfgs))


def scen_power(r, key, model="p_phys_w", idle_a=None):
    """Servo electrical power (W) in one scenario. idle_a: replace the 6 mA/servo idle draw (all 18 servos, at their rail voltage)."""
    sc = r["scenarios"]
    if key == "trot5":
        p = [sc["trot_fwd_5"][model], sc["trot_side_5"][model]]
    elif key == "trot10":
        p = [sc["trot_fwd_10"][model], sc["trot_side_10"][model]]
    else:
        p = [sc[key][model]]
    w = sum(p) / len(p)
    if idle_a is not None:
        vleg = 7.4 if r["servo"].startswith("STS3215") else 5.0
        w += 12 * (idle_a - 0.006) * vleg      # arm servos keep the repo's value
    return w


def mix_power(r, model="p_phys_w", idle_a=None):
    return sum(w * scen_power(r, k, model, idle_a) for k, w in MIX.items())


def batt_power(servo_w):
    return (servo_w + COMPUTE_W) / EFF


def need_mah(r, batt_v=7.4, model="p_phys_w", idle_a=None):
    wh = batt_power(mix_power(r, model, idle_a)) * TARGET_MIN / 60 / USABLE
    return wh / batt_v * 1000


def runtime_min(r, mah, batt_v=7.4, model="p_phys_w", idle_a=None, key=None):
    p = batt_power(mix_power(r, model, idle_a) if key is None else scen_power(r, key, model, idle_a))
    return USABLE * mah / 1000 * batt_v / p * 60


def build_all():
    base = run_designs([{"name": "today", "s": 1.0, "servo": "STS3032", "batt_mah": 850}])[0]
    mah = {s: 1500.0 for s in SCALES}
    res = {}
    for it in range(3):
        cfgs = [{"name": f"s{s:.1f}", "s": s, "servo": "STS3215-m1", "batt_mah": mah[s], "extra_electronics": 0.010}
                for s in SCALES]
        res = {r["cfg"]["s"]: r for r in run_designs(cfgs)}
        new = {s: math.ceil(need_mah(r) / 100) * 100.0 for s, r in res.items()}
        print(f"pass {it}: battery mAh", new, flush=True)
        if all(abs(new[s] - mah[s]) <= 100 for s in SCALES):
            break
        mah = new
    return base, res


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_tables(base, res):
    ds = [("today STS3032", base)] + [(f"s={s:.1f} STS3215", res[s]) for s in SCALES]
    t = {}
    rows = []
    for n, r in ds:
        d = r["cfg"]
        L, W, H = r["span_cm"]
        its = r["mass_items_g"]
        rows.append([n, f"{r['gait']['S']:.2f}", f"{L:.1f} x {W:.1f} x {H:.1f}", f"{r['mass_kg']*1000:.0f}",
                     f"{its['leg servos (12)']:.0f}", f"{its['printed legs']+its['printed arms']+its['printed body']:.0f}",
                     f"{its['electronics']:.0f}", f"{its['battery']:.0f} ({d['batt_mah']:.0f} mAh)",
                     f"{r['geometry']['hip_height']*100:.1f}"])
    t["size"] = table(rows, ["design", "gait scale S", "span L x W x H (cm)", "mass (g)", "12 leg servos (g)",
                              "printed (g)", "electronics (g)", "battery (g)", "hip height (cm)"])
    rows = []
    for n, r in ds:
        sv = r["static"]
        rows.append([n] + [f"{sv[k]['tau']*100:.1f}" for k in ("coxa", "femur", "tibia")]
                    + [f"{sv['femur']['of_stall']*100:.0f}%", f"{sv['femur']['of_model_stall']*100:.0f}%",
                       f"{sv['femur']['of_rated']*100:.0f}%"])
    t["static"] = table(rows, ["design", "coxa N.cm", "femur N.cm", "tibia N.cm", "femur / datasheet stall",
                                "femur / model stall", "femur / rated"])
    rows = []
    for n, r in ds:
        row = [n]
        for key in ("trot_fwd_5", "trot_side_5", "trot_fwd_10", "trot_side_10"):
            s = r["scenarios"][key]
            vmax = max(s["joint"].values(), key=lambda j: j["peak_over_stall"])
            row.append(f"{s['fwd'] if 'fwd_' in key else s['side']:.3f} m/s; peak {vmax['peak_over_stall']*100:.0f}% stall, "
                       f"rms {max(j['rms_over_rated'] for j in s['joint'].values())*100:.0f}% rated; "
                       f"{s['peak_speed']:.1f} rad/s")
        rows.append(row)
    t["trot"] = table(rows, ["design", "forward 5 cm/s", "sideways 5 cm/s", "forward 10 cm/s", "sideways 10 cm/s"])
    rows = []
    for n, r in ds:
        sv = r["servo"]
        lim = 5.29 if sv.startswith("STS3215") else 7.9
        sc = r["scenarios"]
        cap = r["cap_speed"]
        rows.append([n, f"{lim:.1f}"] + [f"{sc[k]['peak_speed']:.1f} ({sc[k]['peak_cmd_speed']:.1f})" for k in
                     ("trot_fwd_5", "trot_side_5", "trot_fwd_10", "trot_side_10")]
                    + [f"{cap['trot_fwd']*100:.1f} / {cap['trot_side']*100:.1f}",
                       f"{sc['trot_fwd_cap']['fwd']*100:.1f} / {sc['trot_side_cap']['side']*100:.1f}",
                       f"{sc['trot_fwd_cap']['peak_speed']:.1f}"])
    t["speed"] = table(rows, ["design", "servo firmware limit rad/s", "fwd 5 (cmd) rad/s", "side 5", "fwd 10", "side 10",
                               "gait speed cap fwd / side (cm/s)", "achieved at cap (cm/s)", "peak rad/s at cap"])
    rows = []
    for n, r in ds:
        mah = r["cfg"]["batt_mah"]
        rows.append([n, f"{scen_power(r,'stand'):.1f}", f"{scen_power(r,'idle'):.1f}", f"{scen_power(r,'trot5'):.1f}",
                     f"{scen_power(r,'trot10'):.1f}", f"{mix_power(r):.1f}", f"{mix_power(r,'p_ub_w'):.1f}",
                     f"{batt_power(mix_power(r)):.1f}", f"{runtime_min(r, mah):.0f}",
                     f"{runtime_min(r, mah, idle_a=IDLE_SENS):.0f}", f"{need_mah(r):.0f}"])
    t["power"] = table(rows, ["design", "stand W", "idle emote W", "trot 5 cm/s W", "trot 10 cm/s W", "mix servo W",
                               "mix servo W (V x I bound)", "mix battery W", "runtime min (model)",
                               "runtime min (150 mA/servo)", "mAh needed for 40 min"])
    rows = []
    for n, r in ds:
        m = r["mass_kg"]
        e = m * G * 0.75
        rows.append([n, f"{m*1000:.0f}", f"{e:.1f}", f"{e/ (base['mass_kg']*G*0.75):.2f}x"])
    t["fall"] = table(rows, ["design", "mass g", "energy from 75 cm (J)", "vs today"])
    rows = []
    for n, r in ds:
        sc = r["scenarios"]
        rows.append([n] + [f"{sc[k]['min_z']*100:.1f}/{sc[k]['tilt']:.0f} deg" for k in ("trot_fwd_5", "trot_side_10", "trot_fwd_cap", "trot_side_cap")])
    t["fall_walk"] = table(rows, ["design", "min body z cm / max tilt: fwd 5", "side 10", "fwd cap", "side cap"])
    return t


def main():
    if "--tables-only" in sys.argv:
        base = json.loads((OUT / "today.json").read_text())
        res = {s: json.loads((OUT / f"s{s:.1f}.json").read_text()) for s in SCALES}
    else:
        base, res = build_all()
    t = fmt_tables(base, res)
    text = "\n\n".join(f"### {k}\n\n{v}" for k, v in t.items())
    (OUT / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
