"""Follow-up runs on the sweep: a slower trot (period x k) and the rotor-inertia diagnostics.

    python sim/sizing/variants.py            # needs the sweep's out/ first; writes out/variants.md

* period x 1.5 and x 2.0 of the gait's own (Froude-scaled) trot period, for every STS3215 scale.
* armature diagnostics: the STS3215 at s = 1.5 with the rotor inertia set to 0, and today's STS3032 with its
  (guessed) armature raised x52 to the STS3215's measured 0.026 kg*m^2 (same reflected-inertia class).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep import OUT, SCALES, mix_power, need_mah, run_designs, scen_power, table  # noqa: E402

KS = (1.5, 2.0)


def main():
    cfgs = [{"name": f"s{s:.1f}_k{k}", "s": s, "servo": "STS3215-m1", "batt_mah": 1100, "extra_electronics": 0.010,
             "period_mult": k} for s in SCALES for k in KS]
    cfgs += [{"name": "s1.5_arm0", "s": 1.5, "servo": "STS3215-m1", "batt_mah": 1100, "extra_electronics": 0.010,
              "armature_scale": 0.0},
             {"name": "s1.5_m6", "s": 1.5, "servo": "STS3215-m6", "batt_mah": 1100, "extra_electronics": 0.010},
             {"name": "today_arm52", "s": 1.0, "servo": "STS3032", "batt_mah": 850, "armature_scale": 52.0}]
    res = {r["cfg"]["name"]: r for r in run_designs(cfgs)}
    for n in ("today", *(f"s{s:.1f}" for s in SCALES)):
        res[n] = json.loads((OUT / f"{n}.json").read_text())

    def row(label, r):
        out = [label]
        for key in ("trot_fwd_5", "trot_side_5", "trot_fwd_10", "trot_side_10"):
            s = r["scenarios"][key]
            pk = max(j["peak_over_stall"] for j in s["joint"].values())
            rms = max(j["rms_over_rated"] for j in s["joint"].values())
            v = s["fwd"] if "fwd" in key else s["side"]
            out.append(f"{v*100:.1f} cm/s: {pk*100:.0f}% / {rms*100:.0f}% / {s['peak_speed']:.1f} / {s['p_phys_w']:.1f} W")
        out.append(f"{r['cap_speed']['trot_fwd']*100:.1f}")
        out.append(f"{mix_power(r):.1f}")
        out.append(f"{need_mah(r):.0f}")
        return out

    head = ["design", "fwd 5", "side 5", "fwd 10", "side 10", "gait cap cm/s", "mix servo W", "mAh for 40 min"]
    rows = [row("today STS3032", res["today"])]
    for s in SCALES:
        for k in (1.0, *KS):
            rows.append(row(f"s={s:.1f} period x{k}", res[f"s{s:.1f}" if k == 1.0 else f"s{s:.1f}_k{k}"]))
    t1 = table(rows, head)
    diag = [row("STS3215 s=1.5 (as swept)", res["s1.5"]), row("STS3215 s=1.5, armature 0", res["s1.5_arm0"]),
            row("STS3215 s=1.5, BAM m6", res["s1.5_m6"]), row("today STS3032 (armature 0.0005)", res["today"]),
            row("today STS3032, armature x52", res["today_arm52"])]
    t2 = table(diag, head)
    text = ("Cells: speed reached: peak motor torque as % of datasheet stall / RMS torque as % of rated / peak joint "
            "speed rad/s / servo electrical power W.\n\n### period variants\n\n" + t1 + "\n\n### armature diagnostics\n\n" + t2)
    (OUT / "variants.md").write_text(text)
    print(text)
    print({k: round(scen_power(res["s1.5"], k), 2) for k in ("stand", "idle", "trot5", "trot10")})


if __name__ == "__main__":
    main()
