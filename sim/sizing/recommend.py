"""The recommended design (s = 1.6, trot period x1.5, 1500 mAh 2S) next to today's robot: the numbers the note quotes.

    python sim/sizing/recommend.py        # needs out/today.json from sweep.py; writes out/recommend.md
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import Design  # noqa: E402
from servos import sts3215  # noqa: E402
from sweep import (  # noqa: E402
    COMPUTE_W,
    EFF,
    IDLE_SENS,
    MIX,
    OUT,
    TARGET_MIN,
    USABLE,
    G,
    batt_power,
    mix_power,
    need_mah,
    run_designs,
    runtime_min,
    scen_power,
)

S, K, MAH = 1.6, 1.5, 1500


def main():
    cfgs = [{"name": "rec", "s": S, "servo": "STS3215-m1", "batt_mah": MAH, "extra_electronics": 0.010, "period_mult": K},
            {"name": "rec_p2", "s": S, "servo": "STS3215-m1", "batt_mah": MAH, "extra_electronics": 0.010,
             "period_mult": K, "print_exp": 2.0},
            {"name": "rec_m6", "s": S, "servo": "STS3215-m6", "batt_mah": MAH, "extra_electronics": 0.010, "period_mult": K}]
    rec, rec2, recm6 = run_designs(cfgs)
    base = json.loads((OUT / "today.json").read_text())
    lines = []

    def p(*a):
        lines.append(" ".join(str(x) for x in a))
        print(lines[-1])

    for tag, r in (("today", base), ("recommended (m1)", rec), ("recommended, printed parts ~s^2 (m1)", rec2),
                   ("recommended (m6)", recm6)):
        its = r["mass_items_g"]
        p(f"## {tag}: mass {r['mass_kg']*1000:.0f} g; items " + ", ".join(f"{k} {v:.0f}" for k, v in its.items()))
        p(f"span {r['span_cm'][0]:.1f} x {r['span_cm'][1]:.1f} x {r['span_cm'][2]:.1f} cm, hip height {r['geometry']['hip_height']*100:.1f} cm,"
          f" claw pitch-axis height {(r['geometry']['hip_height']+0.004)*100:.1f} cm (arm reach 5.4 cm)")
        p(f"fall from 75 cm: {r['mass_kg']*G*0.75:.1f} J, impact speed {(2*G*0.75)**0.5:.2f} m/s")
        st = r["static"]
        p("static femur: {:.1f} N.cm = {:.0f}% datasheet stall, {:.0f}% model stall, {:.0f}% rated".format(
            st["femur"]["tau"] * 100, st["femur"]["of_stall"] * 100, st["femur"]["of_model_stall"] * 100, st["femur"]["of_rated"] * 100))
        for k in ("trot_fwd_5", "trot_side_5", "trot_fwd_10", "trot_side_10", "trot_fwd_cap", "trot_side_cap"):
            s = r["scenarios"][k]
            p(f"  {k}: reached {s['fwd'] if 'fwd' in k else s['side']:.3f} m/s, peak {max(j['peak_over_stall'] for j in s['joint'].values())*100:.0f}% stall "
              f"({max(j['peak_over_model_stall'] for j in s['joint'].values())*100:.0f}% model), rms "
              f"{max(j['rms_over_rated'] for j in s['joint'].values())*100:.0f}% rated, peak {s['peak_speed']:.1f} rad/s, {s['p_phys_w']:.1f} W, tilt {s['tilt']:.0f}")
        mah = r["cfg"]["batt_mah"]
        p(f"power W: stand {scen_power(r,'stand'):.2f} idle {scen_power(r,'idle'):.2f} trot5 {scen_power(r,'trot5'):.2f} "
          f"trot10 {scen_power(r,'trot10'):.2f} mix {mix_power(r):.2f} servo, battery {batt_power(mix_power(r)):.1f} W; "
          f"mAh for 40 min {need_mah(r):.0f}; runtime on {mah:.0f} mAh: {runtime_min(r, mah):.0f} min "
          f"(150 mA/servo idle: {runtime_min(r, mah, idle_a=IDLE_SENS):.0f}); standing only: {runtime_min(r, mah, key='stand'):.0f} min")
        # idle current per leg servo at which the mix just lasts 40 min on this pack
        lo, hi = 0.006, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if runtime_min(r, mah, idle_a=mid) >= TARGET_MIN else (lo, mid)
        p(f"extra idle draw tolerated per leg servo for 40 min: {lo*1000:.0f} mA")
    p(f"mix {MIX}, compute {COMPUTE_W} W, regulator {EFF}, usable {USABLE}")
    d = Design("rec", S, sts3215("m1"), MAH, 7.4, 3.0, 0.010)
    p(f"links coxa {d.coxa*1000:.1f} femur {d.femur*1000:.1f} tibia {d.tibia*1000:.1f} mm; hips +-{d.hip[0]*1000:.1f}, +-{d.hip[1]*1000:.1f} mm; "
      f"chassis half {[round(v*1000,1) for v in d.chassis_half]} mm; pack thickness {d.pack_thickness*1000:.1f} mm; "
      f"body height {d.body_height_mm:.1f} mm; battery {d.battery_mass*1000:.0f} g")
    (OUT / "recommend.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
