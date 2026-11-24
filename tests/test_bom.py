"""The parts list (hardware/bom.py): the v2 style B robot's parts, the owner's decisions of 2026-11-21 and the budget."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pinch_bom", ROOT / "hardware" / "bom.py")
bom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bom)
ROWS = bom.rows()


def _qty(word, order=None):
    return sum(x["qty"] for x in ROWS if word.lower() in x["item"].lower() and (order is None or x["order"] == order))


def test_the_whole_list_fits_the_owners_limit_with_the_orders_adding_up():
    total = bom.tot(ROWS)
    assert total <= bom.LIMIT == 106000, total
    by_order = [bom.tot([x for x in ROWS if x["order"] == n]) for n in bom.ORDER_NAMES.values()]
    assert abs(sum(by_order) - total) < 0.01 and all(0 < o < bom.LIMIT for o in by_order)
    assert bom.tot(ROWS, ("required",)) <= total


def test_the_arms_are_six_bus_servos_and_the_oled_and_pwm_servos_are_gone():
    assert _qty("STS3032") == 6 and _qty("STS3032", bom.ORDER_NAMES[1]) == 1       # one on the bench, no spare
    assert _qty("ST3215") == 14 and _qty("ST3215", bom.ORDER_NAMES[1]) == 2      # 12 legs + 2 spares, two on the bench
    assert not any("GS-4308" in x["item"] or "OLED" in x["item"] for x in ROWS)
    assert _qty("5264 3-pin servo cable") * 6 == 18                                    # three packs of six: one cable per servo
    assert any("GS-4308MG" in r[0] for r in bom.REMOVED) and any("OLED" in r[0] for r in bom.REMOVED)


def test_the_bench_order_has_what_the_plan_needs():
    one = [x for x in ROWS if x["order"] == bom.ORDER_NAMES[1]]
    text = " | ".join(x["item"] for x in one)
    for need in ("ST3215", "STS3032", "driver board", "Li-ion pack", "charger", "fuse", "T-plug", "soldering", "multimeter",
                 "6.25 inch", "buck"):
        assert need in text, f"order 1 lacks {need}"
    assert not any("Pendulum" in x["item"] for x in ROWS)                          # the owner skipped the bench fit (2026-11-22)
    assert any("Pendulum" in r[0] for r in bom.REMOVED)
    assert sum(x["qty"] for x in one if "6.25 inch" in x["item"]) == 1             # weighed before the full build
    assert not any("6.25 inch" in x["item"] for x in ROWS if x["order"] != bom.ORDER_NAMES[1])


def test_the_fixes_of_the_parts_decision_are_in():
    shunt = next(x for x in ROWS if "current-sense" in x["item"])
    assert "0.002 ohm" in shunt["item"] and shunt["checked"] == bom.U              # an allowance, marked as such
    pogo = next(x for x in ROWS if "pogo" in x["item"])
    assert pogo["unit_inr"] == 94.0 and pogo["store"] == bom.RC
    hub = next(x for x in ROWS if "perfboard" in x["item"])
    assert hub["unit_inr"] == 114.0                                                # perfboard, not a custom PCB
    assert next(x for x in ROWS if "VL53L0X" in x["item"])["unit_inr"] == 132.0
    prints = next(x for x in ROWS if x["section"] == "3D prints")
    assert prints["total_inr"] == bom.PRINTS_MID == 7000 and bom.PRINTS_LOW < bom.PRINTS_MID < bom.PRINTS_HIGH


def test_every_row_says_where_its_price_came_from():
    for x in ROWS:
        assert x["checked"] in (bom.N, bom.AG, bom.T, bom.D, bom.U) or x["checked"].startswith(bom.T), x["item"]
        assert x["need"] in ("required", "recommended", "optional") and x["unit_inr"] > 0 and x["qty"] >= 1
    assert bom.tot([x for x in ROWS if x["checked"] == bom.U]) < 0.35 * bom.tot(ROWS)   # estimates stay a minority


def test_the_price_round_of_2026_11_24_lowered_the_total_and_names_each_stores_delivery():
    assert bom.tot(ROWS) < bom.BEFORE_PRICE_ROUND == 105695
    servos = [x for x in ROWS if "ST3215" in x["item"] or "STS3032" in x["item"]]
    assert all(x["checked"] == bom.N for x in servos)                               # the big rows were opened by hand
    stores = {x["store"] for x in ROWS} - {bom.LO, bom.MO}
    assert stores <= set(bom.SHIPPING), stores - set(bom.SHIPPING)
