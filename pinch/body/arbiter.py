"""Body parts and their owners (phase 4; configs/mind/channels.toml). A pure function: claims in, one owner per body
part out, plus the claims it refused and why - so the rules are cheap to test exhaustively.

Rules (after Anki Vector's locked animation tracks and the Cozmo SDK, where an action wanting a track another
action holds fails with TRACKS_LOCKED; docs/research/v2_1/mind_architecture_notes.md):
  * every body part has at most one owner;
  * the higher layer wins (reflex > command > reaction > activity > ambient); at an equal layer the claim already
    holding the part keeps it and the newcomer is refused;
  * a claim is all or nothing: refused on any part, it gets none (a walk can't run with half its legs);
  * ambient life never refuses or is refused: it fills the parts no one owns, and the parts an owner lets it have.
"""

import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "configs" / "mind" / "channels.toml"


@cache
def config():
    with open(PATH, "rb") as f:
        c = tomllib.load(f)
    for kind, s in c["steps"].items():
        bad = (set(s["claims"]) | set(s["ambient"])) - set(c["channels"])
        if bad:
            raise ValueError(f"{PATH.name} [{kind}]: unknown or doubly listed parts {sorted(bad) or s}")
    return c


def layer_rank(layer):
    return config()["layers"].index(layer)          # 0 = highest


@dataclass(frozen=True)
class Claim:
    owner: str                                      # who: "walk (explore)", "cliff reflex", ...
    layer: str
    parts: frozenset
    ambient_ok: frozenset = field(default_factory=frozenset)   # parts it lets ambient life move on top
    since: float = 0.0                              # when it was granted (incumbent first at equal layers)


def claim_for(step_kind, owner, layer, since=0.0):
    s = config()["steps"].get(step_kind, {"claims": ["legs", "claw_L", "claw_R"], "ambient": []})
    return Claim(owner, layer, frozenset(s["claims"]), frozenset(s["ambient"]), since)


def arbitrate(claims):
    """claims: iterable of Claim. Returns (owners: {part: Claim or None}, refused: [(Claim, part, holder Claim)])."""
    owners = {p: None for p in config()["channels"]}
    refused = []
    for c in sorted(claims, key=lambda c: (layer_rank(c.layer), c.since)):
        clash = [(p, owners[p]) for p in c.parts if owners[p] is not None]
        if clash:
            refused += [(c, p, h) for p, h in clash]
            continue
        for p in c.parts:
            owners[p] = c
    return owners, refused


def ambient_parts(owners):
    """The parts ambient life may move now: unowned, or owned by a claim that lets it."""
    return {p for p, c in owners.items() if c is None or p in c.ambient_ok}
