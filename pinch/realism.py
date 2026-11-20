"""Realism metrics of a recorded session (pinch/recorder.py): numbers that track how alive and how un-mechanical
Pinch looks, so changes can be compared across seeds and versions. The owner watching the video stays the judge
(the Godspeed animacy / likeability scales, Bartneck et al. 2009); these are the measurable proxies.

  action_entropy        Shannon entropy (bits) of the motions it chose, and normalised by log2(#distinct) - low
                        means a few motions dominate
  next_action_entropy   entropy of the next motion given the last (bigram) - low means predictable sequences
  activity_entropy      the same for the activities it switched to
  repetition_distance   for the expressive motions (emotes, poses): median number of other motions between two
                        uses of the same one, and the shortest time between two uses
  repeat_rate           share of expressive motions identical to the previous expressive motion
  gaze_agreement        share of the judged samples (a targeted activity, standing) with the eyes within 0.35 of
                        the target (readable intent: Takayama, Dooley & Ju 2011)
  expression_agreement  share of samples where a face with a valence agrees with the feeling's valence
  sound_fit             share of sounds that fit the feeling and the moment (pinch/observer.py sound_misfits)
  jerk_rms              rms jerk of the joint targets sent (rad/s^3), moving and idle (Flash & Hogan 1985: natural
                        motion minimises jerk)
  ldlj                  median log dimensionless jerk per step kind (Balasubramanian et al. 2012: duration- and
                        amplitude-free smoothness; closer to 0 is smoother)
  idle_motion           rms joint-target speed while idle (rad/s): breathing, not twitching
  abrupt_ticks          ticks where a requested target jumped past what the servo can follow
  reactions_per_min, steps_per_min, decisions_per_min, step_success
  activity_share, state_share  share of awake time per activity and per state; task_share (chores), pet_share
                        (idle life, play, social, curiosity); grasp_success over the picks that ended
"""

import math
from collections import Counter

import numpy as np

from pinch.observer import (
    EXPRESSIVE,
    NEG_FACES,
    NEG_FEELINGS,
    PET_STATES,
    POS_FACES,
    POS_FEELINGS,
    REACTIONS,
    activity_mix,
    gaze_judged,
    sound_misfits,
)


def entropy(labels):
    """(bits, normalised 0..1) of a sequence's symbol distribution."""
    c = Counter(labels)
    n = sum(c.values())
    if n == 0:
        return 0.0, 0.0
    h = -sum(v / n * math.log2(v / n) for v in c.values())
    return h, (h / math.log2(len(c)) if len(c) > 1 else 0.0)


def conditional_entropy(labels):
    """H(next | previous) in bits."""
    if len(labels) < 2:
        return 0.0
    pairs = Counter(zip(labels, labels[1:]))
    prev = Counter(labels[:-1])
    n = len(labels) - 1
    return -sum(v / n * math.log2(v / prev[a]) for (a, _), v in pairs.items())


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


def realism_metrics(session):
    ev, sm = session["events"], session.get("samples") or {}
    t = sm.get("t", np.zeros(0))
    dur = float(t[-1]) if len(t) else max([e["t"] for e in ev] or [1.0])
    mins = max(dur / 60.0, 1e-6)
    steps = [e for e in ev if e["kind"] == "step_start" and e.get("source") == "mind"]
    labels = [e["step"] for e in steps]
    acts = [e["to"] for e in ev if e["kind"] == "decision"]
    h, hn = entropy(labels)
    ah, ahn = entropy(acts)
    m = dict(duration_s=_r(dur, 1), decisions_per_min=_r(len(acts) / mins, 2), steps_per_min=_r(len(steps) / mins, 2),
             action_entropy=_r(h), action_entropy_norm=_r(hn), next_action_entropy=_r(conditional_entropy(labels)),
             distinct_actions=len(set(labels)), activity_entropy=_r(ah), activity_entropy_norm=_r(ahn))
    # repetition of the expressive motions
    expr = [(e["t"], e["step"]) for e in steps if e["motion"] in EXPRESSIVE]
    gaps, gap_s, last = [], [], {}
    for i, (tt, lbl) in enumerate(expr):
        if lbl in last:
            gaps.append(i - last[lbl][0] - 1)
            gap_s.append(tt - last[lbl][1])
        last[lbl] = (i, tt)
    m.update(expressive_motions=len(expr), repetition_distance=_r(float(np.median(gaps))) if gaps else None,
             min_repeat_s=_r(min(gap_s), 1) if gap_s else None,
             repeat_rate=_r(sum(a[1] == b[1] for a, b in zip(expr, expr[1:])) / (len(expr) - 1)) if len(expr) > 1 else None)
    # eyes, face, voice
    if "gaze_want" in sm:
        judged, err = gaze_judged(sm)
        m.update(gaze_judged_s=_r(judged.sum() * float(session["header"].get("sample_dt", 0.06)), 1),
                 gaze_agreement=_r(float((err[judged] < 0.35).mean())) if judged.any() else None)
    if "face" in sm and "feeling" in sm:
        fs = np.array([1 if str(f) in POS_FACES else -1 if str(f) in NEG_FACES else 0 for f in sm["face"]])
        gs = np.array([1 if str(f) in POS_FEELINGS else -1 if str(f) in NEG_FEELINGS else 0 for f in sm["feeling"]])
        both = (fs != 0) & (gs != 0)
        m["expression_agreement"] = _r(float((fs[both] == gs[both]).mean())) if both.any() else None
    sounds = [e for e in ev if e["kind"] == "sound"]
    m.update(sounds=len(sounds), sound_fit=_r(1.0 - len(sound_misfits(ev)) / len(sounds)) if sounds else None)
    # smoothness
    if "jerk_rms" in sm:
        busy = sm["busy"].astype(bool)
        j2 = sm["jerk_rms"].astype(float) ** 2
        m.update(jerk_rms_moving=_r(math.sqrt(j2[busy].mean()), 1) if busy.any() else None,
                 jerk_rms_idle=_r(math.sqrt(j2[~busy].mean()), 1) if (~busy).any() else None,
                 abrupt_ticks=int(sm["clamped"].sum()) if "clamped" in sm else None,
                 max_target_jump=_r(float(sm["max_jump"].max())) if "max_jump" in sm else None)
        if "q_cmd" in sm and len(t) > 2:
            q = sm["q_cmd"].astype(float)
            v = np.sqrt(np.mean((np.diff(q, axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]) ** 2, axis=1))
            idle = ~busy[1:]
            m["idle_motion"] = _r(float(np.sqrt(np.mean(v[idle] ** 2)))) if idle.any() else None
    ldlj = {}
    for e in ev:
        if e["kind"] == "step_end" and e.get("ldlj") is not None:
            ldlj.setdefault(e["motion"], []).append(e["ldlj"])
    m["ldlj"] = {k: _r(float(np.median(v)), 2) for k, v in sorted(ldlj.items())}
    ends = [e for e in ev if e["kind"] == "step_end" and e.get("source") == "mind"]
    m.update(step_success=_r(sum(e["ok"] for e in ends) / len(ends)) if ends else None,
             reactions_per_min=_r(sum(a in REACTIONS for a in acts) / mins, 2),
             upside_down_s=_r(float((sm["up"] < 0.3).sum()) * float(session["header"].get("sample_dt", 0.06)), 1)
             if "up" in sm else None)
    acts_share, states_share, awake_s = activity_mix(sm)
    picks = [e for e in ev if e["kind"] == "step_end" and e.get("motion") == "pick" and e.get("end") in ("done", "failed")]
    m.update(awake_s=_r(awake_s, 1), task_share=states_share.get("task", 0.0) if states_share else None,
             pet_share=_r(sum(states_share.get(s, 0.0) for s in PET_STATES)) if states_share else None,
             state_share=states_share, activity_share=acts_share,
             grasp_success=_r(sum(e["ok"] for e in picks) / len(picks)) if picks else None, grasp_attempts=len(picks))
    return m


def report_text(session, findings, metrics):
    hd = session.get("header", {})
    cfg = hd.get("config", {})
    lines = [f"Session {session.get('path', '')}",
             f"  scenario {cfg.get('scenario', '?')}, seed {cfg.get('seed', '?')}, {metrics['duration_s']} s, "
             f"commit {hd.get('git', {}).get('commit', '')[:10]}{' (dirty)' if hd.get('git', {}).get('dirty') else ''}",
             "", "Realism metrics:"]
    for k, v in metrics.items():
        lines.append(f"  {k:24s} {v}")
    lines += ["", f"Observer findings ({len(findings)}):"]
    if not findings:
        lines.append("  none")
    for f in findings:
        lines.append(f"  [{f.severity:5s}] {f.kind:24s} {f.t0:7.1f}-{f.t1:7.1f} s  {f.evidence}")
    return "\n".join(lines) + "\n"
