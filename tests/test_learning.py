"""It grows with its person (pinch/mind/learning.py): praise, petting and scolding slowly shape what it chooses,
credited to what it is doing or just did, bounded, fading without feedback, per person."""

from pinch.memory import Memory
from pinch.mind import Mind
from pinch.mind.learning import CFG, FeedbackLearner

DAY = 86400.0


def test_credit_goes_to_what_it_is_doing_and_just_did():
    lr = FeedbackLearner()
    lr.ended(100.0, "show_off")
    c = lr.credit(102.0, "tidy")
    assert c["tidy"] == 1.0 and 0.4 < c["show_off"] < 0.6          # 2 s ago: about half
    assert "show_off" not in lr.credit(100.0 + CFG["window_s"] + 1, "tidy")
    assert lr.credit(102.0, "rest") == {"show_off": c["show_off"]}  # resting itself is never learned


def test_praise_raises_scolding_lowers_within_bounds():
    lr = FeedbackLearner()
    clock = 0.0
    for _ in range(200):
        clock += 3600.0
        lr.feedback("praised", clock, clock, "tidy", "morning:here")
        lr.feedback("scolded", clock + 30, clock + 30, "show_off", "morning:here")
    up, down = lr.bias("tidy", "morning:here"), lr.bias("show_off", "morning:here")
    assert 0.05 < up <= CFG["max_bias"] and -CFG["max_bias"] <= down < -0.05
    assert lr.shape("show_off", 0.3, "morning:here") >= CFG["floor"] * 0.3     # never scolded out of existence
    assert lr.shape("show_off", 0.0, "morning:here") == 0.0                     # no want from nothing
    assert lr.bias("tidy", "evening:away") > 0                                  # some of it carries over


def test_an_owner_who_praises_everything_teaches_little():
    lr = FeedbackLearner()
    for k in range(60):
        lr.feedback("praised", k * 3600.0, k * 3600.0, ("tidy", "play", "explore")[k % 3], "afternoon:here")
    late = [abs(d) for _, _, _, out in lr.log[-6:] for d in out.values()]
    first = [abs(d) for _, _, _, out in lr.log[:3] for d in out.values()]
    assert max(late) < 0.5 * max(first)                 # the baseline caught up with the always-praise


def test_a_flurry_counts_once_and_it_fades_without_feedback():
    lr = FeedbackLearner()
    outs = [lr.feedback("stroked", 1.0 + 0.5 * k, 1.0 + 0.5 * k, "play", "evening:here") for k in range(6)]
    assert sum(1 for o in outs if o) == CFG["burst_max"]
    b0 = lr.bias("play", "evening:here")
    lr._fade(1.0 + CFG["half_life_days"] * DAY)
    assert abs(lr.bias("play", "evening:here") - 0.5 * b0) < 0.02 * abs(b0) + 1e-6


def test_two_people_teach_it_different_things_and_it_remembers():
    lr = FeedbackLearner()
    for k in range(20):
        t = k * 3600.0
        lr.feedback("praised", t, t, "play", "afternoon:here", person="ana")
        lr.feedback("scolded", t + 60, t + 60, "play", "afternoon:here", person="ben")
    assert lr.bias("play", "afternoon:here", "ana") > 0 > lr.bias("play", "afternoon:here", "ben")
    again = FeedbackLearner(lr.to_dict())
    assert again.bias("play", "afternoon:here", "ana") == lr.bias("play", "afternoon:here", "ana")


def test_the_mind_learns_from_good_crab_while_it_tidies():
    m = Mind(memory=Memory(path=None), llm=False, clock=lambda: 1.8e9)
    m._start("tidy", "test")
    m.hear_text("good crab")
    assert m.learner.log and "tidy" in m.learner.log[-1][3] and m.learner.log[-1][3]["tidy"] > 0
    assert "learned" in m.memory.data
    m._start("sulk", "test")
    m.t += 2.0
    m.hear_text("bad crab")                          # sulking itself isn't learned; the tidy 2 s ago is
    assert "sulk" not in m.learner.log[-1][3] and m.learner.log[-1][3].get("tidy", 0) < 0


def test_it_is_purely_itself_a_fixed_share_of_the_time():
    from pinch.mind.learning import own_block
    for act in ("play", "explore", "tidy"):
        share = sum(own_block(act, k * CFG["own_block_s"]) for k in range(4000)) / 4000
        assert abs(share - CFG["own_share"]) < 0.03, (act, share)
    assert own_block("play", 1234.0) == own_block("play", 1234.0)          # replays exactly
