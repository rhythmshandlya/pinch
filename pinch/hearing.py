"""Hearing: turns microphone audio into what a pet would notice. All local; audio is never stored.

Per 32 ms frame:  loudness, voice activity (energy + spectral shape + zero crossings), pitch (autocorrelation),
                  claps / bangs (sharp broadband transients).
Per second:       a VoiceSummary: how much speech, how loud, pitch level and wobble, talking speed.
Over minutes:     OwnerVoiceMood - is my person calm, excited, stressed or low (prosody only; words
                  from speech-to-text refine it in pinch/language.py)
                  MeetingDetector - long stretches of talking with call-like turn taking, call phrases
                  ("you're on mute"), or being told, mean "my person is in a meeting".

Two microphones (left/right) also give a direction: AudioDirection, from the time difference.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

RATE = 16000
FRAME = 512                       # 32 ms


@dataclass
class FrameFeatures:
    db: float
    voiced: bool
    pitch: float                  # Hz, 0 if none
    transient: bool               # clap / knock / bang


class FrameAnalyzer:
    def __init__(self, rate=RATE):
        self.rate = rate
        self.noise_db = -60.0     # adaptive noise floor
        self.prev_db = -90.0

    def __call__(self, x: np.ndarray) -> FrameFeatures:
        x = np.asarray(x, float)
        x = x - x.mean()
        rms = np.sqrt(np.mean(x ** 2)) + 1e-9
        db = 20 * np.log10(rms)
        # noise floor follows quiet moments quickly, loud ones slowly
        self.noise_db += (db - self.noise_db) * (0.05 if db < self.noise_db else 0.002)
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-9
        freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
        band = (freqs > 120) & (freqs < 3800)
        speech_ratio = spec[band].sum() / spec.sum()
        flatness = np.exp(np.mean(np.log(spec[band]))) / np.mean(spec[band])
        zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2
        loud_enough = db > self.noise_db + 9
        pitch = self._pitch(x) if loud_enough else 0.0
        voiced = bool(loud_enough and speech_ratio > 0.6 and flatness < 0.45 and zcr < 0.25 and pitch > 0)
        transient = bool(db - self.prev_db > 18 and flatness > 0.3 and db > self.noise_db + 20)
        self.prev_db = db
        return FrameFeatures(db, voiced, pitch, transient)

    def _pitch(self, x):
        lo, hi = int(self.rate / 400), int(self.rate / 70)
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        if ac[0] <= 0:
            return 0.0
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        return float(self.rate / (lo + k)) if seg[k] / ac[0] > 0.35 else 0.0


@dataclass
class VoiceSummary:
    t: float
    speech: float                 # fraction of the second with voice
    db: float                     # mean loudness of voiced frames (dBFS)
    pitch: float                  # mean pitch of voiced frames (Hz)
    pitch_var: float              # semitone spread (lively vs flat)
    rate: float                   # voiced on-sets per second (rough talking speed)
    transients: int


class Hearing:
    """Feed raw audio chunks; get a VoiceSummary every second and events (claps, bangs)."""

    def __init__(self, rate=RATE):
        self.fa = FrameAnalyzer(rate)
        self.buf = np.zeros(0)
        self.frames = []
        self.t = 0.0
        self.rate = rate
        self.events = []
        self.claps = deque(maxlen=4)

    def feed(self, audio: np.ndarray):
        out = []
        self.buf = np.concatenate([self.buf, np.asarray(audio, float).ravel()])
        while len(self.buf) >= FRAME:
            f = self.fa(self.buf[:FRAME])
            self.buf = self.buf[FRAME:]
            self.t += FRAME / self.rate
            self.frames.append(f)
            if f.transient:
                self.claps.append(self.t)
                if f.db > -12:
                    self.events.append(("loud_noise", self.t))
                elif len(self.claps) >= 2 and self.claps[-1] - self.claps[-2] < 0.6:
                    self.events.append(("double_clap", self.t))
                    self.claps.clear()
            if len(self.frames) * FRAME >= self.rate:
                out.append(self._summarize())
                self.frames = []
        return out

    def _summarize(self):
        fr = self.frames
        v = [f for f in fr if f.voiced]
        speech = len(v) / max(1, len(fr))
        pitches = np.array([f.pitch for f in v if f.pitch > 0])
        semis = 12 * np.log2(pitches / max(np.median(pitches), 1)) if len(pitches) > 2 else np.zeros(1)
        onsets = sum(1 for a, b in zip(fr, fr[1:]) if b.voiced and not a.voiced)
        return VoiceSummary(self.t, speech, float(np.mean([f.db for f in v])) if v else -90.0,
                            float(pitches.mean()) if len(pitches) else 0.0, float(np.std(semis)),
                            float(onsets), sum(f.transient for f in fr))

    def take_events(self):
        e, self.events = self.events, []
        return e


class OwnerVoiceMood:
    """Prosody -> arousal and a rough mood label. Words (when there are any) can override it."""

    def __init__(self):
        self.base_db, self.base_pitch = None, None
        self.arousal, self.label = 0.3, "calm"

    n = 0                                          # voiced seconds heard so far

    def update(self, s: VoiceSummary):
        if s.speech < 0.25:
            return self.label
        self.n += 1
        self.base_db = s.db if self.base_db is None else 0.98 * self.base_db + 0.02 * s.db
        self.base_pitch = s.pitch if self.base_pitch is None else 0.98 * self.base_pitch + 0.02 * s.pitch
        loud = (s.db - self.base_db) / 6.0
        high = 12 * np.log2(max(s.pitch, 1) / max(self.base_pitch, 1)) / 3.0
        lively = (s.pitch_var - 2.0) / 2.0
        fast = (s.rate - 3.0) / 2.0
        a = 0.5 + 0.18 * (loud + high + lively + fast)
        self.arousal = float(np.clip(0.8 * self.arousal + 0.2 * a, 0, 1))
        if self.n < 30:                            # learn this voice's normal first
            self.label = "calm"
        elif self.arousal > 0.72:
            self.label = "stressed" if lively < 0 or loud > 0.8 else "excited"
        elif self.arousal < 0.3 and s.pitch_var < 1.5:
            self.label = "sad"                     # quiet, flat and slow: low
        else:
            self.label = "calm"
        return self.label


class MeetingDetector:
    """Is my person on a call?  Evidence adds up over minutes and fades when it stops.

    * talking a lot (over a third of the time) for minutes, in turns with pauses, not at Pinch
    * call phrases heard by speech-to-text ("you're on mute", "next slide", ...)
    * being told ("I'm in a meeting") - immediate
    * the hour: if meetings usually happen now (memory), it suspects one sooner
    It ends after about a minute of quiet, or when told / "bye everyone".
    """

    def __init__(self):
        self.p = 0.0
        self.in_meeting = False
        self.speech = deque(maxlen=120)          # last two minutes of per-second speech fractions
        self.quiet_s = 0.0

    def told(self, start: bool):
        self.p = 1.0 if start else 0.0
        self.in_meeting = start
        self.quiet_s = 0.0

    def update(self, s: VoiceSummary | None, call_cue=False, addressed_to_pinch=False, usual_time=False, dt=1.0):
        if s is not None:
            self.speech.append(s.speech)
            self.quiet_s = self.quiet_s + dt if s.speech < 0.1 else 0.0
        talk = np.mean(list(self.speech)[-60:]) if len(self.speech) >= 30 else 0.0
        turns = sum(1 for a, b in zip(self.speech, list(self.speech)[1:]) if (a > 0.3) != (b > 0.3))
        evidence = 0.0
        if talk > 0.33 and len(self.speech) >= 45:
            evidence += 0.02 * (1.5 if turns > 12 else 1.0) * (1.4 if usual_time else 1.0)
        if call_cue:
            evidence += 0.35
        if addressed_to_pinch:
            evidence -= 0.05                     # talking to me isn't a meeting
        self.p = float(np.clip(self.p + evidence - (0.004 if talk < 0.2 else 0.0)
                               - (0.03 if self.quiet_s > 60 else 0.0), 0, 1))
        if not self.in_meeting and self.p > 0.6:
            self.in_meeting = True
        elif self.in_meeting and (self.p < 0.25 or self.quiet_s > 90):
            self.in_meeting = False
            self.p = min(self.p, 0.2)
        return self.in_meeting


class AudioDirection:
    """Two mics ~70 mm apart (left/right in the dome): angle of a voice from the arrival-time difference."""

    def __init__(self, spacing=0.07, rate=RATE):
        self.spacing, self.rate = spacing, rate

    def angle(self, left: np.ndarray, right: np.ndarray):
        """Radians, + = to Pinch's left. GCC-PHAT on one frame."""
        n = len(left) + len(right)
        L, R = np.fft.rfft(left, n), np.fft.rfft(right, n)
        cross = L * np.conj(R)
        cc = np.fft.irfft(cross / (np.abs(cross) + 1e-9), n)
        max_lag = int(self.spacing / 343.0 * self.rate) + 1
        cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
        lag = (np.argmax(cc) - max_lag) / self.rate
        return float(np.arcsin(np.clip(-lag * 343.0 / self.spacing, -1, 1)))    # left mic hears it first -> +


# ---------------------------------------------------------------------- synthetic voices for tests
def synth_speech(seconds, rate=RATE, pitch=140.0, loud=0.1, talk=0.6, lively=1.0, seed=0, syllables=4.0):
    """Speech-like audio: voiced harmonic 'syllables' with pitch movement, separated by pauses.
    talk = fraction of time with voice. Used by the tests and the virtual pet's 'meeting' button."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    out = rng.normal(0, 0.002, n)
    t = 0
    while t < n:
        if rng.random() < talk:
            dur = int(rate / syllables * rng.uniform(0.6, 1.4))
            tt = np.arange(dur) / rate
            f0 = pitch * 2 ** (lively * rng.normal(0, 2) / 12) * (1 + 0.05 * np.sin(2 * np.pi * 3 * tt))
            ph = 2 * np.pi * np.cumsum(f0) / rate
            env = np.sin(np.pi * np.arange(dur) / dur) ** 0.7
            sig = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 8))
            m = min(dur, n - t)
            out[t:t + m] += (loud * env * sig)[:m]
            t += dur
        else:
            t += int(rate * rng.uniform(0.15, 0.6))
    return out
