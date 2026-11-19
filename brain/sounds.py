"""The crab's voice: short procedural pet tones (it never speaks words).

    wav_bytes = render("happy_trill")        # 16-bit mono WAV
    play("curious_chirp")                    # on the robot (aplay) or saves to docs/sounds/

Each sound is a few pitch-swept beeps with an envelope - like Cozmo or R2-D2.
"""

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 22050
SOUNDS = {
    # name: list of (start_hz, end_hz, seconds, gap_after_s)
    "greeting_beep":   [(880, 1320, 0.08, 0.04), (1320, 1760, 0.10, 0.0)],
    "happy_trill":     [(1200, 1500, 0.05, 0.02)] * 4 + [(1500, 2000, 0.12, 0.0)],
    "curious_chirp":   [(700, 1400, 0.18, 0.06), (900, 1700, 0.12, 0.0)],
    "startled_squeak": [(2400, 3200, 0.06, 0.0)],
    "sad_tone":        [(900, 500, 0.45, 0.0)],
    "sleepy_hum":      [(300, 260, 0.6, 0.1), (280, 220, 0.7, 0.0)],
    "angry_buzz":      [(180, 160, 0.12, 0.03)] * 3,
    # Pinch's wider vocabulary
    "purr":            [(220, 240, 0.09, 0.0), (240, 220, 0.09, 0.0)] * 4,
    "giggle":          [(1400, 1900, 0.05, 0.03), (1500, 2000, 0.05, 0.03), (1600, 2200, 0.06, 0.0)],
    "whimper":         [(1100, 800, 0.25, 0.08), (950, 650, 0.35, 0.0)],
    "attention_chirp": [(1000, 1600, 0.07, 0.05)] * 2 + [(1200, 2000, 0.14, 0.0)],
    "fanfare":         [(784, 784, 0.10, 0.02), (988, 988, 0.10, 0.02), (1175, 1175, 0.10, 0.02), (1568, 1568, 0.25, 0.0)],
    "yawn":            [(500, 900, 0.3, 0.0), (900, 350, 0.6, 0.0)],
    "sneeze":          [(600, 900, 0.15, 0.12), (2500, 1200, 0.07, 0.0)],
    "uh_oh":           [(900, 900, 0.12, 0.06), (700, 650, 0.2, 0.0)],
    "boop":            [(1800, 1300, 0.06, 0.0)],
}
NAMES = sorted(SOUNDS)


def synth(name: str) -> np.ndarray:
    parts = []
    for f0, f1, dur, gap in SOUNDS[name]:
        n = int(RATE * dur)
        f = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(f) / RATE
        tone = np.sin(phase)
        if name == "angry_buzz":                      # rough, buzzy timbre
            tone = np.sign(tone) * 0.6 + 0.4 * np.sin(3 * phase)
        env = np.minimum(1, np.minimum(np.arange(n) / (0.01 * RATE), (n - np.arange(n)) / (0.02 * RATE)))
        parts += [0.5 * tone * env, np.zeros(int(RATE * gap))]
    return np.concatenate(parts)


def render(name: str) -> bytes:
    pcm = (np.clip(synth(name), -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def play(name: str, out_dir=None):
    """On the Pi: play through the I2S amp with aplay. Elsewhere: save docs/sounds/<name>.wav."""
    if name not in SOUNDS:
        return None
    data = render(name)
    if shutil.which("aplay") and out_dir is None:
        subprocess.run(["aplay", "-q", "-"], input=data, check=False)
        return None
    out = Path(out_dir or Path(__file__).resolve().parents[1] / "docs" / "sounds")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.wav"
    path.write_bytes(data)
    return path


if __name__ == "__main__":
    for n in NAMES:
        print("wrote", play(n, out_dir=Path(__file__).resolve().parents[1] / "docs" / "sounds"))
