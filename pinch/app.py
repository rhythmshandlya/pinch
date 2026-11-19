"""Pinch, the virtual pet, on your laptop.

    .venv\\Scripts\\python -m pinch.app            # 3D desk + control panel
    .venv\\Scripts\\python -m pinch.app --llm      # also let the local LLM (Ollama) read what you say
    .venv\\Scripts\\python -m pinch.app --headless --seconds 60 --say "hi pinch" --say "dance"   # no windows

Two windows:
  * the digital twin: Pinch on its desk in 3D, with its live face. Drag with the mouse to look around;
    double-click Pinch, then Ctrl + right-drag to pick it up or push it (it notices!).
  * the panel: Pinch's face, how it feels and what it needs, its diary, a box to talk to it, touch
    buttons (pet, stroke, tickle, boop, hold its claw), desk actions (mess up the desk, pen near the edge,
    a meeting), and "Listen" to let it hear you through your microphone (on-device only, never saved).

Pinch remembers you between runs (~/.pinch/memory.json).
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.sounds import render as render_sound  # noqa: E402
from pinch.hearing import RATE, synth_speech  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.sim_runner import DT, PinchSim  # noqa: E402

NEEDS = ["social", "play", "curiosity", "sleep", "energy", "tidy", "safety"]
NEED_LABEL = {"social": "wants company", "play": "wants to play", "curiosity": "curious", "sleep": "sleepy",
              "energy": "hungry (battery)", "tidy": "desk is messy", "safety": "scared"}
FEELING_EMOJI = {"happy": "😊", "excited": "🤩", "love": "🥰", "content": "😌", "curious": "🤔", "surprised": "😮",
                 "sad": "😢", "lonely": "🥺", "angry": "😠", "grumpy": "😒", "scared": "😨", "shy": "😳",
                 "sleepy": "😪", "bored": "😑", "quiet": "🤫", "dizzy": "😵", "proud": "😎", "squeeze": "😆",
                 "neutral": "🙂", "joy": "😄", "focused": "🧐", "asleep": "😴", "dead": "💀", "music": "🎶"}


class Pinch:
    """The simulation thread + everything the UI can ask of it (thread-safe through a queue)."""

    def __init__(self, llm=False, memory_path=None, viewer=True):
        self.mem = Memory(path=memory_path) if memory_path is not False else Memory(path=None)
        self.sim = PinchSim(memory=self.mem, llm=llm, policies=True)
        self.actions = queue.Queue()
        self.sound_q = queue.Queue()
        self.muted = False
        self.running = True
        self.viewer = None
        self.last_user = time.time()
        self.meeting_audio = None              # generator of 1 s chunks while a pretend call is on
        self.mic_q = queue.Queue()
        self.mic_stream = None
        self.mic_level = 0.0
        self._played = 0
        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.sim.m, self.sim.d, show_left_ui=False, show_right_ui=False)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0.0, -0.02, 0.02]
                self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 0.75, -90, -32
        self.tex = self.sim.m.texture("face_tex").id
        self.sim.user_active = True
        threading.Thread(target=self._audio_out, daemon=True).start()

    # -- things you do (called from the UI thread) ----------------------------------------
    def do(self, fn):
        self.last_user = time.time()
        self.actions.put(fn)

    def say(self, text):
        self.do(lambda: self.sim.say(text))

    def touch(self, gesture):
        self.do(lambda: self.sim.touch(gesture))

    def mess_up(self):
        def f():
            rng = self.sim.rng
            x, y, _ = self.sim.pose()
            for name in ("pen", "eraser", "cap", "ball"):
                for _ in range(20):
                    p = (rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15))
                    if np.hypot(p[0] - x, p[1] - y) > 0.12:
                        break
                self.sim.place(name, p, rng.uniform(-np.pi, np.pi))
        self.do(f)

    def pen_to_edge(self):
        self.do(lambda: self.sim.place("pen", (0.05, -0.205), 0.0))

    def meeting(self, on):
        def f():
            if on:
                def talk():                                  # 1 s chunks of a pretend call, forever
                    k = 0
                    while True:
                        k += 1
                        yield synth_speech(1.0, talk=0.55, seed=int(time.time()) + k)
                self.meeting_audio = talk()
                self.sim.say("can you hear me? let's get started")       # what speech-to-text would hear
            else:
                self.meeting_audio = None
                self.sim.say("ok bye everyone, the meeting is over")
        self.do(f)

    def presence(self, here):
        def f():
            self.sim.user_active = here
            o = self.sim.mind.world.owner
            o.present = here
            if not here:
                o.last_seen = o.last_voice = o.last_touch = -1e9
        self.do(f)

    def listen(self, on):
        """Your microphone -> Pinch's hearing (loudness, voice activity, mood, meetings, claps)."""
        if on and self.mic_stream is None:
            import sounddevice as sd

            def cb(indata, frames, t, status):
                self.mic_q.put(indata[:, 0].copy())
            self.mic_stream = sd.InputStream(samplerate=RATE, channels=1, blocksize=RATE // 10, callback=cb)
            self.mic_stream.start()
        elif not on and self.mic_stream is not None:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None

    # -- the loop ----------------------------------------------------------------------------
    def _audio_out(self):
        import winsound
        while self.running:
            name = self.sound_q.get()
            if not self.muted:
                try:
                    winsound.PlaySound(render_sound(name), winsound.SND_MEMORY)
                except Exception:
                    pass

    def step_block(self, n=5):
        sim = self.sim
        while not self.actions.empty():
            self.actions.get()()
        if time.time() - self.last_user > 300 and sim.user_active:     # nobody around for 5 minutes
            sim.user_active = False
        if self.meeting_audio is not None and sim.k % int(1.0 / DT) == 0:
            sim.hear_audio(next(self.meeting_audio))
        while not self.mic_q.empty():
            chunk = self.mic_q.get()
            self.mic_level = float(np.sqrt(np.mean(chunk ** 2)))
            sim.hear_audio(chunk)
        for _ in range(n):
            if self.viewer is not None:
                with self.viewer.lock():
                    sim.step()
            else:
                sim.step()
        for t, s in sim.sounds_played[self._played:]:
            self.sound_q.put(s)
        self._played = len(sim.sounds_played)

    def run(self, seconds=None):
        t0 = time.perf_counter()
        k = 0
        frame = 0
        while self.running and (seconds is None or self.sim.t < seconds):
            self.step_block(5)
            k += 5
            if self.viewer is not None:
                if not self.viewer.is_running():
                    self.running = False
                    break
                frame += 1
                self.viewer.update_texture(self.tex)
                self.viewer.set_texts((None, None, f"Pinch feels {self.sim.mind.affect.feeling()}",
                                       f"doing: {self.sim.mind.activity.name if self.sim.mind.activity else '-'}"))
                self.viewer.sync()
                lag = t0 + k * DT - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        self.running = False
        self.mem.save()


# --------------------------------------------------------------------------------------------- UI
def build_ui(p: Pinch):
    import tkinter as tk
    from tkinter import ttk

    from PIL import ImageTk

    root = tk.Tk()
    root.title("Pinch")
    root.configure(bg="#14161b")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", padding=4)
    bg, fg = "#14161b", "#e8eaf0"

    face_lbl = tk.Label(root, bg="black")
    face_lbl.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 4))
    feel = tk.Label(root, font=("Segoe UI", 16, "bold"), bg=bg, fg=fg)
    feel.grid(row=1, column=0, columnspan=3)
    doing = tk.Label(root, font=("Segoe UI", 10), bg=bg, fg="#9aa3b5")
    doing.grid(row=2, column=0, columnspan=3)

    bars = {}
    fr = tk.Frame(root, bg=bg)
    fr.grid(row=3, column=0, columnspan=3, padx=10, pady=6, sticky="ew")
    for i, n in enumerate(NEEDS):
        tk.Label(fr, text=NEED_LABEL[n], bg=bg, fg="#c8ccd6", width=16, anchor="w").grid(row=i, column=0)
        pb = ttk.Progressbar(fr, length=220, maximum=1.0)
        pb.grid(row=i, column=1, pady=1)
        bars[n] = pb
    owner = tk.Label(root, font=("Segoe UI", 9), bg=bg, fg="#9aa3b5", justify="left")
    owner.grid(row=4, column=0, columnspan=3, sticky="w", padx=10)

    entry = ttk.Entry(root, width=48)
    entry.grid(row=5, column=0, columnspan=2, padx=10, pady=6, sticky="ew")
    entry.insert(0, "")

    def send(_=None):
        text = entry.get().strip()
        if text:
            p.say(text)
            entry.delete(0, "end")
    entry.bind("<Return>", send)
    ttk.Button(root, text="Say", command=send).grid(row=5, column=2, padx=(0, 10))

    def buttons(row, items):
        f = tk.Frame(root, bg=bg)
        f.grid(row=row, column=0, columnspan=3, padx=10, pady=2, sticky="w")
        for i, (label, fn) in enumerate(items):
            ttk.Button(f, text=label, command=fn).grid(row=0, column=i, padx=2)

    buttons(6, [("Pet head", lambda: p.touch("hold")), ("Stroke back", lambda: p.touch("stroke")),
                ("Tickle", lambda: p.touch("tickle")), ("Boop", lambda: p.touch("boop")),
                ("Hold claw", lambda: p.touch("claw_hold")), ("Ruffle", lambda: p.touch("ruffle"))])
    buttons(7, [("Mess up desk", p.mess_up), ("Pen to edge", p.pen_to_edge),
                ("Start meeting", lambda: p.meeting(True)), ("End meeting", lambda: p.meeting(False))])
    listen_var = tk.BooleanVar(value=False)
    mute_var = tk.BooleanVar(value=False)

    def toggle_listen():
        try:
            p.listen(listen_var.get())
        except Exception as e:                          # no microphone: say so, keep going
            listen_var.set(False)
            doing.config(text=f"microphone not available: {e}")

    f = tk.Frame(root, bg=bg)
    f.grid(row=8, column=0, columnspan=3, padx=10, pady=2, sticky="w")
    ttk.Button(f, text="I'm leaving", command=lambda: p.presence(False)).grid(row=0, column=0, padx=2)
    ttk.Button(f, text="I'm back", command=lambda: (p.presence(True), p.say("I'm back!"))).grid(row=0, column=1, padx=2)
    ttk.Checkbutton(f, text="Listen (mic)", variable=listen_var, command=toggle_listen).grid(row=0, column=2, padx=6)
    ttk.Checkbutton(f, text="Mute", variable=mute_var,
                    command=lambda: setattr(p, "muted", mute_var.get())).grid(row=0, column=3, padx=6)

    diary = tk.Listbox(root, height=7, width=64, bg="#1d2027", fg="#c8ccd6", borderwidth=0, font=("Consolas", 9))
    diary.grid(row=9, column=0, columnspan=3, padx=10, pady=(4, 10))
    root.bind_all("<Motion>", lambda e: setattr(p, "last_user", time.time()))
    state = {"img": None, "log": 0}

    def refresh():
        if not p.running:
            root.destroy()
            return
        m = p.sim.mind
        img = p.sim.face.img.convert("RGB").resize((384, 192))
        arr = np.array(img)
        arr[np.all(arr > 128, axis=-1)] = (235, 245, 255)
        from PIL import Image
        state["img"] = ImageTk.PhotoImage(Image.fromarray(arr))
        face_lbl.config(image=state["img"])
        f = m.affect.feeling()
        feel.config(text=f"{FEELING_EMOJI.get(f, '')}  Pinch feels {f}")
        doing.config(text=f"doing: {m.activity.name if m.activity else '-'}   |   mood {m.affect.valence:+.2f}, "
                          f"energy {m.affect.arousal:.2f}" + (f"   |   mic {p.mic_level:.3f}" if p.mic_stream else ""))
        for n in NEEDS:
            bars[n]["value"] = getattr(m.affect.needs, n)
        o = m.world.owner
        name = p.mem.owner_name or "(it doesn't know your name yet - tell it!)"
        owner.config(text=f"You: {name}   present={o.present}   busy={o.busy or 'no'}   your mood={o.mood}"
                          f"   meeting p={m.meeting.p:.2f}")
        log = m.log
        for t, kind, text in log[state["log"]:]:
            if kind in ("activity", "heard", "meeting", "owner", "touch", "tell_name", "llm"):
                diary.insert(0, f"{t:7.1f}s  {kind:9s} {text}")
        state["log"] = len(log)
        root.after(80, refresh)

    root.protocol("WM_DELETE_WINDOW", lambda: setattr(p, "running", False))
    root.after(200, refresh)
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="let the local LLM (Ollama) understand speech too")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0, help="headless: how long")
    ap.add_argument("--say", action="append", default=[], help="headless: things to say, spaced out")
    ap.add_argument("--fresh", action="store_true", help="start without memories (not saved)")
    ap.add_argument("--quit-after", type=float, default=0, help="close the app after N seconds (smoke tests)")
    ap.add_argument("--snapshot", help="save a picture of the panel here before closing (smoke tests)")
    a = ap.parse_args()
    if a.headless:
        p = Pinch(llm=a.llm, memory_path=False if a.fresh else None, viewer=False)
        for i, text in enumerate(a.say):
            p.sim.run(4.0)
            p.sim.say(text)
        p.run(a.seconds)
        for t, kind, text in p.sim.mind.log:
            if kind in ("activity", "heard", "meeting", "touch", "tell_name"):
                print(f"{t:7.1f}s  {kind:9s} {text}")
        print(f"sounds: {[s for _, s in p.sim.sounds_played][:12]} | falls {p.sim.falls} | feeling {p.sim.mind.affect.feeling()}")
        return
    p = Pinch(llm=a.llm, memory_path=False if a.fresh else None, viewer=True)
    threading.Thread(target=p.run, daemon=True).start()
    root = build_ui(p)
    if a.quit_after:
        def close():
            if a.snapshot:
                from PIL import ImageGrab
                root.update()
                x, y = root.winfo_rootx(), root.winfo_rooty()
                ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height())).save(a.snapshot)
            p.running = False
        root.after(int(a.quit_after * 1000), close)
    root.mainloop()
    p.running = False
    p.listen(False)
    p.mem.save()


if __name__ == "__main__":
    main()
