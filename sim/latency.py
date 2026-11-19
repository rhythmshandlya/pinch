"""Command latency for the CPU simulator: servo targets arrive `delay` seconds late,
like on the real bus (Rhoban BAM measured ~1-5 ms for the STS3215, plus bus time)."""

from collections import deque

import numpy as np


class DelayLine:
    def __init__(self, delay_s: float, dt: float, initial):
        n = max(0, int(round(delay_s / dt)))
        self.buf = deque([np.array(initial, float)] * n)

    def __call__(self, ctrl):
        if not self.buf:
            return ctrl
        self.buf.append(np.array(ctrl, float))
        return self.buf.popleft()
