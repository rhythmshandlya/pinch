"""Behaviour settings files (configs/mind/*.toml, phase 4): loaded once, shared by the mind, the body and the face.
Each file carries a `version`; pinch/recorder.py writes every file's version and hash into each session."""

import tomllib
from functools import cache
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "configs" / "mind"


@cache
def load(name):
    """configs/mind/<name>.toml as a dict."""
    with open(DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)
