"""Procedural surface textures (numpy): wood grain, floor planks, tiles, fabric noise, rug stripes and checks.

Randomised textures are the standard domain-randomisation lever for camera-based skills (Tobin et al. 2017,
arXiv 1703.06907): a detector that has seen many surfaces doesn't key on one. Each pattern takes a base colour and a
seed, so a world's textures are as reproducible as its layout.
"""

import numpy as np

N = 128          # texture size (px); materials repeat them across a surface


def _noise(rng, n, octaves=4):
    """Value noise in 0..1: sums of upsampled random grids (cheap, tileable enough for repeats)."""
    out = np.zeros((n, n))
    for o in range(octaves):
        k = 2 ** (o + 2)
        g = rng.random((k + 1, k + 1))
        idx = np.linspace(0, k, n, endpoint=False)
        i0 = idx.astype(int)
        f = idx - i0
        a = g[i0][:, i0] * (1 - f)[None, :] + g[i0][:, i0 + 1] * f[None, :]
        b = g[i0 + 1][:, i0] * (1 - f)[None, :] + g[i0 + 1][:, i0 + 1] * f[None, :]
        out += (a * (1 - f)[:, None] + b * f[:, None]) / 2 ** o
    out -= out.min()
    return out / (out.max() + 1e-9)


def pattern(kind, rgb, seed, n=N):
    """(n, n, 3) uint8 image of a pattern in colours around rgb (0..1)."""
    rng = np.random.default_rng(seed)
    base = np.array(rgb, float)[None, None, :]
    y, x = np.mgrid[0:n, 0:n] / n
    if kind == "wood":
        warp = _noise(rng, n, 3)
        grain = 0.5 + 0.5 * np.sin((y * rng.uniform(18, 40) + 4 * warp) * 2 * np.pi)
        v = 0.82 + 0.18 * grain + 0.08 * (_noise(rng, n) - 0.5)
    elif kind == "planks":
        rows = int(rng.integers(4, 8))
        plank = np.floor(y * rows)
        shade = rng.uniform(0.85, 1.1, rows + 1)[plank.astype(int)]
        grain = 0.5 + 0.5 * np.sin((x * rng.uniform(10, 25) + 3 * _noise(rng, n, 3)) * 2 * np.pi)
        seam = (np.abs(y * rows - np.round(y * rows)) < 0.03).astype(float)
        v = shade * (0.88 + 0.12 * grain) * (1 - 0.35 * seam)
    elif kind == "tiles":
        k = int(rng.integers(2, 5))
        gx, gy = (x * k) % 1, (y * k) % 1
        grout = ((gx < 0.04) | (gy < 0.04)).astype(float)
        v = (0.95 + 0.08 * (_noise(rng, n) - 0.5)) * (1 - 0.3 * grout)
    elif kind == "stripes":
        k = int(rng.integers(3, 9))
        band = np.floor(x * k).astype(int) % 3
        second = np.array(rng.uniform(0.3, 1.0, 3))
        img = np.where((band == 1)[..., None], base, second[None, None, :] * 0.6 + base * 0.4)
        return _to8(img * (0.9 + 0.1 * _noise(rng, n))[..., None])
    elif kind == "checker":
        k = int(rng.integers(2, 6))
        c = ((np.floor(x * k) + np.floor(y * k)) % 2)[..., None]
        other = np.clip(base * rng.uniform(0.5, 0.8) + rng.uniform(0, 0.2), 0, 1)
        return _to8((c * base + (1 - c) * other) * (0.92 + 0.08 * _noise(rng, n))[..., None])
    elif kind == "noise":
        v = 0.88 + 0.2 * (_noise(rng, n) - 0.5) + 0.06 * (rng.random((n, n)) - 0.5)
    else:                                               # flat
        v = np.full((n, n), 1.0) + 0.02 * (rng.random((n, n)) - 0.5)
    return _to8(base * v[..., None])


def _to8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
