"""Rounded edges for the printed parts (owner, 2026-11-22: no sharp edges; children under 6 handle it, EN 71-1 asks
for rounded edges): the parts are built from rounded boxes."""

from build123d import fillet


def RBox(length, width, height, r=None, **kw):       # noqa: N802 (named like build123d's Box, which it replaces)
    """A box with rounded edges (radius 1.2 mm, or 40 % of its thinnest side): the printed parts are built from
    these so they come out without sharp edges (rounding a finished part makes OpenCascade crash)."""
    from build123d import Box
    b = Box(length, width, height, **kw)
    r = min(1.2, 0.4 * min(length, width, height)) if r is None else r
    return fillet(b.edges(), r) if r > 0.05 else b
