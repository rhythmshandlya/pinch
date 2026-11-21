"""Shapes of the v2 body (style B, the chubbier cartoon crab): the superellipsoid shell and the helpers the model
builder, the fit check and the renders share. Pure numpy.

A superellipsoid with half sizes a, b, c and exponents (e_v vertical, e_h horizontal) is the surface
    x = a sp(cos u, e_v) sp(cos v, e_h),  y = b sp(cos u, e_v) sp(sin v, e_h),  z = c sp(sin u, e_v)
with sp(w, e) = sign(w) |w|^e; exponents under 1 make it boxier (1 would be an ellipsoid). Its inside is the set
    ( (|x/a|^(2/e_h) + |y/b|^(2/e_h))^(e_h/e_v) + |z/c|^(2/e_v) ) <= 1.
"""

import numpy as np


def sp(w, e):
    return np.sign(w) * np.abs(w) ** e


def superellipsoid(a, b, c, e_v=0.35, e_h=0.35, n=40, taper=None):
    """Mesh (vertices, triangle faces) of a rounded box. taper(x_norm in 0..1) scales y and z along x."""
    us = np.linspace(-np.pi / 2, np.pi / 2, n)
    vs = np.linspace(-np.pi, np.pi, 2 * n, endpoint=False)
    U, V = np.meshgrid(us, vs, indexing="ij")
    x = a * sp(np.cos(U), e_v) * sp(np.cos(V), e_h)
    y = b * sp(np.cos(U), e_v) * sp(np.sin(V), e_h)
    z = c * sp(np.sin(U), e_v)
    if taper is not None:
        k = taper((x / a + 1) / 2)
        y, z = y * k, z * k
    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    faces = []
    m = 2 * n
    for i in range(n - 1):
        for j in range(m):
            p, q, r, s = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            faces += [(p, r, q), (q, r, s)]
    return verts, np.array(faces, int)


def c_claw(R=0.026, r=0.009, gap_deg=70, n=36, m=12):
    """A cartoon C-shaped claw: a torus arc in the x-z plane opening forward (+x)."""
    th = np.radians(np.linspace(gap_deg / 2, 360 - gap_deg / 2, n))
    ph = np.linspace(0, 2 * np.pi, m, endpoint=False)
    verts = []
    for t in th:
        cx, cz = R * np.cos(t), R * np.sin(t)
        for p in ph:
            rr = r * (0.75 + 0.25 * np.sin(t / 2))                 # thicker at the back, slimmer tips
            verts.append((cx + rr * np.cos(p) * np.cos(t), rr * np.sin(p), cz + rr * np.cos(p) * np.sin(t)))
    faces = []
    for i in range(n - 1):
        for j in range(m):
            p, q, a, b = i * m + j, i * m + (j + 1) % m, (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            faces += [(p, a, q), (q, a, b)]
    for k in (0, n - 1):                                          # round caps
        verts.append(tuple(np.mean(verts[k * m:(k + 1) * m], axis=0)))
        c = len(verts) - 1
        for j in range(m):
            faces.append((c, k * m + j, k * m + (j + 1) % m) if k else (c, k * m + (j + 1) % m, k * m + j))
    return np.array(verts, float), np.array(faces, int)


def area(verts, faces):
    """Surface area of a triangle mesh (same unit as the vertices, squared)."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())


class Shell:
    """The pill shell of the body (all lengths in mm, body frame): centre, half sizes, exponents, wall."""

    def __init__(self, center, half, exponents=(0.65, 0.6), wall=2.0):
        self.c = np.asarray(center, float)
        self.h = np.asarray(half, float)
        self.ev, self.eh = float(exponents[0]), float(exponents[1])
        self.wall = float(wall)

    @classmethod
    def from_chassis(cls, ch):
        return cls(ch["shell_center_mm"], ch["shell_half_mm"], ch["shell_exponents"], ch["wall_mm"])

    def f(self, p, grow=0.0):
        """Implicit function: < 1 inside, 1 on the surface of the shell shrunk by `grow` mm (every half size minus grow)."""
        q = np.abs((np.asarray(p, float) - self.c) / np.maximum(self.h - grow, 1e-9))
        return (q[..., 0] ** (2 / self.eh) + q[..., 1] ** (2 / self.eh)) ** (self.eh / self.ev) + q[..., 2] ** (2 / self.ev)

    def inside(self, p, margin=0.0):
        """Inside the shell's interior: the shell shrunk by the wall and the margin (mm)."""
        return self.f(p, self.wall + margin) <= 1.0

    def depth(self, p):
        """About how far a point lies below the outer surface (mm; negative = outside): the shrink of every half size
        at which the point is on the surface, found by bisection (exact on the axes, close elsewhere)."""
        p = np.asarray(p, float)
        lo, hi = -80.0, float(self.h.min()) - 1e-3
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.f(p, mid) <= 1.0:
                lo = mid                         # still inside the shrunk shell: the surface is deeper
            else:
                hi = mid
        return lo

    def mesh(self, n=40):
        v, f = superellipsoid(*(self.h / 1000.0), self.ev, self.eh, n=n)
        return v + self.c / 1000.0, f

    def area_m2(self, n=120):
        v, f = self.mesh(n)
        return area(v, f)

    def surface_z(self, x, y):
        """Height of the outer surface (mm, body frame) above the point (x, y) on the upper half, or None if (x, y) is off the shell."""
        lo, hi = self.c[2], self.c[2] + self.h[2]
        if self.f(np.array([x, y, lo])) > 1.0:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.f(np.array([x, y, mid])) <= 1.0:
                lo = mid
            else:
                hi = mid
        return lo

    def ray(self, o, d, tmax=500.0):
        """Distance along the ray o + t d (unit d) to the shell's outer surface, starting from a point outside, or None."""
        o, d = np.asarray(o, float), np.asarray(d, float)
        ts = np.linspace(0.0, tmax, 2000)
        inside = self.f(o + ts[:, None] * d) <= 1.0
        if not inside.any():
            return None
        k = int(np.argmax(inside))
        lo, hi = ts[max(k - 1, 0)], ts[k]
        for _ in range(30):
            mid = (lo + hi) / 2
            if self.f(o + mid * d) <= 1.0:
                hi = mid
            else:
                lo = mid
        return hi


def surface_y(shell, x, z):
    """y (mm) of the shell's outer surface at (x, z) on the +y side, or None if (x, z) is off the shell."""
    lo, hi = 0.0, float(shell.h[1])
    if shell.f(np.array([x, lo + shell.c[1], z])) > 1.0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if shell.f(np.array([x, shell.c[1] + mid, z])) <= 1.0:
            lo = mid
        else:
            hi = mid
    return lo
