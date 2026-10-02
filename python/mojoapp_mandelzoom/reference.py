"""The pure-Python escape-time reference: the definition of correct.

This is the algorithm from `src/mandel.mojo` written the obvious way -- flat
arithmetic, an explicit loop, no cleverness. It is deliberately NOT vectorised
and NOT the fast path. Two things depend on it:

* `tests/test_parity.py` asserts the compiled kernel agrees with it, escape
  value and orbit trap, pixel for pixel. If they disagree, one of them is wrong
  and the test says which.
* `bench/bench.py` measures the compiled core against it. That ratio is the
  headline number in the README, so it has to be measured against something
  obviously correct rather than against another compiled thing.

The arithmetic here is the specification the Mojo mirrors:

    z <- z*z + c, from z = 0, until |z|^2 > 4 or max_iter is reached
    mu    = hit + 1 - log2(log2(|z|))   when it escaped
    mu    = max_iter + 1                 when it did not
    trap  = min |z|^2 over the whole orbit, floored at 1e-300

and the colour mapping, identical to the kernel's:

    interior (mu > max_iter) -> palette[0]
    t       = repeat*log(1 + trap^2*trap_scale) + trap_shift
    p       = int(mu*pscale + t) mod 256
"""

from __future__ import annotations

import math

import numpy as np

#: A few real points on the set, used by the demo and the tests as sanity
#: anchors for the view transform.
KNOWN_POINTS = {
    "cusp": (-2.0, 0.0),
    "cardioid_tip": (0.25, 0.0),
    "centre": (0.0, 0.0),
    "seahorse_valley": (-0.743643887037151, 0.131825904205330),
    "double_spiral": (-0.1592, 1.0317),
}


class Reference:
    """Escape-time renderer in plain Python, one pixel plane at a time."""

    def __init__(self, width: int, height: int):
        if width < 1 or height < 1:
            raise ValueError("a frame must be at least 1x1")
        self.w = width
        self.h = height

    # ------------------------------------------------------------- the loop
    def escape(self, cx: float, cy: float, scale: float, max_iter: int) -> np.ndarray:
        """Return a `(h, w)` float64 array of escape values."""
        out = np.empty((self.h, self.w), dtype=np.float64)
        inv_w = 1.0 / self.w
        inv_h = 1.0 / self.h
        for y in range(self.h):
            sy = 1.0 - 2.0 * y * inv_h
            cyy = cy + sy * scale
            for x in range(self.w):
                cxx = cx + (-1.0 + 2.0 * x * inv_w) * scale
                out[y, x] = _escape(cxx, cyy, max_iter)[0]
        return out

    def escape_plane(
        self, cx: float, cy: float, scale: float, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mu, trap)` as two `(h, w)` float64 arrays."""
        mu = np.empty((self.h, self.w), dtype=np.float64)
        trap = np.empty((self.h, self.w), dtype=np.float64)
        inv_w = 1.0 / self.w
        inv_h = 1.0 / self.h
        for y in range(self.h):
            sy = 1.0 - 2.0 * y * inv_h
            cyy = cy + sy * scale
            for x in range(self.w):
                m, t = _escape(cx + (-1.0 + 2.0 * x * inv_w) * scale, cyy, max_iter)
                mu[y, x] = m
                trap[y, x] = t
        return mu, trap

    def render(
        self,
        cx: float,
        cy: float,
        scale: float,
        max_iter: int,
        palette: np.ndarray,
        pscale: float = 8.0,
        repeat: float = 6.0,
        trap_scale: float = 400.0,
        trap_shift: float = 0.0,
        trap_mode: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(rgb, idx)` -- `(h, w, 3)` uint8 and `(h, w)` uint8."""
        mu, trap = self.escape_plane(cx, cy, scale, max_iter)
        idx = index_plane(
            mu, trap, max_iter, pscale, repeat, trap_scale, trap_shift, trap_mode
        )
        rgb = np.ascontiguousarray(palette[idx], dtype=np.uint8)
        return rgb, idx


def _escape(cxx: float, cyy: float, max_iter: int) -> tuple[float, float]:
    """Escape value and orbit trap for one point. The specification loop."""
    zr = 0.0
    zi = 0.0
    trap = 1e300
    hit = -1
    r2 = 0.0
    i = 0
    while i < max_iter:
        zr, zi = zr * zr - zi * zi + cxx, 2.0 * zr * zi + cyy
        r2 = zr * zr + zi * zi
        if r2 < trap:
            trap = r2
        if r2 > 4.0:
            hit = i
            break
        i += 1
    if hit >= 0:
        mu = hit + 1 - math.log2(0.5 * math.log2(r2))
    else:
        mu = float(max_iter) + 1.0
    if trap < 1e-300:
        trap = 1e-300
    return mu, trap


def index_plane(
    mu: np.ndarray,
    trap: np.ndarray,
    max_iter: int,
    pscale: float,
    repeat: float,
    trap_scale: float,
    trap_shift: float,
    trap_mode: int,
) -> np.ndarray:
    """Map `(mu, trap)` to palette indices. The kernel's exact colouring rule.

    Kept as a free function so the tests can pin the mapping on synthetic
    planes without rendering a frame.
    """
    mu = np.asarray(mu, dtype=np.float64)
    trap = np.asarray(trap, dtype=np.float64)
    if trap_mode == 1:
        t = trap * trap_scale + trap_shift
    elif trap_mode == 2:
        h = mu.shape[0]
        inv_h = 1.0 / h
        yy = np.arange(h, dtype=np.float64)
        sy = 1.0 - 2.0 * yy * inv_h
        t = (1.0 - sy)[:, None] * trap_scale + trap_shift
    else:
        t = repeat * np.log(1.0 + trap * trap * trap_scale) + trap_shift
    q = np.trunc(mu * pscale + t).astype(np.int64)
    q[q < 0] = 0
    idx = (q % 256).astype(np.uint8)
    # Interior (never escaped) is palette[0]. The kernel uses the same test.
    idx[mu > max_iter] = 0
    return idx
