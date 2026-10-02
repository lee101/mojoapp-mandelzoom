"""Palettes: 256 RGB triples, deterministic, hand-written.

Every palette is generated from a formula rather than a literal table so the
bytes are reproducible from the source alone and a new palette is one line. The
kernel maps an escape value straight into this table, so the palette is the
only thing that decides what a render looks like.

The generator interpolates in linear-ish space through a list of anchor colours
and returns a `(256, 3)` uint8 array. `to_bytes` flattens it for the FFI.
"""

from __future__ import annotations

import numpy as np

#: Anchor colours per palette, as `#rrggbb` hex. `get` interpolates these to
#: 256 entries, so an anchor list is the whole definition.
ANCHORS: dict[str, list[str]] = {
    # Deep-space blue through cyan to a hot white core.
    "cosmic": ["#000308", "#001b4d", "#0069a8", "#12a5c4", "#6ee7d8", "#f2fbff"],
    # Classic Ultra Fractal: indigo to magenta to amber.
    "ember": ["#08051a", "#2b1a6b", "#7b2d8e", "#c94b7c", "#f0913f", "#ffe9b0"],
    # The saturated rainbow a fractal is expected to have.
    "rainbow": ["#12002e", "#002fa8", "#00a3e0", "#38d16b", "#f7e225", "#ff5f6d"],
    # Monochrome, for text cards and for seeing the structure without colour.
    "mono": ["#000000", "#ffffff"],
    # High contrast poster: black, then a single hot ramp.
    "noir": ["#000000", "#1a1a1a", "#8c1c13", "#e4572e", "#ffd166", "#fffaf0"],
    # Pastel, for wallpaper.
    "pastel": ["#0b1026", "#3b2f63", "#8f6bab", "#d9a5c3", "#f7d9c4", "#fff6ea"],
    # Copper and verdigris, for deep zooms of the seahorse valley.
    "verdigris": ["#04100e", "#0d3b34", "#1f7a63", "#6fbf8f", "#cfe8c0", "#f2f7e8"],
    # Fire.
    "inferno": ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
}

DEFAULT_PALETTE = "cosmic"


def names() -> list[str]:
    return sorted(ANCHORS)


def get(name: str) -> np.ndarray:
    """Return the `(256, 3)` uint8 palette table for `name`."""
    if name not in ANCHORS:
        raise ValueError(
            f"unknown palette {name!r}; pick one of {', '.join(names())}"
        )
    anchors = [_hex(c) for c in ANCHORS[name]]
    n = len(anchors)
    # Piecewise-linear interpolation from anchor to anchor across 256 entries.
    xs = np.linspace(0.0, 1.0, 256)
    pos = xs * (n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    frac = (pos - lo)[:, None]
    arr = np.array(anchors, dtype=np.float64)
    out = arr[lo] * (1.0 - frac) + arr[hi] * frac
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def to_bytes(pal: np.ndarray) -> bytes:
    """Flatten a `(256, 3)` palette to the 768 bytes the kernel reads."""
    a = np.ascontiguousarray(pal, dtype=np.uint8)
    if a.shape != (256, 3):
        raise ValueError(f"expected a (256, 3) palette, got {a.shape}")
    return a.tobytes()


def _hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
