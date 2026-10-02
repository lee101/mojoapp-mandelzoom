"""Mandel Zoom -- fractal wallpaper and deep-zoom frame renderer.

The compute-heavy core is compiled Mojo (`src/mandel.mojo`, built to
`dist/libmojoapp-mandelzoom.so` and called through ctypes): SIMD escape-time
over pixels, `parallelize` over row bands above a size threshold. This package
is the front end: views, zoom sequences, palettes, and PNG/GIF encoding.

Quick start:

    from mojoapp_mandelzoom import Frame, View, preset

    f = Frame(1920, 1080, max_iter=800)
    rgb, idx = f.render(preset("seahorse"), "cosmic")
    rgb.tofile("wallpaper.raw")

`reference.Reference` is the same algorithm in plain Python. It is the
definition of correct (`tests/test_parity.py` asserts the two agree pixel for
pixel) and the baseline `bench/bench.py` measures against.
"""

from __future__ import annotations

from . import gif, palettes, png, presets, reference
from .presets import PRESETS
from .reference import KNOWN_POINTS, Reference
from .render import (
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_MAX_ITER,
    DEFAULT_SCALE,
    Frame,
    View,
    calibrate,
    default_workers,
    parallel_threshold_pixels,
    par_min_pixels,
    preset,
    presets_names,
    simd_width,
    workers,
    zoom_sequence,
)

__version__ = "0.1.0"

__all__ = [
    "Frame",
    "View",
    "Reference",
    "PRESETS",
    "KNOWN_POINTS",
    "gif",
    "palettes",
    "png",
    "presets",
    "reference",
    "calibrate",
    "default_workers",
    "parallel_threshold_pixels",
    "par_min_pixels",
    "preset",
    "presets_names",
    "simd_width",
    "workers",
    "zoom_sequence",
    "DEFAULT_CX",
    "DEFAULT_CY",
    "DEFAULT_SCALE",
    "DEFAULT_MAX_ITER",
    "__version__",
]
