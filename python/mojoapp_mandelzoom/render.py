"""The render object: NumPy buffers in, compiled Mojo out.

This is the whole bridge. It owns the two float scratch planes the kernel
computes into, hands their addresses to the shared library as `Int`, and keeps
the palette bytes alive for the duration of the call -- the FFI cannot check
any of that, so the tests do.

Scheduling is a measurement, not a constant. `calibrate()` times the serial
and parallel paths against each other on this host and returns the cell count
at which the launch starts paying for itself; below it everything runs
serially. That is why the same code is fast on a quiet machine and still
correct on a contended one.
"""

from __future__ import annotations

import ctypes
import math
import os
import time

import numpy as np

from . import palettes, reference
from ._lib import addr, lib

#: Every buffer the kernel writes is C-contiguous; a non-contiguous view handed
#: over as a raw address would be read in the wrong order, silently. Stated as
#: a helper rather than a comment because that failure produces a plausible
#: wrong picture instead of a crash.
_CONTIG = np.ascontiguousarray

DEFAULT_CX = reference.KNOWN_POINTS["cardioid_tip"][0]
DEFAULT_CY = 0.0
DEFAULT_SCALE = 0.6
DEFAULT_MAX_ITER = 500
DEFAULT_WORKERS = 16
#: Default colour parameters. `pscale` maps the escape value to a palette slot,
#: `repeat` is how fast the orbit-trap term winds the bands, `trap_scale` is how
#: strongly the trap shows.
DEFAULT_PSCALE = 8.0
DEFAULT_REPEAT = 6.0
DEFAULT_TRAP_SCALE = 400.0
DEFAULT_TRAP_SHIFT = 0.0
DEFAULT_TRAP_MODE = 0

#: The one-time serial/parallel calibration is done at this size: large on
#: purpose, because at a small frame the launch cost swamps the work and the
#: measurement is noise. A 512x512 frame at 300 iterations is a few tens of
#: milliseconds serial, so calibration stays cheap and still sees a signal.
_PROBE_SIDE = 512
_PROBE_ITERS = 300
_PROBE_REPEATS = 5

_calibration: dict | None = None


# ----------------------------------------------------------------- kernel info
def _info() -> dict:
    """Query the build-time constants through the library's `mz_info`."""
    lib_ = lib()
    out = [np.zeros(1, dtype=np.float64) for _ in range(5)]
    lib_.mz_info(*(addr(a) for a in out))
    return {
        "workers": int(out[0][0]),
        "par_min_pixels": int(out[1][0]),
        "simd_width": int(out[2][0]),
        "aspect_w": int(out[3][0]),
        "aspect_h": int(out[4][0]),
    }


def workers() -> int:
    """Default worker count the parallel path uses."""
    return _info()["workers"]


def par_min_pixels() -> int:
    """Pixels below which the serial loop runs instead of `parallelize`."""
    return _info()["par_min_pixels"]


def simd_width() -> int:
    """Lanes in the `Float64` vector body.

    Queried rather than restated in Python so the two cannot drift: the tests
    probe widths either side of this number to land on the vector tail.
    """
    return _info()["simd_width"]


def default_workers() -> int:
    """Worker count actually used: the kernel's cap, bounded by this machine."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(workers(), n))


# ---------------------------------------------------------------- calibration
def calibrate(repeats: int = _PROBE_REPEATS) -> dict:
    """Measure where the serial loop stops beating the parallel one.

    Two numbers decide it: how long one frame costs in the serial loop, and how
    much a `parallelize` launch adds on top. A launch costs a fixed amount per
    call, so the break-even pixel count is a ratio of the two -- which is why
    it is measured rather than hard-coded. A constant written on one machine is
    wrong on every other one, and on a contended host the difference is not a
    rounding error.

    The two paths are sampled alternately rather than in blocks, so drift in
    host load lands on both instead of on whichever ran second. Each takes its
    own minimum: the question is how fast the frame goes when it is not
    competing with something else.

    Runs once per process and caches. The paths are asserted pixel-for-pixel
    identical in `tests/test_parity.py`, so choosing one or the other cannot
    change an answer -- only how long it takes.
    """
    global _calibration
    if _calibration is not None:
        return _calibration

    r = Frame(_PROBE_SIDE, _PROBE_SIDE, max_iter=_PROBE_ITERS)
    wk = default_workers()
    # A fixed view, the whole set, so the calibration measures the kernel rather
    # than whichever preset happens to be in front of it.
    probe_view = View(DEFAULT_CX, DEFAULT_CY, DEFAULT_SCALE, "calibration probe")

    def sample(path_workers: int) -> float:
        r.workers = path_workers
        t0 = time.perf_counter()
        r.render(probe_view, palettes.DEFAULT_PALETTE)
        return time.perf_counter() - t0

    sample(1)
    sample(wk)

    serial_s = math.inf
    parallel_s = math.inf
    for _ in range(repeats):
        serial_s = min(serial_s, sample(1))
        parallel_s = min(parallel_s, sample(wk))

    pixels = _PROBE_SIDE * _PROBE_SIDE
    serial_per_px = serial_s / pixels
    parallel_per_px = parallel_s / pixels
    launch_overhead = parallel_per_px - serial_per_px

    if launch_overhead <= 0:
        # The parallel path was not slower at the probe size: the launch is
        # cheap here and it wins from the kernel's own floor upward.
        threshold = float(par_min_pixels())
    else:
        threshold = pixels * serial_per_px / launch_overhead

    _calibration = {
        "probe_pixels": pixels,
        "probe_workers": wk,
        "probe_iters": _PROBE_ITERS,
        "serial_ns_per_pixel": serial_per_px * 1e9,
        "parallel_ns_per_pixel": parallel_per_px * 1e9,
        "parallel_wins_at_probe": parallel_per_px < serial_per_px,
        "launch_ns_per_frame": launch_overhead * 1e9 * pixels,
        "threshold_pixels": threshold,
    }
    return _calibration


def parallel_threshold_pixels() -> float:
    """Pixels per frame at or above which the parallel loop wins.

    `inf` means it never wins on this host, and `Frame` renders everything
    serially. That is a real, measured answer rather than a missing feature.
    """
    return calibrate()["threshold_pixels"]


# -------------------------------------------------------------------- the view
class View:
    """A centre, a half-height in complex units, and a label.

    A deep zoom is a sequence of these, and the sequence is what makes the
    artifact shareable: the JSON the demo writes is a list of views, and
    `zoom_sequence` is the same list computed from a seed. Keeping it a value
    object (rather than four loose floats) is what lets the demo, the CLI and
    the GIF agree on what "the view" means.
    """

    __slots__ = ("cx", "cy", "scale", "label")

    def __init__(self, cx: float, cy: float, scale: float, label: str = ""):
        if not (scale > 0.0) or not math.isfinite(scale):
            raise ValueError("scale must be a positive finite number")
        if not (math.isfinite(cx) and math.isfinite(cy)):
            raise ValueError("centre must be finite")
        self.cx = float(cx)
        self.cy = float(cy)
        self.scale = float(scale)
        self.label = label

    def zoom(self, factor: float) -> "View":
        """A new view `factor` times deeper at the same centre."""
        if not (factor > 0.0):
            raise ValueError("factor must be positive")
        return View(self.cx, self.cy, self.scale / factor, self.label)

    def as_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "scale": self.scale, "label": self.label}

    @staticmethod
    def from_dict(d: dict) -> "View":
        return View(d["cx"], d["cy"], d["scale"], d.get("label", ""))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, View):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self) -> str:
        return (
            f"View(cx={self.cx!r}, cy={self.cy!r}, scale={self.scale!r}, "
            f"label={self.label!r})"
        )


def preset(name: str) -> View:
    """One of the named starting views, at its natural scale."""
    from .presets import PRESETS

    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; pick one of {', '.join(presets_names())}")
    return PRESETS[name]


def presets_names() -> list[str]:
    from .presets import PRESETS

    return sorted(PRESETS)


def zoom_sequence(
    view: View, frames: int, factor: float = 1.35
) -> list[View]:
    """A deterministic zoom path: `frames` views from `view` inward.

    `factor` is the per-frame magnification, so a 1.35 over 30 frames is
    1.35**30 ~= 4600x deeper at the end -- a real deep zoom rather than a
    handful of steps. The path is the same list the demo writes into its JSON
    and the same list the GIF walks, so a shared GIF and a shared JSON describe
    one thing.
    """
    if frames < 1:
        raise ValueError("a zoom needs at least one frame")
    out = []
    v = view
    for i in range(frames):
        out.append(View(v.cx, v.cy, v.scale, f"{view.label}#{i}" if view.label else f"#{i}"))
        v = v.zoom(factor)
    return out


# ------------------------------------------------------------------- the frame
class Frame:
    """A renderable image buffer backed by the compiled kernels.

    `width` and `height` are the pixel dimensions. `max_iter` bounds the
    escape loop; `workers` picks the schedule, and `workers=1` forces the
    serial path -- which is the benchmark's lever, not a fallback for frames
    too small to parallelise.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_iter: int = DEFAULT_MAX_ITER,
        workers: int | None = None,
        pscale: float = DEFAULT_PSCALE,
        repeat: float = DEFAULT_REPEAT,
        trap_scale: float = DEFAULT_TRAP_SCALE,
        trap_shift: float = DEFAULT_TRAP_SHIFT,
        trap_mode: int = DEFAULT_TRAP_MODE,
    ):
        if width < 1 or height < 1:
            raise ValueError("a frame must be at least 1x1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not 0 <= trap_mode <= 2:
            raise ValueError("trap_mode must be 0, 1 or 2")
        self.w = int(width)
        self.h = int(height)
        self.max_iter = int(max_iter)
        self.workers = default_workers() if workers is None else int(workers)
        self.pscale = float(pscale)
        self.repeat = float(repeat)
        self.trap_scale = float(trap_scale)
        self.trap_shift = float(trap_shift)
        self.trap_mode = int(trap_mode)
        # Scratch planes the kernel computes into. Reused across renders; the
        # kernel writes every element before reading any, so reuse is safe and
        # saves an allocation per frame.
        self._mu = np.zeros(self.w * self.h, dtype=np.float64)
        self._trap = np.zeros(self.w * self.h, dtype=np.float64)
        self._rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._idx = np.zeros((self.h, self.w), dtype=np.uint8)

    # ---------------------------------------------------------------- helpers
    @property
    def pixels(self) -> int:
        return self.w * self.h

    def _workers_for(self) -> int:
        """The worker count the schedule will actually use.

        A frame below the kernel's pixel floor runs serial no matter what was
        asked for, and `calibrate()` can raise the bar further on a host where
        the launch does not pay. An explicit `workers` overrides the
        calibration -- it is a request for exactly that path, and letting the
        calibration overrule it would make `workers=1` useless as a lever.
        """
        if self.workers <= 1:
            return 1
        if self.pixels < par_min_pixels():
            return 1
        return self.workers

    @property
    def uses_parallel(self) -> bool:
        return self._workers_for() > 1

    # ----------------------------------------------------------------- render
    def render(
        self,
        view: View,
        palette: str | np.ndarray = palettes.DEFAULT_PALETTE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render `view` into this frame. Returns `(rgb, idx)`.

        The two arrays are views into the frame's own buffers and are replaced
        (not written through) on the next render, so copy them if you intend to
        keep them. The CLI and the demo do, which is why they are handed to a
        writer immediately.
        """
        if not isinstance(view, View):
            raise TypeError("render() takes a View")
        pal = _resolve_palette(palette)
        # The palette crosses as a raw address, so the ctypes array holding it
        # must outlive the call. It is held on the Frame (one slot, replaced
        # each render) rather than in a module-level list, which would grow for
        # the life of the process in any loop that renders many frames.
        self._pal = (ctypes.c_uint8 * 768).from_buffer_copy(palettes.to_bytes(pal))
        rc = lib().mz_render(
            addr(self._mu),
            addr(self._trap),
            addr(self._rgb.reshape(-1)),
            addr(self._idx.reshape(-1)),
            self.w,
            self.h,
            view.cx,
            view.cy,
            view.scale,
            self.max_iter,
            ctypes.addressof(self._pal),
            self.pscale,
            self.repeat,
            self.trap_scale,
            self.trap_shift,
            self.trap_mode,
            self._workers_for(),
        )
        if rc != 0:
            raise RuntimeError(f"mz_render rejected the frame (rc={rc})")
        return self._rgb, self._idx

    def escape_plane(self, view: View) -> tuple[np.ndarray, np.ndarray]:
        """Just the float escape planes, as `(h, w)` views, for analysis.

        A render that keeps only the escape value (the histogram, the ASCII
        card) does not need the colour pass, but the kernel always runs it, so
        this is just the plane view of an ordinary render.
        """
        self.render(view, palettes.DEFAULT_PALETTE)
        return (
            self._mu.reshape(self.h, self.w),
            self._trap.reshape(self.h, self.w),
        )

    @property
    def escape(self) -> np.ndarray:
        """The last render's escape plane, as an `(h, w)` view.

        Read-only by convention: it aliases the frame's scratch buffer and is
        replaced by the next render. The CLI's text card and the demo's stats
        both read it after a render, which is the only point it is meaningful
        at.
        """
        return self._mu.reshape(self.h, self.w)

    def histogram(self, view: View) -> np.ndarray:
        """Escape-value histogram: `max_iter + 2` bins summing to `w*h`.

        The last bin is the interior (never-escaped) count.
        """
        out = np.zeros(self.max_iter + 2, dtype=np.uint32)
        n = lib().mz_iteration_histogram(
            addr(out),
            self.w,
            self.h,
            view.cx,
            view.cy,
            view.scale,
            self.max_iter,
            self._workers_for(),
        )
        if n != self.max_iter + 2:
            raise RuntimeError(f"histogram kernel returned {n} bins")
        return out

    def __repr__(self) -> str:
        return (
            f"Frame({self.w}x{self.h}, max_iter={self.max_iter}, "
            f"workers={self.workers})"
        )


def _resolve_palette(palette: str | np.ndarray) -> np.ndarray:
    if isinstance(palette, str):
        return palettes.get(palette)
    a = _CONTIG(palette, dtype=np.uint8)
    if a.shape != (256, 3):
        raise ValueError(f"a palette must be (256, 3), got {a.shape}")
    return a
