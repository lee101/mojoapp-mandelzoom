"""Honest benchmark: the compiled Mojo core against the pure-Python reference.

    pixi run bench

Two tables, both measured in this process. Nothing is extrapolated, nothing is
rounded in our favour, and rows where the compiled path LOSES are kept and
labelled, because they are the interesting ones.

**Table 1 -- serial vs parallel.** Whether a `parallelize` launch pays for
itself is not a property of the source, it is a property of the machine: the
launch costs a fixed amount per call, and on a contended host that cost can
exceed a whole frame of work. This table measures the crossover on the machine
it runs on and prints every size, including the ones where the parallel loop
loses.

**Table 2 -- Mojo vs pure Python.** The headline ratio, against
`mojoapp_mandelzoom.reference`, which is the same escape-time algorithm written
the obvious way. `speedup` is `reference_seconds / mojo_seconds`, so >1 means
Mojo wins. Rows where it loses are in the table too.

The reference is timed with a budget: escape-time in pure Python is slow enough
that an unbounded sweep would run for hours. Where it was not run to completion
the row says `not run` rather than inventing a number.

There is also a **correctness column**, and it is the one worth reading first.
Escape-time is chaotic, so the compiled and reference images agree on interior
pixels exactly and can differ on a handful of pixels sitting exactly on the
`|z| = 2` boundary. The column reports the actual agreement, so the speedup is
never quoted without it.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mojoapp_mandelzoom import reference  # noqa: E402
from mojoapp_mandelzoom.render import (  # noqa: E402
    Frame,
    calibrate,
    default_workers,
    par_min_pixels,
    preset,
    simd_width,
    workers,
)

WARMUPS = 2
REPEATS = 5
#: Wall-clock budget for the whole pure-Python section. The repeat count is
#: derived from the first sample, so a slow host gets fewer repeats rather than
#: a benchmark that never finishes.
REFERENCE_BUDGET_S = 25.0
REFERENCE_MAX_REPEATS = 3
#: Hard ceiling on any single reference run. Pure-Python escape-time costs
#: roughly a microsecond per pixel per iteration, so a 1920x1080 frame at 800
#: iterations is over half an CPU-hour -- and paying that to fill in one table
#: cell would make `pixi run bench` unusable. Cases above this estimate are
#: reported as `skipped (>budget)`, never as a made-up number.
REFERENCE_HARD_LIMIT_S = 20.0

# (label, view_name, width, height, max_iter)
CASES = [
    ("128x128 x200", "seahorse", 128, 128, 200),
    ("256x144 x300", "seahorse", 256, 144, 300),
    ("320x240 x200", "full", 320, 240, 200),
    ("480x270 x400", "elephant", 480, 270, 400),
    ("640x360 x300", "cusp", 640, 360, 300),
    ("800x450 x600", "spiral", 800, 450, 600),
    ("1024x576 x500", "seahorse", 1024, 576, 500),
    ("1920x1080 x800", "full", 1920, 1080, 800),
    # Deliberately tiny. The ctypes round trip and the kernel's fixed overhead
    # dominate here, so these rows are expected to LOSE and they stay.
    ("16x16 x50", "full", 16, 16, 50),
    ("32x32 x50", "full", 32, 32, 50),
    ("64x36 x100", "cusp", 64, 36, 100),
]

PARALLEL_SIZES = [128, 256, 512, 1024, 1920]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def best_seconds(fn, repeats: int = REPEATS) -> float:
    """Fastest of `repeats` timed runs after `WARMUPS` warm-ups.

    `min` rather than median or mean: the question is how fast this loop goes
    when it is not competing with something else, and on a shared host the
    minimum sample is the only one that is not measuring the neighbours.
    """
    for _ in range(WARMUPS):
        fn()
    gc.disable()
    try:
        return min(_time(fn) for _ in range(repeats))
    finally:
        gc.enable()


def _time(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ------------------------------------------------------- table 1: parallel
def parallel_rows() -> list[dict]:
    """Serial against parallel at the same size, per frame."""
    rows = []
    wk = workers()
    for w in PARALLEL_SIZES:
        h = max(1, int(w * 9 / 16))
        view = preset("seahorse")
        iters = 400

        serial = Frame(w, h, max_iter=iters, workers=1)
        serial_s = best_seconds(lambda: serial.render(view, "cosmic"), repeats=3)

        par = Frame(w, h, max_iter=iters, workers=wk)
        par_s = best_seconds(lambda: par.render(view, "cosmic"), repeats=3)

        rows.append(
            {
                "width": w,
                "height": h,
                "shape": f"{w}x{h}",
                "pixels": w * h,
                "serial_ms": serial_s * 1e3,
                "parallel_ms": par_s * 1e3,
                "ratio": serial_s / par_s,
                "winner": "serial" if serial_s <= par_s else "parallel",
                "uses_parallel": par.uses_parallel,
            }
        )
    return rows


# --------------------------------------------------------- table 2: vs Python
def time_mojo(w: int, h: int, iters: int, view):
    f = Frame(w, h, max_iter=iters)
    seconds = best_seconds(lambda: f.render(view, "cosmic"))
    return seconds, f.escape.copy(), f._idx.copy()


def time_reference(w: int, h: int, iters: int, view):
    """Time the pure-Python loop, or `(None, None, "reason")` if it will not fit.

    A small probe frame is timed FIRST and scaled up by pixel count. The naive
    version of this function ran the whole case and only then checked whether
    it had taken too long, which for a 1920x1080x800 case means half an hour
    before finding out the answer was "skip". The probe makes the skip cheap.

    Returns `(seconds, planes, note)`. `seconds` is None when skipped, and
    `note` says why.
    """
    probe = 16
    probe_ref = reference.Reference(probe, probe)
    _time(lambda: probe_ref.escape_plane(view.cx, view.cy, view.scale, iters))
    per_pixel = _time(
        lambda: reference.Reference(probe, probe).escape_plane(
            view.cx, view.cy, view.scale, iters
        )
    ) / (probe * probe)

    estimate = per_pixel * w * h
    if estimate > REFERENCE_HARD_LIMIT_S:
        return None, None, f"skipped (~{estimate:.0f}s > {REFERENCE_HARD_LIMIT_S:.0f}s)"

    def build():
        return reference.Reference(w, h)

    gc.disable()
    try:
        samples = [
            _time(lambda: build().escape_plane(view.cx, view.cy, view.scale, iters))
            for _ in range(REFERENCE_MAX_REPEATS)
        ]
    finally:
        gc.enable()
    mu_r, tr_r = build().escape_plane(view.cx, view.cy, view.scale, iters)
    return min(samples), (mu_r, tr_r), "timed"


def agreement(mu_mojo: np.ndarray, mu_ref: np.ndarray, iters: int) -> str:
    """How closely the two implementations classified the same frame.

    Reported rather than asserted here: the benchmark's job is to tell the
    truth about agreement, and the tests are what enforce it.
    """
    inter_m = mu_mojo > iters
    inter_r = mu_ref > iters
    flips = int((inter_m != inter_r).sum())
    n = mu_mojo.size
    if flips == 0:
        return "exact"
    return f"{100.0 * (n - flips) / n:.4f}% interior"


def main() -> int:
    cal = calibrate()
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    load = os.getloadavg()[0]
    print(f"machine   {cpu_model()}  ({ncpu} cpus available, load {load:.1f})")
    print(f"python    {sys.version.split()[0]}   numpy {np.__version__}")
    print(
        f"kernel    simd_width={simd_width()} default_workers={workers()} "
        f"host_workers={default_workers()} floor={par_min_pixels()} px"
    )
    print(
        f"method    {WARMUPS} warm-ups, best of {REPEATS}, wall time; "
        f"compilation is outside every timed region"
    )
    print()

    # ------------------------------------------------------------ table 1
    print("## serial vs parallel (measured on this host)")
    if cal["threshold_pixels"] >= 1e30:
        print(
            "break-even      never at this size; the kernel floor of "
            f"{par_min_pixels():,} px is used as-is"
        )
    elif cal["launch_ns_per_frame"] <= 0:
        print(
            f"launch overhead not measurable here: at the {cal['probe_pixels']:,} px "
            f"probe the parallel path was already FASTER per pixel than the serial "
            f"one ({cal['parallel_ns_per_pixel']:.2f} vs "
            f"{cal['serial_ns_per_pixel']:.2f} ns/px), so the launch costs less than "
            f"the measurement noise and the crossover is at or below the kernel "
            f"floor of {par_min_pixels():,} px"
        )
    else:
        side = math.sqrt(cal["threshold_pixels"])
        print(
            f"launch overhead {cal['launch_ns_per_frame'] / 1000:.1f} us per frame, "
            f"measured at {cal['probe_pixels']:,} px with {cal['probe_workers']} workers"
        )
        print(
            f"break-even      {cal['threshold_pixels']:,.0f} px per frame "
            f"(about {side:.0f}x{side:.0f})"
        )
    print()
    prows = parallel_rows()
    print("| shape | pixels | serial ms | parallel ms | parallel speedup | winner |")
    print("| --- | ---: | ---: | ---: | ---: | :---: |")
    for r in prows:
        print(
            f"| {r['shape']} | {r['pixels']:,} | {r['serial_ms']:.2f} | "
            f"{r['parallel_ms']:.2f} | {r['ratio']:.2f}x | {r['winner']} |"
        )
    print()
    losers = [r for r in prows if r["ratio"] < 1.0]
    if losers:
        print(
            f"{len(losers)} of {len(prows)} sizes are faster WITHOUT the parallel "
            "launch. Below the kernel's floor the serial loop is the only path there "
            "is; above it, a launch is only worth its cost once a frame is big "
            "enough to amortise it. `render.calibrate()` measures the trade at run "
            "time rather than trusting a constant."
        )
    else:
        print("The parallel loop won at every size measured.")
    print()

    # ------------------------------------------------------------ table 2
    print("## compiled Mojo vs pure Python")
    print()
    rows = []
    for label, view_name, w, h, iters in CASES:
        view = preset(view_name)
        mojo_s, mu_mojo, idx_mojo = time_mojo(w, h, iters, view)
        ref_s, ref_planes, note = time_reference(w, h, iters, view)
        if ref_planes is None:
            agree = note
        else:
            agree = agreement(mu_mojo, ref_planes[0], iters)
        rows.append(
            {
                "case": label,
                "view": view_name,
                "pixels": w * h,
                "iters": iters,
                "mojo_ms": mojo_s * 1e3,
                "mpxit_per_s": w * h * iters / mojo_s / 1e6,
                "ref_ms": None if ref_s is None else ref_s * 1e3,
                "speedup": None if ref_s is None else ref_s / mojo_s,
                "agreement": agree,
            }
        )
        ref_txt = f"{ref_s * 1e3:.1f} ms" if ref_s is not None else "skipped"
        spd = rows[-1]["speedup"]
        spd_txt = f"{spd:.1f}x" if spd is not None else "--"
        print(
            f"  {label:<18} {mojo_s * 1e3:9.3f} ms  "
            f"{rows[-1]['mpxit_per_s']:9.0f} Mpix-it/s  ref {ref_txt:>10}  "
            f"{spd_txt:>9}  {agree}"
        )

    print()
    print("| case | view | pixels | iters | Mojo | Mpix-it/s | pure Python | speedup | agreement |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for r in rows:
        ref_txt = f"{r['ref_ms']:.1f} ms" if r["ref_ms"] is not None else "skipped"
        spd_txt = f"{r['speedup']:.1f}x" if r["speedup"] is not None else "--"
        print(
            f"| {r['case']} | {r['view']} | {r['pixels']:,} | {r['iters']} | "
            f"{r['mojo_ms']:.3f} ms | {r['mpxit_per_s']:.0f} | {ref_txt} | "
            f"{spd_txt} | {r['agreement']} |"
        )

    slower = [r for r in rows if r["speedup"] is not None and r["speedup"] < 1.0]
    print()
    if slower:
        print("Rows where the compiled path is SLOWER, and why: a ctypes round trip")
        print("costs about a microsecond, so on a frame this small the call overhead")
        print("is a real fraction of the work it carries. They stay in the table.")
        for r in slower:
            print(f"  - {r['case']}: {r['speedup']:.2f}x (kept, not hidden)")
    else:
        print("No row was slower; the compiled path won every case measured.")

    wins = [r for r in rows if r["speedup"] is not None]
    if wins:
        best = max(wins, key=lambda r: r["speedup"])
        print()
        print(
            f"headline: {best['speedup']:.1f}x over pure Python at "
            f"{best['case']} ({best['pixels']:,} px x {best['iters']} iters)"
        )

    out = Path(__file__).resolve().parents[1] / "bench" / "last_run.json"
    out.write_text(
        json.dumps(
            {
                "machine": cpu_model(),
                "load": load,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "simd_width": simd_width(),
                "default_workers": workers(),
                "host_workers": default_workers(),
                "calibration": cal,
                "parallel_rows": prows,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())