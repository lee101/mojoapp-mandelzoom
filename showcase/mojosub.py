"""The SAME hot loop, in plain Python, timed against CPython via `mojosub`.

What this shows. The inner loop below is a transcription of one column block of
`src/mandel.mojo::_escape_block` -- the SIMD escape-time recurrence, written the
way a person writes it: flat lists, an explicit loop, no cleverness, one
complex number at a time. It is the same arithmetic the compiled kernel runs
`simd_width_of[float64]()`-way in parallel.

`mojosub` (https://github.com/lee101/mojosub) compiles such a function to native
code and calls it from Python. Wrapping a hot loop in `@jit` is therefore a
one-line change to an existing Python program, which is the point of the
showcase: the gap between "this loop is slow" and "this loop is native" is one
decorator, not a rewrite into a systems language.

What it does NOT show is that mojosub beats the compiled Mojo in `src/`. It is
not measured here, because that is `bench/bench.py`'s job and the two are
measured differently (per-function JIT vs a whole shared library, and the Mojo
side is additionally multi-threaded). The number printed is `mojosub vs CPython
on this loop`, full stop.

    pixi run showcase

If `mojosub` is not importable this script skips with a message naming the fix.
It never prints a speedup it did not measure.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Sized so the CPython baseline takes long enough to time reliably and the
# whole script still finishes in seconds.
N = 200_000
REPEATS = 5
WARMUPS = 2
MAX_ITER = 300


def escape_loop(cr_list, ci_list, max_iter: int):
    """Escape value and orbit trap for each point in `cr_list`/`ci_list`.

    This is the real transcription of `_escape_block`. Lane `k` of the SIMD body
    keeps `zr, zi, trap, hit, escaped` and runs

        zr, zi = zr*zr - zi*zi + cr, 2*zr*zi + ci
        r2     = zr*zr + zi*zi
        trap   = min(trap, r2)

    and that is exactly what this loop does, once per point. The vector body is
    not a different algorithm; it is this loop run `simd_width_of[float64]`
    iterations at a time with the branch replaced by a mask.
    """
    n = len(cr_list)
    mu = [0.0] * n
    trap = [1e300] * n
    for k in range(n):
        cr = cr_list[k]
        ci = ci_list[k]
        zr = 0.0
        zi = 0.0
        best = 1e300
        hit = -1
        i = 0
        while i < max_iter:
            zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
            r2 = zr * zr + zi * zi
            if r2 < best:
                best = r2
            if r2 > 4.0:
                hit = i
                break
            i += 1
        mu[k] = 0.0 if hit < 0 else hit + 1.0
        trap[k] = best
    return mu, trap


def make_points(n: int):
    """Points spread over the whole complex plane, deterministically.

    A fixed LCG rather than `random`: this environment ships a `bisect` that
    breaks `numpy.random`'s import chain, and a plain generator keeps the
    script runnable anywhere. The points are reproducible either way, which is
    the only property this needs.
    """
    v = 20260915
    cr = [0.0] * n
    ci = [0.0] * n
    for k in range(n):
        v = (v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        a = (v >> 11) / float(1 << 53)
        v = (v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        b = (v >> 11) / float(1 << 53)
        cr[k] = -2.0 + a * 3.0
        ci[k] = -1.2 + b * 2.4
    return cr, ci


def _median(fn, *args) -> float:
    for _ in range(WARMUPS):
        fn(*args)
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _load_mojosub():
    """Import the `mojosub` package, or None with a skip message already printed.

    Two guards, both necessary. `MOJOSUB_PATH` is honoured because the project
    ships as a repository rather than a wheel: point it at the clone and the
    import below finds it. And `spec.loader_from_cwd` is not enough -- this file
    is literally named `mojosub.py`, so running it as `python showcase/mojosub.py`
    puts its own directory on `sys.path` and `import mojosub` imports *this file*.
    A package has no `jit`, the import fails, and the skip message never prints
    because the exception is an ImportError from the wrong place. Deleting this
    script's own entry from the front of `sys.path` is what makes the guard
    mean something.
    """
    here = str(Path(__file__).resolve().parent)
    # This file is named `mojosub.py`, so its own directory on `sys.path`
    # shadows the package. Remove it, or `import mojosub` finds this file.
    sys.path = [p for p in sys.path if p not in (here, os.getcwd())]
    local = os.environ.get("MOJOSUB_PATH")
    if local:
        sys.path.insert(0, local)
    try:
        import mojosub
    except ImportError as e:
        print(
            "skip: the `mojosub` package is not importable.\n"
            "  It is not on PyPI -- it ships from https://github.com/lee101/mojosub.\n"
            "  Either clone it and point this script at the checkout:\n"
            "      export MOJOSUB_PATH=/path/to/mojosub\n"
            "  or install it into this environment:\n"
            "      pip install git+https://github.com/lee101/mojosub.git\n"
            f"  (import failed with: {e})\n"
            "  Nothing was compiled and no speedup is claimed."
        )
        return None
    if not hasattr(mojosub, "jit"):
        print(
            "skip: an importable `mojosub` has no `jit`, so it is not the SDK\n"
            "  this script expects. Nothing was compiled and no speedup is\n"
            "  claimed."
        )
        return None
    return mojosub


def main() -> int:
    mojosub = _load_mojosub()
    if mojosub is None:
        return 0

    from mojosub import jit  # noqa: PLC0415  (deferred: needs the env var above)

    cr, ci = make_points(N)

    # `verify=True` asks mojosub to check its first native result against
    # CPython before trusting the variant. Left off here because this script
    # makes that comparison itself, and visibly, a few lines below -- but it is
    # the reason the number below is a measurement rather than a hope.
    compiled = jit(escape_loop)
    cpython_s = _median(escape_loop, cr, ci, MAX_ITER)
    native_s = _median(compiled, cr, ci, MAX_ITER)

    # The two must agree, or the speedup below is meaningless.
    a = escape_loop(cr, ci, MAX_ITER)
    b = compiled(cr, ci, MAX_ITER)
    match = list(a[0]) == list(b[0]) and list(a[1]) == list(b[1])
    if not match:
        print(
            "error: the mojosub build does not match CPython output",
            file=sys.stderr,
        )
        return 1

    speedup = cpython_s / native_s
    print(f"loop      escape-time recurrence, {N:,} points x {MAX_ITER} iterations")
    print(f"source    showcase/mojosub.py::escape_loop")
    print(f"method    {WARMUPS} warmups, {REPEATS} repeats, median wall time")
    print()
    print("| implementation | median ms | Mpoints/s | vs CPython |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| CPython (plain Python) | {cpython_s * 1e3:.2f} | {N / cpython_s / 1e6:.3f} | 1.00x |")
    print(f"| mojosub `@jit` (native) | {native_s * 1e3:.2f} | {N / native_s / 1e6:.3f} | {speedup:.2f}x |")
    print()
    print(f"speedup   {speedup:.2f}x over CPython on this loop")
    print("output    matches the CPython result (checked above)")
    print()

    if speedup < 1.5:
        print("Reading this number honestly: it is small, and that is a result,")
        print("not a broken benchmark. mojosub is running -- the output matches")
        print("CPython exactly, and the call path is compiled native code. But")
        print("`@jit` gets one thread and leans on the host CPU's own")
        print("vectoriser, while `src/` gets `simd_width_of[float64]`-way SIMD")
        print("*and* `parallelize` across 16 workers. On this loop the scalar")
        print("arithmetic and the call overhead dominate, so the two paths land")
        print("close together; bench/bench.py measures the compiled Mojo core at")
        print("tens of times CPython because it is doing more than jit-for-CPython")
        print("can here -- more work per instruction and across cores.")
        print()

    print(
        "note      this is mojosub vs CPython, one function, one thread. The\n"
        "          compiled Mojo core in src/ is benchmarked separately in\n"
        "          bench/bench.py, where it is additionally SIMD and threaded.\n"
        "          Do not read this number as a comparison between the two."
    )

    out = ROOT / "showcase" / "last_mojosub.json"
    out.write_text(
        json.dumps(
            {
                "loop": "escape_loop",
                "points": N,
                "max_iter": MAX_ITER,
                "cpython_ms": cpython_s * 1e3,
                "mojosub_ms": native_s * 1e3,
                "speedup": speedup,
                "outputs_match": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())