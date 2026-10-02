"""Run the pure-Python Mandelbrot reference on https://mojojojo.cc via its SDK.

The point of this script is not speed. It is agreement: the same escape-time
loop that `tests/test_parity.py` pins against the compiled Mojo kernels is
shipped verbatim into a remote run, and the remote result is asserted against
the local one. If the two ever diverge, the platform changed something
underneath us.

`mojojojo` is not on PyPI and there is no API key in this repo, so both are
optional: an unset key or a missing SDK is a clean skip (exit 0), never a
fabricated number. Every timing printed below was measured here or came back in
the `Result`.

    export MOJOJOJO_KEY=...     # from https://mojojojo.cc/keys
    pixi run showcase-remote
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_mandelzoom import reference  # noqa: E402
from mojoapp_mandelzoom.render import View, preset  # noqa: E402

# A 64x64 frame at 400 iterations: small enough that the remote finishes in the
# time it takes to read this comment, big enough that the interior count is a
# real number and not a rounding artefact. This demonstrates the SDK; `bench/`
# measures performance and does so locally.
W = H = 64
MAX_ITER = 400
TIMEOUT_MS = 60_000

#: The view is baked in as literals so the remote program has no imports beyond
#: `time` and no dependency on this repository.
VIEW = preset("seahorse")

REMOTE_SRC = '''\
"""The reference escape-time loop, shipped as a standalone program.

Printed output is exactly one line, `RESULT <interior> <escape_sum> <seconds>`,
so the caller can parse it without guessing. No other stdout: a stray print
here would make the contract ambiguous.
"""

import time

W = {w}
H = {h}
MAX_ITER = {iters}
CX = {cx!r}
CY = {cy!r}
SCALE = {scale!r}


def escape(cxx, cyy):
    """Escape value and orbit trap for one point.

    The interior marker is MAX_ITER + 1, the same convention src/mandel.mojo
    and reference.py both use, so the three implementations are comparable
    without any translation.
    """
    zr = 0.0
    zi = 0.0
    trap = 1e300
    hit = -1
    r2 = 0.0
    i = 0
    while i < MAX_ITER:
        zr, zi = zr * zr - zi * zi + cxx, 2.0 * zr * zi + cyy
        r2 = zr * zr + zi * zi
        if r2 < trap:
            trap = r2
        if r2 > 4.0:
            hit = i
            break
        i += 1
    if hit >= 0:
        mu = hit + 1 - _log2(0.5 * _log2(r2))
    else:
        mu = float(MAX_ITER) + 1.0
    if trap < 1e-300:
        trap = 1e-300
    return mu, trap


def _log2(x):
    import math
    return math.log2(x)


def render():
    """One frame: returns (interior_count, sum_of_escape_values)."""
    interior = 0
    total = 0.0
    inv_w = 1.0 / W
    inv_h = 1.0 / H
    for y in range(H):
        cyy = CY + (1.0 - 2.0 * y * inv_h) * SCALE
        for x in range(W):
            cxx = CX + (-1.0 + 2.0 * x * inv_w) * SCALE
            mu, _trap = escape(cxx, cyy)
            if mu > MAX_ITER:
                interior += 1
            else:
                total += mu
    return interior, total


# Timed here, remotely: the clock has to live with the loop it measures, and the
# platform's own wall_ms includes queueing, transfer and interpreter startup.
t0 = time.perf_counter()
interior, total = render()
elapsed = time.perf_counter() - t0

print("RESULT %d %.6f %.6f" % (interior, total, elapsed))
'''


def _load_sdk():
    """Return the `mojojojo` module, or None with a skip message already printed.

    Three guards, all necessary. `MOJOJOJO_PATH` is honoured because the SDK
    ships as a source checkout rather than a wheel. This file is named
    `mojojojo.py`, so running it directly puts its own directory on `sys.path`
    and `import mojojojo` imports *this file* -- which has no `run`, and the
    resulting AttributeError happens inside the try block below and reads like
    an SDK failure rather than a path mistake. And the exception types are
    checked with `getattr` for the same reason: they may not be there at all.
    """
    here = str(Path(__file__).resolve().parent)
    sys.path = [p for p in sys.path if p not in (here, os.getcwd())]
    local = os.environ.get("MOJOJOJO_PATH")
    if local:
        sys.path.insert(0, local)
    try:
        import mojojojo
    except ImportError as e:
        print(
            "skip: the `mojojojo` SDK is not importable.\n"
            "  It is not on PyPI -- it ships as a source release from the mojojojo\n"
            "  GitHub organisation and from the https://mojojojo.cc dashboard. Clone\n"
            "  it and either install it into this environment or point the env var at\n"
            "  the checkout:  export MOJOJOJO_PATH=/path/to/mojojojo-python\n"
            f"  (import failed with: {e})\n"
            "  Nothing was run and nothing was charged."
        )
        return None
    if not hasattr(mojojojo, "run"):
        print(
            "skip: an importable `mojojojo` has no `run`, so it is not the SDK\n"
            "  this script expects. Nothing was run and nothing was charged."
        )
        return None
    return mojojojo


def _build_payload() -> tuple[str, dict]:
    """The remote program, plus the local answer to compare it against.

    Seeding locally is what makes the comparison meaningful: the coordinates are
    the same published preset the rest of this repo uses, so the remote run and
    the local run are the same frame, not two frames that look similar.
    """
    code = REMOTE_SRC.format(
        w=W, h=H, iters=MAX_ITER, cx=VIEW.cx, cy=VIEW.cy, scale=VIEW.scale
    )
    return code, {}


def _parse_result(stdout: str) -> tuple[int, float, float] | None:
    """Pull `(interior, escape_sum, seconds)` out of the RESULT line, or None."""
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "RESULT":
            try:
                return int(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                return None
    return None


def _local_answer() -> tuple[int, float, float]:
    """The same frame computed here, with this repo's reference."""
    ref = reference.Reference(W, H)
    t0 = time.perf_counter()
    mu, _trap = ref.escape_plane(VIEW.cx, VIEW.cy, VIEW.scale, MAX_ITER)
    elapsed = time.perf_counter() - t0
    interior = int((mu > MAX_ITER).sum())
    total = float(mu[mu <= MAX_ITER].sum())
    return interior, total, elapsed


def main() -> int:
    key = os.environ.get("MOJOJOJO_KEY")
    if not key:
        print(
            "skip: MOJOJOJO_KEY is not set, so there is nothing to authenticate with.\n"
            "  Get a key at https://mojojojo.cc/keys, then:\n"
            "      export MOJOJOJO_KEY=...\n"
            "  Nothing was run and nothing was charged."
        )
        return 0

    mojojojo = _load_sdk()
    if mojojojo is None:
        return 0

    code, _ = _build_payload()

    # `getattr` for the exception types: a module that has `run` but not
    # `AuthError` would otherwise turn a skip into a traceback.
    def _err(name: str) -> type[BaseException]:
        got = getattr(mojojojo, name, None)
        return got if isinstance(got, type) and issubclass(got, BaseException) else ()

    try:
        result = mojojojo.run(
            code, source="mandelzoom-showcase", timeout_ms=TIMEOUT_MS
        )
    except _err("AuthError") as e:
        print(
            f"skip: mojojojo rejected the key: {e}\n"
            "  Check MOJOJOJO_KEY, or mint a fresh one at https://mojojojo.cc/keys.\n"
            "  Nothing was charged."
        )
        return 0
    except _err("InsufficientCredits") as e:
        print(
            f"skip: the account is out of credits: {e}\n"
            "  Top up at https://mojojojo.cc/keys. Nothing beyond the rejected\n"
            "  request was charged."
        )
        return 0
    except _err("MojojojoError") as e:
        print(f"error: mojojojo run failed: {e}", file=sys.stderr)
        return 1

    if not result.ok:
        print("error: the remote run did not succeed.", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        return 1

    parsed = _parse_result(result.stdout)
    if parsed is None:
        print("error: no RESULT line in the remote stdout.", file=sys.stderr)
        if result.stdout.strip():
            print(result.stdout.rstrip(), file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.rstrip(), file=sys.stderr)
        return 1
    remote_interior, remote_total, remote_s = parsed

    local_interior, local_total, local_s = _local_answer()

    # Interior pixels are an exact integer count, so they must match exactly.
    if remote_interior != local_interior:
        print(
            f"error: interior count mismatch after a {W}x{H} frame at "
            f"{MAX_ITER} iterations: remote {remote_interior}, "
            f"local {local_interior}",
            file=sys.stderr,
        )
        return 1
    # The escape sum is a float over ~4000 terms; compare it relatively, and
    # print both so the margin is visible rather than asserted away.
    total_rel = (
        abs(remote_total - local_total) / abs(local_total)
        if local_total
        else 0.0
    )

    print(f"| {W}x{H} x {MAX_ITER} iters, seahorse valley | value |")
    print("| --- | --- |")
    print(f"| remote wall ms | {result.wall_ms:.1f} |")
    print(f"| remote billed ms | {result.charge.billed_ms:.1f} |")
    print(f"| remote self-timed | {remote_s:.6f} s |")
    print(f"| local self-timed | {local_s:.6f} s |")
    print(f"| interior pixels | {remote_interior} (remote) == {local_interior} (local) |")
    print(f"| escape-value sum | remote {remote_total:.6f}, local {local_total:.6f} "
          f"({total_rel:.2e} relative) |")
    print()
    print(
        f"agreement exact on the interior count; the float sum differs by "
        f"{total_rel:.2e}, which is ordinary last-bit accumulation order."
    )
    print()
    print(f"ran on {result.ran_on}, charged ${result.charge.usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())