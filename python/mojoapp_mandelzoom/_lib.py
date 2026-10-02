"""Loads (and if necessary builds) the compiled Mojo library.

Mirrors the loader style used across the mojo-* ports: a signature table, a
stale-checked build, and a single cached `ctypes.CDLL`. Buffers cross as `Int`
addresses, so the only thing this module knows about Mojo is the ABI.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOAPP_MANDELZOOM_LIB") or os.path.join(
    ROOT, "dist", "libmojoapp-mandelzoom.so"
)

I = ctypes.c_int64
D = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    # out-params: workers, min_pixels, simd_width, aspect_w, aspect_h
    "mz_info": ([I] * 5, None),
    # mu, trap, out_rgb, out_idx, w, h, cx, cy, scale, max_iter, pal,
    # pscale, repeat, trap_scale, trap_shift, trap_mode, workers
    "mz_render": ([I] * 4 + [I, I] + [D, D, D, I] + [I, D, D, D, D, I, I], I),
    "mz_escape_one": ([I, I, D, D, I], D),
    # mu, trap, y, w, h, cx, cy, scale, max_iter
    "mz_row": ([I, I, I, I, I, D, D, D, I], None),
    # out_mu, out_trap, cx, cy, scale, half_width, max_iter
    "mz_row_escape_list": ([I, I, D, D, D, I, I], I),
    # out_hist, w, h, cx, cy, scale, max_iter, workers
    "mz_iteration_histogram": ([I, I, I, D, D, D, I, I], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOAPP_MANDELZOOM_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOAPP_MANDELZOOM_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojoapp-mandelzoom.so` if stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(a) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojoapp_mandelzoom._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
