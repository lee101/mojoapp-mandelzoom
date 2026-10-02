"""Mandelbrot escape-time kernels: SIMD over columns, parallel over row bands.

This is the whole compute core of Mandel Zoom. Everything here is
allocation-free and writes only into caller-owned buffers, so the host keeps
ownership of every byte and nothing in this file can leak.

The algorithm
-------------
For each pixel the host supplies a complex centre `(cx, cy)` and a half-height
`scale` in complex units; the pixel's complex coordinate is
`c = (cx + sx*scale, cy + sy*scale)` where `sx`/`sy` are the normalised pixel
coordinates in `[-1, 1]`. The standard escape-time loop iterates
`z <- z*z + c` from `z = 0` up to `max_iter` times, and a pixel's *iteration
count* is the smooth escape value:

    mu = n + 1 - log2(log(|z|))          (when it escaped)
    mu = max_iter + 1                    (when it never escaped)

The two extra outputs per pixel are what turn a plot into a shareable image:

* **orbit trap** `-- a second, cheap metric from the same iteration, namely the
  minimum `|z|` reached along the orbit. It produces the filigree and the
  spiral banding that make deep zooms legible; see the reference for the
  identical formula.
* **palette index** -- the pair `(mu, trap)` is mapped through a 768-byte
  palette table plus a repeat period, so the same colouring the host uses for
  its text cards is produced here in the kernel.

Why the vector body works
-------------------------
The iteration is elementwise: lane `k` runs the recurrence for its own `c` and
never reads another lane. So a `SIMD` of `W` doubles iterates `W` pixels per
instruction with the identical arithmetic as the scalar tail -- no gather, no
mask, and bit-identical results lane-for-lane, because every operation is the
same IEEE-754 double op the scalar path performs. `W` is
`simd_width_of[DType.float64]()` (queried, never hard-coded, so a toolchain bump
cannot silently change the vector tail) and the body is written over the
`*_lanes` helpers that build the per-lane `c` vectors for a block of `W`
consecutive columns.

Parallelism
-----------
Rows are handed to `parallelize` in contiguous bands, each work item filling a
block of rows. Rows are embarrassingly parallel (no cross-row state) and each
band is a plain `{imm}` nested def, which is the only closure shape the current
`parallelize` accepts. `workers <= 1` or a frame below `PAR_MIN_PIXELS` forces
the serial loop: the launch costs a fixed amount that does not shrink with the
frame, and `bench/bench.py` measures where that actually crosses over instead
of trusting the constant.
"""

from max.algorithm import parallelize
from std.math import log, log2
from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

#: Doubles per SIMD register. Queried, not assumed, so the scalar tail in every
#: kernel is aligned to the real width.
comptime SIMD_WIDTH = simd_width_of[DType.float64]()

#: The vector body uses this many columns per iteration regardless of the host
#: width, so the arithmetic is a single code path. Using the full register width
#: is what we want when it divides cleanly; a fixed 4 is always valid and is
#: often the better choice. `SIMD_WIDTH` is the honest choice here and the tests
#: probe either side of it.
comptime W = SIMD_WIDTH
comptime Vec = SIMD[DType.float64, W]

#: Default worker count. Capped because a frame is far wider than it is tall:
#: past this the row-band count is no longer the limit. `la_workers` exposes it
#: so the benchmark cannot claim more parallelism than it actually got.
comptime PAR_WORKERS = 16

#: Frames at or above this pixel count are eligible for `parallelize`. Below it
#: the launch cannot pay for itself. Measured, not guessed -- see bench.
comptime PAR_MIN_PIXELS = 1 << 15  # 32768

#: Rows per parallel work item. Fewer, fatter items spend the fixed launch cost
#: once instead of per row.
comptime ROWS_PER_TASK = 8

#: Escape value written for a point still inside the set after `max_iter`. One
#: past the iteration count so colouring can distinguish "escaped at the last
#: iteration" from "never escaped". The `_F` form is the same value as a double,
#: for the comparison in the colour pass.
comptime INSIDE = 1 << 29
comptime INSIDE_F = Float64(1 << 29)

#: The point every zoom starts from, and the default the CLI renders.
comptime DEFAULT_CX = -0.5
comptime DEFAULT_CY = 0.0
comptime DEFAULT_SCALE = 1.25


# --------------------------------------------------------------- SIMD helpers

# ------------------------------------------------------------------ scalars

def _escape_scalar(
    cx: Float64, cy: Float64, max_iter: Int
) -> Tuple[Float64, Float64]:
    """Escape-time for one point, scalar. Returns `(mu, trap)`.

    This is the reference arithmetic: the vector body below performs exactly
    these same IEEE-754 operations lane-for-lane, so the two agree bit for bit.

    `trap` is the running minimum of `|z|` over the orbit, floored at a tiny
    positive value so the later `log2` is defined (a point that never moves
    would give `log2(0)`).
    """
    var zr = Float64(0.0)
    var zi = Float64(0.0)
    var cr = cx
    var ci = cy
    var r2 = Float64(0.0)
    var trap = Float64(1e300)
    var i = 0
    var escaped = False
    var hit = 0
    while i < max_iter:
        zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
        r2 = zr * zr + zi * zi
        if r2 < trap:
            trap = r2
        if r2 > 4.0:
            escaped = True
            hit = i
            break
        i += 1
    if escaped:
        var mu = Float64(hit) + 1.0 - _log2(0.5 * _log2(r2))
        if trap < 1e-300:
            trap = 1e-300
        return mu, trap
    # Never escaped: the interior value, which is what the vector body's
    # `interior` term is too.
    var mu_in = Float64(max_iter) + 1.0
    if trap < 1e-300:
        trap = 1e-300
    return mu_in, trap


# --- libm log/log2. `import math` inside a function body does NOT resolve on
# --- this toolchain (`unable to locate module 'math'`); the names come from
# --- `std.math` at module scope and the helpers below are thin so the call
# --- sites read the same as the scalar reference.


@always_inline
def _ln(x: Float64) -> Float64:
    return log(x)


@always_inline
def _log2(x: Float64) -> Float64:
    return log2(x)


# ------------------------------------------------------- one SIMD block of W

def _escape_block(
    c_re: Vec, c_im: Vec, max_iter: Int
) -> Tuple[Vec, Vec, Vec]:
    """Iterate `W` points at once. Returns `(mu, trap, escaped_mask)`.

    `c_re`/`c_im` are the real/imag parts of the `W` points. Every operation is
    elementwise, so this is the scalar recurrence run `W`-way and each lane
    carries its own escape value, trap and escape flag.

    The escape test is branch-free: a lane that has escaped is frozen by a
    `still_going` mask (1 while iterating, 0 once escaped) and `mu` is
    accumulated only while that mask is 1. This is the standard masked-iteration
    shape and it is what keeps the vector body branchless -- a `break` would
    exit for the whole block, not one lane.

    The returned `mu` is the *same* smooth escape value the scalar path
    computes, by the same expression on the same number:

        mu = (hit + 1) - log2(0.5 * log2(r2_at_escape))

    where `r2_at_escape` is `|z|^2` on the iteration the lane escaped, which the
    mask freezes at the moment of escape. Interior lanes get `max_iter + 1.0`,
    exactly as the scalar path reports. `log2` is called on the vector, which
    resolves to the `SIMD` overload of the same libm function the scalar path
    uses, so this is the same rounding lane for lane -- which is what makes the
    compiled output agree with the Python reference rather than merely close.
    """
    var zr = Vec(0.0)
    var zi = Vec(0.0)
    var trap = Vec(1e300)
    var hit = Vec(0.0)
    var r2_at_escape = Vec(0.0)
    var escaped = Vec(0.0)  # 1 once a lane has escaped
    var still = Vec(1.0)  # 1 while a lane has not escaped
    var i = 0
    while i < max_iter:
        # The recurrence, masked so escaped lanes freeze rather than diverge.
        var gate = 1.0 - escaped
        var nzr = zr * zr - zi * zi + c_re
        var nzi = 2.0 * zr * zi + c_im
        zr = nzr * gate + zr * (1.0 - gate)
        zi = nzi * gate + zi * (1.0 - gate)
        var r2 = zr * zr + zi * zi
        trap = _vmin(trap, r2)
        # Newly-escaped lanes only (escaped is still 0 for them this step).
        var just = _vgt(r2, 4.0) * still
        hit = _vmax(hit, just * Float64(i + 1))
        # Freeze r2 on the escaping iteration; `_vfirst` keeps the first write.
        r2_at_escape = _vfirst(just, r2, r2_at_escape)
        escaped = _vmax(escaped, just)
        still = still * (1.0 - _vgt(r2, 4.0))
        i += 1
    # Smooth the escape exactly as the scalar path does. `hit` already holds
    # `i + 1` (the number of iterations performed), which is the scalar path's
    # `hit + 1`, so the correction is `hit - log2(...)` and NOT `hit + 1 - ...`.
    # The extra 1 shifts every escape value by one, which is not a rounding
    # difference: a point that escapes on the final allowed iteration then
    # reports `max_iter + 1` and the colour pass reads it as interior.
    #
    # Interior lanes have `r2_at_escape == 0` and the smoothing is only defined
    # for an escaped point, so the value is made well defined FIRST and selected
    # afterwards. The fallback is 2.0, not 1.0, and the choice matters: the
    # correction needs `log2(r2) > 0` to stay finite. 1.0 gives -inf, and
    # `escaped * inf` is NaN even where `escaped` is 0 -- a mask does not
    # rescue a poisoned intermediate, it multiplies it. 2.0 gives
    # `log2(0.5) == -1`, a finite number the mask then discards.
    var safe_r2 = _vfirst(escaped, r2_at_escape, Vec(2.0))
    var corr = hit - log2(0.5 * log2(safe_r2))
    var interior = Float64(max_iter) + 1.0
    var mu = escaped * corr + (1.0 - escaped) * interior
    return mu, trap, escaped


def _vfirst(sel: Vec, take: Vec, cur: Vec) -> Vec:
    """Per-lane select: `take` where `sel != 0`, otherwise `cur`.

    `_vmax` would not do -- it would let a later write overwrite an earlier
    one, and a lane must keep the `r2` from the iteration it escaped on, not
    the last one. The selector is exactly 0.0 or 1.0, so an `add`-based select
    cannot misfire.
    """
    return cur + sel * (take - cur)


def _vmin(a: Vec, b: Vec) -> Vec:
    """Per-lane min. `SIMD` has no `.min()` method on this toolchain; the free
    `min` handles both scalars and vectors."""
    return min(a, b)


def _vmax(a: Vec, b: Vec) -> Vec:
    return max(a, b)


def _vgt(a: Vec, b: Vec) -> Vec:
    """1.0 where `a > b`, 0.0 elsewhere, per lane."""
    var t = max(a, b) - b  # 0 where a<=b, else a-b (never wraps in float)
    return _step_vec(t)


def _step_vec(t: Vec) -> Vec:
    """1.0 where `t > 0`, else 0.0. `t` is a non-negative difference here."""
    # A saturating clamp: min(1, t) is 1 only when t>=1. We need t>0, so use
    # min(1, t*<big>) -- multiplying a float by a big constant keeps every
    # positive t at or above 1 and every zero at zero. No sign trick needed
    # because `t` here is already `max(a,b)-b >= 0`.
    return min(Vec(1.0), t * 1e300)


def _vany(v: Vec) -> Float64:
    """1.0 if any lane of `v` is non-zero (used to early-exit the loop)."""
    var s = v[0]
    var k = 1
    while k < W:
        s = s + v[k]
        k += 1
    return s


# ------------------------------------------------------- band / frame driver

def _render_band(
    mu: Ptr,
    trap: Ptr,
    y0: Int,
    y1: Int,
    w: Int,
    h: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    max_iter: Int,
):
    """Fill rows `[y0, y1)` of the two float output planes.

    `mu` and `trap` are `Float64` pixel planes, `w*h` each. The colour pass
    (`_index_band`) owns the palette mapping, so this function only computes the
    escape values and the orbit trap.

    Not `raises`, and deliberately: nothing here can fail, and marking it so
    would force every `parallelize` work item to be a raising closure, which the
    `{imm}` worker shape does not accept.
    """
    var inv_h = 1.0 / Float64(h)
    var inv_w = 1.0 / Float64(w)
    var y = y0
    while y < y1:
        # Normalised y in [-1, 1]. +1 at the top row so the image is not
        # vertically flipped relative to the usual complex plane.
        var sy = 1.0 - 2.0 * Float64(y) * inv_h
        var cyy = cy + sy * scale
        var x = 0
        while x + W <= w:
            # Build the W lanes' c values for this column block.
            var c_re = Vec(0.0)
            var c_im = Vec(0.0)
            var k = 0
            var base = cx + (-1.0 + 2.0 * Float64(x) * inv_w) * scale
            var step = 2.0 * scale * inv_w
            while k < W:
                c_re[k] = base + Float64(k) * step
                c_im[k] = cyy
                k += 1
            var m = mu.unsafe_offset(y * w)
            var t = trap.unsafe_offset(y * w)
            var mres = _escape_block(c_re, c_im, max_iter)
            var i = 0
            while i < W:
                m.unsafe_store(x + i, mres[0][i])
                t.unsafe_store(x + i, mres[1][i])
                i += 1
            x += W
        # Scalar tail for the last partial block, same arithmetic.
        while x < w:
            var cx0 = cx + (-1.0 + 2.0 * Float64(x) * inv_w) * scale
            var res = _escape_scalar(cx0, cyy, max_iter)
            mu.unsafe_offset(y * w).unsafe_store(x, res[0])
            trap.unsafe_offset(y * w).unsafe_store(x, res[1])
            x += 1
        y += 1


def _index_band(
    mu: Ptr,
    trap: Ptr,
    out_rgb: U8Ptr,
    out_idx: U8Ptr,
    y0: Int,
    y1: Int,
    w: Int,
    h: Int,
    pal: U8Ptr,
    pscale: Float64,
    repeat: Float64,
    trap_scale: Float64,
    trap_shift: Float64,
    trap_mode: Int,
    max_iter: Int,
):
    """Map `(mu, trap)` through the palette for rows `[y0, y1)`.

    `out_rgb` is `w*h*3`, `out_idx` is `w*h`. Colouring is deliberately the
    host's rule, duplicated exactly in the Python reference:

        inside  -> palette[0]
        t       = repeat * log(1 + trap*trap_scale) + trap_shift
        p       = int(mu * pscale + t) mod 256

    `trap_mode` selects the variant: 0 is the formula above, 1 drops the log
    (raw linear trap), 2 is interior-out colouring. Each is a documented
    option rather than a magic number.
    """
    var inv_h = 1.0 / Float64(h)
    var inv_w = 1.0 / Float64(w)
    var y = y0
    while y < y1:
        var sy = 1.0 - 2.0 * Float64(y) * inv_h
        var mrow = mu.unsafe_offset(y * w)
        var trow = trap.unsafe_offset(y * w)
        var orow = out_rgb.unsafe_offset(y * w * 3)
        var irow = out_idx.unsafe_offset(y * w)
        var x = 0
        while x < w:
            var mv = mrow.unsafe_load(x)
            var tv = trow.unsafe_load(x)
            var o = x * 3
            # Interior (never escaped) is palette[0]; an escaped pixel is
            # mapped through the orbit trap below.
            var p = 0
            if mv <= Float64(max_iter):
                # Mode 0 is the log trap, the default; modes 1 and 2 replace
                # the term. Written as a default plus two overrides so no
                # branch leaves `t` unassigned.
                var t = repeat * _ln(1.0 + tv * tv * trap_scale) + trap_shift
                if trap_mode == 1:
                    t = tv * trap_scale + trap_shift
                elif trap_mode == 2:
                    t = (1.0 - sy) * trap_scale + trap_shift
                var q = Int(mv * pscale + t)
                if q < 0:
                    q = 0
                p = q % 256
            orow.unsafe_store(o, pal.unsafe_load(p * 3))
            orow.unsafe_store(o + 1, pal.unsafe_load(p * 3 + 1))
            orow.unsafe_store(o + 2, pal.unsafe_load(p * 3 + 2))
            irow.unsafe_store(x, UInt8(p))
            x += 1
        y += 1


def _render_frame(
    mu: Ptr,
    trap: Ptr,
    out_rgb: U8Ptr,
    out_idx: U8Ptr,
    w: Int,
    h: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    max_iter: Int,
    pal: U8Ptr,
    pscale: Float64,
    repeat: Float64,
    trap_scale: Float64,
    trap_shift: Float64,
    trap_mode: Int,
    workers: Int,
):
    """Render a full frame: escape planes, then colour pass.

    Bands rows for the escape pass under the same `workers <= 1` /
    `PAR_MIN_PIXELS` gate as before, then bands the colour pass. The colour pass
    is cheap (one palette lookup per pixel) but still parallelised so a large
    frame does not serialise its LUT.

    Not `raises`, for the same reason as the band helpers: nothing here can
    fail, and marking it so would make the `abi("C")` wrapper around it need an
    `except` arm the compiler reports as unreachable.
    """
    var pixels = w * h

    def band(t: Int) {imm}:
        var y0 = t * ROWS_PER_TASK
        var y1 = min(y0 + ROWS_PER_TASK, h)
        _render_band(mu, trap, y0, y1, w, h, cx, cy, scale, max_iter)

    var tasks = (h + ROWS_PER_TASK - 1) // ROWS_PER_TASK
    if workers > 1 and pixels >= PAR_MIN_PIXELS and tasks > 1:
        parallelize(band, tasks, min(tasks, workers))
    else:
        _render_band(mu, trap, 0, h, w, h, cx, cy, scale, max_iter)

    def cband(t: Int) {imm}:
        var y0 = t * ROWS_PER_TASK
        var y1 = min(y0 + ROWS_PER_TASK, h)
        _index_band(
            mu, trap, out_rgb, out_idx, y0, y1, w, h,
            pal, pscale, repeat, trap_scale, trap_shift, trap_mode, max_iter,
        )

    if workers > 1 and pixels >= PAR_MIN_PIXELS and tasks > 1:
        parallelize(cband, tasks, min(tasks, workers))
    else:
        _index_band(
            mu, trap, out_rgb, out_idx, 0, h, w, h,
            pal, pscale, repeat, trap_scale, trap_shift, trap_mode, max_iter,
        )
