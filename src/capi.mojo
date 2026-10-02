"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address: `@export` rejects parametric
functions, and a pointer with an inferred origin is parametric. Pointers are
non-nullable here, so each is constructed inside the branch that uses it.
Callers own all memory, so nothing here allocates and nothing here can leak.
This is the only file where the Mojo divergence lives; the algorithm stays in
`mandel.mojo`.
"""

from mandel import (
    DEFAULT_CX,
    DEFAULT_CY,
    DEFAULT_SCALE,
    PAR_MIN_PIXELS,
    PAR_WORKERS,
    SIMD_WIDTH,
    _escape_scalar,
    _index_band,
    _render_band,
    _render_frame,
)

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]


@export("mz_info")
def mz_info(
    workers_out: Int, min_pixels_out: Int, simd_out: Int, w_out: Int, h_out: Int
) abi("C"):
    """Report the kernel's build-time constants through out-params.

    The host queries these instead of restating them, so a toolchain bump cannot
    silently disagree with the Python side. Passing 0 for an out-param writes
    nothing for that field.
    """
    if workers_out != 0:
        var p = Ptr(unsafe_from_address=workers_out)
        p.unsafe_store(0, Float64(PAR_WORKERS))
    if min_pixels_out != 0:
        var p = Ptr(unsafe_from_address=min_pixels_out)
        p.unsafe_store(0, Float64(PAR_MIN_PIXELS))
    if simd_out != 0:
        var p = Ptr(unsafe_from_address=simd_out)
        p.unsafe_store(0, Float64(SIMD_WIDTH))
    if w_out != 0:
        var p = Ptr(unsafe_from_address=w_out)
        p.unsafe_store(0, Float64(16))
    if h_out != 0:
        var p = Ptr(unsafe_from_address=h_out)
        p.unsafe_store(0, Float64(9))


@export("mz_render")
def mz_render_c(
    mu: Int,
    trap: Int,
    out_rgb: Int,
    out_idx: Int,
    w: Int,
    h: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    max_iter: Int,
    pal: Int,
    pscale: Float64,
    repeat: Float64,
    trap_scale: Float64,
    trap_shift: Float64,
    trap_mode: Int,
    workers: Int,
) abi("C") -> Int:
    """Render one frame into host-owned buffers. Returns 0 on success, 1 on bad
    arguments (which leaves every output untouched).

    `mu` and `trap` are `w*h` Float64 scratch planes the host supplies; the
    colour pass reads them and writes `out_rgb` (`w*h*3` UInt8) and `out_idx`
    (`w*h` UInt8). Nothing is allocated here.
    """
    if mu == 0 or trap == 0 or out_rgb == 0 or out_idx == 0 or pal == 0:
        return Int(1)
    if w < 1 or h < 1 or max_iter < 1:
        return Int(1)
    if scale <= 0.0:
        return Int(1)
    var pmu = Ptr(unsafe_from_address=mu)
    var ptrap = Ptr(unsafe_from_address=trap)
    var prgb = U8Ptr(unsafe_from_address=out_rgb)
    var pidx = U8Ptr(unsafe_from_address=out_idx)
    var ppal = U8Ptr(unsafe_from_address=pal)
    _render_frame(
        pmu,
        ptrap,
        prgb,
        pidx,
        w,
        h,
        cx,
        cy,
        scale,
        max_iter,
        ppal,
        pscale,
        repeat,
        trap_scale,
        trap_shift,
        trap_mode,
        workers,
    )
    return Int(0)


@export("mz_escape_one")
def mz_escape_one_c(
    out_mu: Int, out_trap: Int, cx: Float64, cy: Float64, max_iter: Int
) abi("C") -> Float64:
    """Escape-time for a single point via the scalar path. Returns 1 on success.

    The parity tests use this to compare the reference's per-point scalar result
    against the kernel's, without allocating a frame.
    """
    if out_mu == 0 or out_trap == 0 or max_iter < 1:
        return Float64(1.0)
    var pmu = Ptr(unsafe_from_address=out_mu)
    var ptrap = Ptr(unsafe_from_address=out_trap)
    var res = _escape_scalar(cx, cy, max_iter)
    pmu.unsafe_store(0, res[0])
    ptrap.unsafe_store(0, res[1])
    return Float64(0.0)


@export("mz_row")
def mz_row_c(
    mu: Int,
    trap: Int,
    y: Int,
    w: Int,
    h: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    max_iter: Int,
) abi("C"):
    """Escape-time for a single image row, serial. Returns 0 on success.

    Exposed so the host can profile one row against the reference loop and so
    the tests can drive the kernel a row at a time without a full frame.
    """
    if mu == 0 or trap == 0 or w < 1 or h < 1 or y < 0 or y >= h or max_iter < 1:
        return
    var pmu = Ptr(unsafe_from_address=mu)
    var ptrap = Ptr(unsafe_from_address=trap)
    _render_band(pmu, ptrap, y, y + 1, w, h, cx, cy, scale, max_iter)


@export("mz_row_escape_list")
def mz_row_escape_list_c(
    out_mu: Int,
    out_trap: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    half_width: Int,
    max_iter: Int,
) abi("C") -> Int:
    """Escape-time for `2*half_width+1` points along the real axis at `cy`.

    The centre row of the view: a cheap, deterministic probe the demo and the
    tests use to detect a broken view transform (a bad `scale` or `cx` moves
    these). Returns the number of points written, 0 on bad arguments.
    """
    if out_mu == 0 or out_trap == 0 or half_width < 0 or max_iter < 1:
        return Int(0)
    var pmu = Ptr(unsafe_from_address=out_mu)
    var ptrap = Ptr(unsafe_from_address=out_trap)
    var count = 2 * half_width + 1
    var i = 0
    while i < count:
        var px = cx + Float64(i - half_width) * scale
        var res = _escape_scalar(px, cy, max_iter)
        pmu.unsafe_store(i, res[0])
        ptrap.unsafe_store(i, res[1])
        i += 1
    return count


@export("mz_iteration_histogram")
def mz_iteration_histogram_c(
    out_hist: Int, w: Int, h: Int, cx: Float64, cy: Float64, scale: Float64,
    max_iter: Int, workers: Int,
) abi("C") -> Int:
    """Histogram of escape values into `out_hist` (`max_iter+2` UInt32 slots).

    Escape value `k` lands in bin `k`; the interior bin is `max_iter+1`. The
    sum of the histogram is exactly `w*h`, which the tests assert, so a
    mis-sized bin table cannot pass silently. Returns the number of bins.
    """
    if out_hist == 0 or w < 1 or h < 1 or max_iter < 1:
        return Int(0)
    var ph = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=out_hist)
    var bins = max_iter + 2
    var i = 0
    while i < bins:
        ph.unsafe_store(i, UInt32(0))
        i += 1
    # Bin row by row through the scalar path, which needs no scratch planes and
    # no palette. The histogram is a reporting convenience, not the timed path,
    # so it stays serial by design.
    var y = 0
    while y < h:
        _bin_row(ph, w, cx, cy, scale, max_iter, y, h)
        y += 1
    return Int(bins)


def _bin_row(
    ph: Pointer[UInt32, AnyOrigin[mut=True]],
    w: Int,
    cx: Float64,
    cy: Float64,
    scale: Float64,
    max_iter: Int,
    y: Int,
    h: Int,
):
    """Bin one row of escape values. Serial; the histogram is a reporting
    convenience, not the timed path.

    Not `raises`: nothing here can fail, and a raising body would force every
    caller to be a raising context for no reason.
    """
    var inv_h = 1.0 / Float64(h)
    var inv_w = 1.0 / Float64(w)
    var sy = 1.0 - 2.0 * Float64(y) * inv_h
    var cyy = cy + sy * scale
    var x = 0
    while x < w:
        var cxx = cx + (-1.0 + 2.0 * Float64(x) * inv_w) * scale
        var res = _escape_scalar(cxx, cyy, max_iter)
        var mv = res[0]
        var b = Int(mv)
        if b > max_iter:
            b = max_iter + 1
        if b < 0:
            b = 0
        ph.unsafe_store(b, ph.unsafe_load(b) + UInt32(1))
        x += 1
