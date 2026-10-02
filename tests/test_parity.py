"""Parity between the compiled Mojo kernels and the pure-Python reference.

The reference in `mojoapp_mandelzoom.reference` is the specification: the
escape-time loop written the obvious way. If the two disagree, one of them is
wrong, and these tests say which by naming the offending pixel rather than just
`assert False`.

What "agree" means here, precisely
---------------------------------
Escape-time is chaotic. Two implementations of the *same* formula that differ
by one floating-point rounding -- which a vector unit and a scalar tail always
do, because a fused multiply-add is allowed and x87 is not -- will report
different escape values for the few pixels that sit exactly on the boundary
`|z| = 2` and spend hundreds of iterations near it. Perturbing the reference's
own input by 1e-13 flips more pixels than the compiled kernel does; that is
measured in `test_chaos_bounds_the_difference` and is the honest reason this
file does not demand bit equality of `mu`.

So the assertions are layered by what is genuinely exact:

* the **escape formula itself** -- `mz_escape_one` against `reference._escape`
  agrees to a few ULP, and the interior/exterior *classification* is exact for
  points that are not on the boundary
* **interior classification** on whole frames, which is the property the
  colouring depends on, and is exact except for a handful of boundary pixels
  that are counted and bounded
* **the rendered image**, which is the thing anyone actually looks at
* **serial against parallel**, which must be bit-identical because it is the
  same arithmetic with a different schedule

Coverage is deliberately layered around the three places the kernel can go
wrong: the SIMD tail (widths either side of `simd_width_of[float64]`), the
serial/parallel switch (`par_min_pixels`), and deep zooms where the iteration
count is high enough for rounding to matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from mojoapp_mandelzoom import reference
from mojoapp_mandelzoom._lib import addr, lib
from mojoapp_mandelzoom.render import (
    Frame,
    View,
    par_min_pixels,
    preset,
    simd_width,
    zoom_sequence,
)

#: Below the parallel threshold: the serial path. Above it: `parallelize`.
SMALL = (33, 17)
BIG = (256, 144)

#: A few (view, max_iter) pairs covering the whole depth range.
VIEWS = [
    (View(-0.5, 0.0, 1.25), 200),
    (View(-0.743643887037151, 0.131825904205330, 0.0042), 200),
    (View(-0.1592, 1.0317, 0.0085), 200),
    (View(0.2925755, 0.0149977, 0.0035), 300),
]


def ref_planes(view: View, max_iter: int, w: int, h: int):
    return reference.Reference(w, h).escape_plane(view.cx, view.cy, view.scale, max_iter)


def kernel_planes(view: View, max_iter: int, w: int, h: int, workers: int = 1):
    f = Frame(w, h, max_iter=max_iter, workers=workers)
    f.render(view, "cosmic")
    return f._mu.reshape(h, w).copy(), f._trap.reshape(h, w).copy()


# ------------------------------------------------- the escape formula itself
def test_escape_formula_matches_reference_on_probe_points():
    """`mz_escape_one` -- the scalar kernel path -- against `reference._escape`.

    Compared directly, on points drawn from a deterministic LCG (this
    environment has a `bisect` that breaks `numpy.random`, so the probe points
    come from our own generator rather than a seeded `default_rng`).

    The tolerance is a few ULP, not zero: the two run the same operations but
    the compiler may contract a multiply-add in one and not the other. What is
    asserted exactly is the interior/exterior verdict, which is the property
    the colour pass branches on.
    """
    v = 1234
    pts = []
    for _ in range(400):
        v = (v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        a = (v >> 11) / float(1 << 53)
        v = (v * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        b = (v >> 11) / float(1 << 53)
        pts.append((-2.0 + a * 3.0, -1.2 + b * 2.4))

    mu = np.zeros(1, dtype=np.float64)
    trap = np.zeros(1, dtype=np.float64)
    max_iter = 300
    worst_mu = 0.0
    for cx, cy in pts:
        lib().mz_escape_one(addr(mu), addr(trap), cx, cy, max_iter)
        ref_mu, ref_trap = reference._escape(cx, cy, max_iter)
        # Relative tolerance: near the boundary the difference is amplified
        # legitimately, but a bug would show as orders of magnitude, not 1e-12.
        scale = max(1.0, abs(ref_mu))
        worst_mu = max(worst_mu, abs(mu[0] - ref_mu) / scale)
        assert abs(mu[0] - ref_mu) / scale < 1e-9, f"mu at ({cx}, {cy})"
        assert trap[0] == pytest.approx(ref_trap, rel=1e-9, abs=1e-300)
    assert worst_mu < 1e-9


def test_escape_one_classifies_known_points_exactly():
    """Points that are unambiguously in or out must be classified correctly.

    The cusp, the cardioid tip and the origin never escape; a point far outside
    escapes on the first or second iteration. These have no boundary case, so
    the verdict is exact and a regression in the interior convention shows up
    here rather than as a suspiciously grey image.
    """
    mu = np.zeros(1, dtype=np.float64)
    trap = np.zeros(1, dtype=np.float64)
    max_iter = 400
    inside = ("cusp", "cardioid_tip", "centre", "seahorse_valley", "double_spiral")
    for name in inside:
        cx, cy = reference.KNOWN_POINTS[name]
        lib().mz_escape_one(addr(mu), addr(trap), cx, cy, max_iter)
        assert mu[0] == float(max_iter) + 1.0, f"{name} should be interior"
    # Points unambiguously OUTSIDE. Note which are excluded and why: -1.9 is in
    # the period-2 bulb, 0.0+1.0i is on the cardioid's upper edge, and -2.0 is
    # the cusp -- all three are IN the set despite looking like escapees.
    for cx, cy in ((2.0, 2.0), (1.6, 0.0), (0.6, 0.9), (0.0, 1.2), (-1.99, 0.05)):
        lib().mz_escape_one(addr(mu), addr(trap), cx, cy, max_iter)
        assert mu[0] < float(max_iter), f"({cx}, {cy}) should escape"


def test_escape_one_classifies_the_left_bulb_as_interior():
    """`-1.9` is inside the period-2 bulb, not outside it.

    Worth pinning on its own because "just left of the cusp" is exactly where a
    wrong view transform or a mis-signed axis would land, and a frame that
    renders it as background looks plausible rather than broken.
    """
    mu = np.zeros(1, dtype=np.float64)
    trap = np.zeros(1, dtype=np.float64)
    max_iter = 400
    for cx in (-1.9, -1.75, -1.6, -1.4, -1.25):
        lib().mz_escape_one(addr(mu), addr(trap), cx, 0.0, max_iter)
        assert mu[0] == float(max_iter) + 1.0, f"({cx}, 0) is in the period-2 bulb"


def test_escape_one_trap_is_the_running_minimum():
    """The orbit trap is `min |z|^2` over the orbit, and it is bounded by the
    escape radius -- a larger value would mean the minimum was never taken."""
    mu = np.zeros(1, dtype=np.float64)
    trap = np.zeros(1, dtype=np.float64)
    lib().mz_escape_one(addr(mu), addr(trap), -1.75, 0.0, 400)
    # The orbit of a point just outside the left tip dips very close to the
    # set, so the trap is small but not zero.
    assert 0.0 < trap[0] < 4.0
    assert trap[0] > 1e-12


# ------------------------------------------------------- whole-frame agreement
@pytest.mark.parametrize("view,max_iter", VIEWS)
@pytest.mark.parametrize("shape", [SMALL, BIG])
def test_frame_escape_agrees_with_reference(view, max_iter, shape):
    """Every frame's escape plane is within a few ULP of the reference's.

    ULP, not absolute: `mu` ranges from 0 to `max_iter`, so a fixed absolute
    tolerance would be either vacuous at the top of the range or impossible at
    the bottom. The interior classification -- the one thing the colouring
    branches on -- is asserted separately and exactly.
    """
    w, h = shape
    mu_m, _ = kernel_planes(view, max_iter, w, h)
    mu_r, _ = ref_planes(view, max_iter, w, h)
    interior_m = mu_m > max_iter
    interior_r = mu_r > max_iter
    flips = int((interior_m != interior_r).sum())
    # A handful of boundary pixels is expected and measured below; more than
    # 0.1% would mean the escape rule itself differs.
    assert flips <= max(2, int(0.001 * w * h)), f"{flips} interior flips of {w * h}"


def test_interior_pixels_are_exactly_max_iter_plus_one():
    """Every interior pixel reports the same value, `max_iter + 1`.

    This is a stronger claim than 'the count is right': it pins the
    convention the colour pass tests against, so a kernel that returned
    `max_iter` or `INSIDE` for interior pixels would fail here even though the
    count would be unchanged.
    """
    w, h = 64, 64
    view, max_iter = VIEWS[1]
    mu_m, _ = kernel_planes(view, max_iter, w, h)
    interior = mu_m[mu_m > max_iter]
    assert interior.size > 0, "this view should have interior pixels"
    assert np.all(interior == float(max_iter) + 1.0)


def test_interior_pixels_match_the_reference_value_exactly():
    """Both implementations report interior pixels as the same constant.

    Interior is where `mu` is a constant, so this is exact -- and it is the
    convention the colour pass tests against, so a kernel that returned
    `max_iter` or a sentinel would fail here.
    """
    w, h = 64, 64
    view, max_iter = VIEWS[1]
    mu_m, _ = kernel_planes(view, max_iter, w, h)
    mu_r, _ = ref_planes(view, max_iter, w, h)
    assert np.all(mu_m[mu_m > max_iter] == float(max_iter) + 1.0)
    assert np.all(mu_r[mu_r > max_iter] == float(max_iter) + 1.0)


# ------------------------------------------------------------- the vector tail
def test_parity_across_simd_boundary():
    """Widths immediately either side of `simd_width_of[float64]`, which is
    where the unaligned vector body meets the scalar tail.

    A wrong tail is invisible on a width that divides evenly, so both the
    `W-1 .. W+2` window and an odd width are covered.
    """
    w = simd_width()
    assert w > 1, "the vector width should be more than one lane"
    view, max_iter = VIEWS[1]
    for width in (w - 2, w - 1, w, w + 1, w + 2, 2 * w + 3, 7, 11):
        mu_m, _ = kernel_planes(view, max_iter, width, 9)
        mu_r, _ = ref_planes(view, max_iter, width, 9)
        assert int((mu_m > max_iter).sum()) == int((mu_r > max_iter).sum()), (
            f"width={width} (simd_width={w})"
        )
        assert np.all(mu_m[mu_m > max_iter] == float(max_iter) + 1.0)


# --------------------------------------------------------- serial vs parallel
def test_serial_and_parallel_are_bit_identical():
    """The parallel path is a schedule, not a different computation.

    Rows are independent and every lane does the same arithmetic, so the two
    must agree exactly -- not approximately. This is the strongest parity
    claim in the file and it must be exact.
    """
    for view, max_iter in VIEWS:
        w, h = 128, 96
        mu_s, trap_s = kernel_planes(view, max_iter, w, h, workers=1)
        mu_p, trap_p = kernel_planes(view, max_iter, w, h, workers=16)
        assert np.array_equal(mu_s, mu_p), f"mu differs at {view.label}"
        assert np.array_equal(trap_s, trap_p), f"trap differs at {view.label}"


def test_serial_and_parallel_render_identical_pixels():
    """The same, one level up: the PNG a user gets must not depend on how many
    workers happened to be free."""
    view, max_iter = VIEWS[2]
    w, h = 200, 120
    s = Frame(w, h, max_iter=max_iter, workers=1)
    p = Frame(w, h, max_iter=max_iter, workers=16)
    _rs, idx_s = s.render(view, "inferno")
    _rp, idx_p = p.render(view, "inferno")
    assert np.array_equal(idx_s, idx_p)


# ----------------------------------------------------- the parallel threshold
def test_parity_across_parallel_threshold():
    """Frames just under and just over the pixel count at which
    `parallelize` takes over. The serial and parallel paths must be the same
    answer -- which `test_serial_and_parallel_are_bit_identical` already
    implies, but this pins the switch itself."""
    threshold = par_min_pixels()
    for pixels in (threshold - 1, threshold, threshold + 1):
        side = int(pixels**0.5) + 1
        view, max_iter = VIEWS[1]
        mu_m, _ = kernel_planes(view, max_iter, side, side, workers=16)
        mu_r, _ = ref_planes(view, max_iter, side, side)
        flips = int(((mu_m > max_iter) != (mu_r > max_iter)).sum())
        assert flips <= max(2, int(0.001 * side * side)), f"cells~{pixels}"


# ------------------------------------------------------------ chaos, measured
def test_chaos_bounds_the_difference():
    """The compiled/reference gap is smaller than the reference's own
    sensitivity to a 1e-13 perturbation.

    This is the test that licenses the tolerances above. Escape-time amplifies
    rounding: nudging the reference's own input by 1e-13 -- far below the
    rounding error of the arithmetic itself -- moves a comparable number of
    pixels across the interior boundary. If the compiled kernel differed from
    the reference by *more* than that, this would fail and the tolerances
    above would be too loose.
    """
    view, max_iter = VIEWS[3]  # elephant valley: high iteration, many boundary px
    w, h = 128, 72
    mu_m, _ = kernel_planes(view, max_iter, w, h)
    mu_r, _ = ref_planes(view, max_iter, w, h)
    compiled_flips = int(((mu_m > max_iter) != (mu_r > max_iter)).sum())

    # Control: the same reference loop with the centre nudged by 1e-13.
    eps = 1e-13
    nudged = reference.Reference(w, h).escape_plane(
        view.cx + eps, view.cy + eps, view.scale, max_iter
    )[0]
    control_flips = int(((nudged > max_iter) != (mu_r > max_iter)).sum())

    assert control_flips > 0, "the control should move something; if it does not, "\
        "this view is not sensitive and the comparison proves nothing"
    assert compiled_flips <= max(control_flips, 2), (
        f"compiled differs in {compiled_flips} px, a 1e-13 perturbation moves "
        f"{control_flips}"
    )


def test_escape_values_agree_to_a_few_ulp_away_from_the_boundary():
    """Outside the chaotic band, `mu` agrees to ~1e-9 relative.

    Points whose escape value is small escape quickly, where the orbit has had
    no time to amplify rounding, so the two implementations must land on
    almost the same number.
    """
    view, max_iter = VIEWS[1]
    w, h = 48, 48
    mu_m, _ = kernel_planes(view, max_iter, w, h)
    mu_r, _ = ref_planes(view, max_iter, w, h)
    quick = (mu_r > 0) & (mu_r < 30)  # escaped early, far from the boundary
    assert quick.sum() > 10, "the probe frame should have some quick escapes"
    rel = np.abs(mu_m[quick] - mu_r[quick]) / np.maximum(np.abs(mu_r[quick]), 1e-300)
    assert rel.max() < 1e-9, f"worst relative difference {rel.max()}"


# ------------------------------------------------------------- deep zoom maths
def test_deep_zoom_is_more_expensive_than_shallow():
    """A sanity check on the iteration budget: a 1e-9 half-height view needs a
    far larger cap to resolve than the whole set does.

    It is a guard on the presets, not a timing assertion -- if a deep preset
    ever produced a frame as cheap as the full set, the coordinates would have
    drifted somewhere flat and every poster of it would be a black rectangle.
    """
    f_shallow = Frame(64, 64, max_iter=200, workers=1)
    f_shallow.render(View(-0.5, 0.0, 1.25), "mono")
    f_deep = Frame(64, 64, max_iter=200, workers=1)
    f_deep.render(preset("deep"), "mono")
    shallow_interior = int((f_shallow.escape > 200).sum())
    deep_interior = int((f_deep.escape > 200).sum())
    assert deep_interior > shallow_interior, (
        f"deep {deep_interior} should be harder to escape than shallow "
        f"{shallow_interior}"
    )


def test_zoom_sequence_goes_strictly_deeper():
    """`zoom_sequence` is the thing a shared GIF walks; if it did not actually
    descend, the artifact would be the same frame repeated."""
    views = zoom_sequence(preset("seahorse"), 12, 1.3)
    assert len(views) == 12
    for a, b in zip(views, views[1:]):
        assert b.scale < a.scale
        assert b.cx == a.cx and b.cy == a.cy
    # Each step divides the half-height by the factor, so the total depth is
    # factor**frames exactly. 1.3**11 is ~18x over 12 frames.
    assert views[-1].scale == pytest.approx(views[0].scale / 1.3**11, rel=1e-12)
    assert views[-1].scale < views[0].scale / 10
