"""The Python API, the CLI, and the encoders.

This file covers the surface a user touches: `Frame`, `View`, the presets, the
calibration, the palette and encoder helpers, and the command line. The
parity file covers the kernels; the golden file covers the bytes.

The tests are written so that a failure names the thing that broke. A ctypes
ABI that silently passes a wrong address produces a plausible-looking image, so
the buffer-size and shape checks here are not decoration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mojoapp_mandelzoom import gif, palettes, png, presets as presets_mod, reference
from mojoapp_mandelzoom import render as render_mod
from mojoapp_mandelzoom.render import (
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

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the view
def test_view_rejects_a_non_positive_scale():
    with pytest.raises(ValueError):
        View(-0.5, 0.0, 0.0)
    with pytest.raises(ValueError):
        View(-0.5, 0.0, -1.0)
    with pytest.raises(ValueError):
        View(-0.5, 0.0, float("inf"))


def test_view_rejects_a_non_finite_centre():
    with pytest.raises(ValueError):
        View(float("nan"), 0.0, 1.0)
    with pytest.raises(ValueError):
        View(0.0, float("inf"), 1.0)


def test_view_zoom_shrinks_the_scale_and_keeps_the_centre():
    v = View(-0.75, 0.13, 1e-4)
    z = v.zoom(10.0)
    assert z.scale == pytest.approx(1e-5)
    assert (z.cx, z.cy) == (v.cx, v.cy)
    with pytest.raises(ValueError):
        v.zoom(0.0)


def test_view_roundtrips_through_a_dict():
    v = View(-0.743643887037151, 0.13182590420533, 0.0042, "seahorse")
    assert View.from_dict(v.as_dict()) == v


def test_every_preset_is_a_sane_view():
    for name in presets_names():
        v = presets_mod.PRESETS[name]
        assert isinstance(v, View)
        assert v.scale > 0, name
        assert -2.2 < v.cx < 1.2, f"{name} cx off the real axis of the set"
        assert -1.2 < v.cy < 1.2, f"{name} cy off the imaginary axis of the set"
        assert v.label, name


def test_preset_lookup_rejects_an_unknown_name():
    with pytest.raises(ValueError) as excinfo:
        preset("not-a-preset")
    # The message names the offender AND the alternatives, so a typo on the
    # command line is fixable from the error alone.
    assert "not-a-preset" in str(excinfo.value)
    assert "seahorse" in str(excinfo.value)


def test_zoom_sequence_shape_and_monotonicity():
    views = zoom_sequence(preset("cusp"), 8, 1.5)
    assert len(views) == 8
    scales = [v.scale for v in views]
    assert all(b < a for a, b in zip(scales, scales[1:]))
    with pytest.raises(ValueError):
        zoom_sequence(preset("cusp"), 0)


# ------------------------------------------------------------------ the frame
def test_frame_rejects_bad_dimensions_and_iterations():
    with pytest.raises(ValueError):
        Frame(0, 10)
    with pytest.raises(ValueError):
        Frame(10, 0)
    with pytest.raises(ValueError):
        Frame(10, 10, max_iter=0)


def test_frame_rejects_an_unknown_trap_mode():
    with pytest.raises(ValueError):
        Frame(10, 10, trap_mode=7)


def test_render_requires_a_view():
    f = Frame(16, 16, max_iter=50)
    with pytest.raises(TypeError):
        f.render((-0.5, 0.0, 1.0))  # a tuple is not a View


def test_render_returns_the_right_shapes():
    f = Frame(37, 23, max_iter=80, workers=1)
    rgb, idx = f.render(preset("cusp"), "cosmic")
    assert rgb.shape == (23, 37, 3)
    assert idx.shape == (23, 37)
    assert rgb.dtype == np.uint8 and idx.dtype == np.uint8


def test_render_is_deterministic():
    """Same view, same frame, twice: identical bytes.

    A render is a pure function of (view, size, max_iter, palette) -- there is
    no RNG in the kernel -- so this has to hold, and it is what makes every
    published artifact reproducible.
    """
    f1 = Frame(64, 48, max_iter=200, workers=1)
    f2 = Frame(64, 48, max_iter=200, workers=1)
    v = preset("seahorse")
    a = f1.render(v, "ember")
    b = f2.render(v, "ember")
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_a_reused_frame_does_not_accumulate_state():
    """Rendering a second view into the same Frame must fully overwrite the
    first -- no ghosting from the previous escape plane."""
    f = Frame(48, 32, max_iter=150, workers=1)
    f.render(preset("full"), "cosmic")
    first = f._idx.copy()
    f.render(preset("cusp"), "cosmic")
    second = f._idx.copy()
    assert not np.array_equal(first, second)
    # And the second frame must equal a fresh frame rendering the same view.
    fresh = Frame(48, 32, max_iter=150, workers=1)
    _r, idx = fresh.render(preset("cusp"), "cosmic")
    assert np.array_equal(second, idx)


def test_rgb_matches_the_palette_applied_to_the_index():
    """`rgb` is exactly `palette[idx]`. The kernel writes both, and if the two
    ever drifted the PNG and the histogram would describe different images."""
    f = Frame(40, 24, max_iter=120, workers=1)
    rgb, idx = f.render(preset("elephant"), "verdigris")
    pal = palettes.get("verdigris")
    assert np.array_equal(rgb, pal[idx])


def test_a_custom_palette_array_is_accepted():
    pal = np.zeros((256, 3), dtype=np.uint8)
    pal[:, 0] = np.arange(256, dtype=np.uint8)  # a red ramp
    f = Frame(32, 32, max_iter=100, workers=1)
    rgb, idx = f.render(preset("full"), pal)
    assert np.array_equal(rgb, pal[idx])
    assert rgb[..., 1].max() == 0 and rgb[..., 2].max() == 0


def test_a_malformed_palette_array_is_rejected():
    f = Frame(16, 16, max_iter=50)
    with pytest.raises(ValueError):
        f.render(preset("full"), np.zeros((128, 3), dtype=np.uint8))


# ------------------------------------------------------------------ scheduling
def test_small_frames_stay_serial():
    """Below the kernel's pixel floor the serial loop is the only path."""
    f = Frame(32, 32, max_iter=50)
    assert f.uses_parallel is False
    assert f._workers_for() == 1


def test_explicit_workers_disable_the_calibration():
    """An explicit count is a request for exactly that path. Letting the
    calibration overrule it would make `workers=1` useless as a benchmark
    lever, and would mean a caller could not pin the serial path."""
    f = Frame(512, 512, max_iter=50, workers=1)
    assert f._workers_for() == 1
    assert f.uses_parallel is False
    forced = Frame(512, 512, max_iter=50, workers=4)
    assert forced._workers_for() == 4


def test_default_workers_is_bounded_by_the_machine():
    assert 1 <= default_workers() <= workers()
    assert default_workers() <= (len(__import__("os").sched_getaffinity(0)) or 1)


def test_simd_width_is_a_real_vector_width():
    """`mz_info` reports the compiled width, so a toolchain bump cannot
    silently disagree with the Python side."""
    assert simd_width() >= 2


def test_calibration_reports_both_paths():
    cal = calibrate()
    assert cal["serial_ns_per_pixel"] > 0
    assert cal["parallel_ns_per_pixel"] > 0
    assert cal["probe_pixels"] >= par_min_pixels()
    assert cal["threshold_pixels"] > 0
    assert isinstance(cal["parallel_wins_at_probe"], bool)


def test_calibration_is_cached():
    """A calibration per call would put a multi-millisecond probe inside every
    `Frame` and dominate every small benchmark."""
    assert calibrate() is calibrate()


def test_threshold_is_never_below_the_kernel_floor():
    assert parallel_threshold_pixels() >= par_min_pixels()


# ----------------------------------------------------------------- histogram
def test_histogram_sums_to_the_pixel_count():
    f = Frame(64, 48, max_iter=200, workers=1)
    hist = f.histogram(preset("seahorse"))
    assert hist.size == 202
    assert int(hist.sum()) == 64 * 48


def test_histogram_interior_bin_matches_the_escape_plane():
    f = Frame(64, 48, max_iter=200, workers=1)
    v = preset("seahorse")
    f.render(v, "cosmic")
    from_plane = int((f.escape > 200).sum())
    hist = f.histogram(v)
    assert int(hist[-1]) == from_plane


def test_histogram_bin_count_is_returned_by_the_kernel():
    f = Frame(32, 32, max_iter=100, workers=1)
    assert f.histogram(preset("cusp")).size == 102


# ------------------------------------------------------------------ palettes
def test_every_palette_is_256_by_3():
    for name in palettes.names():
        pal = palettes.get(name)
        assert pal.shape == (256, 3), name
        assert pal.dtype == np.uint8, name


def test_palette_lookup_rejects_an_unknown_name():
    with pytest.raises(ValueError):
        palettes.get("no-such-palette")


def test_palette_to_bytes_roundtrips():
    pal = palettes.get("cosmic")
    raw = palettes.to_bytes(pal)
    assert len(raw) == 768
    assert np.array_equal(np.frombuffer(raw, np.uint8).reshape(256, 3), pal)
    with pytest.raises(ValueError):
        palettes.to_bytes(np.zeros((128, 3), dtype=np.uint8))


def test_palettes_are_not_all_identical():
    """A bug that returned the same table for every name would pass every
    other palette test."""
    tables = {n: palettes.get(n).tobytes() for n in palettes.names()}
    assert len(set(tables.values())) == len(tables)


# ------------------------------------------------------------------- encoders
def _small_render(w: int = 24, h: int = 20):
    f = Frame(w, h, max_iter=80, workers=1)
    f.render(preset("cusp"), "ember")
    return f._rgb.copy(), f._idx.copy()


def test_png_encoders_are_pure_functions_of_their_input():
    rgb, idx = _small_render()
    pal_b = palettes.to_bytes(palettes.get("ember"))
    assert png.encode_rgb(rgb) == png.encode_rgb(rgb.copy())
    assert png.encode_indexed(idx, pal_b) == png.encode_indexed(idx.copy(), pal_b)


def test_gif_encoder_is_a_pure_function_of_its_input():
    f = Frame(24, 24, max_iter=120, workers=1)
    seq = zoom_sequence(preset("cusp"), 5, 1.4)
    frames = np.stack([f.render(v, "mono")[1].copy() for v in seq])
    pal_b = palettes.to_bytes(palettes.get("mono"))
    assert gif.encode(frames, pal_b) == gif.encode(frames.copy(), pal_b)


def test_png_rejects_a_wrongly_shaped_array():
    with pytest.raises(ValueError):
        png.encode_rgb(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        png.encode_indexed(np.zeros((4, 4, 3), dtype=np.uint8), b"\x00" * 768)


def test_gif_rejects_a_wrongly_shaped_stack():
    pal_b = palettes.to_bytes(palettes.get("mono"))
    with pytest.raises(ValueError):
        gif.encode(np.zeros((4, 4), dtype=np.uint8), pal_b)
    with pytest.raises(ValueError):
        gif.encode(np.zeros((0, 4, 4), dtype=np.uint8), pal_b)


# ----------------------------------------------------------------------- CLI
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mojoapp_mandelzoom", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=900,
    )


def test_cli_help_lists_all_subcommands():
    proc = _run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for name in ("still", "zoom", "card", "stats", "presets", "info"):
        assert name in proc.stdout


def test_cli_version(capsys):
    from mojoapp_mandelzoom import __main__ as cli

    assert cli.main(["--version"]) == 0
    assert "mojoapp-mandelzoom" in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys):
    from mojoapp_mandelzoom import __main__ as cli

    assert cli.main([]) == 0
    assert "still" in capsys.readouterr().out


def test_cli_presets_lists_names(capsys):
    from mojoapp_mandelzoom import __main__ as cli

    assert cli.main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in presets_names():
        assert name in out
    for name in palettes.names():
        assert name in out


def test_cli_rejects_a_bad_palette():
    proc = _run_cli("still", "--preset", "cusp", "--palette", "nope", "-o", "/dev/null")
    assert proc.returncode == 2
    assert "nope" in proc.stderr


def test_cli_still_writes_a_png(tmp_path):
    out = tmp_path / "cli.png"
    proc = _run_cli(
        "still", "--preset", "cusp", "--width", "96", "--height", "64",
        "--max-iter", "120", "-o", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_card_prints_a_boxed_frame(capsys):
    from mojoapp_mandelzoom import __main__ as cli

    rc = cli.main(
        ["card", "--preset", "cusp", "--width", "40", "--height", "10",
         "--max-iter", "80"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "mandel-zoom" in out
    assert out.count("\n") >= 10


def test_cli_stats_emits_valid_json(capsys):
    from mojoapp_mandelzoom import __main__ as cli

    rc = cli.main(
        ["stats", "--preset", "cusp", "--width", "48", "--height", "32",
         "--max-iter", "100"]
    )
    record = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert record["pixels"] == 48 * 32
    assert record["histogram"]["sum"] == 48 * 32
    assert 0.0 <= record["interior_fraction"] <= 1.0
    assert record["kernel"]["simd_width"] >= 2


def test_cli_zoom_writes_a_gif(tmp_path):
    out = tmp_path / "cli.gif"
    proc = _run_cli(
        "zoom", "--preset", "cusp", "--width", "64", "--height", "48",
        "--max-iter", "120", "--frames", "5", "--factor", "1.4", "-o", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_bytes()[:6] in (b"GIF89a", b"GIF87a")


def test_cli_zoom_rejects_a_factor_that_does_not_zoom():
    proc = _run_cli("zoom", "--preset", "cusp", "--frames", "3", "--factor", "0.5",
                    "-o", "/dev/null")
    assert proc.returncode == 2
    assert "factor" in proc.stderr


# ------------------------------------------------------------------ reference
def test_reference_rejects_a_tiny_frame():
    with pytest.raises(ValueError):
        reference.Reference(0, 10)


def test_reference_escape_plane_shape():
    r = reference.Reference(12, 9)
    mu, trap = r.escape_plane(-0.5, 0.0, 1.25, 50)
    assert mu.shape == (9, 12)
    assert trap.shape == (9, 12)
    assert mu.dtype == np.float64


def test_reference_index_plane_marks_the_interior_as_zero():
    mu = np.array([[1.0, 5.0], [100.0, 0.5]])
    trap = np.ones((2, 2))
    idx = reference.index_plane(mu, trap, 50, 8.0, 6.0, 400.0, 0.0, 0)
    assert idx[1, 0] == 0, "an interior pixel (mu > max_iter) is palette[0]"


def test_reference_index_plane_wraps_modulo_256():
    mu = np.array([[300.0]])
    trap = np.array([[0.0]])
    idx = reference.index_plane(mu, trap, 10_000, 8.0, 6.0, 400.0, 0.0, 0)
    assert 0 <= int(idx[0, 0]) < 256
    assert int(idx[0, 0]) == (int(300.0 * 8.0) % 256)


def test_known_points_are_reported_as_interior():
    for name in ("cusp", "cardioid_tip", "centre"):
        cx, cy = reference.KNOWN_POINTS[name]
        mu, _trap = reference._escape(cx, cy, 500)
        assert mu == 501.0, name
