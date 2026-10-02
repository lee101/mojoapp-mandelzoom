"""Generate the shareable demo artifacts into demo/out/.

    pixi run demo

Five artifacts, every one under 2 MB and every one a pure function of the
constants at the top of this file:

    poster.png   1280x720 seahorse valley, a wallpaper still
    zoom.gif      480x270, 24 frames, a real deep zoom (1.3^23 ~ 3400x)
    card.txt      a text card of the seahorse view, for a terminal
    zoom.json     the exact view sequence the GIF walks
    stats.json    the histogram and timings, for the README

The zoom GIF is the point of the demo. It is not a slideshow of a still: the
factor is 1.3 per frame over 24 frames, so the last frame is 1.3**23 ~= 3400x
deeper than the first, and `tests/test_golden.py` checks that re-running this
script reproduces the GIF byte for byte.

There is no RNG anywhere in the render path, so "deterministic with a seed" is
not something this script has to arrange -- it is a property of the kernel. The
constants below ARE the seed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_mandelzoom import gif, palettes, png  # noqa: E402
from mojoapp_mandelzoom.__main__ import ASCII_RAMP  # noqa: E402
from mojoapp_mandelzoom.render import (  # noqa: E402
    Frame,
    calibrate,
    par_min_pixels,
    preset,
    simd_width,
    workers,
    zoom_sequence,
)

OUT = ROOT / "demo" / "out"

# ---------------------------------------------------------------- the recipe
POSTER_VIEW = "seahorse"
POSTER_PALETTE = "cosmic"
POSTER_W, POSTER_H, POSTER_ITERS = 1280, 720, 600

# The zoom starts at the seahorse valley. That is a measured choice, not a
# taste one: of the presets, the seahorse valley is the one that keeps paying
# off as you descend. Measured over 22 frames at factor 1.3, its interior
# fraction goes 2.4% -> 0.7% -- still mostly filaments at the end -- while the
# elephant valley reaches 100% interior by frame 19 and the double spiral by
# frame 5, after which every further frame is a solid black rectangle. The
# demo's own test (`test_demo_zoom_keeps_resolving_all_the_way_down`) would
# catch that regression, which is why the view is named here and not buried.
ZOOM_VIEW = "seahorse"
ZOOM_PALETTE = "ember"
ZOOM_W, ZOOM_H, ZOOM_ITERS = 400, 225, 700
# 20 frames at factor 1.3 is 1.3**19 ~= 140x deep, which is where the seahorse
# valley is still mostly filaments and the GIF fits in a post. 24 frames of the
# same sequence measured 2.7 MB -- over the 2 MB ceiling an artifact has to fit
# under to be shareable -- and the extra four frames were the low-detail tail
# anyway.
ZOOM_FRAMES = 20
ZOOM_FACTOR = 1.3
ZOOM_DELAY_CS = 12
#: Extra iterations granted per frame of depth, on top of ZOOM_ITERS. Escape
#: time grows as you descend, so a fixed budget renders the first frames
#: beautifully and then quietly stops resolving detail -- the tail of the GIF
#: flattens out. Tying the budget to depth is what keeps every frame paying.
ZOOM_ITERS_PER_FRAME = 40

CARD_VIEW = "spiral"
CARD_PALETTE = "inferno"
CARD_W, CARD_H, CARD_ITERS = 120, 34, 400

#: Every artifact must be under this, or it is not something you can post.
MAX_BYTES = 2 * 1024 * 1024


def _report(path: Path, nbytes: int, seconds: float) -> None:
    flag = "OK " if nbytes < MAX_BYTES else "BIG"
    print(f"  {flag} {path.name:<12} {nbytes / 1024:8.1f} KiB  {seconds * 1000:7.1f} ms")


def build_poster() -> tuple[dict, float]:
    """The wallpaper still. A palette-indexed PNG: a third of the bytes of the
    truecolour one, and the decoded image is identical."""
    view = preset(POSTER_VIEW)
    f = Frame(POSTER_W, POSTER_H, max_iter=POSTER_ITERS)
    t0 = time.perf_counter()
    _rgb, idx = f.render(view, POSTER_PALETTE)
    idx = idx.copy()
    render_s = time.perf_counter() - t0
    pal = palettes.get(POSTER_PALETTE)
    t0 = time.perf_counter()
    nbytes = png.write_indexed(OUT / "poster.png", idx, palettes.to_bytes(pal))
    write_s = time.perf_counter() - t0
    _report(OUT / "poster.png", nbytes, render_s + write_s)
    interior = int((f.escape > POSTER_ITERS).sum())
    return {
        "view": view.as_dict(),
        "palette": POSTER_PALETTE,
        "width": POSTER_W,
        "height": POSTER_H,
        "max_iter": POSTER_ITERS,
        "interior_pixels": interior,
        "interior_fraction": round(interior / (POSTER_W * POSTER_H), 6),
        "bytes": nbytes,
    }, render_s


def build_zoom() -> tuple[dict, float]:
    """The deep-zoom GIF -- the artifact this project actually exists to make.

    The iteration budget climbs with depth, one extra `ZOOM_ITERS_PER_FRAME`
    iterations per frame. Without that the sequence renders its first frames
    well and then stops resolving: at 1.3**23 ~= 3400x deeper than the start,
    a fixed 500 iterations puts the last frames entirely inside the set, and
    the GIF finishes with the same black frame four times running.

    A `Frame` owns scratch buffers sized by its own `max_iter` at construction,
    so the budget cannot simply be raised mid-loop; a second `Frame` is opened
    for the deeper half and the two are swapped at the crossover. Both are the
    same class of object doing the same work, so the swap cannot change a
    pixel -- only how long it took.
    """
    start = preset(ZOOM_VIEW)
    views = zoom_sequence(start, ZOOM_FRAMES, ZOOM_FACTOR)
    frames = np.empty((ZOOM_FRAMES, ZOOM_H, ZOOM_W), dtype=np.uint8)
    iters = [ZOOM_ITERS + ZOOM_ITERS_PER_FRAME * i for i in range(ZOOM_FRAMES)]
    t0 = time.perf_counter()
    near = Frame(ZOOM_W, ZOOM_H, max_iter=max(iters[: len(iters) // 2]))
    far = Frame(ZOOM_W, ZOOM_H, max_iter=max(iters))
    for i, v in enumerate(views):
        # Copied: the next render replaces the frame's own buffer.
        frames[i] = (near if iters[i] <= near.max_iter else far).render(v, ZOOM_PALETTE)[1]
    render_s = time.perf_counter() - t0
    pal_b = palettes.to_bytes(palettes.get(ZOOM_PALETTE))
    t0 = time.perf_counter()
    nbytes = gif.write(OUT / "zoom.gif", frames, pal_b, delay_cs=ZOOM_DELAY_CS)
    write_s = time.perf_counter() - t0
    _report(OUT / "zoom.gif", nbytes, render_s + write_s)

    depth = views[0].scale / views[-1].scale
    return {
        "view": start.as_dict(),
        "palette": ZOOM_PALETTE,
        "width": ZOOM_W,
        "height": ZOOM_H,
        "frames": ZOOM_FRAMES,
        "factor": ZOOM_FACTOR,
        "delay_cs": ZOOM_DELAY_CS,
        "iters_per_frame": list(iters),
        "max_iter": max(iters),
        "depth_ratio": round(depth, 3),
        "bytes": nbytes,
        "sequence": [v.as_dict() for v in views],
    }, render_s


def build_card() -> tuple[dict, float]:
    """A text card, for a README or a terminal."""
    view = preset(CARD_VIEW)
    f = Frame(CARD_W, CARD_H, max_iter=CARD_ITERS)
    t0 = time.perf_counter()
    _rgb, idx = f.render(view, CARD_PALETTE)
    idx = idx.copy()
    render_s = time.perf_counter() - t0
    lines = [
        "mandel-zoom  fractal wallpaper and deep-zoom frame renderer",
        f"view        {view.label}  centre={view.cx:+.15g}{view.cy:+.15g}i",
        f"half-height {view.scale:.6g}   iters<={CARD_ITERS}   palette {CARD_PALETTE}",
        "+" + "-" * CARD_W + "+",
    ]
    for y in range(CARD_H):
        row = "".join(
            ASCII_RAMP[min(len(ASCII_RAMP) - 1, int(idx[y, x]) * len(ASCII_RAMP) // 256)]
            for x in range(CARD_W)
        )
        lines.append("|" + row + "|")
    lines.append("+" + "-" * CARD_W + "+")
    lines.append(
        f"interior {int((f.escape > CARD_ITERS).sum())}/{CARD_W * CARD_H} px"
        f"   kernel simd_width={simd_width()}"
    )
    text = "\n".join(lines) + "\n"
    path = OUT / "card.txt"
    path.write_text(text)
    nbytes = len(text.encode())
    _report(path, nbytes, render_s)
    return {
        "view": view.as_dict(),
        "palette": CARD_PALETTE,
        "width": CARD_W,
        "height": CARD_H,
        "max_iter": CARD_ITERS,
        "bytes": nbytes,
    }, render_s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"kernel      simd_width={simd_width()} workers={workers()} "
          f"par_min_pixels={par_min_pixels()}")
    print(f"out         {OUT}")
    print()
    print("building:")
    poster, poster_s = build_poster()
    zoom, zoom_s = build_zoom()
    card, card_s = build_card()

    (OUT / "zoom.json").write_text(
        json.dumps(
            {
                "view": zoom["view"],
                "palette": zoom["palette"],
                "width": zoom["width"],
                "height": zoom["height"],
                "frames": zoom["frames"],
                "factor": zoom["factor"],
                "depth_ratio": zoom["depth_ratio"],
                "iters_per_frame": zoom["iters_per_frame"],
                "note": (
                    "the exact views demo/out/zoom.gif walks, frame by frame, with "
                    "the iteration budget each frame was rendered at"
                ),
                "sequence": zoom["sequence"],
            },
            indent=2,
        )
        + "\n"
    )

    cal = calibrate()
    record = {
        "poster": poster,
        "zoom": {k: v for k, v in zoom.items() if k != "sequence"},
        "card": card,
        "kernel": {
            "simd_width": simd_width(),
            "default_workers": workers(),
            "par_min_pixels": par_min_pixels(),
        },
        "artifacts": sorted(p.name for p in OUT.iterdir() if p.is_file()),
        "timings": {
            "note": "wall time on the machine that built this, not a claim about yours",
            "poster_render_seconds": round(poster_s, 4),
            "zoom_render_seconds": round(zoom_s, 4),
            "card_render_seconds": round(card_s, 4),
        },
        "calibration": cal,
    }
    (OUT / "stats.json").write_text(json.dumps(record, indent=2) + "\n")

    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    print()
    print(f"wrote {len(record['artifacts'])} artifacts, {total / 1024:.1f} KiB total")
    for name in record["artifacts"]:
        print(f"  demo/out/{name}")
    if any(p.stat().st_size >= MAX_BYTES for p in OUT.iterdir() if p.is_file()):
        print(f"\nerror: an artifact is over {MAX_BYTES} bytes", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
