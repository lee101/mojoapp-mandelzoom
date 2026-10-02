"""`python -m mojoapp_mandelzoom` -- the command line.

Six subcommands, each of which writes one artifact:

    still     a PNG of one view
    zoom      a GIF of a deterministic zoom sequence
    card      a text card of a view, for terminals and READMEs
    stats     escape histogram and timings, as JSON
    presets   list the built-in views and palettes
    info      what the compiled kernel was built with

Every render is a pure function of (view, size, max_iter, palette), so all of
them are reproducible: there is no RNG anywhere in the render path, and the
"seed" a zoom takes is the factor and the view, both given on the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import gif, palettes, png, presets as presets_mod, reference
from .render import (
    DEFAULT_MAX_ITER,
    Frame,
    View,
    par_min_pixels,
    preset,
    presets_names,
    simd_width,
    workers,
    zoom_sequence,
)

#: Ramp for the text card: escape value -> glyph. Dense enough to see the
#: filigree, short enough to stay monospaced in a terminal.
ASCII_RAMP = " .:-=+*#%@"


def _view_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="full",
                   help=f"named view, one of {', '.join(presets_names())} (default full)")
    p.add_argument("--cx", type=float, help="view centre real part (overrides --preset)")
    p.add_argument("--cy", type=float, help="view centre imaginary part (overrides --preset)")
    p.add_argument("--scale", type=float, help="half-height in complex units (overrides --preset)")


def _size_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--width", type=int, default=1280, help="frame width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="frame height (default 720)")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help=f"iteration cap (default {DEFAULT_MAX_ITER})")
    p.add_argument("--palette", default=palettes.DEFAULT_PALETTE,
                   help=f"palette, one of {', '.join(palettes.names())}")
    p.add_argument("--workers", type=int, help="worker count (default: the calibrated choice)")
    p.add_argument("-o", "--out", help="output path")


def _color_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pscale", type=float, default=8.0, help="escape->palette scale (default 8)")
    p.add_argument("--repeat", type=float, default=6.0, help="orbit-trap winding (default 6)")
    p.add_argument("--trap-scale", type=float, default=400.0,
                   help="orbit-trap strength (default 400)")
    p.add_argument("--trap-shift", type=float, default=0.0, help="palette offset (default 0)")
    p.add_argument("--trap-mode", type=int, default=0, choices=(0, 1, 2),
                   help="0 log trap, 1 linear trap, 2 interior-out (default 0)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mojoapp_mandelzoom",
        description="Mandelbrot wallpaper and deep-zoom frame renderer, "
                    "compiled Mojo core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m mojoapp_mandelzoom still --preset seahorse -o deep.png\n"
            "  python -m mojoapp_mandelzoom zoom --preset seahorse --frames 24 "
            "--factor 1.3 -o zoom.gif\n"
            "  python -m mojoapp_mandelzoom still --width 1920 --height 1080 "
            "--max-iter 1200 --palette inferno -o wallpaper.png\n"
            "  python -m mojoapp_mandelzoom card --preset spiral\n"
        ),
    )
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    q = sub.add_parser("still", help="write a PNG of one view")
    _view_args(q)
    _size_args(q)
    _color_args(q)

    q = sub.add_parser("zoom", help="write an animated GIF of a zoom sequence")
    _view_args(q)
    _size_args(q)
    _color_args(q)
    q.add_argument("--frames", type=int, default=20,
                   help="frames in the sequence (default 20)")
    q.add_argument("--factor", type=float, default=1.3,
                   help="magnification per frame (default 1.3)")
    q.add_argument("--iters-per-frame", type=int, default=40,
                   help="extra iterations per frame of depth (default 40; 0 holds "
                        "--max-iter fixed, which stalls a deep sequence)")
    q.add_argument("--delay", type=int, default=10,
                   help="frame delay in hundredths of a second (default 10)")

    q = sub.add_parser("card", help="print a text card of one view")
    _view_args(q)
    _size_args(q)
    _color_args(q)
    q.add_argument("--columns", type=int, default=96,
                   help="text card width (default 96)")

    q = sub.add_parser("stats", help="print a JSON record: histogram and timings")
    _view_args(q)
    _size_args(q)
    _color_args(q)

    sub.add_parser("presets", help="list the built-in views and palettes")
    sub.add_parser("info", help="report the compiled kernel's constants")
    return p


# ------------------------------------------------------------------- helpers
def _resolve_view(args) -> View:
    """Explicit coordinates win over the preset name, one at a time or all."""
    any_coord = args.cx is not None or args.cy is not None or args.scale is not None
    if not any_coord:
        return preset(args.preset)
    base = preset(args.preset)
    return View(
        args.cx if args.cx is not None else base.cx,
        args.cy if args.cy is not None else base.cy,
        args.scale if args.scale is not None else base.scale,
        base.label,
    )


def _frame(args) -> Frame:
    if args.width < 1 or args.height < 1:
        raise ValueError("frame must be at least 1x1")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1")
    palettes.get(args.palette)  # resolve now, so a typo is a message not a traceback
    return Frame(
        args.width,
        args.height,
        max_iter=args.max_iter,
        workers=args.workers,
        pscale=args.pscale,
        repeat=args.repeat,
        trap_scale=args.trap_scale,
        trap_shift=args.trap_shift,
        trap_mode=args.trap_mode,
    )


# ------------------------------------------------------------------ commands
def cmd_still(args) -> int:
    f = _frame(args)
    view = _resolve_view(args)
    t0 = time.perf_counter()
    rgb, idx = f.render(view, args.palette)
    render_s = time.perf_counter() - t0
    out = args.out or "mandel.png"
    # Indexed PNG: same decoded image as the RGB one, a third of the bytes.
    nbytes = png.write_indexed(out, idx, palettes.to_bytes(palettes.get(args.palette)))
    mpirate = _mpirate(render_s, args.width * args.height, args.max_iter)
    print(
        f"wrote {out}  {nbytes / 1024:.1f} KiB  {args.width}x{args.height}  "
        f"max_iter {args.max_iter}  {render_s * 1000:.1f} ms  {mpirate:.0f} Mpix-it/s"
    )
    return 0


def cmd_zoom(args) -> int:
    if args.frames < 1:
        raise ValueError("--frames must be at least 1")
    if not (args.factor > 1.0):
        raise ValueError("--factor must be greater than 1 (a zoom goes inward)")
    if args.iters_per_frame < 0:
        raise ValueError("--iters-per-frame must not be negative")
    start = _resolve_view(args)
    views = zoom_sequence(start, args.frames, args.factor)
    # Escape time grows with depth, so the budget climbs with it. Holding
    # --max-iter fixed makes a deep sequence render its early frames well and
    # then stop resolving -- the tail of the GIF goes black and repeats.
    iters = [args.max_iter + args.iters_per_frame * i for i in range(args.frames)]
    frames = np.empty((args.frames, args.height, args.width), dtype=np.uint8)
    # A Frame's scratch is sized by its own max_iter, so a budget that climbs
    # needs a second Frame for the deeper half. Two Frames of the same class
    # doing the same work: the split cannot change a pixel, only how long the
    # shallow frames take.
    shallow = _frame(args)
    shallow.max_iter = min(iters)
    deep = shallow if max(iters) == min(iters) else _frame(args)
    if deep is not shallow:
        deep.max_iter = max(iters)
    t0 = time.perf_counter()
    for i, v in enumerate(views):
        _rgb, idx = (shallow if iters[i] <= shallow.max_iter else deep).render(
            v, args.palette
        )
        frames[i] = idx  # copy: the next render replaces the frame's buffer
    render_s = time.perf_counter() - t0
    out = args.out or "mandel.gif"
    pal_b = palettes.to_bytes(palettes.get(args.palette))
    nbytes = gif.write(out, frames, pal_b, delay_cs=args.delay)
    depth = views[-1].scale / views[0].scale
    print(
        f"wrote {out}  {nbytes / 1024:.1f} KiB  {args.frames} frames  "
        f"iters {min(iters)}..{max(iters)}  {render_s * 1000:.1f} ms total  "
        f"{depth:.4g}x deeper than it started"
    )
    return 0


def cmd_card(args) -> int:
    # The card is rendered small and blown up to the requested column count,
    # so the terminal layout is the same whatever the render size was.
    f = _frame(args)
    view = _resolve_view(args)
    rgb, idx = f.render(view, args.palette)
    h, w = idx.shape
    lines = []
    for y in range(h):
        row = "".join(
            ASCII_RAMP[min(len(ASCII_RAMP) - 1, int(idx[y, x]) * len(ASCII_RAMP) // 256)]
            for x in range(w)
        )
        lines.append(row)
    header = (
        f"mandel-zoom  view={view.label or 'custom'}  "
        f"centre={view.cx:+.15g}{view.cy:+.15g}i  half-height={view.scale:.6g}  "
        f"iters<={args.max_iter}  palette={args.palette}"
    )
    footer = (
        f"interior {int((f.escape > args.max_iter).sum())}/{w * h} px  "
        f"kernel simd_width={simd_width()} workers={f._workers_for()}"
    )
    body = [header, "+" + "-" * w + "+"]
    body += ["|" + line + "|" for line in lines]
    body += ["+" + "-" * w + "+", footer]
    print("\n".join(body))
    if args.out:
        with open(args.out, "w") as f_out:
            f_out.write("\n".join(body) + "\n")
        print(f"wrote {args.out}")
    return 0


def cmd_stats(args) -> int:
    f = _frame(args)
    view = _resolve_view(args)
    t0 = time.perf_counter()
    f.render(view, args.palette)
    render_s = time.perf_counter() - t0
    hist = f.histogram(view)
    interior = int(hist[-1])
    total = int(hist.sum())
    record = {
        "view": view.as_dict(),
        "width": args.width,
        "height": args.height,
        "pixels": args.width * args.height,
        "max_iter": args.max_iter,
        "palette": args.palette,
        "color": {
            "pscale": args.pscale,
            "repeat": args.repeat,
            "trap_scale": args.trap_scale,
            "trap_shift": args.trap_shift,
            "trap_mode": args.trap_mode,
        },
        "interior_pixels": interior,
        "interior_fraction": round(interior / total, 6) if total else None,
        "histogram": {
            "bins": int(hist.size),
            "sum": total,
            "nonzero_bins": int((hist > 0).sum()),
            "peak_bin": int(np.argmax(hist)),
            "peak_count": int(hist.max()),
        },
        "render_seconds": round(render_s, 6),
        "mpxit_per_second": (
            round(args.width * args.height * args.max_iter / render_s / 1e6, 1)
            if render_s > 0 else None
        ),
        "kernel": {
            "simd_width": simd_width(),
            "default_workers": workers(),
            "workers_used": f._workers_for(),
            "par_min_pixels": par_min_pixels(),
        },
        "numpy": np.__version__,
    }
    print(json.dumps(record, indent=2))
    if args.out:
        with open(args.out, "w") as f_out:
            json.dump(record, f_out, indent=2)
            f_out.write("\n")
        print(f"wrote {args.out}")
    return 0


def cmd_presets(_args) -> int:
    print("presets:")
    for name in presets_names():
        v = presets_mod.PRESETS[name]
        print(f"  {name:<12} cx={v.cx:+.15g}  cy={v.cy:+.15g}  "
              f"half-height={v.scale:.6g}   {v.label}")
    print("\npalettes:")
    for name in palettes.names():
        pal = palettes.get(name)
        print(f"  {name:<10} {pal[0].tolist()} -> {pal[-1].tolist()}")
    print(f"\ndefault palette: {palettes.DEFAULT_PALETTE}")
    return 0


def cmd_info(_args) -> int:
    from .render import calibrate, default_workers

    cal = calibrate()
    record = {
        "version": __import__("mojoapp_mandelzoom").__version__,
        "simd_width": simd_width(),
        "default_workers": workers(),
        "host_workers": default_workers(),
        "par_min_pixels": par_min_pixels(),
        "calibration": cal,
        "numpy": np.__version__,
    }
    print(json.dumps(record, indent=2))
    return 0


def _mpirate(seconds: float, pixels: int, max_iter: int) -> float:
    """Megapixel-iterations per second, the unit this kernel is measured in.

    `0.0` when the clock did not advance, which keeps the status line honest
    rather than printing a rate derived from a zero denominator.
    """
    if seconds <= 0:
        return 0.0
    return pixels * max_iter / seconds / 1e6


COMMANDS = {
    "still": cmd_still,
    "zoom": cmd_zoom,
    "card": cmd_card,
    "stats": cmd_stats,
    "presets": cmd_presets,
    "info": cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__

        print(f"mojoapp-mandelzoom {__version__}")
        return 0
    if not args.command:
        p.print_help()
        return 0
    try:
        return COMMANDS[args.command](args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
