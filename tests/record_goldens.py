"""Regenerate the SHA-256 digests in tests/test_golden.py.

    python tests/record_goldens.py

Prints the four digests and rewrites the four `GOLDEN_*` constants in place.
Read the diff before committing it: a changed digest is either an intentional
change to the renderer or a bug, and this script cannot tell you which.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from mojoapp_mandelzoom import gif, palettes, png  # noqa: E402
from mojoapp_mandelzoom.render import Frame, preset, zoom_sequence  # noqa: E402

GOLDEN_FILE = ROOT / "tests" / "test_golden.py"


def ascii_card(view, palette, w: int, h: int, max_iter: int) -> str:
    from mojoapp_mandelzoom.__main__ import ASCII_RAMP

    f = Frame(w, h, max_iter=max_iter, workers=1)
    _rgb, idx = f.render(view, palette)
    lines = [
        f"mandel-zoom  view={view.label}  centre={view.cx:+.15g}{view.cy:+.15g}i  "
        f"half-height={view.scale:.6g}  iters<={max_iter}  palette={palette}",
        "+" + "-" * w + "+",
    ]
    for y in range(h):
        row = "".join(
            ASCII_RAMP[min(len(ASCII_RAMP) - 1, int(idx[y, x]) * len(ASCII_RAMP) // 256)]
            for x in range(w)
        )
        lines.append("|" + row + "|")
    lines.append("+" + "-" * w + "+")
    lines.append(f"interior {int((f.escape > max_iter).sum())}/{w * h} px")
    return "\n".join(lines) + "\n"


def compute() -> dict[str, str]:
    f = Frame(24, 20, max_iter=80, workers=1)
    rgb, _idx = f.render(preset("cusp"), "ember")
    rgb = rgb.copy()
    _rgb2, idx = f.render(preset("cusp"), "ember")
    idx = idx.copy()
    pal_b = palettes.to_bytes(palettes.get("ember"))

    g = Frame(24, 24, max_iter=120, workers=1)
    seq = zoom_sequence(preset("cusp"), 6, 1.4)
    frames = np.stack([g.render(v, "mono")[1].copy() for v in seq])
    mono_b = palettes.to_bytes(palettes.get("mono"))

    return {
        "GOLDEN_PNG_SHA256": hashlib.sha256(png.encode_rgb(rgb)).hexdigest(),
        "GOLDEN_INDEXED_PNG_SHA256": hashlib.sha256(
            png.encode_indexed(idx, pal_b)
        ).hexdigest(),
        "GOLDEN_GIF_SHA256": hashlib.sha256(
            gif.encode(frames, mono_b, delay_cs=8)
        ).hexdigest(),
        "GOLDEN_CARD_SHA256": hashlib.sha256(
            ascii_card(preset("cusp"), "ember", 48, 12, 80).encode()
        ).hexdigest(),
    }


def main() -> int:
    values = compute()
    text = GOLDEN_FILE.read_text()
    changed = []
    for name, value in values.items():
        pattern = re.compile(rf'^{name} = "[0-9a-f]{{64}}"|^{name} = "[^"]*"', re.M)
        new_text, n = pattern.subn(f'{name} = "{value}"', text)
        if n != 1:
            print(f"error: could not find exactly one {name} line", file=sys.stderr)
            return 1
        if new_text != text:
            changed.append(name)
        text = new_text
    GOLDEN_FILE.write_text(text)
    for name, value in values.items():
        print(f"{name} = {value}")
    if changed:
        print(f"\nupdated: {', '.join(changed)}")
    else:
        print("\nno change -- the goldens already matched this build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
