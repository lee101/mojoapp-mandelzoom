"""Golden outputs and the demo's reproducibility.

The parity file pins the kernels against the reference. The API file pins the
surface a user touches. This one pins the *bytes*: what a published PNG, GIF or
text card actually contains.

A SHA-256 digest, not a "does it decode" check: the encoders are hand-rolled
and deterministic by design, so a change to either the pixels or the byte
packing is a change worth seeing before it ships. Regenerate with
`python tests/record_goldens.py` and read the diff.

The demo artifacts are checked two ways: they exist and are small enough to
post, and re-running the demo on the same repository reproduces them byte for
byte -- which is the whole claim of a deterministic render.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mojoapp_mandelzoom import gif, palettes, png
from mojoapp_mandelzoom.render import Frame, preset, zoom_sequence

ROOT = Path(__file__).resolve().parents[1]
DEMO_OUT = ROOT / "demo" / "out"

# Recorded from the deterministic encoders in this repo: a fixed view, a fixed
# palette, fixed colour parameters. Regenerate with `python tests/record_goldens.py`.
GOLDEN_PNG_SHA256 = "5787bf3268a666646cc581f1444c1081b5645b17812fa7c2dbc13a871c1bc6f5"
GOLDEN_INDEXED_PNG_SHA256 = "0e27e4f31d006b5e1ed3bd6bd85efec55573ba7ef51fe47c690b4d2e1b48501b"
GOLDEN_GIF_SHA256 = "aaa39981ecb1d2bc34947f8e5f998da7d4f8eebe4234f1ce41ea441e75cd816c"
GOLDEN_CARD_SHA256 = "1b4498f2d969990326186b8b39e6540acb166d93f198f0001a711a9024e712a0"


# ------------------------------------------------------------------ the card
def ascii_card(view, palette, w: int, h: int, max_iter: int) -> str:
    """The exact text the CLI's `card` command prints, minus the timing lines.

    Reproduced here rather than imported so the golden pins the *output*, not
    whatever the formatter happens to emit today.
    """
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


# ------------------------------------------------------------------ goldens
def test_png_bytes_are_golden():
    f = Frame(24, 20, max_iter=80, workers=1)
    rgb, _idx = f.render(preset("cusp"), "ember")
    rgb = rgb.copy()
    digest = hashlib.sha256(png.encode_rgb(rgb)).hexdigest()
    assert digest == GOLDEN_PNG_SHA256, (
        "PNG bytes changed; if that is intended, update GOLDEN_PNG_SHA256 in "
        "tests/test_golden.py"
    )


def test_indexed_png_bytes_are_golden():
    f = Frame(24, 20, max_iter=80, workers=1)
    _rgb, idx = f.render(preset("cusp"), "ember")
    idx = idx.copy()
    pal_b = palettes.to_bytes(palettes.get("ember"))
    digest = hashlib.sha256(png.encode_indexed(idx, pal_b)).hexdigest()
    assert digest == GOLDEN_INDEXED_PNG_SHA256


def test_gif_bytes_are_golden():
    """The LZW encoder is the most easily-broken piece in the repo, and every
    GIF this project publishes goes through it."""
    f = Frame(24, 24, max_iter=120, workers=1)
    seq = zoom_sequence(preset("cusp"), 6, 1.4)
    frames = np.stack([f.render(v, "mono")[1].copy() for v in seq])
    pal_b = palettes.to_bytes(palettes.get("mono"))
    digest = hashlib.sha256(gif.encode(frames, pal_b, delay_cs=8)).hexdigest()
    assert digest == GOLDEN_GIF_SHA256


def test_card_text_is_golden():
    text = ascii_card(preset("cusp"), "ember", 48, 12, 80)
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert digest == GOLDEN_CARD_SHA256


def test_every_golden_is_actually_recorded():
    """A placeholder left in the file would make the goldens above vacuous --
    they would be comparing against themselves."""
    for name, value in (
        ("GOLDEN_PNG_SHA256", GOLDEN_PNG_SHA256),
        ("GOLDEN_INDEXED_PNG_SHA256", GOLDEN_INDEXED_PNG_SHA256),
        ("GOLDEN_GIF_SHA256", GOLDEN_GIF_SHA256),
        ("GOLDEN_CARD_SHA256", GOLDEN_CARD_SHA256),
    ):
        assert value != "TO_BE_RECORDED", f"{name} was never recorded"
        assert len(value) == 64, f"{name} is not a sha256 hex digest"


# ----------------------------------------------------------------- the demo
def _demo_script_ok() -> bool:
    return (ROOT / "demo" / "make_demo.py").exists()


DEMO_ARTIFACTS = ("poster.png", "zoom.gif", "card.txt", "stats.json", "zoom.json")


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
@pytest.mark.parametrize("name", DEMO_ARTIFACTS)
def test_demo_artifact_exists_and_is_small(name):
    path = DEMO_OUT / name
    assert path.exists(), f"run `pixi run demo`: {name} is missing"
    assert 0 < path.stat().st_size < 2 * 1024 * 1024, (
        f"{name} is {path.stat().st_size} bytes; demo artifacts must be under 2 MB"
    )


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_artifacts_are_reproducible(tmp_path):
    """Re-running the demo on the same repository must produce the same bytes.

    Only the timing fields in stats.json may move, so they are excluded by
    comparison rather than by editing the file: the numbers that describe the
    *image* have to be stable, and the numbers that describe the *machine* are
    allowed not to be.
    """
    before = {n: (DEMO_OUT / n).read_bytes() for n in DEMO_ARTIFACTS
              if (DEMO_OUT / n).exists()}
    assert before, "no demo artifacts to compare against; run `pixi run demo`"

    proc = subprocess.run(
        [sys.executable, str(ROOT / "demo" / "make_demo.py")],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    for name in ("poster.png", "zoom.gif", "card.txt", "zoom.json"):
        if name in before:
            assert (DEMO_OUT / name).read_bytes() == before[name], name

    if "stats.json" in before:
        a = json.loads(before["stats.json"])
        b = json.loads((DEMO_OUT / "stats.json").read_text())
        for key in ("poster", "zoom", "kernel", "artifacts"):
            assert a.get(key) == b.get(key), f"stats.json: {key} moved"


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_poster_decodes_to_the_size_it_claims():
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(DEMO_OUT / "poster.png")
    assert img.mode == "P", "the poster is written as a palette PNG"
    assert img.size == (1280, 720)
    assert img.convert("RGB").getbbox() is not None, "the poster is blank"


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_zoom_gif_is_animated_and_the_frames_differ():
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(DEMO_OUT / "zoom.gif")
    assert img.size == (400, 225)
    assert img.n_frames == 20
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img.convert("RGB")))
    # A zoom that does not move would make every frame identical; that is the
    # single most likely way this artifact silently breaks.
    assert len({f.tobytes() for f in frames}) == img.n_frames


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_zoom_keeps_resolving_all_the_way_down():
    """The tail of a deep zoom must still change frame to frame.

    This is the failure the iteration budget exists to prevent: a fixed budget
    renders the first frames well, then stops resolving, and the GIF ends with
    the same black frame several times running. Every frame must differ from
    its neighbour, not just the first few.
    """
    Image = pytest.importorskip("PIL.Image")
    img = Image.open(DEMO_OUT / "zoom.gif")
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.array(img.convert("RGB")).astype(np.int32))
    diffs = [np.abs(a - b).mean() for a, b in zip(frames, frames[1:])]
    stalled = [i for i, d in enumerate(diffs) if d == 0.0]
    assert not stalled, (
        f"frames {stalled} are identical to their predecessor: the zoom stopped "
        f"resolving, which means --max-iter was too low for the depth"
    )


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_zoom_json_matches_the_gif():
    """The JSON in `zoom.json` is the GIF's recipe. If the two drift, the
    artifact is not reproducible from what is published alongside it."""
    record = json.loads((DEMO_OUT / "zoom.json").read_text())
    assert len(record["sequence"]) == record["frames"]
    assert len(record["iters_per_frame"]) == record["frames"]
    scales = [v["scale"] for v in record["sequence"]]
    assert all(b < a for a, b in zip(scales, scales[1:])), "the sequence ascends"
    assert record["iters_per_frame"] == sorted(record["iters_per_frame"])
    assert record["depth_ratio"] > 100


@pytest.mark.skipif(not _demo_script_ok(), reason="demo script not present")
def test_demo_card_is_not_blank():
    text = (DEMO_OUT / "card.txt").read_text()
    assert "mandel-zoom" in text
    ramp = set(" .:-=+*#%@")
    glyphs = {c for line in text.splitlines() if line.startswith("|") for c in line}
    assert len(glyphs & ramp) >= 3, "the card is nearly uniform"
