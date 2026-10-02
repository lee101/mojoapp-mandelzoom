"""Minimal deterministic PNG writer.

Hand-rolled rather than pulled from Pillow for one reason that matters here:
the golden-image tests compare bytes. A Pillow upgrade must not be able to
change a test outcome, and zlib with a fixed level and filter gives a byte
sequence that depends only on the pixels.

Two flavours, because a fractal frame is naturally palette-indexed:

* `encode_rgb` -- 8-bit truecolour, colour type 2.
* `encode_indexed` -- colour type 3 with a 256-entry PLTE, roughly a third of
  the bytes for the same image. `palette_rgb` is the same table the kernel
  used, so the decoded result is identical to `encode_rgb` of the same state.
"""

from __future__ import annotations

import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def encode_rgb(rgb, compress_level: int = 9) -> bytes:
    """Encode an `(h, w, 3)` uint8 array as a PNG file's bytes."""
    import numpy as np

    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) array, got {a.shape}")
    h, w = a.shape[0], a.shape[1]
    # One filter byte per scanline. Filter 0 (None) on an already-flat palette
    # image compresses as well as anything here and keeps the encoder trivial.
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += a[y].tobytes()
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def encode_indexed(idx, palette_rgb, compress_level: int = 9) -> bytes:
    """Encode an `(h, w)` palette-index array as a colour-type-3 PNG.

    `palette_rgb` is the 768-byte RGB table the renderer used, so the decoded
    image is identical to `encode_rgb` of the same state while being a fraction
    of the size.
    """
    import numpy as np

    a = np.ascontiguousarray(idx, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError(f"expected an (h, w) array, got {a.shape}")
    h, w = a.shape
    pal = np.frombuffer(palette_rgb, dtype=np.uint8)
    if pal.size != 768:
        raise ValueError(f"palette must be 768 bytes, got {pal.size}")
    plte = pal.reshape(256, 3).tobytes()
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += a[y].tobytes()
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
    out += _chunk(b"PLTE", plte)
    out += _chunk(b"IDAT", zlib.compress(bytes(raw), compress_level))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def write(path, rgb) -> int:
    """Write an RGB PNG. Returns the byte count written."""
    data = encode_rgb(rgb)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def write_indexed(path, idx, palette_rgb: bytes) -> int:
    """Write a palette-indexed PNG. Returns the byte count written."""
    data = encode_indexed(idx, palette_rgb)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
