"""Minimal deterministic GIF89a writer for palette-indexed frames.

A deep zoom is a GIF89a because it is the one animated format a chat client
will actually play inline. The renderer already produces palette indices, so a
GIF is "that, plus LZW compression and a 47-byte header". Pillow would do the
same job, but the golden-file tests compare bytes and Pillow's LZW
implementation is not a contract anyone can rely on across upgrades.

Scope is exactly what Mandel Zoom emits: 256-entry global palette, 8-bit
indices, no interlacing, loop forever, per-frame delay.
"""

from __future__ import annotations

import struct


class _BitWriter:
    """GIF's variable-width LZW bit packing: codes go out least-significant bit
    first, and the code stream starts at `min_code_size` bits per code."""

    def __init__(self, min_code_size: int):
        self.out = bytearray()
        self.min_code_size = min_code_size
        self.cur = 0
        self.nbits = 0
        self.clear = 1 << min_code_size
        self.end = self.clear + 1

    def write(self, code: int, width: int) -> None:
        self.cur |= code << self.nbits
        self.nbits += width
        while self.nbits >= 8:
            self.out.append(self.cur & 0xFF)
            self.cur >>= 8
            self.nbits -= 8

    def flush(self) -> None:
        if self.nbits > 0:
            self.out.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0


def _lzw_encode(indices, min_code_size: int = 8) -> bytes:
    """GIF-flavoured LZW: clear code, growing dictionary, end code.

    The dictionary is capped at 12 bits because that is what a GIF89a frame's
    `LZW minimum code size` byte plus the spec allow; hitting the cap emits a
    clear code rather than growing further.
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    w = _BitWriter(min_code_size)
    code_width = min_code_size + 1
    next_code = end_code + 1
    # dict maps (prefix_code, byte) -> code
    table: dict[tuple[int, int], int] = {}
    w.write(clear_code, code_width)
    prefix: int | None = None
    for byte in indices:
        b = int(byte)
        if prefix is None:
            prefix = b
            continue
        key = (prefix, b)
        found = table.get(key)
        if found is not None:
            prefix = found
            continue
        w.write(prefix, code_width)
        if next_code < 4096:
            table[key] = next_code
            next_code += 1
            if next_code > (1 << code_width) and code_width < 12:
                code_width += 1
        else:
            w.write(clear_code, code_width)
            table = {}
            next_code = end_code + 1
            code_width = min_code_size + 1
        prefix = b
    if prefix is not None:
        w.write(prefix, code_width)
    w.write(end_code, code_width)
    w.flush()
    return bytes(w.out)


def _sub_blocks(data: bytes) -> bytes:
    """GIF's run of length-prefixed sub-blocks; a zero-length block ends it."""
    out = bytearray()
    for i in range(0, len(data), 255):
        part = data[i : i + 255]
        out.append(len(part))
        out += part
    out.append(0)
    return bytes(out)


def encode(
    frames,
    palette_rgb: bytes,
    delay_cs: int = 8,
    loop: int = 0,
) -> bytes:
    """Encode `(frames, h, w)` uint8 index arrays as one animated GIF.

    `palette_rgb` is the 768-byte table. `delay_cs` is the per-frame delay in
    hundredths of a second, so 8 is 80ms -- slow enough to read, fast enough not
    to bore.
    """
    import numpy as np

    a = np.asarray(frames)
    if a.ndim != 3:
        raise ValueError(f"expected (frames, h, w), got {a.shape}")
    n_frames, h, w = a.shape
    if n_frames < 1:
        raise ValueError("a GIF needs at least one frame")
    pal = np.frombuffer(palette_rgb, dtype=np.uint8)
    if pal.size != 768:
        raise ValueError(f"palette must be 768 bytes, got {pal.size}")

    out = bytearray(b"GIF89a")
    # Logical screen descriptor: global palette, 8 bits per primary, no GCT
    # colour resolution bump (the pixels are already 8-bit).
    out += struct.pack("<HHBBB", w, h, 0xF7, 0, 0)
    out += pal.reshape(256, 3).tobytes()
    # Netscape looping extension: 0 = forever.
    out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01" + struct.pack("<H", loop) + b"\x00"
    min_code_size = 8
    for f in range(n_frames):
        idx = np.ascontiguousarray(a[f], dtype=np.uint8)
        if idx.shape != (h, w):
            raise ValueError(f"frame {f} is {idx.shape}, expected {(h, w)}")
        # Graphic control extension: no disposal (0 = leave in place), no
        # transparency, user input not expected to be "click to advance".
        out += b"\x21\xF9\x04" + struct.pack("<BHB", 0x00, delay_cs, 0x00) + b"\x00"
        # Image descriptor: local image left at 0,0 with no local table.
        out += b"\x2C" + struct.pack("<HHHHB", 0, 0, w, h, 0x00)
        out += bytes([min_code_size])
        out += _sub_blocks(_lzw_encode(idx.reshape(-1).tolist(), min_code_size))
    out += b"\x3B"
    return bytes(out)


def write(path, frames, palette_rgb: bytes, delay_cs: int = 8) -> int:
    data = encode(frames, palette_rgb, delay_cs)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)
