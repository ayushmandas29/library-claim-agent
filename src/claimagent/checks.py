"""Integrity checks run before a packet is emitted."""
from __future__ import annotations

from pathlib import Path

from .schema import ClaimPacket


def missing_frames(packet: ClaimPacket, frames_dir: str | Path) -> list[str]:
    """Return ids of lines whose frame_ref does not point at a saved file."""
    root = Path(frames_dir)
    bad = []
    for line in [*packet.books, *packet.items]:
        if not (root / line.frame_ref).exists():
            bad.append(line.id)
    return bad
