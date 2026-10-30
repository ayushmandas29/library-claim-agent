"""Safe verification of saved frame evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime

from PIL import Image, UnidentifiedImageError

from .schema import ClaimPacket

_EVIDENCE_CHECK_TOKEN = object()
_VIDEO_TIMECODE = re.compile(r"^\d{2,}:[0-5]\d:[0-5]\d(?:\.\d+)?$")


def valid_frame_timestamp(value: str) -> bool:
    """Accept video timecodes (HH:MM:SS[.fraction]) or ISO timestamps."""
    if _VIDEO_TIMECODE.fullmatch(value):
        return True
    if not re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, init=False)
class EvidenceCheckedPacket:
    """Packet plus per-line evidence validation results for deterministic totals."""

    _packet_json: str
    valid_books: tuple[bool, ...]
    valid_items: tuple[bool, ...]
    problems: tuple[tuple[str, str], ...]

    def __init__(self, packet: ClaimPacket, valid_books: tuple[bool, ...],
                 valid_items: tuple[bool, ...], problems: tuple[tuple[str, str], ...],
                 *, _token: object):
        if _token is not _EVIDENCE_CHECK_TOKEN:
            raise TypeError("EvidenceCheckedPacket must come from validate_packet_evidence")
        object.__setattr__(self, "_packet_json", packet.model_dump_json())
        object.__setattr__(self, "valid_books", valid_books)
        object.__setattr__(self, "valid_items", valid_items)
        object.__setattr__(self, "problems", problems)

    @property
    def packet(self) -> ClaimPacket:
        """Return an independent packet snapshot so checks cannot be invalidated by mutation."""
        return ClaimPacket.model_validate_json(self._packet_json)


def validate_packet_evidence(packet: ClaimPacket, frames_dir: str | Path) -> EvidenceCheckedPacket:
    """Run the evidence checks and return the only accepted input type for totals."""
    problems = tuple(evidence_problems(packet, frames_dir))

    def traceable(line) -> bool:
        if line.frame_ref:
            valid, _ = check_frame(line.frame_ref, frames_dir)
            return valid or valid_frame_timestamp(line.frame_timestamp)
        return valid_frame_timestamp(line.frame_timestamp)

    return EvidenceCheckedPacket(
        packet=packet,
        valid_books=tuple(traceable(book) for book in packet.books),
        valid_items=tuple(traceable(item) for item in packet.items),
        problems=problems,
        _token=_EVIDENCE_CHECK_TOKEN,
    )


def check_frame(frame_ref: str, frames_dir: str | Path) -> tuple[bool, str]:
    """Check containment, existence, and image readability for one relative frame path."""
    if not frame_ref:
        return False, "frame reference is empty"
    root = Path(frames_dir).resolve()
    try:
        candidate = (root / frame_ref).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return False, "frame path is missing or outside the permitted frames directory"
    if not candidate.is_file():
        return False, "frame reference is not a file"
    try:
        with Image.open(candidate) as image:
            image.verify()
        # verify() checks container structure but does not decode image pixels.
        with Image.open(candidate) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, UnidentifiedImageError):
        return False, "frame file cannot be opened as an image"
    return True, ""


def evidence_problems(packet: ClaimPacket, frames_dir: str | Path) -> list[tuple[str, str]]:
    """Return (line id, problem) pairs. Timestamp-only refs remain traceable."""
    problems: list[tuple[str, str]] = []
    for line in [*packet.books, *packet.items]:
        if line.frame_ref:
            valid, reason = check_frame(line.frame_ref, frames_dir)
            if not valid:
                problems.append((line.id, reason))
        elif not valid_frame_timestamp(line.frame_timestamp):
            problems.append((line.id, "no saved frame reference or valid frame timestamp"))
    for index, frame_ref in enumerate(packet.room.frame_refs):
        valid, reason = check_frame(frame_ref, frames_dir)
        if not valid:
            problems.append((f"room-frame-{index + 1}", reason))
    return problems


def missing_frames(packet: ClaimPacket, frames_dir: str | Path) -> list[str]:
    """Backward-compatible helper returning IDs with invalid frame file references."""
    return [ref_id for ref_id, _ in evidence_problems(packet, frames_dir)]
