"""Video frame extraction and reviewed-packet helpers for the Streamlit demo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from .pipeline import build_packet, render_report
from .schema import ReviewEntry


@dataclass(frozen=True)
class ExtractedFrame:
    frame_ref: str
    timestamp_s: float
    timecode: str


@dataclass(frozen=True)
class PriceQuote:
    amount: float
    currency: str
    source: str
    price_type: str
    url: str
    retrieved_at: str


def _format_timecode(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def extract_representative_frames(video_path: str | Path, frames_dir: str | Path,
                                  max_frames: int = 10) -> tuple[list[ExtractedFrame], float]:
    """Extract up to ``max_frames`` evenly spaced JPEGs from a readable MP4."""
    video = Path(video_path)
    if video.suffix.lower() != ".mp4":
        raise ValueError("Upload must be an MP4 video")
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError("Uploaded video is empty or unavailable")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open this MP4; check that the file is a valid supported video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError("Video has no readable frame-rate or frame-count metadata")

        duration_s = frame_count / fps
        sample_count = min(max_frames, frame_count)
        indices = np.unique(np.linspace(0, frame_count - 1, sample_count, dtype=np.int64))
        output = Path(frames_dir)
        output.mkdir(parents=True, exist_ok=True)
        extracted: list[ExtractedFrame] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                continue
            seconds = float(index) / fps
            name = f"video_frame_{len(extracted) + 1:04d}.jpg"
            destination = output / name
            if not cv2.imwrite(str(destination), frame):
                raise ValueError(f"Could not save extracted frame {name}")
            extracted.append(ExtractedFrame(name, seconds, _format_timecode(seconds)))
        if not extracted:
            raise ValueError("No decodable frames were found in the uploaded video")
        return extracted, duration_s
    finally:
        capture.release()


def lookup_google_books_price(title: str, author: str = "", isbn: str = "",
                              country: str = "") -> PriceQuote | None:
    """Return an exact-match Google Books eBook retail price, or Unknown (None).

    The quote is an eBookstore price, not an assertion about a physical copy's
    replacement value. ISBN searches require an exact ISBN match; title searches
    require exact normalized title and, when supplied, author matches.
    """
    clean_isbn = "".join(char for char in isbn if char.isdigit() or char.upper() == "X")
    if not clean_isbn and not title.strip():
        return None
    if clean_isbn:
        query = f"isbn:{clean_isbn}"
    else:
        query = f'intitle:"{title.strip()}"'
        if author.strip():
            query += f' inauthor:"{author.strip()}"'
    params: dict[str, str | int] = {"q": query, "maxResults": 10, "projection": "full"}
    if country.strip():
        params["country"] = country.strip().upper()
    try:
        response = httpx.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=8.0)
        response.raise_for_status()
        results = response.json().get("items", [])
    except (httpx.HTTPError, ValueError, TypeError):
        return None

    expected_title = " ".join(title.casefold().split())
    expected_author = " ".join(author.casefold().split())
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue
        info = item.get("volumeInfo") or {}
        sale = item.get("saleInfo") or {}
        identifiers = info.get("industryIdentifiers") or []
        if clean_isbn and not any(
            "".join(str(identifier.get("identifier", "")).upper().replace("-", "").split()) == clean_isbn
            for identifier in identifiers if isinstance(identifier, dict)
        ):
            continue
        if not clean_isbn:
            result_title = " ".join(str(info.get("title", "")).casefold().split())
            authors = [" ".join(str(name).casefold().split()) for name in info.get("authors", [])]
            if not expected_title or result_title != expected_title:
                continue
            if expected_author and expected_author not in authors:
                continue
        price = sale.get("retailPrice")
        try:
            amount = float(price["amount"])
            currency = str(price["currencyCode"]).upper()
        except (TypeError, KeyError, ValueError):
            continue
        if not math.isfinite(amount) or amount < 0 or not currency:
            continue
        if sale.get("saleability") != "FOR_SALE":
            continue
        link = sale.get("buyLink") or info.get("infoLink") or item.get("selfLink")
        if not isinstance(link, str) or not link.startswith("https://"):
            continue
        return PriceQuote(
            amount=amount,
            currency=currency,
            source="Google Books eBook retail price",
            price_type="eBook retail price",
            url=link,
            retrieved_at=datetime.now(timezone.utc).date().isoformat(),
        )
    return None


def build_reviewed_video_packet(scan_packet, reviewed_books: list[dict[str, Any]],
                                frames_dir: str | Path, timecodes: dict[str, str],
                                sweep: dict[str, Any],
                                excluded_candidates: list[dict[str, Any]] | None = None,
                                rejected_candidates: list[dict[str, Any]] | None = None):
    """Build a claim packet from reviewed candidates, retaining relevant scan notes."""
    books: list[dict[str, Any]] = []
    for observation in reviewed_books:
        row = dict(observation)
        frame_ref = str(row.get("frame_ref") or "")
        if frame_ref in timecodes:
            row["frame_timestamp"] = timecodes[frame_ref]
            row["timestamp_source"] = "uploaded_video_timecode"
        books.append(row)
    accepted_ids = {str(row.get("id") or "") for row in books}
    rejected_ids = {str(row.get("id") or "") for row in rejected_candidates or []}
    remaining_candidates = [candidate.model_dump(mode="python") for candidate in scan_packet.candidates
                            if candidate.id not in accepted_ids and candidate.id not in rejected_ids]
    for candidate in excluded_candidates or []:
        data = dict(candidate)
        data["frame_timestamp"] = timecodes.get(str(data.get("frame_ref") or ""), data.get("frame_timestamp", ""))
        data["timestamp_source"] = "uploaded_video_timecode"
        remaining_candidates.append(data)
    packet = build_packet({"sweep": sweep, "books": books, "candidates": remaining_candidates, "items": []}, frames_dir)
    time_refs = set(timecodes)
    packet.review_queue.extend(
        entry for entry in scan_packet.review_queue
        if not (entry.ref_id in time_refs and "filesystem modification time" in entry.reason)
    )
    for candidate in excluded_candidates or []:
        data = dict(candidate)
        ref_id = str(data.get("id") or "candidate")
        frame_ref = str(data.get("frame_ref") or "")
        data["frame_timestamp"] = timecodes.get(frame_ref, data.get("frame_timestamp", ""))
        data["timestamp_source"] = "uploaded_video_timecode"
        packet.review_queue.append(ReviewEntry(
            ref_id=ref_id,
            reason="low-specificity contour excluded from book observations; retained for manual review",
            details={"candidate": data},
        ))
    for candidate in rejected_candidates or []:
        data = dict(candidate)
        ref_id = str(data.get("id") or "candidate")
        frame_ref = str(data.get("frame_ref") or "")
        data["frame_timestamp"] = timecodes.get(frame_ref, data.get("frame_timestamp", ""))
        data["timestamp_source"] = "uploaded_video_timecode"
        packet.review_queue.append(ReviewEntry(
            ref_id=ref_id, reason="candidate rejected by reviewer as not a book",
            details={"candidate": data},
        ))
    return packet


def packet_downloads(packet) -> tuple[bytes, bytes]:
    """Serialize the JSON claim packet and its Markdown report for download."""
    payload = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return payload.encode("utf-8"), render_report(packet).encode("utf-8")
