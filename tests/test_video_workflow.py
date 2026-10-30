from datetime import datetime
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from claimagent import video_workflow
from claimagent.schema import BookCandidate, ClaimPacket, ReviewEntry
from claimagent.video_workflow import (
    PriceQuote,
    build_reviewed_video_packet,
    extract_representative_frames,
    lookup_google_books_price,
    packet_downloads,
)


class FakeVideoCapture:
    def __init__(self, opened=True, frame_count=11, fps=2.0, fail_indices=()):
        self.opened = opened
        self.frame_count = frame_count
        self.fps = fps
        self.fail_indices = set(fail_indices)
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        if self.position in self.fail_indices:
            return False, None
        return True, np.full((24, 32, 3), self.position, dtype=np.uint8)

    def release(self):
        self.released = True


def test_extracts_evenly_spaced_representative_mp4_frames(monkeypatch, tmp_path):
    video = tmp_path / "shelf.mp4"
    video.write_bytes(b"fake mp4 content")
    fake_capture = FakeVideoCapture(frame_count=11, fps=2)
    monkeypatch.setattr(video_workflow.cv2, "VideoCapture", lambda _: fake_capture)

    frames, duration = extract_representative_frames(video, tmp_path / "frames", max_frames=3)

    assert duration == pytest.approx(5.5)
    assert [frame.timestamp_s for frame in frames] == [0, 2.5, 5]
    assert [frame.timecode for frame in frames] == ["00:00:00.000", "00:00:02.500", "00:00:05.000"]
    assert all((tmp_path / "frames" / frame.frame_ref).is_file() for frame in frames)
    assert fake_capture.released


@pytest.mark.parametrize("suffix, payload", [(".avi", b"video"), (".mp4", b"")])
def test_rejects_non_mp4_or_empty_video(tmp_path, suffix, payload):
    video = tmp_path / f"upload{suffix}"
    video.write_bytes(payload)
    with pytest.raises(ValueError):
        extract_representative_frames(video, tmp_path / "frames")


def test_rejects_unopenable_video_and_releases_capture(monkeypatch, tmp_path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"not really mp4")
    fake_capture = FakeVideoCapture(opened=False)
    monkeypatch.setattr(video_workflow.cv2, "VideoCapture", lambda _: fake_capture)
    with pytest.raises(ValueError, match="could not open"):
        extract_representative_frames(video, tmp_path / "frames")
    assert fake_capture.released


def test_invalid_video_metadata_and_no_decodable_frames_are_errors(monkeypatch, tmp_path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(b"content")
    bad_metadata = FakeVideoCapture(frame_count=0)
    monkeypatch.setattr(video_workflow.cv2, "VideoCapture", lambda _: bad_metadata)
    with pytest.raises(ValueError, match="metadata"):
        extract_representative_frames(video, tmp_path / "frames")
    assert bad_metadata.released

    no_frames = FakeVideoCapture(frame_count=4, fail_indices=(0, 1, 2, 3))
    monkeypatch.setattr(video_workflow.cv2, "VideoCapture", lambda _: no_frames)
    with pytest.raises(ValueError, match="No decodable frames"):
        extract_representative_frames(video, tmp_path / "frames")
    assert no_frames.released


def test_reviewed_video_packet_keeps_video_timecodes_and_exports(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_ref = "video_frame_0001.jpg"
    Image.new("RGB", (20, 20), "white").save(frames_dir / frame_ref)
    scan_packet = ClaimPacket(review_queue=[
        ReviewEntry(ref_id="detected-1", reason="OCR spine text requires human review"),
        ReviewEntry(ref_id=frame_ref, reason="timestamp is filesystem modification time, not verified capture time"),
    ])
    reviewed = [{
        "id": "detected-1", "frame_ref": frame_ref, "status": "identified",
        "title": "Reviewed Book", "author": "Reviewed Author", "id_confidence": 0.95,
        "detection_confidence": 0.7,
    }]
    packet = build_reviewed_video_packet(
        scan_packet, reviewed, frames_dir, {frame_ref: "00:00:03.500"},
        {"id": "sweep-1", "captured_at": "2026-10-30T10:00:00Z", "duration_s": 12,
         "country": "IN", "currency": "INR"},
    )

    assert packet.books[0].title == "Reviewed Book"
    assert packet.books[0].frame_timestamp == "00:00:03.500"
    assert packet.books[0].timestamp_source == "uploaded_video_timecode"
    assert packet.totals.book_count == 1
    assert any("OCR spine text" in entry.reason for entry in packet.review_queue)
    assert not any("filesystem modification time" in entry.reason for entry in packet.review_queue)
    json_bytes, report_bytes = packet_downloads(packet)
    assert json.loads(json_bytes)["books"][0]["frame_timestamp"] == "00:00:03.500"
    assert b"Reviewed Book" in report_bytes


def test_google_books_price_requires_exact_match_and_records_source(monkeypatch):
    payload = {"items": [{
        "volumeInfo": {
            "title": "Example Book", "authors": ["A. Writer"],
            "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9781234567890"}],
            "infoLink": "https://books.google.com/books?id=example",
        },
        "saleInfo": {
            "saleability": "FOR_SALE",
            "retailPrice": {"amount": 12.5, "currencyCode": "USD"},
            "buyLink": "https://books.google.com/books?id=example&buy=1",
        },
    }]}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(video_workflow.httpx, "get", lambda *args, **kwargs: Response())
    quote = lookup_google_books_price("Example Book", "A. Writer", "978-123-456-7890", "US")
    assert isinstance(quote, PriceQuote)
    assert quote.amount == 12.5 and quote.currency == "USD"
    assert quote.source == "Google Books eBook retail price"
    assert quote.price_type == "eBook retail price"
    assert quote.url.startswith("https://books.google.com/")
    assert datetime.fromisoformat(quote.retrieved_at)

    payload["items"][0]["volumeInfo"]["industryIdentifiers"][0]["identifier"] = "9780000000000"
    assert lookup_google_books_price("Different", "", "978-123-456-7890") is None


def test_google_books_lookup_returns_unknown_on_network_or_price_absence(monkeypatch):
    monkeypatch.setattr(
        video_workflow.httpx, "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(video_workflow.httpx.ConnectError("offline")),
    )
    assert lookup_google_books_price("Example Book", "A. Writer") is None


def test_online_quote_is_reported_but_not_counted_as_replacement_value(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_ref = "video_frame_0001.jpg"
    Image.new("RGB", (20, 20), "white").save(frames_dir / frame_ref)
    quote = PriceQuote(
        amount=12.5, currency="USD", source="Google Books eBook retail price",
        price_type="eBook retail price", url="https://books.google.com/books?id=example",
        retrieved_at="2026-10-30",
    )
    packet = build_reviewed_video_packet(
        ClaimPacket(), [{
            "id": "book-1", "frame_ref": frame_ref, "status": "identified", "title": "Example",
            "id_confidence": 0.9, "price_quote": quote.__dict__,
        }], frames_dir, {frame_ref: "00:00:01.000"},
        {"id": "sweep", "captured_at": "2026-10-30T12:00:00Z", "duration_s": 3,
         "country": "US", "currency": "USD"},
    )
    assert packet.books[0].price_quote.amount == 12.5
    assert packet.totals.books_replacement_cost is None
    _, report = packet_downloads(packet)
    assert b"eBook retail price, not included in replacement totals" in report




def test_filtered_low_specificity_candidate_remains_in_review_queue(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_ref = "video_frame_0001.jpg"
    Image.new("RGB", (20, 20), "white").save(frames_dir / frame_ref)
    candidate = {
        "id": "candidate-flagged", "frame_ref": frame_ref, "frame_timestamp": "old-time",
        "detection_box": [10, 10, 15, 18], "candidate_warnings": ["possible shelf edge"],
    }
    packet = build_reviewed_video_packet(
        ClaimPacket(), [], frames_dir, {frame_ref: "00:00:02.250"},
        {"id": "sweep", "captured_at": "2026-10-30T12:00:00Z", "duration_s": 3,
         "country": "US", "currency": "USD"},
        excluded_candidates=[candidate],
    )
    assert packet.books == []
    assert len(packet.candidates) == 1
    assert packet.candidates[0].id == "candidate-flagged"
    filtered = next(entry for entry in packet.review_queue if entry.ref_id == "candidate-flagged")
    assert "retained for manual review" in filtered.reason
    assert filtered.details["candidate"]["frame_ref"] == frame_ref
    assert filtered.details["candidate"]["frame_timestamp"] == "00:00:02.250"
    assert candidate["frame_timestamp"] == "old-time"


def test_packet_json_and_report_counts_separate_candidates_from_confirmed_books(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_ref = "video_frame_0001.jpg"
    Image.new("RGB", (20, 20), "white").save(frames_dir / frame_ref)
    scan = ClaimPacket(candidates=[
        BookCandidate(id="candidate-a", frame_ref=frame_ref, frame_timestamp="00:00:01.000",
                      detection_box=(1, 2, 3, 4), detection_confidence=0.7, spine_text="possible OCR"),
        BookCandidate(id="candidate-b", frame_ref=frame_ref, frame_timestamp="00:00:02.000"),
    ])
    packet = build_reviewed_video_packet(
        scan, [{"id": "candidate-a", "frame_ref": frame_ref, "status": "unidentified"}],
        frames_dir, {frame_ref: "00:00:01.000"},
        {"id": "sweep", "captured_at": "2026-10-30T12:00:00Z", "duration_s": 3,
         "country": "US", "currency": "USD"},
    )
    json_bytes, report_bytes = packet_downloads(packet)
    exported = json.loads(json_bytes)
    report = report_bytes.decode("utf-8")
    assert exported["totals"]["book_count"] == 1
    assert exported["totals"]["books_identified"] == 0
    assert exported["totals"]["books_unidentified"] == 1
    assert exported["candidate_count"] == len(exported["candidates"]) == 1
    assert exported["candidates"][0]["id"] == "candidate-b"
    assert "Confirmed books: 1 (0 identified, 1 confirmed but unidentified)" in report
    assert "Unreviewed detected candidates: 1" in report
    assert "dimensions: Unknown; price: Unknown" in report
    assert "bounding box=Unknown" in report
    assert "New replacement price: Unknown" in report
    assert packet.totals.shelf_run_m is None
    assert packet.totals.books_replacement_cost is None


def test_reviewed_rejection_removes_false_detection_from_pending_candidates(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_ref = "video_frame_0001.jpg"
    Image.new("RGB", (20, 20), "white").save(frames_dir / frame_ref)
    rejected = BookCandidate(id="false-positive", frame_ref=frame_ref, frame_timestamp="00:00:01.000",
                             detection_box=(1, 2, 3, 4), candidate_warnings=["possible shelf edge"])
    scan = ClaimPacket(candidates=[rejected])
    packet = build_reviewed_video_packet(
        scan, [], frames_dir, {frame_ref: "00:00:01.000"},
        {"id": "sweep", "captured_at": "2026-10-30T12:00:00Z", "duration_s": 3,
         "country": "US", "currency": "USD"},
        rejected_candidates=[rejected.model_dump(mode="python")],
    )
    assert packet.candidates == []
    entry = next(item for item in packet.review_queue if item.ref_id == "false-positive")
    assert entry.reason == "candidate rejected by reviewer as not a book"
    assert entry.details["candidate"]["detection_box"] == (1, 2, 3, 4)
