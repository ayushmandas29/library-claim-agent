import json
import io
import struct
import zlib
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from claimagent.checks import check_frame
from claimagent.pipeline import build_packet, process_file, render_report
from claimagent.schema import Book, BookReplacementCost, ClaimPacket, Dimensions, ItemReplacementCost


def make_frame(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def make_truncated_pixel_data_frame(path: Path) -> Path:
    """Write a PNG with valid chunk CRCs but an incomplete compressed image stream."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    source = buffer.getvalue()
    signature, chunks, offset = source[:8], [], 8
    while offset < len(source):
        length = struct.unpack(">I", source[offset:offset + 4])[0]
        kind = source[offset + 4:offset + 8]
        payload = source[offset + 8:offset + 8 + length]
        chunks.append((kind, payload))
        offset += 12 + length
    rebuilt = bytearray(signature)
    for kind, payload in chunks:
        if kind == b"IDAT":
            payload = payload[: max(2, len(payload) // 2)]
        rebuilt.extend(struct.pack(">I", len(payload)))
        rebuilt.extend(kind)
        rebuilt.extend(payload)
        rebuilt.extend(struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rebuilt)
    return path


def base_payload(frame="book.jpg"):
    return {
        "sweep": {"id": "s-1", "captured_at": "2026-10-29T10:00:00Z", "country": "IN", "currency": "INR", "duration_s": 90},
        "room": {"length_m": 4, "width_m": 3, "height_m": 2.5, "shelved_wall_area_m2": 5, "scale_method": "known wall dimensions", "confidence": 0.8},
        "books": [{
            "id": "b-1", "frame_ref": frame, "frame_timestamp": "00:00:04.200", "shelf": "A",
            "status": "identified", "title": "Observed Title", "author": "Observed Author", "publisher": "Observed Publisher",
            "id_confidence": 0.91, "spine_height_cm": 20, "spine_thickness_cm": 3,
            "measurement_method": "marker scale", "replacement_cost": {
                "amount": 200, "currency": "INR", "source": "Retail listing", "url": "https://example.test/book",
                "retrieved_at": "2026-10-29"
            }, "used_value": {
                "amount": 80, "currency": "INR", "source": "Used listing", "url": "https://example.test/used",
                "retrieved_at": "2026-10-29", "condition_assumed": "good"
            }
        }, {
            "id": "b-2", "frame_timestamp": "00:00:06.100", "shelf": "B", "status": "unidentified",
            "spine_thickness_cm": 2, "orientation": "flat"
        }],
        "items": [{
            "id": "i-1", "frame_ref": "item.jpg", "category": "lamp", "description": "Desk lamp",
            "dimensions_cm": {"w": 10, "h": 30, "d": 12}, "measurement_method": "marker scale",
            "status": "range", "replacement_cost": {
                "low": 1000, "high": 1500, "currency": "INR", "source": "Retail listings",
                "url": "https://example.test/lamp", "retrieved_at": "2026-10-29"
            }
        }]
    }


def test_valid_packet_totals_and_json_round_trip(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    make_frame(frames / "item.jpg")
    packet = build_packet(base_payload(), frames)
    assert packet.totals.book_count == 2
    assert packet.totals.books_identified == 1
    assert packet.totals.books_unidentified == 1
    assert packet.totals.shelf_run_m == pytest.approx(0.03)
    assert packet.totals.books_replacement_cost == 200
    assert packet.totals.books_used_value == 80
    assert (packet.totals.items_replacement_cost_low, packet.totals.items_replacement_cost_high) == (1000, 1500)
    assert packet.room.floor_area_m2 == 12
    assert packet.room.wall_area_m2 == 35
    raw = packet.model_dump(mode="json")
    assert set(raw) >= {"sweep", "room", "books", "items", "totals", "review_queue"}
    restored = type(packet).model_validate_json(json.dumps(raw))
    assert restored == packet
    assert packet.books[1].status == "unidentified" and packet.books[1].title == ""


def test_frame_paths_missing_unreadable_and_outside_are_rejected(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    make_frame(tmp_path / "outside.jpg")
    (frames / "broken.jpg").write_text("not an image", encoding="utf-8")
    assert check_frame("missing.jpg", frames)[0] is False
    assert check_frame("broken.jpg", frames)[0] is False
    assert check_frame("../outside.jpg", frames)[0] is False
    payload = base_payload()
    payload["books"] = [
        {"id": "missing", "frame_ref": "missing.jpg", "status": "unidentified"},
        {"id": "broken", "frame_ref": "broken.jpg", "status": "unidentified"},
        {"id": "escape", "frame_ref": "../outside.jpg", "status": "unidentified"},
    ]
    payload["items"] = []
    packet = build_packet(payload, frames)
    assert len(packet.books) == 3
    assert sum("evidence problem" in e.reason for e in packet.review_queue) == 3


def test_frame_with_valid_structure_but_truncated_pixels_is_rejected(tmp_path):
    frames = tmp_path / "frames"
    corrupted = make_truncated_pixel_data_frame(frames / "truncated.png")
    # Pillow's structure-only verification succeeds; full decoding must fail safely.
    with Image.open(corrupted) as image:
        image.verify()
    valid, reason = check_frame("truncated.png", frames)
    assert not valid
    assert "cannot be opened" in reason


def test_unsupported_prices_are_removed_from_totals_and_queued(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    make_frame(frames / "item.jpg")
    payload = base_payload()
    payload["books"][0]["replacement_cost"] = {"amount": 999, "source": "", "url": "", "retrieved_at": "", "currency": "INR"}
    payload["items"][0]["replacement_cost"] = {"low": 100, "high": 50, "source": "Listing", "url": "https://example.test", "retrieved_at": "2026-10-29", "currency": "INR"}
    packet = build_packet(payload, frames)
    assert packet.books[0].replacement_cost is None
    assert packet.totals.books_replacement_cost is None
    assert packet.items[0].replacement_cost is None
    assert packet.totals.items_replacement_cost_low is None
    assert any("unsupported replacement_cost" in e.reason for e in packet.review_queue)
    assert any("without a supported" in e.reason or "missing; excluded" in e.reason for e in packet.review_queue)


def test_invalid_dimensions_are_not_confirmed_and_ranges_rejected(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    make_frame(frames / "item.jpg")
    payload = base_payload()
    payload["books"][0]["spine_height_cm"] = -1
    payload["items"][0]["dimensions_cm"]["w"] = "wide"
    packet = build_packet(payload, frames)
    assert packet.books[0].spine_height_cm is None
    assert packet.items[0].dimensions_cm.w is None
    assert any("invalid numeric measurement" in e.reason for e in packet.review_queue)
    with pytest.raises(ValidationError):
        Dimensions(w=-1)
    with pytest.raises(ValidationError):
        Dimensions(w=0)
    with pytest.raises(ValidationError):
        ItemReplacementCost(low=10, high=9, source="x", url="https://example.test", retrieved_at="2026-10-29", currency="INR")


def test_incomplete_and_empty_input_generates_reviewable_packet(tmp_path):
    packet = build_packet({}, tmp_path)
    assert packet.books == [] and packet.items == []
    assert packet.totals.book_count == 0
    assert packet.review_queue
    assert any(entry.ref_id == "sweep" for entry in packet.review_queue)
    assert any(entry.ref_id == "room" for entry in packet.review_queue)
    assert any(entry.ref_id == "books" for entry in packet.review_queue)
    assert any(entry.ref_id == "items" for entry in packet.review_queue)


def test_unidentified_book_is_explicitly_queued_for_identification_review(tmp_path):
    payload = base_payload()
    payload["books"] = [{
        "id": "unknown-book", "frame_timestamp": "00:00:08.200", "status": "unidentified"
    }]
    payload["items"] = []
    packet = build_packet(payload, tmp_path)

    identification_review = [
        entry for entry in packet.review_queue
        if entry.ref_id == "unknown-book" and "identification requires human review" in entry.reason
    ]
    assert len(identification_review) == 1
    reasons = [entry.reason for entry in packet.review_queue if entry.ref_id == "unknown-book"]
    assert "one or more spine measurements are missing" in reasons
    assert "replacement price missing; excluded from replacement total" in reasons
    assert "used price missing; excluded from used-value total" in reasons

    report = render_report(packet)
    for entry in packet.review_queue:
        assert f"- [{entry.ref_id}] {entry.reason}" in report
    section = report.split("## Appraisal and human review", 1)[1]
    assert "[unknown-book] book identification requires human review" in section
    assert "None explicitly flagged." not in section


def test_identified_status_without_identity_is_counted_as_confirmed_unidentified(tmp_path):
    packet = build_packet({
        "sweep": {"id": "s", "captured_at": "2026-10-29T10:00:00Z", "country": "IN", "currency": "INR", "duration_s": 30},
        "books": [{"id": "accepted", "status": "identified", "frame_timestamp": "00:00:02.000", "id_confidence": 0.9}],
        "items": [],
    }, tmp_path)
    assert packet.totals.book_count == 1
    assert packet.totals.books_identified == 0
    assert packet.totals.books_unidentified == 1
    assert packet.books[0].status == "unidentified"
    assert "confirmed but unidentified" in render_report(packet)


def test_duplicates_do_not_double_count(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    payload = base_payload()
    payload["books"] = [payload["books"][0], dict(payload["books"][0])]
    payload["items"] = []
    packet = build_packet(payload, frames)
    assert len(packet.books) == 1
    assert packet.totals.book_count == 1
    assert any("duplicate book" in entry.reason for entry in packet.review_queue)


def test_file_outputs_repeat_deterministically(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    make_frame(frames / "item.jpg")
    source = tmp_path / "reviewed.json"
    source.write_text(json.dumps(base_payload()), encoding="utf-8")
    output = tmp_path / "out"
    json_path, report_path = process_file(source, frames, output)
    first_json, first_report = json_path.read_bytes(), report_path.read_bytes()
    exported_packet = ClaimPacket.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert exported_packet.totals == build_packet(base_payload(), frames).totals
    process_file(source, frames, output)
    assert json_path.read_bytes() == first_json
    assert report_path.read_bytes() == first_report
    report = first_report.decode("utf-8")
    assert "Method: known wall dimensions" in report
    assert "marker scale" in report
    assert "Retail listings" in report


def test_measurements_without_methods_are_not_counted(tmp_path):
    payload = base_payload()
    payload["books"] = [{"id": "b-1", "status": "unidentified", "spine_thickness_cm": 8}]
    payload["items"] = []
    payload["room"] = {"length_m": 5, "width_m": 4}
    packet = build_packet(payload, tmp_path)
    assert packet.books[0].spine_thickness_cm is None
    assert packet.totals.shelf_run_m is None
    assert packet.room.length_m is None and packet.room.floor_area_m2 is None
    assert any("method is missing" in entry.reason for entry in packet.review_queue)


def test_non_rectangular_room_area_is_not_inferred(tmp_path):
    payload = base_payload()
    payload["room"]["shape_note"] = "L-shaped room"
    payload["books"] = []
    payload["items"] = []
    packet = build_packet(payload, tmp_path)
    assert packet.room.floor_area_m2 is None
    assert packet.room.wall_area_m2 is None


def test_measurements_without_valid_frame_or_timestamp_are_not_counted(tmp_path):
    payload = base_payload()
    payload["books"] = [{
        "id": "b-1", "frame_ref": "missing.jpg", "status": "unidentified",
        "spine_thickness_cm": 6, "measurement_method": "reviewed caliper"
    }]
    payload["items"] = []
    packet = build_packet(payload, tmp_path / "frames")
    assert packet.books[0].spine_thickness_cm is None
    assert packet.totals.shelf_run_m is None
    issue = next(entry for entry in packet.review_queue if "measurements omitted because no valid frame" in entry.reason)
    assert issue.details["measurements"]["spine_thickness_cm"] == 6


def test_invalid_timestamp_does_not_preserve_measurements_when_frame_is_missing(tmp_path):
    payload = base_payload()
    payload["books"] = [{
        "id": "b-invalid-time", "frame_ref": "missing.jpg", "frame_timestamp": "not-a-timecode",
        "status": "unidentified", "spine_thickness_cm": 6, "measurement_method": "reviewed caliper",
    }]
    payload["items"] = []

    packet = build_packet(payload, tmp_path / "frames")

    assert packet.books[0].spine_thickness_cm is None
    assert packet.totals.shelf_run_m is None
    issue = next(entry for entry in packet.review_queue
                 if entry.ref_id == "b-invalid-time" and "measurements omitted because no valid frame" in entry.reason)
    assert issue.details["measurements"]["spine_thickness_cm"] == 6


def test_prices_in_another_currency_stay_out_of_sweep_totals(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    payload = base_payload()
    payload["books"] = [payload["books"][0]]
    payload["books"][0]["replacement_cost"]["currency"] = "USD"
    payload["books"][0].pop("used_value")
    payload["items"] = []
    packet = build_packet(payload, frames)
    assert packet.books[0].replacement_cost.amount == 200
    assert packet.totals.books_replacement_cost is None
    assert any("currency differs" in entry.reason for entry in packet.review_queue)


def test_invalid_or_unlabelled_price_provenance_is_rejected():
    with pytest.raises(ValidationError):
        BookReplacementCost(amount=20, source="seller", url="not-a-url", retrieved_at="2026-10-29", currency="INR")
    with pytest.raises(ValidationError):
        BookReplacementCost(amount=20, source="seller", url="https://example.test/book", retrieved_at="2026-10-29")
    with pytest.raises(ValidationError):
        BookReplacementCost(amount=20, source="seller", url="https://example.test/book", retrieved_at="2026-10-29", currency="INR", converted=True)
    converted = BookReplacementCost(
        amount=200, currency="INR", source="seller", url="https://example.test/book",
        retrieved_at="2026-10-29", converted=True, original_amount=2,
        original_currency="USD", conversion_rate=100, conversion_source="reviewed FX source",
        conversion_date="2026-10-29",
    )
    assert converted.amount == 200
    with pytest.raises(ValidationError):
        BookReplacementCost(
            amount=201, currency="INR", source="seller", url="https://example.test/book",
            retrieved_at="2026-10-29", converted=True, original_amount=2,
            original_currency="USD", conversion_rate=100, conversion_source="reviewed FX source",
            conversion_date="2026-10-29",
        )


def test_malformed_retrieval_date_price_is_excluded_from_totals(tmp_path):
    frames = tmp_path / "frames"
    make_frame(frames / "book.jpg")
    make_frame(frames / "item.jpg")
    payload = base_payload()
    payload["books"] = [payload["books"][0]]
    payload["books"][0]["replacement_cost"]["retrieved_at"] = "2026-01-01not-a-date"
    payload["items"] = []
    packet = build_packet(payload, frames)
    assert packet.books[0].replacement_cost is None
    assert packet.totals.books_replacement_cost is None
    assert any("unsupported replacement_cost" in entry.reason for entry in packet.review_queue)
    timestamp_price = BookReplacementCost(
        amount=10, currency="INR", source="seller", url="https://example.test/item",
        retrieved_at="2026-01-01T09:30:00Z",
    )
    assert timestamp_price.amount == 10


def test_invalid_first_book_id_does_not_suppress_valid_later_book(tmp_path):
    payload = base_payload()
    payload["books"] = [
        {"id": "same-book", "status": "not-a-status"},
        {"id": "same-book", "status": "identified", "title": "Reviewed title",
         "id_confidence": 0.9, "frame_timestamp": "00:00:03.000"},
    ]
    payload["items"] = []
    packet = build_packet(payload, tmp_path)
    assert len(packet.books) == 1
    assert packet.books[0].id == "same-book"
    assert packet.books[0].title == "Reviewed title"
    assert not any("duplicate book" in entry.reason for entry in packet.review_queue)


def test_invalid_first_item_id_does_not_suppress_valid_later_item(tmp_path):
    payload = base_payload()
    payload["books"] = []
    payload["items"] = [
        {"id": "same-item", "category": "lamp", "status": "not-a-status"},
        {"id": "same-item", "category": "lamp", "status": "needs_appraisal",
         "frame_timestamp": "00:00:05.000", "confidence": 0.8},
    ]
    packet = build_packet(payload, tmp_path)
    assert len(packet.items) == 1
    assert packet.items[0].id == "same-item"
    assert packet.items[0].status == "needs_appraisal"
    assert not any("duplicate item" in entry.reason for entry in packet.review_queue)


