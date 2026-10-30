"""Offline reviewed-observation ingestion and claim packet export."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .checks import check_frame, evidence_problems, valid_frame_timestamp, validate_packet_evidence
from .schema import (
    Book, BookCandidate, BookReplacementCost, BookUsedValue, ClaimPacket, Item,
    ItemReplacementCost, ReviewEntry, Room, Sweep,
)
from .totals import finalize


PRICE_FIELDS = ("replacement_cost", "used_value")
BOOK_MEASURE_FIELDS = ("spine_height_cm", "spine_thickness_cm")
ROOM_MEASURE_FIELDS = (
    "length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2",
    "shelved_wall_area_m2",
)
ITEM_MEASURE_FIELDS = ("w", "h", "d")
IDENTIFICATION_FIELDS = ("title", "author", "publisher", "edition", "isbn")


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _valid_measure(value: Any, *, allow_zero: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and (number >= 0 if allow_zero else number > 0)


def _clean_measures(data: dict[str, Any], fields: tuple[str, ...], ref_id: str,
                    review: list[ReviewEntry], *, nested_key: str | None = None,
                    method_field: str | None = None) -> None:
    target = data
    if nested_key:
        candidate = data.get(nested_key)
        if not isinstance(candidate, dict):
            if candidate is not None:
                review.append(ReviewEntry(ref_id=ref_id, reason=f"invalid {nested_key}; measurement values omitted",
                                          details={"observation": _json_copy(candidate)}))
            data[nested_key] = {}
            return
        target = candidate
    for field in fields:
        if field in target and target[field] is not None and not _valid_measure(target[field]):
            bad_value = target.pop(field)
            review.append(ReviewEntry(ref_id=ref_id, reason=f"invalid numeric measurement: {field}",
                                      details={"field": field, "rejected_value": _json_copy(bad_value)}))
    if method_field and not data.get(method_field):
        unsupported = {field: target.pop(field) for field in fields if target.get(field) is not None}
        if unsupported:
            review.append(ReviewEntry(ref_id=ref_id, reason="measurement values omitted because their method is missing",
                                      details={"measurements": _json_copy(unsupported)}))


def _clean_price(data: dict[str, Any], field: str, model: type, ref_id: str,
                 review: list[ReviewEntry]) -> None:
    raw = data.get(field)
    if raw is None:
        return
    try:
        if not isinstance(raw, dict):
            raise ValueError("price record must be an object")
        model.model_validate(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        data[field] = None
        review.append(ReviewEntry(ref_id=ref_id, reason=f"unsupported {field}; excluded from totals",
                                  details={"validation_error": str(exc), "rejected_price": _json_copy(raw)}))


def _reference_problem(line: Book | Item, frames_dir: Path,
                       review: list[ReviewEntry]) -> None:
    packet = ClaimPacket(books=[line] if isinstance(line, Book) else [],
                         items=[line] if isinstance(line, Item) else [])
    for ref_id, reason in evidence_problems(packet, frames_dir):
        review.append(ReviewEntry(ref_id=ref_id, reason=f"evidence problem: {reason}"))


def _has_traceable_frame(line: Book | Item, frames_dir: Path) -> bool:
    if line.frame_ref:
        valid, _ = check_frame(line.frame_ref, frames_dir)
        return valid or valid_frame_timestamp(line.frame_timestamp)
    return valid_frame_timestamp(line.frame_timestamp)


def _room(payload: dict[str, Any], review: list[ReviewEntry]) -> Room:
    raw = payload.get("room")
    if raw is None:
        review.append(ReviewEntry(ref_id="room", reason="room measurements are missing"))
        return Room()
    if not isinstance(raw, dict):
        review.append(ReviewEntry(ref_id="room", reason="room observation must be an object",
                                  details={"observation": _json_copy(raw)}))
        return Room()
    data = _json_copy(raw)
    _clean_measures(data, ROOM_MEASURE_FIELDS, "room", review, method_field="scale_method")
    try:
        room = Room.model_validate(data)
    except ValidationError as exc:
        review.append(ReviewEntry(ref_id="room", reason="invalid room observation",
                                  details={"validation_error": str(exc), "observation": raw}))
        return Room()
    if any(getattr(room, field) is not None for field in ROOM_MEASURE_FIELDS) and not room.scale_method:
        review.append(ReviewEntry(ref_id="room", reason="room measurement method is missing"))
    if room.length_m is not None and room.width_m is not None and not room.shape_note:
        calculated_floor = room.length_m * room.width_m
        if room.floor_area_m2 is None:
            room.floor_area_m2 = calculated_floor
            review.append(ReviewEntry(ref_id="room", reason="floor area derived from length × width",
                                      details={"floor_area_m2": calculated_floor}))
        elif not math.isclose(room.floor_area_m2, calculated_floor, rel_tol=0.02, abs_tol=0.05):
            review.append(ReviewEntry(ref_id="room", reason="supplied floor area conflicts with length × width",
                                      details={"supplied_m2": room.floor_area_m2, "calculated_m2": calculated_floor}))
    if room.length_m is not None and room.width_m is not None and room.height_m is not None and not room.shape_note:
        calculated_wall = 2 * (room.length_m + room.width_m) * room.height_m
        if room.wall_area_m2 is None:
            room.wall_area_m2 = calculated_wall
            review.append(ReviewEntry(ref_id="room", reason="wall area derived from rectangular dimensions",
                                      details={"wall_area_m2": calculated_wall}))
        elif not math.isclose(room.wall_area_m2, calculated_wall, rel_tol=0.02, abs_tol=0.05):
            review.append(ReviewEntry(ref_id="room", reason="supplied wall area conflicts with rectangular dimensions",
                                      details={"supplied_m2": room.wall_area_m2, "calculated_m2": calculated_wall}))
    for field in ROOM_MEASURE_FIELDS:
        if getattr(room, field) is None:
            review.append(ReviewEntry(ref_id="room", reason=f"room measurement missing: {field}"))
    return room


def _sweep(payload: dict[str, Any], review: list[ReviewEntry]) -> Sweep:
    raw = payload.get("sweep")
    if raw is None:
        review.append(ReviewEntry(ref_id="sweep", reason="sweep metadata is missing"))
        return Sweep()
    if not isinstance(raw, dict):
        review.append(ReviewEntry(ref_id="sweep", reason="sweep metadata must be an object",
                                  details={"observation": _json_copy(raw)}))
        return Sweep()
    data = _json_copy(raw)
    if "duration_s" in data and not _valid_measure(data["duration_s"], allow_zero=True):
        review.append(ReviewEntry(ref_id="sweep", reason="invalid sweep duration", details={"value": data["duration_s"]}))
        data.pop("duration_s")
    try:
        sweep = Sweep.model_validate(data)
    except ValidationError as exc:
        review.append(ReviewEntry(ref_id="sweep", reason="invalid sweep metadata",
                                  details={"validation_error": str(exc), "observation": raw}))
        return Sweep()
    for field in ("id", "captured_at", "country", "currency"):
        if not getattr(sweep, field):
            review.append(ReviewEntry(ref_id="sweep", reason=f"sweep metadata missing: {field}"))
    if sweep.duration_s is None:
        review.append(ReviewEntry(ref_id="sweep", reason="sweep duration is missing"))
    if sweep.duration_s is not None and sweep.duration_s >= 180:
        review.append(ReviewEntry(ref_id="sweep", reason="sweep duration is not under three minutes",
                                  details={"duration_s": sweep.duration_s}))
    return sweep


def build_packet(payload: Any, frames_dir: str | Path) -> ClaimPacket:
    """Validate user-reviewed data, retain problems in review_queue, and total valid lines."""
    review: list[ReviewEntry] = []
    if not isinstance(payload, dict):
        payload = {}
        review.append(ReviewEntry(ref_id="input", reason="input must be a JSON object"))
    sweep = _sweep(payload, review)
    room = _room(payload, review)
    frames_root = Path(frames_dir)

    books: list[Book] = []
    candidates: list[BookCandidate] = []
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        review.append(ReviewEntry(ref_id="candidates", reason="candidates must be a list"))
        raw_candidates = []
    for index, raw in enumerate(raw_candidates, 1):
        try:
            candidates.append(BookCandidate.model_validate(raw))
        except (ValidationError, TypeError) as exc:
            review.append(ReviewEntry(ref_id=f"candidate-{index}", reason="candidate rejected by schema",
                                      details={"validation_error": str(exc)}))
    seen_book_ids: set[str] = set()
    raw_books = payload.get("books", [])
    if not raw_books and not raw_candidates:
        review.append(ReviewEntry(ref_id="books", reason="no book observations supplied"))
    if not isinstance(raw_books, list):
        review.append(ReviewEntry(ref_id="books", reason="books must be a list"))
        raw_books = []
    for index, raw in enumerate(raw_books, 1):
        if not isinstance(raw, dict):
            review.append(ReviewEntry(ref_id=f"book-{index}", reason="book observation must be an object",
                                      details={"observation": _json_copy(raw)}))
            continue
        data = _json_copy(raw)
        ref_id = str(data.get("id") or f"book-{index}")
        if data.get("id") and ref_id in seen_book_ids:
            review.append(ReviewEntry(ref_id=ref_id, reason="duplicate book observation excluded",
                                      details={"observation": data}))
            continue
        if not data.get("id"):
            data["id"] = ref_id
            review.append(ReviewEntry(ref_id=ref_id, reason="book has no reliable source identifier; generated local line ID"))
        if data.get("status") == "identified" and not any(data.get(field) for field in IDENTIFICATION_FIELDS):
            data["status"] = "unidentified"
            data["id_confidence"] = None
            review.append(ReviewEntry(ref_id=ref_id,
                                      reason="book marked identified without identification details; retained as confirmed but unidentified"))
        _clean_measures(data, BOOK_MEASURE_FIELDS, ref_id, review, method_field="measurement_method")
        for field, model in (("replacement_cost", BookReplacementCost), ("used_value", BookUsedValue)):
            _clean_price(data, field, model, ref_id, review)
        if data.get("status", "unidentified") != "identified":
            for price_field in PRICE_FIELDS:
                if data.get(price_field) is not None:
                    review.append(ReviewEntry(ref_id=ref_id, reason=f"{price_field} excluded because book is not identified",
                                              details={"rejected_price": data[price_field]}))
                    data[price_field] = None
        if data.get("status", "unidentified") == "unidentified" and any(data.get(k) for k in IDENTIFICATION_FIELDS):
            rejected = {key: data.get(key) for key in IDENTIFICATION_FIELDS if data.get(key)}
            for key in IDENTIFICATION_FIELDS:
                data.pop(key, None)
            review.append(ReviewEntry(ref_id=ref_id, reason="identification omitted because status is unidentified",
                                      details={"rejected_identification": rejected}))
        try:
            book = Book.model_validate(data)
        except ValidationError as exc:
            review.append(ReviewEntry(ref_id=ref_id, reason="book observation rejected by schema",
                                      details={"validation_error": str(exc), "observation": raw}))
            continue
        if raw.get("id"):
            seen_book_ids.add(ref_id)
        books.append(book)
        _reference_problem(book, frames_root, review)
        if not _has_traceable_frame(book, frames_root) and (
            book.spine_height_cm is not None or book.spine_thickness_cm is not None
        ):
            measurements = {"spine_height_cm": book.spine_height_cm,
                            "spine_thickness_cm": book.spine_thickness_cm}
            book = book.model_copy(update={"spine_height_cm": None, "spine_thickness_cm": None})
            books[-1] = book
            review.append(ReviewEntry(ref_id=book.id,
                                      reason="book measurements omitted because no valid frame or timestamp supports them",
                                      details={"measurements": measurements}))
        if book.status == "identified" and book.id_confidence is None:
            review.append(ReviewEntry(ref_id=book.id, reason="identification confidence is missing"))
        if book.status == "identified" and not any(getattr(book, field) for field in IDENTIFICATION_FIELDS):
            review.append(ReviewEntry(ref_id=book.id, reason="identified status has no title, author, publisher, edition, or ISBN"))
        if book.status == "unidentified":
            review.append(ReviewEntry(ref_id=book.id,
                                      reason="book identification requires human review; title and author remain unknown"))
        if not book.measurement_method and (book.spine_height_cm is not None or book.spine_thickness_cm is not None):
            review.append(ReviewEntry(ref_id=book.id, reason="book measurement method is missing"))
        if book.spine_height_cm is None or book.spine_thickness_cm is None:
            review.append(ReviewEntry(ref_id=book.id, reason="one or more spine measurements are missing"))
        if book.status == "needs_appraisal":
            review.append(ReviewEntry(ref_id=book.id, reason="book requires human appraisal"))
        if not book.replacement_cost or book.replacement_cost.amount is None:
            review.append(ReviewEntry(ref_id=book.id, reason="replacement price missing; excluded from replacement total"))
        elif book.replacement_cost.currency != sweep.currency:
            review.append(ReviewEntry(ref_id=book.id, reason="replacement price currency differs from sweep; excluded from total"))
        if not book.used_value or book.used_value.amount is None:
            review.append(ReviewEntry(ref_id=book.id, reason="used price missing; excluded from used-value total"))
        elif book.used_value.currency != sweep.currency:
            review.append(ReviewEntry(ref_id=book.id, reason="used price currency differs from sweep; excluded from total"))

    items: list[Item] = []
    seen_item_ids: set[str] = set()
    raw_items = payload.get("items", [])
    if not raw_items:
        review.append(ReviewEntry(ref_id="items", reason="no non-book item observations supplied"))
    if not isinstance(raw_items, list):
        review.append(ReviewEntry(ref_id="items", reason="items must be a list"))
        raw_items = []
    for index, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            review.append(ReviewEntry(ref_id=f"item-{index}", reason="item observation must be an object",
                                      details={"observation": _json_copy(raw)}))
            continue
        data = _json_copy(raw)
        ref_id = str(data.get("id") or f"item-{index}")
        if data.get("id") and ref_id in seen_item_ids:
            review.append(ReviewEntry(ref_id=ref_id, reason="duplicate item observation excluded",
                                      details={"observation": data}))
            continue
        if not data.get("id"):
            data["id"] = ref_id
            review.append(ReviewEntry(ref_id=ref_id, reason="item has no reliable source identifier; generated local line ID"))
        _clean_measures(data, ITEM_MEASURE_FIELDS, ref_id, review, nested_key="dimensions_cm",
                        method_field="measurement_method")
        _clean_price(data, "replacement_cost", ItemReplacementCost, ref_id, review)
        try:
            item = Item.model_validate(data)
        except ValidationError as exc:
            # Keep a valid observation line, but reject an inconsistent price.
            if data.get("replacement_cost") is not None and any(term in str(exc).lower() for term in ("priced item", "range item")):
                rejected_cost = data["replacement_cost"]
                data["replacement_cost"] = None
                data["status"] = "needs_appraisal"
                review.append(ReviewEntry(ref_id=ref_id, reason="inconsistent item price excluded; human review required",
                                          details={"validation_error": str(exc), "rejected_price": rejected_cost}))
                try:
                    item = Item.model_validate(data)
                except ValidationError as second:
                    review.append(ReviewEntry(ref_id=ref_id, reason="item observation rejected by schema",
                                              details={"validation_error": str(second), "observation": raw}))
                    continue
            else:
                review.append(ReviewEntry(ref_id=ref_id, reason="item observation rejected by schema",
                                          details={"validation_error": str(exc), "observation": raw}))
                continue
        if raw.get("id"):
            seen_item_ids.add(ref_id)
        items.append(item)
        _reference_problem(item, frames_root, review)
        if not _has_traceable_frame(item, frames_root) and any(
            getattr(item.dimensions_cm, field) is not None for field in ITEM_MEASURE_FIELDS
        ):
            measurements = item.dimensions_cm.model_dump()
            item = item.model_copy(update={"dimensions_cm": Dimensions()})
            items[-1] = item
            review.append(ReviewEntry(ref_id=item.id,
                                      reason="item dimensions omitted because no valid frame or timestamp supports them",
                                      details={"measurements_cm": measurements}))
        if any(getattr(item.dimensions_cm, field) is not None for field in ITEM_MEASURE_FIELDS) and not item.measurement_method:
            review.append(ReviewEntry(ref_id=item.id, reason="item measurement method is missing"))
        if all(getattr(item.dimensions_cm, field) is None for field in ITEM_MEASURE_FIELDS):
            review.append(ReviewEntry(ref_id=item.id, reason="item dimensions are missing"))
        if item.status == "needs_appraisal":
            review.append(ReviewEntry(ref_id=item.id, reason="item requires human appraisal"))
        if not item.replacement_cost or (item.replacement_cost.low is None and item.replacement_cost.high is None):
            review.append(ReviewEntry(ref_id=item.id, reason="replacement price missing; excluded from item total"))
        elif item.replacement_cost.currency != sweep.currency:
            review.append(ReviewEntry(ref_id=item.id, reason="item price currency differs from sweep; excluded from total"))

    packet = ClaimPacket(sweep=sweep, room=room, books=books, candidates=candidates, items=items, review_queue=review)
    for index, frame_ref in enumerate(room.frame_refs, 1):
        valid, reason = check_frame(frame_ref, frames_root)
        if not valid:
            packet.review_queue.append(ReviewEntry(ref_id=f"room-frame-{index}",
                                                   reason=f"evidence problem: {reason}"))
    return finalize(validate_packet_evidence(packet, frames_root))


def _fmt(value: float | None, unit: str) -> str:
    return "Unknown" if value is None else f"{value:g} {unit}".strip()


def render_report(packet: ClaimPacket) -> str:
    """Create a human-readable report without filling unknown values."""
    t, r = packet.totals, packet.room
    evidence_problem_ids = {
        entry.ref_id for entry in packet.review_queue if entry.reason.startswith("evidence problem:")
    }
    confirmed_books = [b for b in packet.books if b.id not in evidence_problem_ids]
    uncertain_books = [b for b in packet.books if b not in confirmed_books]
    confirmed_items = [i for i in packet.items if i.status in ("priced", "range") and
                       i.confidence is not None and i.id not in evidence_problem_ids]
    uncertain_items = [i for i in packet.items if i not in confirmed_items]
    lines = [
        "# Library Contents Inventory Report", "",
        f"Sweep: {packet.sweep.id or 'unknown'} | Captured: {packet.sweep.captured_at or 'unknown'} | Duration: {_fmt(packet.sweep.duration_s, 's')}",
        f"Country/currency: {packet.sweep.country or 'unknown'} / {packet.sweep.currency or 'unknown'}", "",
        "## Room measurements", "",
        f"Method: {r.scale_method or 'not recorded'}; confidence: {_fmt(r.confidence, '')}",
    ]
    for label, value, unit in (
        ("Length", r.length_m, "m"), ("Width", r.width_m, "m"), ("Height", r.height_m, "m"),
        ("Floor area", r.floor_area_m2, "m²"), ("Wall area", r.wall_area_m2, "m²"),
        ("Shelved wall area", r.shelved_wall_area_m2, "m²"),
    ):
        imperial = ""
        if value is not None:
            imperial = f" ({value * 3.28084:.2f} ft)" if unit == "m" else f" ({value * 10.7639:.2f} ft²)"
        lines.append(f"- {label}: {_fmt(value, unit)}{imperial}")
    if r.shape_note:
        lines.append(f"- Shape note: {r.shape_note}")
    if r.frame_refs or r.frame_timestamps:
        lines.append(f"- Room evidence: frames={', '.join(r.frame_refs) or 'none'}; timestamps={', '.join(r.frame_timestamps) or 'none'}")
    lines += ["", "## Inventory totals", "", f"- Confirmed books: {t.book_count} ({t.books_identified} identified, {t.books_unidentified} confirmed but unidentified)",
              f"- Unreviewed detected candidates: {packet.candidate_count} (excluded from confirmed inventory)",
              f"- Upright shelf run: {_fmt(t.shelf_run_m, 'm')}",
              f"- Book replacement cost supported in {packet.sweep.currency or 'unknown currency'}: {_fmt(t.books_replacement_cost, packet.sweep.currency)}",
              f"- Book used value supported in {packet.sweep.currency or 'unknown currency'}: {_fmt(t.books_used_value, packet.sweep.currency)}",
              f"- Item replacement range supported in {packet.sweep.currency or 'unknown currency'}: {_fmt(t.items_replacement_cost_low, packet.sweep.currency)}–{_fmt(t.items_replacement_cost_high, packet.sweep.currency)}",
              f"- Lines excluded from all money totals: {t.excluded_from_totals}",
              f"- Lines excluded from all calculations for invalid evidence: {t.lines_without_valid_evidence}",
              f"- Books without supported replacement price: {t.books_without_replacement_price}",
              f"- Books without supported used price: {t.books_without_used_price}",
              f"- Items without supported price: {t.items_without_price}", "", "## Confirmed observations", ""]
    for book in confirmed_books:
        label = book.title or "Confirmed but unidentified book"
        authors = f" — {book.author}" if book.author else ""
        lines.append(f"- [{book.id}] {label}{authors}; status={book.status}; shelf={book.shelf or 'unknown'}; orientation={book.orientation}; frame={book.frame_ref or book.frame_timestamp or 'missing'}")
        lines.append(f"  - Dimensions: height {_fmt(book.spine_height_cm, 'cm')}, thickness {_fmt(book.spine_thickness_cm, 'cm')}; method={book.measurement_method or 'not recorded'}")
        if book.replacement_cost and book.replacement_cost.amount is not None:
            c = book.replacement_cost
            lines.append(f"  - New price: {c.amount:g} {c.currency}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
            if c.converted:
                lines.append(f"  - Conversion: {c.original_amount:g} {c.original_currency} × {c.conversion_rate:g} from {c.conversion_source} on {c.conversion_date}")
        else:
            lines.append("  - New replacement price: Unknown")
        if book.used_value and book.used_value.amount is not None:
            c = book.used_value
            lines.append(f"  - Used price: {c.amount:g} {c.currency}; condition={c.condition_assumed}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
            if c.converted:
                lines.append(f"  - Conversion: {c.original_amount:g} {c.original_currency} × {c.conversion_rate:g} from {c.conversion_source} on {c.conversion_date}")
        else:
            lines.append("  - Used price: Unknown")
        if book.price_quote:
            q = book.price_quote
            lines.append(f"  - Online quote ({q.price_type}, not included in replacement totals): {q.amount:g} {q.currency}; source={q.source}; URL={q.url}; retrieved={q.retrieved_at}")
    for item in confirmed_items:
        lines.append(f"- [{item.id}] {item.category or 'Uncategorized item'}: {item.description or 'no description'}; status={item.status}; frame={item.frame_ref or item.frame_timestamp or 'missing'}")
        lines.append(f"  - Dimensions (w×h×d): {_fmt(item.dimensions_cm.w, 'cm')} × {_fmt(item.dimensions_cm.h, 'cm')} × {_fmt(item.dimensions_cm.d, 'cm')}; method={item.measurement_method or 'not recorded'}")
        if item.replacement_cost and (item.replacement_cost.low is not None or item.replacement_cost.high is not None):
            c = item.replacement_cost
            lines.append(f"  - Replacement: {_fmt(c.low, c.currency)} to {_fmt(c.high, c.currency)}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
        else:
            lines.append("  - Replacement price: Unknown")
    if not packet.books and not packet.items:
        lines.append("- No book or non-book observations were supplied.")
    lines += ["", "## Unreviewed detected candidates", ""]
    if packet.candidates:
        for candidate in packet.candidates:
            box = list(candidate.detection_box) if candidate.detection_box else "Unknown"
            score = _fmt(candidate.detection_confidence, "")
            lines.append(f"- [candidate {candidate.id}] Book identity: Unknown; evidence frame={candidate.frame_ref or 'Unknown'}, timestamp={candidate.frame_timestamp or 'Unknown'}, bounding box={box}, contour score={score} (not a probability).")
            lines.append(f"  - OCR text evidence: {candidate.spine_text or 'Unknown'}; dimensions: Unknown; price: Unknown.")
    else:
        lines.append("- None.")
    lines += ["", "## Uncertain observations and missing information", ""]
    for book in uncertain_books:
        lines.append(f"- [book {book.id}] {book.title or 'Unidentified book'}; status={book.status}; shelf={book.shelf or 'unknown'}; evidence={book.frame_ref or book.frame_timestamp or 'missing'}")
        lines.append(f"  - Dimensions: height {_fmt(book.spine_height_cm, 'cm')}, thickness {_fmt(book.spine_thickness_cm, 'cm')}; method={book.measurement_method or 'not recorded'}")
        if book.replacement_cost and book.replacement_cost.amount is not None:
            c = book.replacement_cost
            lines.append(f"  - New price: {c.amount:g} {c.currency}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
        if book.used_value and book.used_value.amount is not None:
            c = book.used_value
            lines.append(f"  - Used price: {c.amount:g} {c.currency}; condition={c.condition_assumed}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
            if c.converted:
                lines.append(f"  - Conversion: {c.original_amount:g} {c.original_currency} × {c.conversion_rate:g} from {c.conversion_source} on {c.conversion_date}")
        if book.price_quote:
            q = book.price_quote
            lines.append(f"  - Online quote ({q.price_type}, not included in replacement totals): {q.amount:g} {q.currency}; source={q.source}; URL={q.url}; retrieved={q.retrieved_at}")
    for item in uncertain_items:
        lines.append(f"- [item {item.id}] {item.category or 'Uncategorized item'}; status={item.status}; evidence={item.frame_ref or item.frame_timestamp or 'missing'}")
        lines.append(f"  - Dimensions (w×h×d): {_fmt(item.dimensions_cm.w, 'cm')} × {_fmt(item.dimensions_cm.h, 'cm')} × {_fmt(item.dimensions_cm.d, 'cm')}; method={item.measurement_method or 'not recorded'}")
        if item.replacement_cost and (item.replacement_cost.low is not None or item.replacement_cost.high is not None):
            c = item.replacement_cost
            lines.append(f"  - Replacement: {_fmt(c.low, c.currency)} to {_fmt(c.high, c.currency)}; source={c.source}; URL={c.url}; retrieved={c.retrieved_at}")
    if packet.review_queue:
        for entry in packet.review_queue:
            detail = f" — {json.dumps(entry.details, ensure_ascii=False, sort_keys=True)}" if entry.details else ""
            lines.append(f"- [{entry.ref_id}] {entry.reason}{detail}")
    else:
        lines.append("- No review entries.")
    lines += ["", "## Appraisal and human review", ""]
    appraisal = [entry for entry in packet.review_queue if "appraisal" in entry.reason or "human review" in entry.reason]
    if appraisal:
        for entry in appraisal:
            lines.append(f"- [{entry.ref_id}] {entry.reason}")
    else:
        lines.append("- None explicitly flagged.")
    lines += ["", "Totals include only sourced amounts in the sweep currency. Unknown or unsupported amounts are excluded, not treated as confirmed zero-valued observations.", ""]
    return "\n".join(lines)


def process_file(input_path: str | Path, frames_dir: str | Path,
                 output_dir: str | Path) -> tuple[Path, Path]:
    source = Path(input_path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read input JSON {source}: {exc}") from exc
    packet = build_packet(payload, frames_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "claim_packet.json"
    report_path = target / "inventory_report.md"
    json_path.write_text(json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(packet), encoding="utf-8")
    return json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline claim packet from reviewed inventory JSON.")
    parser.add_argument("input", help="Path to reviewed observations JSON")
    parser.add_argument("--frames-dir", default="data/frames", help="Permitted directory for saved evidence images")
    parser.add_argument("--output-dir", default="output", help="Directory for claim_packet.json and inventory_report.md")
    args = parser.parse_args()
    try:
        packet_path, report_path = process_file(args.input, args.frames_dir, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {packet_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
