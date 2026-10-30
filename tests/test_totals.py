import pytest
from pydantic import ValidationError

from claimagent.schema import (
    Book, BookReplacementCost, BookUsedValue, ClaimPacket, Item,
    ItemReplacementCost, Sweep,
)
from claimagent.checks import validate_packet_evidence
from claimagent.totals import compute_totals

SWEEP = Sweep(id="s1", captured_at="2026-10-28T00:00:00Z", country="IN", currency="INR")
SRC = dict(source="example.test", url="https://example.test/p", retrieved_at="2026-10-28", currency="INR")


def book(i, status="identified", thick=3.0, orient="upright", price=None, used=None):
    kw = {}
    if status == "identified":
        kw = dict(title=f"T{i}", author="A")
    return Book(
        id=f"b{i}", frame_ref=f"f{i}.jpg", frame_timestamp="00:00:00.001", status=status, spine_thickness_cm=thick,
        orientation=orient, measurement_method="reviewed caliper",
        replacement_cost=BookReplacementCost(amount=price, **SRC) if price is not None else None,
        used_value=BookUsedValue(amount=used, condition_assumed="good", **SRC) if used is not None else None,
        **kw,
    )


def test_totals_sum_from_lines():
    p = ClaimPacket(sweep=SWEEP, books=[
        book(1, price=500, used=200),
        book(2, price=300),
        book(3, status="unidentified", thick=2.0),
        book(4, status="needs_appraisal", thick=4.0),
        book(5, orient="flat", thick=5.0, price=100),
    ])
    t = compute_totals(validate_packet_evidence(p, "."))
    assert t.book_count == 5
    assert t.books_identified == 3
    assert t.books_unidentified == 2
    assert t.books_replacement_cost == 900
    assert t.books_used_value == 200
    # flat book excluded from shelf run: 3+3+2+4 = 12 cm
    assert t.shelf_run_m == pytest.approx(0.12)
    # unidentified + needs_appraisal contribute nothing
    assert t.excluded_from_totals == 2


def test_item_ranges_and_exclusions():
    p = ClaimPacket(sweep=SWEEP, items=[
        Item(id="i1", category="lamp", frame_ref="f.jpg", frame_timestamp="00:00:00.001", status="range",
             replacement_cost=ItemReplacementCost(low=1000, high=2000, **SRC)),
        Item(id="i2", category="portrait", frame_ref="f.jpg", frame_timestamp="00:00:00.001", status="needs_appraisal"),
    ])
    t = compute_totals(validate_packet_evidence(p, "."))
    assert (t.items_replacement_cost_low, t.items_replacement_cost_high) == (1000, 2000)
    assert t.excluded_from_totals == 1


def test_price_without_source_is_rejected():
    with pytest.raises(ValidationError):
        BookReplacementCost(amount=499)


def test_unidentified_book_cannot_carry_a_guessed_title():
    with pytest.raises(ValidationError):
        Book(id="b", frame_ref="f.jpg", status="unidentified", title="Guess")


def test_compute_totals_requires_evidence_checked_packet():
    packet = ClaimPacket(sweep=SWEEP, books=[book(1, price=500)])
    with pytest.raises(TypeError, match="validate_packet_evidence"):
        compute_totals(packet)


def test_invalid_frame_evidence_excludes_line_from_totals(tmp_path):
    packet = ClaimPacket(sweep=SWEEP, books=[Book(
        id="b-no-frame", status="identified", title="Observed", id_confidence=0.9,
        spine_thickness_cm=4, measurement_method="caliper",
        replacement_cost=BookReplacementCost(amount=700, **SRC),
    )])
    checked = validate_packet_evidence(packet, tmp_path)
    totals = compute_totals(checked)
    assert totals.book_count == 0
    assert totals.books_replacement_cost is None
    assert totals.shelf_run_m is None
    assert totals.lines_without_valid_evidence == 1
    checked.packet.books[0].frame_timestamp = "00:00:00.002"
    assert compute_totals(checked).book_count == 0


def test_invalid_frame_timestamp_does_not_validate_evidence(tmp_path):
    packet = ClaimPacket(sweep=SWEEP, books=[Book(
        id="b-bad-time", status="identified", title="Observed", id_confidence=0.9,
        frame_timestamp="not-a-timecode", spine_thickness_cm=4,
        measurement_method="caliper",
    )])
    totals = compute_totals(validate_packet_evidence(packet, tmp_path))
    assert totals.book_count == 0
    assert totals.shelf_run_m is None
    assert totals.lines_without_valid_evidence == 1
