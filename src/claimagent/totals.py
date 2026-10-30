"""Deterministic totals. Only sourced prices in the sweep currency are summed."""
from __future__ import annotations

import math

from .checks import EvidenceCheckedPacket
from .schema import ClaimPacket, Totals


def compute_totals(checked: EvidenceCheckedPacket) -> Totals:
    """Calculate totals only after evidence checks have produced a checked packet."""
    if not isinstance(checked, EvidenceCheckedPacket):
        raise TypeError("compute_totals requires validate_packet_evidence(packet, frames_dir) output")
    packet = checked.packet
    if len(packet.books) != len(checked.valid_books) or len(packet.items) != len(checked.valid_items):
        raise ValueError("packet lines changed after evidence validation; validate the packet again")
    books = [book for book, valid in zip(packet.books, checked.valid_books) if valid]
    items = [item for item, valid in zip(packet.items, checked.valid_items) if valid]
    currency = packet.sweep.currency
    upright_thickness_cm = [
        b.spine_thickness_cm for b in books
        if b.orientation == "upright" and b.spine_thickness_cm is not None
        and b.measurement_method
    ]
    replacement, used, lows, highs = [], [], [], []
    excluded = 0
    no_replacement = no_used = no_item_price = 0

    for book in books:
        replacement_cost = book.replacement_cost
        has_replacement = bool(
            book.status == "identified" and replacement_cost
            and replacement_cost.amount is not None and replacement_cost.currency == currency
        )
        used_cost = book.used_value
        has_used = bool(
            book.status == "identified" and used_cost
            and used_cost.amount is not None and used_cost.currency == currency
        )
        if has_replacement:
            replacement.append(replacement_cost.amount)
        else:
            no_replacement += 1
        if has_used:
            used.append(used_cost.amount)
        else:
            no_used += 1
        if not (has_replacement or has_used):
            excluded += 1

    for item in items:
        cost = item.replacement_cost
        supported = bool(
            item.status in ("priced", "range") and cost
            and (cost.low is not None or cost.high is not None)
            and cost.currency == currency
        )
        if supported:
            low = cost.low if cost.low is not None else cost.high
            high = cost.high if cost.high is not None else cost.low
            lows.append(low)
            highs.append(high)
        else:
            no_item_price += 1
            excluded += 1

    return Totals(
        book_count=len(books),
        books_identified=sum(b.status == "identified" for b in books),
        books_unidentified=sum(b.status != "identified" for b in books),
        shelf_run_m=round(math.fsum(upright_thickness_cm) / 100, 3) if upright_thickness_cm else None,
        books_replacement_cost=round(math.fsum(replacement), 2) if replacement else None,
        books_used_value=round(math.fsum(used), 2) if used else None,
        items_replacement_cost_low=round(math.fsum(lows), 2) if lows else None,
        items_replacement_cost_high=round(math.fsum(highs), 2) if highs else None,
        excluded_from_totals=excluded,
        books_without_replacement_price=no_replacement,
        books_without_used_price=no_used,
        items_without_price=no_item_price,
        lines_without_valid_evidence=sum(not valid for valid in checked.valid_books)
        + sum(not valid for valid in checked.valid_items),
    )


def finalize(checked: EvidenceCheckedPacket) -> ClaimPacket:
    """Return a copy with totals recomputed from inventory lines."""
    return checked.packet.model_copy(update={"totals": compute_totals(checked)})
