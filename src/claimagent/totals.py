"""Totals are computed here, in code, from the lines. Never by a language model.

Definitions (state these in the README):
  * book_count            = every book line, any status.
  * shelf_run_m           = sum of spine thickness of UPRIGHT books only
                            (flat/stacked books don't consume shelf run).
  * books_replacement_cost / books_used_value
                          = sum over books whose status is "identified" AND that
                            have a sourced amount. needs_appraisal is never summed.
  * items_*_low / high    = sum over items with status "priced" or "range" and a
                            sourced cost. For "priced", low == high.
  * excluded_from_totals  = number of lines (books + items) that contribute
                            nothing to the money totals: unpriced, unidentified,
                            or needs_appraisal.
"""
from __future__ import annotations

import math

from .schema import ClaimPacket, Totals


def compute_totals(packet: ClaimPacket) -> Totals:
    books, items = packet.books, packet.items

    upright_thickness_cm = [
        b.spine_thickness_cm
        for b in books
        if b.orientation == "upright" and b.spine_thickness_cm is not None
    ]

    repl, used = [], []
    excluded = 0
    for b in books:
        counted = False
        if b.status == "identified":
            if b.replacement_cost and b.replacement_cost.amount is not None:
                repl.append(b.replacement_cost.amount)
                counted = True
            if b.used_value and b.used_value.amount is not None:
                used.append(b.used_value.amount)
                counted = True
        if not counted:
            excluded += 1

    lows, highs = [], []
    for it in items:
        c = it.replacement_cost
        if it.status in ("priced", "range") and c and (
            c.low is not None or c.high is not None
        ):
            lo = c.low if c.low is not None else c.high
            hi = c.high if c.high is not None else c.low
            lows.append(lo)
            highs.append(hi)
        else:
            excluded += 1

    return Totals(
        book_count=len(books),
        books_identified=sum(b.status == "identified" for b in books),
        books_unidentified=sum(b.status == "unidentified" for b in books),
        shelf_run_m=round(math.fsum(upright_thickness_cm) / 100, 3),
        books_replacement_cost=round(math.fsum(repl), 2),
        books_used_value=round(math.fsum(used), 2),
        items_replacement_cost_low=round(math.fsum(lows), 2),
        items_replacement_cost_high=round(math.fsum(highs), 2),
        excluded_from_totals=excluded,
    )


def finalize(packet: ClaimPacket) -> ClaimPacket:
    """Return a copy of the packet with totals recomputed from its lines."""
    return packet.model_copy(update={"totals": compute_totals(packet)})
