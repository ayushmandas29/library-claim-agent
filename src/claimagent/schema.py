"""Claim packet schema (the output contract).

Design rules enforced here, not in prompts:
  * Unknown is represented as None, never as a made-up value.
  * A price with an amount MUST carry a source and a retrieval date.
    A price without evidence is rejected at construction time.
  * Totals are NOT stored by hand; see totals.py.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Sweep(BaseModel):
    id: str
    captured_at: str
    device: str = ""
    duration_s: float = 0
    country: str
    currency: str


class Room(BaseModel):
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    floor_area_m2: Optional[float] = None
    wall_area_m2: Optional[float] = None
    shelved_wall_area_m2: Optional[float] = None
    scale_method: str = ""
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    shape_note: str = ""  # how a non-rectangular room was handled


class BookReplacementCost(BaseModel):
    amount: Optional[float] = Field(default=None, ge=0)
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False

    @model_validator(mode="after")
    def _needs_evidence(self):
        if self.amount is not None and not (self.source and self.retrieved_at):
            raise ValueError("price amount requires source and retrieved_at")
        return self


class BookUsedValue(BaseModel):
    amount: Optional[float] = Field(default=None, ge=0)
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    condition_assumed: str = ""
    converted: bool = False

    @model_validator(mode="after")
    def _needs_evidence(self):
        if self.amount is not None and not (
            self.source and self.retrieved_at and self.condition_assumed
        ):
            raise ValueError(
                "used value requires source, retrieved_at and condition_assumed"
            )
        return self


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: Optional[int] = None
    frame_ref: str  # path or timestamp of a saved frame; must be openable
    orientation: Literal["upright", "flat", "stacked"] = "upright"
    status: Literal["identified", "unidentified", "needs_appraisal"]
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    spine_height_cm: Optional[float] = Field(default=None, ge=0)
    spine_thickness_cm: Optional[float] = Field(default=None, ge=0)
    id_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    replacement_cost: Optional[BookReplacementCost] = None
    used_value: Optional[BookUsedValue] = None

    @model_validator(mode="after")
    def _no_title_when_unidentified(self):
        if self.status == "unidentified" and (self.title or self.author):
            raise ValueError("unidentified books must not carry a guessed title/author")
        return self


class Dimensions(BaseModel):
    w: Optional[float] = None
    h: Optional[float] = None
    d: Optional[float] = None


class ItemReplacementCost(BaseModel):
    low: Optional[float] = Field(default=None, ge=0)
    high: Optional[float] = Field(default=None, ge=0)
    source: str = ""
    url: str = ""
    retrieved_at: str = ""

    @model_validator(mode="after")
    def _needs_evidence(self):
        has_amount = self.low is not None or self.high is not None
        if has_amount and not (self.source and self.retrieved_at):
            raise ValueError("price range requires source and retrieved_at")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("low must be <= high")
        return self


class Item(BaseModel):
    id: str
    category: str
    description: str = ""
    brand_model: str = ""
    frame_ref: str
    dimensions_cm: Dimensions = Dimensions()
    status: Literal["priced", "range", "needs_appraisal"]
    replacement_cost: Optional[ItemReplacementCost] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    shelf_run_m: float = 0
    books_replacement_cost: float = 0
    books_used_value: float = 0
    items_replacement_cost_low: float = 0
    items_replacement_cost_high: float = 0
    excluded_from_totals: int = 0


class ReviewEntry(BaseModel):
    ref_id: str
    reason: str


class ClaimPacket(BaseModel):
    sweep: Sweep
    room: Room = Room()
    books: list[Book] = []
    items: list[Item] = []
    totals: Totals = Totals()
    review_queue: list[ReviewEntry] = []
