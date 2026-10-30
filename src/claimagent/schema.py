"""Reviewed inventory output contract. Unknown measurements and values remain null."""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Literal, Optional
from urllib.parse import urlsplit
from pydantic import BaseModel, Field, computed_field, model_validator


def _price_evidence_is_valid(source: str, url: str, retrieved_at: str, currency: str) -> bool:
    parsed = urlsplit(url)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", retrieved_at):
            date.fromisoformat(retrieved_at)
            valid_date = True
        elif re.match(r"^\d{4}-\d{2}-\d{2}T", retrieved_at):
            datetime.fromisoformat(retrieved_at[:-1] + "+00:00" if retrieved_at.endswith("Z") else retrieved_at)
            valid_date = True
        else:
            valid_date = False
    except (ValueError, TypeError):
        valid_date = False
    return bool(source and parsed.scheme in ("http", "https") and parsed.netloc and valid_date and currency)

class Sweep(BaseModel):
    id: str = ""
    captured_at: str = ""
    device: str = ""
    duration_s: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    country: str = ""
    currency: str = ""

class Room(BaseModel):
    length_m: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    width_m: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    height_m: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    floor_area_m2: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    wall_area_m2: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    shelved_wall_area_m2: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    scale_method: str = ""
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    shape_note: str = ""
    frame_refs: list[str] = Field(default_factory=list)
    frame_timestamps: list[str] = Field(default_factory=list)

class BookReplacementCost(BaseModel):
    amount: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str = ""
    original_amount: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    original_currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    conversion_rate: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    conversion_source: str = ""
    conversion_date: str = ""
    @model_validator(mode="after")
    def _needs_evidence(self):
        if self.amount is not None and not _price_evidence_is_valid(self.source, self.url, self.retrieved_at, self.currency):
            raise ValueError("price amount requires source, URL, retrieval date and currency")
        if self.converted and not all((self.original_amount is not None, self.original_currency,
                                       self.conversion_rate, self.conversion_source, self.conversion_date)):
            raise ValueError("converted price requires original amount/currency, rate, conversion source and date")
        if self.converted and self.amount is not None and not math.isclose(
            self.original_amount * self.conversion_rate, self.amount, rel_tol=1e-6, abs_tol=0.01
        ):
            raise ValueError("converted amount does not match original amount × rate")
        return self

class BookUsedValue(BaseModel):
    amount: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str = ""
    original_amount: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    original_currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    condition_assumed: str = ""
    converted: bool = False
    conversion_rate: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    conversion_source: str = ""
    conversion_date: str = ""
    @model_validator(mode="after")
    def _needs_evidence(self):
        if self.amount is not None and not (_price_evidence_is_valid(self.source, self.url, self.retrieved_at, self.currency) and self.condition_assumed):
            raise ValueError("used value requires source, URL, retrieval date, currency and condition")
        if self.converted and not all((self.original_amount is not None, self.original_currency,
                                       self.conversion_rate, self.conversion_source, self.conversion_date)):
            raise ValueError("converted price requires original amount/currency, rate, conversion source and date")
        if self.converted and self.amount is not None and not math.isclose(
            self.original_amount * self.conversion_rate, self.amount, rel_tol=1e-6, abs_tol=0.01
        ):
            raise ValueError("converted amount does not match original amount × rate")
        return self

class BookPriceQuote(BaseModel):
    """A sourced online quote kept separate from physical replacement totals."""
    amount: float = Field(ge=0, allow_inf_nan=False)
    currency: str
    source: str
    price_type: str
    url: str
    retrieved_at: str
    @model_validator(mode="after")
    def _needs_evidence(self):
        if not self.price_type or not _price_evidence_is_valid(self.source, self.url, self.retrieved_at, self.currency):
            raise ValueError("price quote requires type, source, URL, retrieval date and currency")
        return self

class Book(BaseModel):
    id: str = ""
    shelf: str = ""
    position: Optional[int] = Field(default=None, ge=0)
    frame_ref: str = ""
    frame_timestamp: str = ""
    timestamp_source: str = ""
    observed_at: str = ""
    orientation: Literal["upright", "flat", "stacked"] = "upright"
    status: Literal["identified", "unidentified", "needs_appraisal"] = "unidentified"
    title: str = ""
    author: str = ""
    publisher: str = ""
    edition: str = ""
    isbn: str = ""
    spine_height_cm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    spine_thickness_cm: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    measurement_method: str = ""
    measurement_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    id_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    detection_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    detection_box: Optional[tuple[int, int, int, int]] = None
    detection_image_size: Optional[tuple[int, int]] = None
    candidate_warnings: list[str] = Field(default_factory=list)
    ocr_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    spine_text: str = ""
    replacement_cost: Optional[BookReplacementCost] = None
    used_value: Optional[BookUsedValue] = None
    price_quote: Optional[BookPriceQuote] = None
    @model_validator(mode="after")
    def _no_title_when_unidentified(self):
        if self.status == "unidentified" and any((self.title, self.author, self.publisher, self.edition, self.isbn)):
            raise ValueError("unidentified books must not carry an asserted identification")
        return self

class BookCandidate(BaseModel):
    """Machine-detected possibility; excluded from confirmed inventory totals."""
    id: str
    frame_ref: str = ""
    frame_timestamp: str = ""
    timestamp_source: str = ""
    detection_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    detection_box: Optional[tuple[int, int, int, int]] = None
    detection_image_size: Optional[tuple[int, int]] = None
    candidate_warnings: list[str] = Field(default_factory=list)
    ocr_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    spine_text: str = ""

class Dimensions(BaseModel):
    w: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    h: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    d: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

class ItemReplacementCost(BaseModel):
    low: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    high: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    @model_validator(mode="after")
    def _needs_evidence(self):
        has_amount = self.low is not None or self.high is not None
        if has_amount and not _price_evidence_is_valid(self.source, self.url, self.retrieved_at, self.currency):
            raise ValueError("price range requires source, URL, retrieval date and currency")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("low must be <= high")
        return self

class Item(BaseModel):
    id: str = ""
    category: str = ""
    description: str = ""
    brand_model: str = ""
    frame_ref: str = ""
    frame_timestamp: str = ""
    observed_at: str = ""
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    measurement_method: str = ""
    measurement_confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    status: Literal["priced", "range", "needs_appraisal"] = "needs_appraisal"
    replacement_cost: Optional[ItemReplacementCost] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    @model_validator(mode="after")
    def _price_matches_status(self):
        cost = self.replacement_cost
        if cost and (cost.low is not None or cost.high is not None):
            if self.status == "priced" and (cost.low is None or cost.high is None or cost.low != cost.high):
                raise ValueError("priced item requires equal low and high values")
            if self.status == "range" and (cost.low is None or cost.high is None):
                raise ValueError("range item requires both low and high values")
        return self

class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    shelf_run_m: Optional[float] = None
    books_replacement_cost: Optional[float] = None
    books_used_value: Optional[float] = None
    items_replacement_cost_low: Optional[float] = None
    items_replacement_cost_high: Optional[float] = None
    excluded_from_totals: int = 0
    books_without_replacement_price: int = 0
    books_without_used_price: int = 0
    items_without_price: int = 0
    lines_without_valid_evidence: int = 0

class ReviewEntry(BaseModel):
    ref_id: str
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)

class ClaimPacket(BaseModel):
    sweep: Sweep = Field(default_factory=Sweep)
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    candidates: list[BookCandidate] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)

    @computed_field
    @property
    def candidate_count(self) -> int:
        return len(self.candidates)
