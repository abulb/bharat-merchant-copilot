"""Formatting helpers for Indian currency, numbers and dates."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd


def _group_indian(integer_part: str) -> str:
    """Group digits the Indian way: 12,34,567."""
    if len(integer_part) <= 3:
        return integer_part
    last3 = integer_part[-3:]
    rest = integer_part[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups) + "," + last3


def format_inr(amount: float | int | None, decimals: int = 0, signed: bool = False) -> str:
    """Format a number as rupees with Indian digit grouping, e.g. ₹1,23,456."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "—"
    sign = "-" if amount < 0 else ("+" if signed and amount > 0 else "")
    value = abs(float(amount))
    text = f"{value:.{decimals}f}"
    if "." in text:
        integer_part, frac = text.split(".")
        return f"{sign}₹{_group_indian(integer_part)}.{frac}"
    return f"{sign}₹{_group_indian(text)}"


def format_inr_compact(amount: float | int | None) -> str:
    """Short rupee format using lakh / crore, e.g. ₹1.25 L, ₹2.1 Cr."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "—"
    sign = "-" if amount < 0 else ""
    value = abs(float(amount))
    if value >= 1_00_00_000:
        return f"{sign}₹{value / 1_00_00_000:.2f} Cr"
    if value >= 1_00_000:
        return f"{sign}₹{value / 1_00_000:.2f} L"
    return f"{sign}{format_inr(value)}"


def format_number(value: float | int | None) -> str:
    """Indian-grouped integer, e.g. 1,23,456."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    return sign + _group_indian(str(int(round(abs(value)))))


def format_pct(value: float | None, decimals: int = 1, signed: bool = True) -> str:
    """Format a ratio-style percentage (already multiplied by 100)."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{value:+.{decimals}f}%" if signed else f"{value:.{decimals}f}%"


def _to_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, str):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    return value


def format_date(value: date | datetime | pd.Timestamp | str) -> str:
    """Readable Indian-style date, e.g. 10 Oct 2026."""
    return _to_date(value).strftime("%d %b %Y")


def format_date_short(value: date | datetime | pd.Timestamp | str) -> str:
    """Short date with weekday, e.g. Sat, 10 Oct."""
    return _to_date(value).strftime("%a, %d %b")


def format_date_numeric(value: date | datetime | pd.Timestamp | str) -> str:
    """Numeric Indian date format DD-MM-YYYY."""
    return _to_date(value).strftime("%d-%m-%Y")


def format_datetime(value: datetime | pd.Timestamp | str) -> str:
    """e.g. 10 Oct 2026, 07:45 PM."""
    ts = pd.Timestamp(value)
    return ts.strftime("%d %b %Y, %I:%M %p")
