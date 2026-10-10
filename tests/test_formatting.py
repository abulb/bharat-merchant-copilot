from datetime import date

from utils.formatting import (
    format_date, format_date_numeric, format_inr, format_inr_compact, format_number, format_pct,
)


def test_indian_grouping():
    assert format_inr(0) == "₹0"
    assert format_inr(999) == "₹999"
    assert format_inr(1000) == "₹1,000"
    assert format_inr(123456) == "₹1,23,456"
    assert format_inr(12345678) == "₹1,23,45,678"


def test_negative_signed_and_decimals():
    assert format_inr(-52000) == "-₹52,000"
    assert format_inr(1250.5, 2) == "₹1,250.50"
    assert format_inr(7062, signed=True) == "+₹7,062"
    assert format_inr(-10, signed=True) == "-₹10"
    assert format_inr(None) == "—"
    assert format_inr(float("nan")) == "—"


def test_compact_lakh_crore():
    assert format_inr_compact(125000) == "₹1.25 L"
    assert format_inr_compact(21000000) == "₹2.10 Cr"
    assert format_inr_compact(950) == "₹950"


def test_numbers_pct_dates():
    assert format_number(1234567) == "12,34,567"
    assert format_pct(5.0) == "+5.0%"
    assert format_pct(None) == "—"
    assert format_date(date(2026, 10, 10)) == "10 Oct 2026"
    assert format_date_numeric("2026-10-10") == "10-10-2026"
