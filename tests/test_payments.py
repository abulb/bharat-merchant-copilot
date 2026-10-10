import re

import pytest

from services import payments

EXPECTED = {
    "T20261010900001": ("not_visible", "Payment received"),
    "T20261010900002": ("pending", "pending"),
    "T20261010900003": ("failed", "failed"),
    "T20261010900004": ("duplicate", "other attempt failed"),
    "T20261009900006": ("duplicate", "genuine double payment"),
    "T20261008900008": ("refund", "Refund initiated"),
    "T20261001900009": ("refund", "Refund processed"),
}

# Phrases that would wrongly claim an action on a real payment, or ask for secrets
FORBIDDEN = [
    r"has been (reversed|refunded|settled|escalated)",
    r"we (have )?(reversed|refunded|escalated)",
    r"(share|enter|tell|send) (me |us )?(your |the )?(upi )?(pin|otp|password)",
]


def _all_text(g):
    return " ".join([g.headline, *g.findings, *g.steps, *g.escalation])


@pytest.mark.parametrize("txn_id,expected", EXPECTED.items())
def test_demo_scenarios(dataset, txn_id, expected):
    scenario, headline = expected
    txn = payments.lookup_transaction(dataset["txns"], txn_id)
    assert txn is not None and txn["demo_scenario"] == scenario
    g = payments.diagnose(scenario, txn, dataset["txns"])
    assert headline.lower() in g.headline.lower()
    assert g.steps and g.escalation
    text = _all_text(g).lower()
    for pattern in FORBIDDEN:
        assert not re.search(pattern, text), pattern


def test_lookup_by_utr_and_case_insensitive(dataset):
    t = dataset["txns"]
    row = t[t["txn_id"] == "T20261010900001"].iloc[0]
    assert payments.lookup_transaction(t, row["utr"])["txn_id"] == "T20261010900001"
    assert payments.lookup_transaction(t, " t20261010900001 ")["txn_id"] == "T20261010900001"
    assert payments.lookup_transaction(t, "NOPE12345") is None


@pytest.mark.parametrize("scenario", list(payments.SCENARIOS))
def test_unknown_reference_gives_safe_guidance(dataset, scenario):
    g = payments.diagnose(scenario, None, dataset["txns"])
    assert g.tone == "warning" and "no matching" in g.headline.lower()


def test_unknown_scenario_raises(dataset):
    with pytest.raises(ValueError):
        payments.diagnose("bogus", None, dataset["txns"])


def test_reference_validation():
    assert payments.validate_reference("") is not None
    assert payments.validate_reference("12") is not None
    assert payments.validate_reference("abc$%^") is not None
    assert payments.validate_reference("412345678901") is None
    assert payments.validate_reference("T20261010900001") is None


@pytest.mark.parametrize("text", ["my otp is 482913", "UPI PIN 1234", "pin: 123456", "password is hunter2",
                                  "card 4111 1111 1111 1111", "cvv 123"])
def test_sensitive_info_detected(text):
    assert payments.detect_sensitive_info(text)


@pytest.mark.parametrize("text", ["why do PIN errors happen?", "customer did not get OTP",
                                  "412345678901", "T20261010900001", ""])
def test_sensitive_info_not_flagged_for_normal_text(text):
    assert payments.detect_sensitive_info(text) == []


def test_merchant_view_hides_internal_fields(dataset):
    txn = payments.lookup_transaction(dataset["txns"], "T20261010900003")
    view = payments.merchant_view(txn)
    assert view["Status"] == "Failed" and "Failure reason" in view
    assert "demo_scenario" not in view and "****" in view["Customer (masked)"]


def test_disclaimer_mentions_not_connected():
    assert "not connected to Paytm" in payments.DEMO_DISCLAIMER
