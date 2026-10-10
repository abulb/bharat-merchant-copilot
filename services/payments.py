"""Simulated payment-support workflow.

DEMONSTRATION ONLY. This module looks up transactions in the local synthetic
CSV. It is not connected to Paytm, NPCI, any bank, or any payment system, and
it never reverses, settles, refunds or escalates a real payment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from utils.formatting import format_date, format_datetime, format_inr

DEMO_DISCLAIMER = (
    "Demonstration only — this assistant searches synthetic sample data. It is not connected to "
    "Paytm's payment infrastructure, any bank or NPCI, and it cannot reverse, settle, refund or "
    "escalate a real payment."
)
SAFETY_NOTE = (
    "Never share your UPI PIN, OTP, passwords, CVV or full card/bank details with anyone — "
    "including support staff. Genuine support will never ask for them."
)

SCENARIOS = {
    "not_visible": {
        "title": "Customer says payment succeeded, but I can't see it",
        "ask": "Ask the customer for the 12-digit UTR / transaction reference shown in their UPI app.",
    },
    "pending": {"title": "Transaction is pending", "ask": "Enter the transaction ID or UTR."},
    "failed": {"title": "Transaction failed", "ask": "Enter the transaction ID or UTR."},
    "duplicate": {
        "title": "Customer says they were charged twice",
        "ask": "Enter the transaction ID or UTR of either payment.",
    },
    "refund": {"title": "Refund status query", "ask": "Enter the original transaction ID or UTR."},
}

# Flag text that appears to *contain* a secret (keyword next to a number), not
# questions that merely mention the word, e.g. "why do PIN errors happen?".
_SENSITIVE_PATTERNS = [
    (re.compile(r"\b(?:upi\s*)?m?pin\b\D{0,15}\d{4,6}\b|\b\d{4,6}\D{0,15}\b(?:upi\s*)?m?pin\b", re.I), "UPI PIN"),
    (re.compile(r"\botp\b\D{0,15}\d{4,8}\b|\b\d{4,8}\D{0,15}\botp\b", re.I), "OTP"),
    (re.compile(r"\b(?:password|passcode)\b\s*(?:is|:|=)\s*\S+", re.I), "password"),
    (re.compile(r"\bcvv\b\D{0,10}\d{3,4}\b", re.I), "CVV"),
    (re.compile(r"\b(?:\d[ -]?){15,18}\d\b"), "full card or account number"),
]
_REF_PATTERN = re.compile(r"^[A-Za-z0-9-]{6,30}$")


def detect_sensitive_info(text: str) -> list[str]:
    """Return the kinds of sensitive data that appear to be present in `text`."""
    if not text:
        return []
    found = []
    for pattern, label in _SENSITIVE_PATTERNS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def normalise_reference(ref: str) -> str:
    return re.sub(r"\s+", "", (ref or "")).upper()


def validate_reference(ref: str) -> str | None:
    """Return an error message, or None if the reference format looks valid."""
    ref = normalise_reference(ref)
    if not ref:
        return "Please enter a transaction ID or UTR."
    if not _REF_PATTERN.match(ref):
        return "A reference should be 6–30 letters/digits (e.g. a 12-digit UTR like 412345678901)."
    return None


def lookup_transaction(txns: pd.DataFrame, ref: str) -> pd.Series | None:
    """Find a transaction by transaction ID or UTR (exact match)."""
    ref = normalise_reference(ref)
    if not ref or txns.empty:
        return None
    match = txns[(txns["txn_id"].astype(str).str.upper() == ref) | (txns["utr"].astype(str) == ref)]
    return None if match.empty else match.iloc[0]


def sample_transactions(txns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Planted demo transactions for a scenario."""
    return txns[txns["demo_scenario"] == scenario]


def merchant_view(txn: pd.Series) -> dict:
    """Only the fields a merchant needs (customer identity stays masked)."""
    view = {
        "Transaction ID": txn["txn_id"],
        "UTR / bank reference": txn.get("utr") or "—",
        "Date & time": format_datetime(txn["timestamp"]),
        "Amount": format_inr(txn["amount"], 2),
        "Payment method": txn["payment_method"],
        "Status": str(txn["status"]).capitalize(),
        "Customer (masked)": txn.get("customer_ref") or "—",
    }
    if txn["status"] == "failed" and txn.get("failure_reason"):
        view["Failure reason"] = txn["failure_reason"]
    settle = txn.get("settlement_status")
    if settle == "settled":
        view["Settlement"] = f"Settled to bank on {format_date(txn['settlement_date'])}"
    elif settle == "in_progress":
        view["Settlement"] = f"In progress — expected {format_date(txn['settlement_date'])}"
    if txn.get("refund_status") and txn["refund_status"] != "none":
        view["Refund"] = f"{str(txn['refund_status']).capitalize()} — {format_inr(txn['refund_amount'], 2)}"
        if txn.get("refund_id"):
            view["Refund reference"] = txn["refund_id"]
        if txn.get("refund_date"):
            view["Refund processed on"] = format_date(txn["refund_date"])
    return view


def find_possible_duplicates(txns: pd.DataFrame, txn: pd.Series, window_minutes: int = 15) -> pd.DataFrame:
    """Other payments from the same masked customer, same amount, close in time."""
    if txn.get("payment_method") == "Cash" or not txn.get("customer_ref"):
        return txns.iloc[0:0]
    near = txns[
        (txns["customer_ref"] == txn["customer_ref"])
        & (txns["amount"] == txn["amount"])
        & ((txns["timestamp"] - txn["timestamp"]).abs() <= pd.Timedelta(minutes=window_minutes))
        & (txns["txn_id"] != txn["txn_id"])
    ]
    return near


@dataclass
class Guidance:
    headline: str
    tone: str  # "success", "warning", "error", "info"
    findings: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    escalation: list[str] = field(default_factory=list)
    related: pd.DataFrame | None = None


_ESCALATE_GENERIC = [
    "In the real app, raise a ticket from the transaction's Help/Support option, quoting the transaction ID and UTR.",
    "If unresolved after the bank's turnaround time, the customer can raise a complaint with their own bank, "
    "and later via the NPCI dispute portal or RBI Ombudsman (cms.rbi.org.in).",
]


def diagnose(scenario: str, txn: pd.Series | None, txns: pd.DataFrame) -> Guidance:
    """Scenario-specific, safe guidance. Never claims a real action was taken."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    if txn is None:
        steps = [
            "Double-check the reference — UTRs are usually 12 digits; ask the customer to share a screenshot of the payment screen.",
            "Confirm the customer paid to your QR / UPI ID (check the payee name on their screen).",
            "Check the date and amount on their screen and look for it in your transaction list.",
            "Do not hand over goods for an unverified payment; ask for another payment method if needed.",
        ]
        if scenario == "not_visible":
            steps.insert(1, "If their app shows 'Pending', wait — most pending UPI payments resolve within minutes to a few hours.")
        return Guidance("No matching transaction found in your records", "warning",
                        ["We could not find this reference in your (sample) transaction history."],
                        steps, _ESCALATE_GENERIC)

    status = txn["status"]
    amount = format_inr(txn["amount"], 2)
    g: Guidance

    if scenario == "duplicate":
        dups = find_possible_duplicates(txns, txn)
        group = pd.concat([txn.to_frame().T, dups]).sort_values("timestamp")
        n_success = int((group["status"] == "success").sum())
        if dups.empty:
            g = Guidance("Only one payment found for this amount", "info",
                         [f"One {status} payment of {amount}. No other payment of the same amount from this customer within 15 minutes."],
                         ["Ask the customer to check their bank statement for the second debit's UTR and share it.",
                          "A second debit with no matching payment here is usually a failed attempt that their bank reverses automatically."],
                         _ESCALATE_GENERIC, group)
        elif n_success >= 2:
            g = Guidance("Two successful payments found — likely a genuine double payment", "error",
                         [f"{n_success} successful payments of {amount} from the same customer a few minutes apart."],
                         ["Verify with the customer that this was one purchase.",
                          "If so, return the extra amount using the official refund option for one of the transactions "
                          "in your payment app (this demo cannot issue refunds).",
                          "Share the refund reference with the customer once it is issued."],
                         ["If the refund option is unavailable, contact your payment provider's merchant support with both transaction IDs."],
                         group)
        else:
            g = Guidance("Only one payment succeeded — the other attempt failed", "warning",
                         [f"{n_success} successful and {len(group) - n_success} failed attempt(s) of {amount}.",
                          "You received the money once. If the customer's account was debited for the failed attempt, "
                          "their bank typically reverses it automatically."],
                         ["Tell the customer you received one payment and show them the successful transaction ID.",
                          "Under RBI's turnaround-time rules, failed UPI debits are typically auto-reversed by T+1 working day; "
                          "ask them to check with their bank if not.",
                          "Do not refund from your side for the failed attempt — that would pay them twice."],
                         ["If the debit isn't reversed after T+1 working day, the customer should raise a complaint with their bank quoting the failed UTR."],
                         group)
        return g

    if scenario == "refund":
        rs = txn.get("refund_status", "none")
        if rs == "processed":
            return Guidance(f"Refund processed on {format_date(txn['refund_date'])}", "success",
                            [f"Refund of {format_inr(txn['refund_amount'], 2)} (reference {txn['refund_id']}) is marked processed."],
                            ["Share the refund reference with the customer.",
                             "Refunds can take 5–7 working days to appear in the customer's account depending on their bank (cards may take longer)."],
                            ["If the customer hasn't received it after 7 working days, they should contact their bank with the refund reference."])
        if rs == "initiated":
            return Guidance("Refund initiated — not yet processed", "warning",
                            [f"Refund of {format_inr(txn['refund_amount'], 2)} (reference {txn['refund_id']}) is initiated in the records."],
                            ["Tell the customer the refund is in progress and share the refund reference.",
                             "Do not issue a second refund for the same payment."],
                            ["If it stays 'initiated' for more than 3 working days, contact merchant support with the refund reference."])
        return Guidance("No refund found for this payment", "info",
                        [f"This {status} payment of {amount} has no refund recorded."],
                        ["If you agreed to a refund, issue it using the official refund option in your payment app.",
                         "If the payment failed, no refund is needed — the customer's bank reverses failed debits."],
                        _ESCALATE_GENERIC)

    # Status-driven guidance for not_visible / pending / failed
    if status == "success":
        settle = txn.get("settlement_status")
        finding = f"Payment of {amount} was successful."
        if settle == "in_progress":
            finding += f" It will reach your bank account on {format_date(txn['settlement_date'])} (settlement in progress)."
        elif settle == "settled":
            finding += f" It was settled to your bank on {format_date(txn['settlement_date'])}."
        return Guidance("Payment received", "success", [finding],
                        ["Match the amount and time with the customer's screen, then hand over the goods.",
                         "Check 'Today's payments' in your app — not your bank statement — for same-day payments; "
                         "bank credit comes with settlement."],
                        ["If the settlement doesn't arrive by the expected date, contact merchant support with the transaction ID."])
    if status == "pending":
        return Guidance("Payment is pending — not yet confirmed", "warning",
                        [f"Payment of {amount} is pending. Money has not been confirmed to you yet."],
                        ["Do not treat this as paid until it shows Success.",
                         "Most pending UPI payments resolve within a few minutes to a few hours.",
                         "If it fails, any debit from the customer's account is reversed by their bank, typically by T+1 working day.",
                         "If the customer needs the goods now, you may ask for another payment method — and tell them their pending amount will be reversed if it fails."],
                        ["If it remains pending after 24 hours, raise a ticket from the transaction's Help option quoting the transaction ID."])
    # failed
    reason = txn.get("failure_reason") or "not specified"
    return Guidance("Payment failed — you have not received this money", "error",
                    [f"Payment of {amount} failed. Reason: {reason}."],
                    ["Ask the customer to try again or use another payment method.",
                     "If the customer's account was debited, their bank typically reverses it automatically by T+1 working day — "
                     "you don't need to refund.",
                     "If the reason mentions PIN or balance, it is on the customer's side; ask them to retry carefully. Never ask for their PIN."],
                    ["If the customer's money isn't back after T+1 working day, they should complain to their own bank quoting the UTR."])
