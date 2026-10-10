"""Reproducible synthetic data for one fictional Indian kirana merchant.

ALL DATA PRODUCED HERE IS SYNTHETIC. It does not describe any real merchant,
customer or payment.

Run directly to (re)create the CSV files in ./data:

    python -m utils.data_generator            # ends today
    python -m utils.data_generator --end 2026-10-10 --seed 42
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SEED = 42
DEFAULT_DAYS = 90

MERCHANT_PROFILE = {
    "merchant_name": "Sharma General Store",
    "owner_name": "Ramesh Sharma",
    "city": "Pune, Maharashtra",
    "business_type": "Kirana / general store",
    "opening_cash": 25000,
    "min_cash_buffer": 5000,
    "daily_operating_cost": 600,
    "synthetic": True,
}

# Weekday base transaction counts (Mon=0 .. Sun=6)
WEEKDAY_BASE_TXNS = [70, 68, 72, 70, 78, 92, 88]
METHODS = ["UPI", "Cash", "Card", "Wallet"]
METHOD_PROBS = [0.62, 0.25, 0.07, 0.06]
FAILURE_REASONS = [
    "Bank server timeout",
    "Insufficient balance in customer account",
    "Incorrect UPI PIN entered by customer",
    "Transaction limit exceeded at customer bank",
    "Customer bank declined the transaction",
]
FIRST_NAMES = [
    "ravi", "priya", "amit", "sneha", "rahul", "kavita", "suresh", "anita", "vikas", "pooja",
    "arjun", "meena", "deepak", "neha", "sanjay", "asha", "manoj", "rekha", "kiran", "sunil",
    "geeta", "rohit", "lata", "ajay", "swati", "nitin", "usha", "vijay", "seema", "prakash",
]
UPI_HANDLES = ["oksbi", "okhdfcbank", "okicici", "okaxis", "paytm", "ybl", "ibl"]


def _mask_vpa(name: str, handle: str) -> str:
    return f"{name[:2]}****@{handle}"


def _customer_pool(rng: np.random.Generator, size: int = 400) -> list[str]:
    pool = []
    for i in range(size):
        name = FIRST_NAMES[i % len(FIRST_NAMES)] + str(rng.integers(10, 99))
        pool.append(_mask_vpa(name, UPI_HANDLES[rng.integers(len(UPI_HANDLES))]))
    return pool


def _random_times(rng: np.random.Generator, n: int) -> np.ndarray:
    """Minutes after midnight, mixing a morning and an evening peak."""
    morning = rng.normal(9.5 * 60, 70, n)
    evening = rng.normal(19 * 60, 80, n)
    midday = rng.uniform(11 * 60, 17 * 60, n)
    pick = rng.choice(3, size=n, p=[0.38, 0.42, 0.20])
    minutes = np.where(pick == 0, morning, np.where(pick == 1, evening, midday))
    return np.clip(minutes, 7 * 60, 22 * 60 - 1).astype(int)


def _daily_multiplier(day: date, end: date, festival_day: date) -> float:
    m = 1.0
    if day.day <= 7:  # salary week: customers stock up
        m *= 1.10
    if day in (festival_day, festival_day - timedelta(days=1)):
        m *= 1.55
    return m


def generate_transactions(seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS,
                          end_date: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end_date or date.today()
    start = end - timedelta(days=days - 1)
    customers = _customer_pool(rng)

    # Realistic anomalies, placed relative to the end date
    festival_day = end - timedelta(days=38)
    outage_day = end - timedelta(days=19)
    recent_ticket_drop_start = end - timedelta(days=6)

    rows = []
    seq = 0
    for offset in range(days):
        day = start + timedelta(days=offset)
        lam = WEEKDAY_BASE_TXNS[day.weekday()] * _daily_multiplier(day, end, festival_day)
        n = int(rng.poisson(lam))
        minutes = np.sort(_random_times(rng, n))
        median_ticket = 210.0
        if day >= recent_ticket_drop_start:
            median_ticket *= 0.80  # smaller baskets this week
        if day in (festival_day, festival_day - timedelta(days=1)):
            median_ticket *= 1.25
        amounts = np.clip(rng.lognormal(np.log(median_ticket), 0.72, n), 10, 6000).round(0)
        methods = rng.choice(METHODS, size=n, p=METHOD_PROBS)

        for i in range(n):
            seq += 1
            method = methods[i]
            ts = datetime.combine(day, datetime.min.time()) + timedelta(
                minutes=int(minutes[i]), seconds=int(rng.integers(0, 60)))
            status, reason = "success", ""
            if method != "Cash":
                fail_p = 0.55 if (day == outage_day and method == "UPI") else 0.045
                pend_p = 0.02 if day == end else 0.0
                r = rng.random()
                if r < fail_p:
                    status = "failed"
                    reason = ("Bank server timeout" if day == outage_day
                              else FAILURE_REASONS[rng.integers(len(FAILURE_REASONS))])
                elif r < fail_p + pend_p:
                    status = "pending"
            if method == "Cash":
                customer = "Walk-in (cash)"
            elif method == "Card":
                customer = f"Card XXXX-{rng.integers(1000, 9999)}"
            else:
                customer = customers[rng.integers(len(customers))]
            rows.append(_make_row(day, seq, ts, float(amounts[i]), method, status, reason,
                                  customer, end, rng))

    df = pd.DataFrame(rows)
    df = _apply_random_refunds(df, rng, end)
    df = pd.concat([df, _demo_transactions(end, rng)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _make_row(day: date, seq: int, ts: datetime, amount: float, method: str, status: str,
              reason: str, customer: str, end: date, rng: np.random.Generator,
              txn_id: str | None = None) -> dict:
    digital = method != "Cash"
    if not digital:
        settlement_status, settlement_date = "not_applicable", ""
    elif status != "success":
        settlement_status, settlement_date = "not_applicable", ""
    elif day >= end:
        settlement_status, settlement_date = "in_progress", (day + timedelta(days=1)).isoformat()
    else:
        settlement_status, settlement_date = "settled", (day + timedelta(days=1)).isoformat()
    return {
        "txn_id": txn_id or f"T{day.strftime('%Y%m%d')}{seq:06d}",
        "utr": str(rng.integers(10**11, 10**12)) if digital else "",
        "timestamp": ts,
        "amount": amount,
        "payment_method": method,
        "status": status,
        "failure_reason": reason,
        "customer_ref": customer,
        "settlement_status": settlement_status,
        "settlement_date": settlement_date,
        "refund_status": "none",
        "refund_amount": 0.0,
        "refund_id": "",
        "refund_date": "",
        "demo_scenario": "",
    }


def _apply_random_refunds(df: pd.DataFrame, rng: np.random.Generator, end: date) -> pd.DataFrame:
    eligible = df.index[(df["status"] == "success") & (df["payment_method"] != "Cash")]
    picks = rng.choice(eligible, size=max(1, int(len(eligible) * 0.003)), replace=False)
    for idx in picks:
        day = df.at[idx, "timestamp"].date()
        refund_day = day + timedelta(days=3)
        df.at[idx, "refund_amount"] = df.at[idx, "amount"]
        df.at[idx, "refund_id"] = f"RF{rng.integers(10**9, 10**10)}"
        if refund_day <= end:
            df.at[idx, "refund_status"] = "processed"
            df.at[idx, "refund_date"] = refund_day.isoformat()
        else:
            df.at[idx, "refund_status"] = "initiated"
    return df


def _demo_transactions(end: date, rng: np.random.Generator) -> pd.DataFrame:
    """Planted transactions that drive the payment-support demo scenarios."""
    d = end.strftime("%Y%m%d")

    def at(day: date, hh: int, mm: int, ss: int = 0) -> datetime:
        return datetime.combine(day, datetime.min.time()) + timedelta(hours=hh, minutes=mm, seconds=ss)

    rows = []

    def add(txn_id, day, ts, amount, method, status, reason, customer, scenario, **extra):
        row = _make_row(day, 0, ts, amount, method, status, reason, customer, end, rng, txn_id)
        row["demo_scenario"] = scenario
        row.update(extra)
        rows.append(row)

    # 1. Customer says paid, merchant can't see it (payment succeeded, settlement in progress)
    add(f"T{d}900001", end, at(end, 18, 42, 10), 1250.0, "UPI", "success", "",
        "pr****@oksbi", "not_visible")
    # 2. Pending
    add(f"T{d}900002", end, at(end, 19, 55, 31), 640.0, "UPI", "pending", "",
        "am****@ybl", "pending")
    # 3. Failed
    add(f"T{d}900003", end, at(end, 17, 18, 5), 980.0, "UPI", "failed", "Bank server timeout",
        "ne****@okicici", "failed")
    # 4a. Duplicate debit concern: one success + one failed attempt a minute apart
    add(f"T{d}900004", end, at(end, 12, 10, 2), 499.0, "UPI", "success", "",
        "su****@okaxis", "duplicate")
    add(f"T{d}900005", end, at(end, 12, 11, 15), 499.0, "UPI", "failed", "Bank server timeout",
        "su****@okaxis", "duplicate")
    # 4b. Genuine double payment: two successful payments for the same bill
    prev = end - timedelta(days=1)
    p = prev.strftime("%Y%m%d")
    add(f"T{p}900006", prev, at(prev, 20, 5, 40), 1835.0, "UPI", "success", "",
        "ka****@paytm", "duplicate")
    add(f"T{p}900007", prev, at(prev, 20, 6, 22), 1835.0, "UPI", "success", "",
        "ka****@paytm", "duplicate")
    # 5. Refunds: one initiated, one processed
    r1 = end - timedelta(days=2)
    add(f"T{r1.strftime('%Y%m%d')}900008", r1, at(r1, 10, 22), 760.0, "UPI", "success", "",
        "vi****@okhdfcbank", "refund", refund_status="initiated", refund_amount=760.0,
        refund_id="RF7700012345", refund_date="")
    r2 = end - timedelta(days=9)
    add(f"T{r2.strftime('%Y%m%d')}900009", r2, at(r2, 16, 40), 2150.0, "Card", "success", "",
        "Card XXXX-4417", "refund", refund_status="processed", refund_amount=2150.0,
        refund_id="RF7700067890", refund_date=(r2 + timedelta(days=4)).isoformat())
    return pd.DataFrame(rows)


def generate_expenses(seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS,
                      end_date: date | None = None) -> pd.DataFrame:
    """Historical expenses paid by the merchant (synthetic)."""
    rng = np.random.default_rng(seed + 1)
    end = end_date or date.today()
    start = end - timedelta(days=days - 1)
    rows = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        if day.weekday() == 1:
            rows.append((day, "Supplier", "Wholesale grocery (Gupta Traders)",
                         float(round(rng.normal(52000, 4000), -2))))
        if day.weekday() == 5:
            rows.append((day, "Supplier", "FMCG distributor",
                         float(round(rng.normal(36000, 3000), -2))))
        if day.day == 5:
            rows.append((day, "Rent", "Shop rent", 22000.0))
        if day.day == 1:
            rows.append((day, "Salary", "Helper salary", 12000.0))
        if day.day == 10:
            rows.append((day, "Utilities", "Electricity bill", float(round(rng.normal(3800, 300), -1))))
        rows.append((day, "Daily operations", "Tea, transport, packaging",
                     float(round(rng.uniform(350, 850), -1))))
    return pd.DataFrame(rows, columns=["date", "category", "description", "amount"])


def generate_planned_expenses(end_date: date | None = None) -> pd.DataFrame:
    """Sample upcoming payments for the 7 days after the data ends (merchant-editable)."""
    end = end_date or date.today()
    d = lambda n: end + timedelta(days=n)  # noqa: E731
    rows = [
        (d(2), "Supplier", "Wholesale grocery (Gupta Traders)", 52000.0),
        (d(4), "Utilities", "Electricity bill", 3800.0),
        (d(5), "Rent", "Shop rent", 22000.0),
        (d(6), "Supplier", "FMCG distributor", 38000.0),
    ]
    return pd.DataFrame(rows, columns=["date", "category", "description", "amount"])


def write_all(data_dir: Path = DATA_DIR, seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS,
              end_date: date | None = None) -> dict:
    end = end_date or date.today()
    data_dir.mkdir(parents=True, exist_ok=True)
    txns = generate_transactions(seed, days, end)
    txns.to_csv(data_dir / "transactions.csv", index=False)
    generate_expenses(seed, days, end).to_csv(data_dir / "expenses.csv", index=False)
    generate_planned_expenses(end).to_csv(data_dir / "planned_expenses.csv", index=False)
    profile = dict(MERCHANT_PROFILE, seed=seed, days=days, data_end_date=end.isoformat(),
                   generated_at=datetime.now().isoformat(timespec="seconds"),
                   note="SYNTHETIC DEMO DATA - not a real merchant or real payments.")
    (data_dir / "merchant_profile.json").write_text(json.dumps(profile, indent=2))
    return {"transactions": len(txns), "end_date": end.isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic merchant data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--end", type=str, default=None, help="Last day of data, YYYY-MM-DD")
    args = parser.parse_args()
    end = date.fromisoformat(args.end) if args.end else None
    print(write_all(seed=args.seed, days=args.days, end_date=end))


if __name__ == "__main__":
    main()
