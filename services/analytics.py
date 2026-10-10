"""Deterministic business analytics. Every number shown in the app comes from here."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils import data_generator
from utils.formatting import format_date, format_inr, format_pct

DATA_DIR = data_generator.DATA_DIR
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REQUIRED_TXN_COLUMNS = {"txn_id", "timestamp", "amount", "payment_method", "status"}


class DataError(Exception):
    """Raised when merchant data is missing or malformed."""


# --------------------------------------------------------------------------- loading

def load_data(data_dir: Path = DATA_DIR, auto_generate: bool = True):
    """Load transactions, expenses, planned expenses and profile.

    Generates synthetic data on first run if the files are missing.
    """
    txn_path = data_dir / "transactions.csv"
    if not txn_path.exists():
        if not auto_generate:
            raise DataError(f"No transaction file found at {txn_path}")
        data_generator.write_all(data_dir)

    txns = pd.read_csv(txn_path, dtype={"utr": str, "txn_id": str}, keep_default_na=False)
    txns = clean_transactions(txns)

    exp_path = data_dir / "expenses.csv"
    expenses = pd.read_csv(exp_path, parse_dates=["date"]) if exp_path.exists() else pd.DataFrame(
        columns=["date", "category", "description", "amount"])
    planned_path = data_dir / "planned_expenses.csv"
    planned = pd.read_csv(planned_path, parse_dates=["date"]) if planned_path.exists() else pd.DataFrame(
        columns=["date", "category", "description", "amount"])

    profile_path = data_dir / "merchant_profile.json"
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else dict(
        data_generator.MERCHANT_PROFILE)
    return txns, expenses, planned, profile


def clean_transactions(txns: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_TXN_COLUMNS - set(txns.columns)
    if missing:
        raise DataError(f"Transaction data is missing columns: {', '.join(sorted(missing))}")
    df = txns.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount"])
    df = df[df["amount"] > 0]
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    for col, default in [("refund_amount", 0.0), ("refund_status", "none"), ("utr", ""),
                         ("failure_reason", ""), ("customer_ref", ""), ("demo_scenario", ""),
                         ("settlement_status", ""), ("settlement_date", ""), ("refund_id", ""),
                         ("refund_date", "")]:
        if col not in df.columns:
            df[col] = default
    df["refund_amount"] = pd.to_numeric(df["refund_amount"], errors="coerce").fillna(0.0)
    df["date"] = df["timestamp"].dt.normalize()
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- daily table

def daily_summary(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day (gaps filled with zero sales)."""
    if txns.empty:
        return pd.DataFrame(columns=["date", "sales", "txn_count", "avg_ticket", "failed_count",
                                     "pending_count", "failed_amount", "digital_sales",
                                     "cash_sales", "refunds", "weekday"])
    ok = txns[txns["status"] == "success"]
    grouped = ok.groupby("date")
    full_range = pd.date_range(txns["date"].min(), txns["date"].max(), freq="D")
    daily = pd.DataFrame(index=full_range)
    daily["sales"] = grouped["amount"].sum()
    daily["txn_count"] = grouped["amount"].count()
    daily["digital_sales"] = ok[ok["payment_method"] != "Cash"].groupby("date")["amount"].sum()
    daily["cash_sales"] = ok[ok["payment_method"] == "Cash"].groupby("date")["amount"].sum()
    failed = txns[txns["status"] == "failed"].groupby("date")["amount"]
    daily["failed_count"] = failed.count()
    daily["failed_amount"] = failed.sum()
    daily["pending_count"] = txns[txns["status"] == "pending"].groupby("date")["amount"].count()
    refunds = txns[txns["refund_status"] == "processed"]
    if not refunds.empty:
        rd = pd.to_datetime(refunds["refund_date"], errors="coerce").dt.normalize()
        daily["refunds"] = refunds.groupby(rd)["refund_amount"].sum()
    else:
        daily["refunds"] = 0.0

    daily = daily.fillna(0.0)
    daily.index.name = "date"
    daily = daily.reset_index()
    for col in ["txn_count", "failed_count", "pending_count"]:
        daily[col] = daily[col].astype(int)
    daily["avg_ticket"] = np.where(daily["txn_count"] > 0, daily["sales"] / daily["txn_count"].clip(lower=1), 0.0)
    daily["weekday"] = daily["date"].dt.day_name()
    return daily


# --------------------------------------------------------------------------- comparisons

def _pct(cur: float, prev: float) -> float | None:
    if prev == 0:
        return None
    return (cur - prev) / prev * 100


def today_metrics(txns: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """KPIs for the latest day vs the same weekday one week earlier."""
    if txns.empty:
        return {}
    as_of = pd.Timestamp(as_of or txns["date"].max()).normalize()
    comp = as_of - pd.Timedelta(days=7)

    def stats(day):
        d = txns[txns["date"] == day]
        ok = d[d["status"] == "success"]
        return {
            "sales": float(ok["amount"].sum()),
            "txn_count": int(len(ok)),
            "avg_ticket": float(ok["amount"].mean()) if len(ok) else 0.0,
            "success_count": int(len(ok)),
            "pending_count": int((d["status"] == "pending").sum()),
            "failed_count": int((d["status"] == "failed").sum()),
            "pending_amount": float(d.loc[d["status"] == "pending", "amount"].sum()),
            "failed_amount": float(d.loc[d["status"] == "failed", "amount"].sum()),
            "attempts": int(len(d)),
        }

    cur, prev = stats(as_of), stats(comp)
    return {
        "date": as_of.date().isoformat(),
        "comparison_date": comp.date().isoformat(),
        **cur,
        "prev": prev,
        "sales_change_pct": _pct(cur["sales"], prev["sales"]),
        "txn_change_pct": _pct(cur["txn_count"], prev["txn_count"]),
        "avg_ticket_change_pct": _pct(cur["avg_ticket"], prev["avg_ticket"]),
    }


@dataclass
class PeriodComparison:
    days: int
    current_start: str
    current_end: str
    previous_start: str
    previous_end: str
    sales: float
    prev_sales: float
    sales_change_pct: float | None
    txn_count: int
    prev_txn_count: int
    txn_change_pct: float | None
    avg_ticket: float
    prev_avg_ticket: float
    avg_ticket_change_pct: float | None
    volume_effect: float  # change in sales explained by number of transactions
    ticket_effect: float  # change in sales explained by average ticket size
    failed_rate: float
    prev_failed_rate: float
    complete: bool  # False when there is not enough history for the previous period

    def to_dict(self) -> dict:
        return asdict(self)


def period_comparison(daily: pd.DataFrame, days: int = 7,
                      end: pd.Timestamp | None = None) -> PeriodComparison | None:
    """Compare the last `days` days with the `days` days before them.

    Sales change is split into a volume effect (more/fewer transactions) and a
    ticket effect (bigger/smaller average bill):
        Δsales = (n1 - n0) * avg0  +  (avg1 - avg0) * n1
    """
    if daily.empty:
        return None
    end = pd.Timestamp(end or daily["date"].max()).normalize()
    cur_start = end - pd.Timedelta(days=days - 1)
    prev_end = cur_start - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    cur = daily[(daily["date"] >= cur_start) & (daily["date"] <= end)]
    prev = daily[(daily["date"] >= prev_start) & (daily["date"] <= prev_end)]
    if cur.empty:
        return None

    s1, n1 = float(cur["sales"].sum()), int(cur["txn_count"].sum())
    s0, n0 = float(prev["sales"].sum()), int(prev["txn_count"].sum())
    a1 = s1 / n1 if n1 else 0.0
    a0 = s0 / n0 if n0 else 0.0
    f1 = cur["failed_count"].sum() / max(1, cur["failed_count"].sum() + n1)
    f0 = prev["failed_count"].sum() / max(1, prev["failed_count"].sum() + n0)
    return PeriodComparison(
        days=days,
        current_start=cur_start.date().isoformat(), current_end=end.date().isoformat(),
        previous_start=prev_start.date().isoformat(), previous_end=prev_end.date().isoformat(),
        sales=s1, prev_sales=s0, sales_change_pct=_pct(s1, s0),
        txn_count=n1, prev_txn_count=n0, txn_change_pct=_pct(n1, n0),
        avg_ticket=a1, prev_avg_ticket=a0, avg_ticket_change_pct=_pct(a1, a0),
        volume_effect=(n1 - n0) * a0, ticket_effect=(a1 - a0) * n1,
        failed_rate=float(f1 * 100), prev_failed_rate=float(f0 * 100),
        complete=len(prev) == days,
    )


# --------------------------------------------------------------------------- profiles

def weekday_profile(daily: pd.DataFrame, weeks: int | None = None) -> pd.DataFrame:
    """Average sales and transactions per weekday (optionally last N weeks)."""
    d = daily
    if weeks:
        d = daily[daily["date"] > daily["date"].max() - pd.Timedelta(weeks=weeks)]
    prof = d.groupby("weekday").agg(avg_sales=("sales", "mean"), avg_txns=("txn_count", "mean"),
                                    days=("sales", "size"))
    prof = prof.reindex([w for w in WEEKDAYS if w in prof.index]).reset_index()
    prof["avg_ticket"] = prof["avg_sales"] / prof["avg_txns"].replace(0, np.nan)
    return prof


def hourly_profile(txns: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["hour", "avg_sales", "avg_txns"])
    cutoff = txns["date"].max() - pd.Timedelta(days=days - 1)
    ok = txns[(txns["status"] == "success") & (txns["date"] >= cutoff)]
    n_days = max(1, ok["date"].nunique())
    g = ok.groupby(ok["timestamp"].dt.hour)["amount"].agg(["sum", "count"])
    g = g.reindex(range(7, 22), fill_value=0)
    return pd.DataFrame({"hour": g.index, "avg_sales": g["sum"].values / n_days,
                         "avg_txns": g["count"].values / n_days})


def payment_method_mix(txns: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["payment_method", "sales", "share_pct", "count"])
    cutoff = txns["date"].max() - pd.Timedelta(days=days - 1)
    ok = txns[(txns["status"] == "success") & (txns["date"] >= cutoff)]
    mix = ok.groupby("payment_method")["amount"].agg(sales="sum", count="count").reset_index()
    mix["share_pct"] = mix["sales"] / mix["sales"].sum() * 100 if mix["sales"].sum() else 0
    return mix.sort_values("sales", ascending=False).reset_index(drop=True)


def status_breakdown(txns: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if txns.empty:
        return pd.DataFrame(columns=["status", "count", "amount"])
    cutoff = txns["date"].max() - pd.Timedelta(days=days - 1)
    d = txns[(txns["date"] >= cutoff) & (txns["payment_method"] != "Cash")]
    out = d.groupby("status")["amount"].agg(count="count", amount="sum").reset_index()
    return out


def detect_anomalies(daily: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """Days whose sales differ strongly from the usual level for that weekday."""
    if len(daily) < 21:
        return daily.iloc[0:0].assign(expected=[], z=[], kind=[])
    d = daily.copy()
    stats = d.groupby("weekday")["sales"].agg(["median", "std"])
    d["expected"] = d["weekday"].map(stats["median"])
    d["z"] = (d["sales"] - d["expected"]) / d["weekday"].map(stats["std"]).replace(0, np.nan)
    out = d[d["z"].abs() >= z_threshold].copy()
    out["kind"] = np.where(out["z"] > 0, "spike", "dip")
    return out[["date", "weekday", "sales", "expected", "z", "kind", "failed_count", "txn_count"]]


def detect_failure_spikes(daily: pd.DataFrame, multiple: float = 3.0, min_failed: int = 8) -> pd.DataFrame:
    """Days where the payment failure rate was at least `multiple` times the usual (median) rate."""
    if daily.empty:
        return daily.assign(failed_rate=[], typical_rate=[])
    d = daily.copy()
    attempts = (d["txn_count"] + d["failed_count"]).replace(0, np.nan)
    d["failed_rate"] = d["failed_count"] / attempts * 100
    typical = float(d["failed_rate"].median())
    d["typical_rate"] = typical
    return d[(d["failed_rate"] >= typical * multiple) & (d["failed_count"] >= min_failed)]


# --------------------------------------------------------------------------- insights

@dataclass
class Insight:
    title: str
    detail: str
    kind: str  # "positive", "negative", "neutral", "watch"
    basis: str  # "fact" (measured) or "hypothesis" (possible explanation to check)


def generate_insights(txns: pd.DataFrame, daily: pd.DataFrame) -> list[Insight]:
    """Rule-based insights computed from data. No AI involved."""
    insights: list[Insight] = []
    if daily.empty or len(daily) < 2:
        return [Insight("Not enough data yet", "Insights appear after a few days of sales.",
                        "neutral", "fact")]

    wk = period_comparison(daily, 7)
    if wk and wk.complete and wk.sales_change_pct is not None:
        direction = "up" if wk.sales_change_pct >= 0 else "down"
        kind = "positive" if wk.sales_change_pct >= 3 else "negative" if wk.sales_change_pct <= -3 else "neutral"
        insights.append(Insight(
            f"Sales are {direction} {abs(wk.sales_change_pct):.1f}% this week",
            f"Last 7 days: {format_inr(wk.sales)} vs {format_inr(wk.prev_sales)} in the 7 days before.",
            kind, "fact"))
        if abs(wk.sales - wk.prev_sales) > 0:
            if abs(wk.volume_effect) >= abs(wk.ticket_effect):
                driver = ("more" if wk.volume_effect > 0 else "fewer") + " transactions"
            else:
                driver = ("bigger" if wk.ticket_effect > 0 else "smaller") + " average bills"
            insights.append(Insight(
                "What moved your sales",
                f"Change from number of transactions: {format_inr(wk.volume_effect, signed=True)}. "
                f"Change from average bill size: {format_inr(wk.ticket_effect, signed=True)}. "
                f"The bigger driver is {driver}. "
                f"Transactions {format_pct(wk.txn_change_pct)}, average bill {format_pct(wk.avg_ticket_change_pct)} "
                f"({format_inr(wk.prev_avg_ticket)} → {format_inr(wk.avg_ticket)}).",
                "neutral", "fact"))
            if wk.avg_ticket_change_pct is not None and wk.avg_ticket_change_pct <= -5:
                insights.append(Insight(
                    "Customers are buying smaller baskets",
                    "Possible reasons to check: a popular high-value item out of stock, price changes, "
                    "a competitor offer nearby, or customers splitting purchases. This is a hypothesis — "
                    "the data shows smaller bills, not why.",
                    "watch", "hypothesis"))

    prof = weekday_profile(daily, weeks=8)
    if not prof.empty:
        best = prof.loc[prof["avg_sales"].idxmax()]
        worst = prof.loc[prof["avg_sales"].idxmin()]
        insights.append(Insight(
            f"{best['weekday']} is your strongest day",
            f"Average {format_inr(best['avg_sales'])} on {best['weekday']}s vs "
            f"{format_inr(worst['avg_sales'])} on {worst['weekday']}s (last 8 weeks). "
            "Keep fast-moving stock full before your busiest days.",
            "positive", "fact"))

    if wk and wk.failed_rate - wk.prev_failed_rate >= 1.5:
        insights.append(Insight(
            "More payment failures this week",
            f"{wk.failed_rate:.1f}% of payment attempts failed vs {wk.prev_failed_rate:.1f}% the week before.",
            "negative", "fact"))

    for _, f in detect_failure_spikes(daily).tail(2).iterrows():
        insights.append(Insight(
            f"Payment failures spiked on {format_date(f['date'])}",
            f"{int(f['failed_count'])} payments failed ({f['failed_rate']:.0f}% of attempts) vs a usual "
            f"{f['typical_rate']:.0f}%. Sales that day were {format_inr(f['sales'])}. A bank or UPI network issue "
            "is a likely factor (hypothesis — check if customers complained that day). Keep a backup "
            "payment option visible during outages.",
            "watch", "fact"))

    anomalies = detect_anomalies(daily)
    for _, a in anomalies.tail(3).iterrows():
        if a["kind"] == "dip":
            extra = ""
            if a["failed_count"] >= 8:
                extra = f" {int(a['failed_count'])} payments failed that day — a payment network issue is a likely factor (to confirm)."
            insights.append(Insight(
                f"Unusually low sales on {format_date(a['date'])}",
                f"{format_inr(a['sales'])} vs a typical {a['weekday']} of {format_inr(a['expected'])}.{extra}",
                "watch", "fact"))
        else:
            insights.append(Insight(
                f"Sales spike on {format_date(a['date'])}",
                f"{format_inr(a['sales'])} vs a typical {a['weekday']} of {format_inr(a['expected'])}. "
                "If this was a festival or local event, plan stock for the next one.",
                "positive", "fact"))

    mix = payment_method_mix(txns)
    if not mix.empty:
        upi = mix.loc[mix["payment_method"] == "UPI", "share_pct"]
        if len(upi):
            insights.append(Insight(
                f"UPI is {upi.iloc[0]:.0f}% of your sales",
                "Digital payments settle to your bank account (assumed next day), not into the cash drawer — "
                "keep this in mind when planning supplier payments.",
                "neutral", "fact"))
    return insights


# --------------------------------------------------------------------------- AI context

def build_ai_context(txns: pd.DataFrame, daily: pd.DataFrame, profile: dict) -> dict:
    """Compact, verified facts passed to Claude. Claude must not go beyond these."""
    if daily.empty:
        return {"data_available": False}
    today = today_metrics(txns)
    wk = period_comparison(daily, 7)
    mo = period_comparison(daily, 30)
    prof = weekday_profile(daily, weeks=8)
    hours = hourly_profile(txns)
    mix = payment_method_mix(txns)
    anomalies = detect_anomalies(daily)
    monthly = daily.groupby(daily["date"].dt.to_period("M")).agg(
        sales=("sales", "sum"), txns=("txn_count", "sum"), days=("sales", "size"))
    weekly = daily.set_index("date").resample("W-SUN").agg(
        sales=("sales", "sum"), txn_count=("txn_count", "sum"), days=("sales", "size"))
    spikes = detect_failure_spikes(daily)

    r = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)  # noqa: E731
    return {
        "data_available": True,
        "data_note": "Synthetic demo data for a fictional merchant. Amounts in INR. "
                     "'sales' = successful payments (gross), not profit.",
        "merchant": {k: profile.get(k) for k in ["merchant_name", "city", "business_type"]},
        "data_range": {"first_day": daily["date"].min().date().isoformat(),
                       "latest_day": daily["date"].max().date().isoformat(), "days": len(daily)},
        "latest_day": {k: (r(v) if isinstance(v, float) else v) for k, v in today.items() if k != "prev"},
        "same_weekday_last_week": {k: r(v) for k, v in today.get("prev", {}).items()},
        "last_7_vs_previous_7": {k: (r(v) if isinstance(v, float) else v) for k, v in wk.to_dict().items()} if wk else None,
        "last_30_vs_previous_30": {k: (r(v) if isinstance(v, float) else v) for k, v in mo.to_dict().items()} if mo else None,
        "weekday_averages_last_8_weeks": [
            {"weekday": row.weekday, "avg_sales": r(row.avg_sales), "avg_txns": r(row.avg_txns),
             "avg_ticket": r(row.avg_ticket)} for row in prof.itertuples()],
        "hourly_avg_last_30_days": [{"hour": int(h.hour), "avg_sales": r(h.avg_sales)} for h in hours.itertuples()],
        "payment_method_share_last_30_days": [
            {"method": m.payment_method, "sales": r(m.sales), "share_pct": r(m.share_pct)} for m in mix.itertuples()],
        "unusual_days": [
            {"date": a.date.date().isoformat(), "weekday": a.weekday, "sales": r(a.sales),
             "typical_for_weekday": r(a.expected), "kind": a.kind, "failed_payments": int(a.failed_count)}
            for a in anomalies.itertuples()],
        "monthly_totals": [{"month": str(p), "sales": r(row.sales), "txns": int(row.txns), "days_in_data": int(row.days)}
                           for p, row in monthly.iterrows()],
        "weekly_totals_mon_to_sun": [{"week_ending": d.date().isoformat(), "sales": r(row.sales), "txns": int(row.txn_count),
                                      "days_in_data": int(row.days)} for d, row in weekly.iterrows()],
        "payment_failure_spikes": [
            {"date": f.date.date().isoformat(), "failed_payments": int(f.failed_count),
             "failed_rate_pct": r(f.failed_rate), "typical_rate_pct": r(f.typical_rate), "sales": r(f.sales)}
            for f in spikes.itertuples()],
        "daily": [{"date": row.date.date().isoformat(), "weekday": row.weekday[:3], "sales": r(row.sales),
                   "txns": int(row.txn_count), "failed": int(row.failed_count)} for row in daily.itertuples()],
        "insights": [asdict(i) for i in generate_insights(txns, daily)],
    }
