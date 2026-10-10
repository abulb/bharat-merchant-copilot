"""Transparent 7-day cash-flow forecast.

Method (deliberately simple so a merchant can follow it):
1. Expected sales for a future day = average sales on the same weekday over the
   last 4 weeks (falls back to the recent daily average if history is short).
2. Uncertainty band comes from backtesting that same method on recent weeks:
   we record how far actual sales landed from the forecast and use the 10th and
   90th percentile errors as the low / high case.
3. Cash in: cash sales arrive the same day; digital sales (UPI, card, wallet)
   arrive in the bank after the settlement lag (default T+1).
4. Cash out: merchant-entered payments on their dates + a daily operating cost.
5. Closing balance = previous closing + cash in - cash out.

Gross sales are NOT profit: restocking costs only appear if the merchant enters
the supplier payments they expect to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HORIZON_DAYS = 7
SEASONAL_WEEKS = 4
BACKTEST_WEEKS = 6
DEFAULT_BAND = 0.20  # ±20% when there is not enough history to backtest


# --------------------------------------------------------------------------- sales forecast

def _baseline_for(history: pd.DataFrame, day: pd.Timestamp) -> float:
    """Average sales on the same weekday over the last SEASONAL_WEEKS weeks of history."""
    if history.empty:
        return 0.0
    same = history[history["date"].dt.weekday == day.weekday()].tail(SEASONAL_WEEKS)
    if len(history) >= 7 * SEASONAL_WEEKS and len(same) >= 2:
        return float(same["sales"].mean())
    return float(history["sales"].tail(7).mean())


def forecast_sales(daily: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["date", "expected_sales"])
    start = daily["date"].max() + pd.Timedelta(days=1)
    days = pd.date_range(start, periods=horizon, freq="D")
    return pd.DataFrame({"date": days, "expected_sales": [_baseline_for(daily, d) for d in days]})


@dataclass
class BacktestResult:
    available: bool
    weeks_tested: int = 0
    mape_pct: float | None = None  # mean absolute daily error, as % of the forecast
    bias_pct: float | None = None  # positive = forecast tends to be too high
    weekly_total_error_pct: float | None = None
    low_ratio: float = 1 - DEFAULT_BAND
    high_ratio: float = 1 + DEFAULT_BAND
    note: str = ""


def backtest(daily: pd.DataFrame, weeks: int = BACKTEST_WEEKS) -> BacktestResult:
    """Rolling-origin backtest: forecast each of the last `weeks` weeks using only earlier data."""
    min_history = 7 * SEASONAL_WEEKS
    available_weeks = (len(daily) - min_history) // 7
    weeks = min(weeks, available_weeks)
    if weeks < 2:
        return BacktestResult(available=False, note=(
            f"Not enough history to backtest (need at least {min_history + 14} days). "
            f"Using a default ±{int(DEFAULT_BAND * 100)}% band."))
    ratios, weekly_errors = [], []
    for w in range(weeks, 0, -1):
        cut = len(daily) - 7 * w
        history, actual = daily.iloc[:cut], daily.iloc[cut:cut + 7]
        fc = np.array([_baseline_for(history, d) for d in actual["date"]])
        act = actual["sales"].to_numpy()
        mask = fc > 0
        ratios.extend(list(act[mask] / fc[mask] - 1))
        if fc.sum() > 0:
            weekly_errors.append(abs(act.sum() - fc.sum()) / fc.sum())
    ratios = np.array(ratios)
    return BacktestResult(
        available=True,
        weeks_tested=weeks,
        mape_pct=float(np.mean(np.abs(ratios)) * 100),
        bias_pct=float(-np.mean(ratios) * 100),
        weekly_total_error_pct=float(np.mean(weekly_errors) * 100) if weekly_errors else None,
        low_ratio=float(1 + np.percentile(ratios, 10)),
        high_ratio=float(1 + np.percentile(ratios, 90)),
        note=f"Tested on the last {weeks} weeks, forecasting each week using only the data before it.",
    )


# --------------------------------------------------------------------------- planned expenses

def validate_planned_expenses(planned: pd.DataFrame, start: pd.Timestamp,
                              horizon: int = HORIZON_DAYS) -> tuple[pd.DataFrame, list[str]]:
    """Clean merchant-entered expenses. Returns (clean rows within horizon, warnings)."""
    warnings: list[str] = []
    cols = ["date", "category", "description", "amount"]
    if planned is None or planned.empty:
        return pd.DataFrame(columns=cols), warnings
    df = planned.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df = df.dropna(how="all")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["category"] = df["category"].fillna("Other").astype(str).str.strip().replace("", "Other")
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    bad_date = df["date"].isna()
    if bad_date.any():
        warnings.append(f"{int(bad_date.sum())} row(s) ignored: missing or invalid date.")
    bad_amount = df["amount"].isna() | (df["amount"] <= 0)
    if (bad_amount & ~bad_date).any():
        warnings.append(f"{int((bad_amount & ~bad_date).sum())} row(s) ignored: amount must be a positive number.")
    df = df[~bad_date & ~bad_amount]

    end = start + pd.Timedelta(days=horizon - 1)
    outside = (df["date"] < start) | (df["date"] > end)
    if outside.any():
        warnings.append(
            f"{int(outside.sum())} row(s) fall outside the forecast window "
            f"({start.strftime('%d %b')} – {end.strftime('%d %b')}) and were not counted.")
    return df[~outside].reset_index(drop=True), warnings


# --------------------------------------------------------------------------- cash-flow

@dataclass
class CashflowResult:
    table: pd.DataFrame
    backtest: BacktestResult
    summary: dict
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_cashflow(daily: pd.DataFrame, planned: pd.DataFrame, opening_cash: float,
                   daily_operating_cost: float = 0.0, settlement_lag_days: int = 1,
                   min_buffer: float = 0.0, horizon: int = HORIZON_DAYS) -> CashflowResult:
    if daily.empty:
        raise ValueError("No sales history available for forecasting.")
    if opening_cash is None or not np.isfinite(opening_cash):
        raise ValueError("Opening cash must be a number.")
    if daily_operating_cost < 0 or min_buffer < 0:
        raise ValueError("Operating cost and minimum buffer cannot be negative.")
    if settlement_lag_days not in (0, 1, 2):
        raise ValueError("Settlement lag must be 0, 1 or 2 days.")

    fc = forecast_sales(daily, horizon)
    bt = backtest(daily)
    start = fc["date"].iloc[0]
    clean, warnings = validate_planned_expenses(planned, start, horizon)

    recent = daily.tail(28)
    total = recent["sales"].sum()
    digital_share = float(recent["digital_sales"].sum() / total) if total else 0.0

    # Digital sales from recent actual days that will still land in the forecast window
    actual_digital = dict(zip(daily["date"], daily["digital_sales"]))

    rows = []
    bal_e = bal_l = bal_h = float(opening_cash)
    expected_by_day = dict(zip(fc["date"], fc["expected_sales"]))
    for d, exp_sales in expected_by_day.items():
        lo_sales, hi_sales = exp_sales * bt.low_ratio, exp_sales * bt.high_ratio
        src = d - pd.Timedelta(days=settlement_lag_days)
        if src in expected_by_day:
            fe = expected_by_day[src]
            settle_e, settle_l, settle_h = (fe * digital_share, fe * bt.low_ratio * digital_share,
                                            fe * bt.high_ratio * digital_share)
        else:  # settlement of an actual, already-recorded day
            settle_e = settle_l = settle_h = float(actual_digital.get(src, 0.0))
        if settlement_lag_days == 0:
            settle_e, settle_l, settle_h = (exp_sales * digital_share, lo_sales * digital_share,
                                            hi_sales * digital_share)
        cash_share = 1 - digital_share
        in_e = exp_sales * cash_share + settle_e
        in_l = lo_sales * cash_share + settle_l
        in_h = hi_sales * cash_share + settle_h

        day_items = clean[clean["date"] == d]
        planned_out = float(day_items["amount"].sum())
        out = planned_out + daily_operating_cost
        bal_e += in_e - out
        bal_l += in_l - out
        bal_h += in_h - out
        if bal_e < 0:
            flag = "Shortfall"
        elif bal_l < 0 or bal_e < min_buffer:
            flag = "At risk"
        else:
            flag = "OK"
        rows.append({
            "date": d, "expected_sales": exp_sales, "sales_low": lo_sales, "sales_high": hi_sales,
            "cash_in": in_e, "cash_in_low": in_l, "cash_in_high": in_h,
            "planned_payments": planned_out, "operating_cost": daily_operating_cost, "cash_out": out,
            "closing_expected": bal_e, "closing_low": bal_l, "closing_high": bal_h, "flag": flag,
            "payments_detail": ", ".join(
                f"{r.description or r.category} (₹{r.amount:,.0f})" for r in day_items.itertuples()),
        })
    table = pd.DataFrame(rows)
    low_idx = table["closing_expected"].idxmin()
    risk_rows = table[table["flag"] != "OK"]
    summary = {
        "opening_cash": float(opening_cash),
        "total_cash_in": float(table["cash_in"].sum()),
        "total_cash_out": float(table["cash_out"].sum()),
        "closing_expected": float(table["closing_expected"].iloc[-1]),
        "closing_low": float(table["closing_low"].iloc[-1]),
        "closing_high": float(table["closing_high"].iloc[-1]),
        "lowest_expected_balance": float(table.loc[low_idx, "closing_expected"]),
        "lowest_expected_date": table.loc[low_idx, "date"].date().isoformat(),
        "lowest_low_case_balance": float(table["closing_low"].min()),
        "first_risk_date": risk_rows["date"].iloc[0].date().isoformat() if not risk_rows.empty else None,
        "shortfall_days": int((table["flag"] == "Shortfall").sum()),
        "risk_days": int((table["flag"] != "OK").sum()),
        "digital_share_pct": digital_share * 100,
        "expected_sales_total": float(table["expected_sales"].sum()),
    }
    assumptions = [
        "Expected sales = average sales on the same weekday over the last 4 weeks.",
        f"{digital_share * 100:.0f}% of sales are digital (UPI/card/wallet, from the last 28 days) and "
        f"reach your bank {settlement_lag_days} day(s) later; the rest is cash received the same day.",
        "Sales are gross takings, not profit. Restocking costs are only included if you enter supplier payments.",
        f"A daily operating cost of ₹{daily_operating_cost:,.0f} is deducted every day.",
        "Low / high case use the 10th / 90th percentile of past forecast errors."
        if bt.available else f"Low / high case use a default ±{int(DEFAULT_BAND * 100)}% band.",
        "Refunds, credit (udhaar) given to customers, loan EMIs and taxes are not included unless entered.",
    ]
    return CashflowResult(table=table, backtest=bt, summary=summary, assumptions=assumptions,
                          warnings=warnings)
