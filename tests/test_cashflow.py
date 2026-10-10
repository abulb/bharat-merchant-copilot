import pandas as pd
import pytest

from services import cashflow
from tests.conftest import make_daily


def test_forecast_uses_same_weekday_average():
    # 5 weeks where each weekday has a fixed value: Mon=100, Tue=200, ... Sun=700
    sales = [100 * (i % 7 + 1) for i in range(35)]
    daily = make_daily(sales, start="2026-01-05")  # a Monday
    fc = cashflow.forecast_sales(daily, 7)
    assert list(fc["expected_sales"]) == [100, 200, 300, 400, 500, 600, 700]
    assert fc["date"].iloc[0] == daily["date"].max() + pd.Timedelta(days=1)


def test_short_history_falls_back_to_recent_average_and_default_band():
    daily = make_daily([1000] * 10)
    fc = cashflow.forecast_sales(daily, 7)
    assert (fc["expected_sales"] == 1000).all()
    bt = cashflow.backtest(daily)
    assert not bt.available and bt.low_ratio == pytest.approx(0.8)


def test_cash_balance_arithmetic_with_settlement_lag():
    daily = make_daily([1000] * 42, digital_share=0.5)
    start = daily["date"].max() + pd.Timedelta(days=1)
    planned = pd.DataFrame([{"date": start + pd.Timedelta(days=1), "category": "Supplier",
                             "description": "Stock", "amount": 3000}])
    res = cashflow.build_cashflow(daily, planned, opening_cash=500, daily_operating_cost=100,
                                  settlement_lag_days=1, min_buffer=0)
    t = res.table
    # constant history -> perfect backtest -> no band
    assert res.backtest.available and res.backtest.mape_pct == pytest.approx(0)
    # each day: 500 cash + 500 settled digital (from previous day) in; 100 out
    assert t["cash_in"].iloc[0] == pytest.approx(1000)
    assert t["closing_expected"].iloc[0] == pytest.approx(500 + 1000 - 100)
    assert t["closing_expected"].iloc[1] == pytest.approx(1400 + 1000 - 3100)  # -700
    assert t["flag"].iloc[1] == "Shortfall"
    assert res.summary["first_risk_date"] == t["date"].iloc[1].date().isoformat()
    assert res.summary["total_cash_out"] == pytest.approx(3000 + 700)
    assert res.summary["closing_expected"] == pytest.approx(500 + 7000 - 3700)


def test_buffer_flags_at_risk():
    daily = make_daily([1000] * 42)
    res = cashflow.build_cashflow(daily, pd.DataFrame(), opening_cash=0, min_buffer=5000)
    assert res.table["flag"].iloc[0] == "At risk"
    assert res.table["flag"].iloc[-1] == "OK"  # 7000 > 5000 by day 7


def test_planned_expense_validation():
    start = pd.Timestamp("2026-10-11")
    planned = pd.DataFrame([
        {"date": "2026-10-12", "category": "Rent", "description": "ok", "amount": 100},
        {"date": "bad", "category": "Rent", "description": "", "amount": 100},
        {"date": "2026-10-13", "category": "", "description": "", "amount": -5},
        {"date": "2026-11-30", "category": "Rent", "description": "", "amount": 100},
        {"date": None, "category": None, "description": None, "amount": None},
    ])
    clean, warnings = cashflow.validate_planned_expenses(planned, start)
    assert len(clean) == 1 and clean.iloc[0]["amount"] == 100
    assert len(warnings) == 3


def test_invalid_inputs_raise():
    daily = make_daily([1000] * 30)
    with pytest.raises(ValueError):
        cashflow.build_cashflow(daily, pd.DataFrame(), opening_cash=0, settlement_lag_days=5)
    with pytest.raises(ValueError):
        cashflow.build_cashflow(daily, pd.DataFrame(), opening_cash=0, daily_operating_cost=-1)
    with pytest.raises(ValueError):
        cashflow.build_cashflow(daily.iloc[0:0], pd.DataFrame(), opening_cash=0)


def test_real_dataset_forecast(dataset):
    p = dataset["profile"]
    res = cashflow.build_cashflow(dataset["daily"], dataset["planned"], p["opening_cash"],
                                  p["daily_operating_cost"], 1, p["min_cash_buffer"])
    assert len(res.table) == 7 and not res.warnings
    assert res.backtest.available and res.backtest.weeks_tested == 6
    assert (res.table["closing_low"] <= res.table["closing_expected"] + 1e-6).all()
    assert (res.table["closing_high"] >= res.table["closing_expected"] - 1e-6).all()
    assert res.summary["total_cash_out"] == pytest.approx(dataset["planned"]["amount"].sum() + 7 * p["daily_operating_cost"])
