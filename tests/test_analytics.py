import pandas as pd
import pytest

from services import analytics


def _txns(rows):
    df = pd.DataFrame(rows, columns=["txn_id", "timestamp", "amount", "payment_method", "status"])
    return analytics.clean_transactions(df)


def test_daily_summary_counts_only_successful_payments():
    t = _txns([
        ("a", "2026-10-01 10:00", 100, "UPI", "success"),
        ("b", "2026-10-01 11:00", 300, "Cash", "success"),
        ("c", "2026-10-01 12:00", 999, "UPI", "failed"),
        ("d", "2026-10-03 12:00", 50, "UPI", "pending"),
    ])
    d = analytics.daily_summary(t)
    assert len(d) == 3  # gap day 2 Oct is filled with zero
    day1 = d.iloc[0]
    assert day1["sales"] == 400 and day1["txn_count"] == 2 and day1["avg_ticket"] == 200
    assert day1["failed_count"] == 1 and day1["digital_sales"] == 100 and day1["cash_sales"] == 300
    assert d.iloc[1]["sales"] == 0
    assert d.iloc[2]["pending_count"] == 1 and d.iloc[2]["sales"] == 0


def test_clean_transactions_rejects_missing_columns_and_bad_rows():
    with pytest.raises(analytics.DataError):
        analytics.clean_transactions(pd.DataFrame({"txn_id": ["x"]}))
    t = _txns([("a", "not a date", 100, "UPI", "success"), ("b", "2026-10-01", -5, "UPI", "success"),
               ("c", "2026-10-01", "abc", "UPI", "success"), ("d", "2026-10-01", 10, "UPI", "SUCCESS ")])
    assert list(t["txn_id"]) == ["d"] and t.iloc[0]["status"] == "success"


def test_daily_totals_match_raw_data(dataset):
    t, d = dataset["txns"], dataset["daily"]
    assert d["sales"].sum() == pytest.approx(t.loc[t["status"] == "success", "amount"].sum())
    assert d["txn_count"].sum() == (t["status"] == "success").sum()
    assert len(d) == 90


def test_period_comparison_decomposition_is_exact(dataset):
    wk = analytics.period_comparison(dataset["daily"], 7)
    assert wk.complete
    assert wk.volume_effect + wk.ticket_effect == pytest.approx(wk.sales - wk.prev_sales)
    assert wk.current_end == "2026-10-10" and wk.previous_start == "2026-09-27"


def test_period_comparison_incomplete_history():
    t = _txns([("a", "2026-10-01 10:00", 100, "UPI", "success"), ("b", "2026-10-05 10:00", 100, "UPI", "success")])
    mo = analytics.period_comparison(analytics.daily_summary(t), 30)
    assert mo is not None and not mo.complete and mo.sales_change_pct is None


def test_today_metrics_compares_same_weekday_last_week(dataset):
    m = analytics.today_metrics(dataset["txns"])
    assert m["date"] == "2026-10-10" and m["comparison_date"] == "2026-10-03"
    t = dataset["txns"]
    today = t[(t["date"] == "2026-10-10")]
    assert m["sales"] == pytest.approx(today.loc[today["status"] == "success", "amount"].sum())
    assert m["pending_count"] == (today["status"] == "pending").sum()


def test_weekday_profile_and_anomalies(dataset):
    prof = analytics.weekday_profile(dataset["daily"], weeks=8)
    assert list(prof["weekday"]) == analytics.WEEKDAYS
    assert prof.loc[prof["avg_sales"].idxmax(), "weekday"] in ("Saturday", "Sunday")
    spikes = analytics.detect_failure_spikes(dataset["daily"])
    assert "2026-09-21" in set(spikes["date"].dt.strftime("%Y-%m-%d"))  # planted UPI outage


def test_insights_and_context_are_built_from_data(dataset):
    ins = analytics.generate_insights(dataset["txns"], dataset["daily"])
    assert ins and all(i.basis in ("fact", "hypothesis") for i in ins)
    ctx = analytics.build_ai_context(dataset["txns"], dataset["daily"], dataset["profile"])
    assert ctx["data_available"] and len(ctx["daily"]) == 90
    assert sum(m["sales"] for m in ctx["monthly_totals"]) == pytest.approx(dataset["daily"]["sales"].sum(), rel=1e-6)


def test_empty_data_is_handled():
    empty = _txns([])
    assert analytics.daily_summary(empty).empty
    assert analytics.today_metrics(empty) == {}
    assert analytics.build_ai_context(empty, analytics.daily_summary(empty), {}) == {"data_available": False}
