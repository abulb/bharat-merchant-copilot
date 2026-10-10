import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services import analytics  # noqa: E402
from utils import data_generator  # noqa: E402

END = date(2026, 10, 10)


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """Synthetic dataset with a fixed end date, written to a temp dir."""
    d = tmp_path_factory.mktemp("data")
    data_generator.write_all(d, seed=42, days=90, end_date=END)
    txns, expenses, planned, profile = analytics.load_data(d, auto_generate=False)
    daily = analytics.daily_summary(txns)
    return {"txns": txns, "expenses": expenses, "planned": planned, "profile": profile, "daily": daily}


def make_daily(sales: list[float], start: str = "2026-01-05", digital_share: float = 0.5) -> pd.DataFrame:
    """Hand-built daily table for exact arithmetic checks."""
    dates = pd.date_range(start, periods=len(sales), freq="D")
    df = pd.DataFrame({"date": dates, "sales": sales})
    df["txn_count"] = 10
    df["digital_sales"] = df["sales"] * digital_share
    df["cash_sales"] = df["sales"] - df["digital_sales"]
    df["failed_count"] = 0
    df["pending_count"] = 0
    df["failed_amount"] = 0.0
    df["refunds"] = 0.0
    df["avg_ticket"] = df["sales"] / 10
    df["weekday"] = df["date"].dt.day_name()
    return df
