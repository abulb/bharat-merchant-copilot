"""Bharat Merchant Copilot — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from services import ai_assistant, analytics, cashflow, payments  # noqa: E402
from utils import charts, data_generator  # noqa: E402
from utils.formatting import (  # noqa: E402
    format_date, format_date_short, format_inr, format_number, format_pct,
)

st.set_page_config(page_title="Bharat Merchant Copilot", page_icon="🛒", layout="wide")

# Design rules: answer first (hero banner), colour = status (always with an icon),
# few big numbers up front, details behind expanders.
st.markdown("""
<style>
.block-container {padding-top:1.6rem; max-width:1200px;}
.stButton > button {border-radius:10px; min-height:2.75rem; font-weight:600;}
section[data-testid="stSidebar"] [role="radiogroup"] label {padding:6px 2px; font-size:1.02rem;}

.hero {display:flex; gap:18px; align-items:center; padding:18px 22px; border-radius:16px; margin-bottom:18px;
       background:#eef5fd; border:1px solid #cde2fb;}
.hero .emoji {font-size:2.5rem; line-height:1;}
.hero .title {font-size:1.45rem; font-weight:700; color:#0b0b0b; line-height:1.25;}
.hero .sub {color:#3d3c39; font-size:1rem; margin-top:4px; line-height:1.45;}
.hero.good {background:#ecf8ec; border-color:#b9e4b9;}
.hero.watch {background:#fff7e6; border-color:#f5d48a;}
.hero.bad {background:#fdeeee; border-color:#f3c1c1;}

.kpi {background:#fff; border:1px solid rgba(11,11,11,0.08); border-radius:14px; padding:14px 16px; height:100%;
      box-shadow:0 1px 3px rgba(11,11,11,0.05);}
.kpi .label {color:#52514e; font-size:0.9rem; font-weight:500;}
.kpi .value {color:#0b0b0b; font-size:1.7rem; font-weight:700; margin-top:4px; line-height:1.2;}
.kpi.small .value {font-size:1.25rem;}
.kpi .delta {font-size:0.82rem; margin-top:6px; color:#6b6a66;}
.chip {display:inline-block; padding:1px 9px; border-radius:999px; font-weight:650; margin-right:4px;}
.chip.up {background:#e3f3e3; color:#006300;} .chip.down {background:#fde8e8; color:#b42318;}

.tiles {display:flex; gap:6px; margin-top:6px;}
.tile {flex:1; border-radius:10px; padding:6px 4px; text-align:center;}
.tile .n {font-size:1.3rem; font-weight:700;} .tile .l {font-size:0.74rem; font-weight:600;}
.tile.ok {background:#e3f3e3; color:#0a5d0a;} .tile.pend {background:#fff2cf; color:#7a5200;}
.tile.fail {background:#fde8e8; color:#9b1c1c;}

.sec {font-size:1.12rem; font-weight:700; color:#0b0b0b; margin:22px 0 2px;}
.hint {color:#6b6a66; font-size:0.88rem; margin-bottom:8px;}

.insight {background:#fff; border:1px solid rgba(11,11,11,0.08); border-left:5px solid #c3c2b7; border-radius:12px;
          padding:12px 14px; margin-bottom:12px; box-shadow:0 1px 3px rgba(11,11,11,0.04);}
.insight.positive {border-left-color:#0ca30c;} .insight.negative {border-left-color:#d03b3b;}
.insight.watch {border-left-color:#fab219;}
.insight .t {font-weight:650; color:#0b0b0b; margin:4px 0 2px;} .insight .d {color:#52514e; font-size:0.92rem;}
.badge {display:inline-block; padding:1px 8px; border-radius:999px; font-size:0.72rem; font-weight:650;}
.badge.fact {background:#e8f1fc; color:#184f95;} .badge.hypothesis {background:#fdf0e1; color:#8a4b00;}

.days {display:grid; grid-template-columns:repeat(7, 1fr); gap:8px; margin:4px 0 6px;}
.day {background:#fff; border:1px solid rgba(11,11,11,0.08); border-top:5px solid #0ca30c; border-radius:12px;
      padding:10px 6px; text-align:center;}
.day.risk {border-top-color:#fab219; background:#fffbf0;} .day.short {border-top-color:#d03b3b; background:#fff5f5;}
.day .d {font-size:0.8rem; color:#52514e; font-weight:650;}
.day .b {font-size:1.05rem; font-weight:700; color:#0b0b0b; margin:4px 0 2px;}
.day .s {font-size:0.74rem; color:#52514e;} .day .pay {font-size:0.72rem; color:#8a4b00; margin-top:4px;}

.kv {display:flex; justify-content:space-between; gap:12px; padding:7px 0; border-bottom:1px solid #efeee9; font-size:0.92rem;}
.kv .k {color:#6b6a66;} .kv .v {color:#0b0b0b; font-weight:600; text-align:right;}
.steps {counter-reset:s; list-style:none; padding-left:0; margin:4px 0 0;}
.steps li {counter-increment:s; position:relative; padding:4px 0 12px 40px; color:#2b2a28; line-height:1.45;}
.steps li::before {content:counter(s); position:absolute; left:0; top:2px; width:28px; height:28px; border-radius:50%;
                   background:#2a78d6; color:#fff; font-weight:700; text-align:center; line-height:28px; font-size:0.9rem;}
.strip {padding:9px 12px; border-radius:10px; font-size:0.86rem; margin-bottom:10px;}
.strip.demo {background:#fff7e6; border:1px solid #f5d48a; color:#6b4a00;}
.strip.safe {background:#eef5fd; border:1px solid #cde2fb; color:#184f95;}

.brand {font-size:1.3rem; font-weight:750; color:#ffffff;} .brand-sub {color:#b7cdea; font-size:0.86rem; margin-bottom:10px;}
.side-pill {padding:8px 10px; border-radius:10px; font-size:0.84rem; margin-bottom:8px; line-height:1.4;}
.side-pill.ai {background:rgba(12,163,12,0.22); color:#e3f8e3;}
.side-pill.demo {background:rgba(255,255,255,0.10); color:#e8eef7;}
.side-pill.syn {background:rgba(250,178,25,0.18); color:#ffe6ad; border:1px solid rgba(250,178,25,0.35);}
@media (max-width: 900px) {.days {grid-template-columns:repeat(4, 1fr);} .hero .title {font-size:1.2rem;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data

def _data_signature() -> float:
    p = analytics.DATA_DIR / "transactions.csv"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner="Loading your shop data…")
def load_everything(_sig: float):
    txns, expenses, planned, profile = analytics.load_data()
    daily = analytics.daily_summary(txns)
    context = analytics.build_ai_context(txns, daily, profile)
    return txns, expenses, planned, profile, daily, context


try:
    txns, expenses, planned_default, profile, daily, ai_context = load_everything(_data_signature())
except analytics.DataError as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if txns.empty or daily.empty:
    st.warning("No transactions found. Use **Regenerate sample data** in the sidebar.")

latest_day = daily["date"].max() if not daily.empty else pd.Timestamp(date.today())
NO_BAR = {"displayModeBar": False}


# --------------------------------------------------------------------------- UI helpers

def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def hero(emoji: str, title: str, sub: str = "", tone: str = ""):
    """One plain-language headline at the top of every page."""
    html(f'<div class="hero {tone}"><div class="emoji">{emoji}</div><div><div class="title">{title}</div>'
         f'<div class="sub">{sub}</div></div></div>')


def section(title: str, hint: str = ""):
    html(f'<div class="sec">{title}</div>' + (f'<div class="hint">{hint}</div>' if hint else ""))


def _chip(delta: float | None, good_when_up: bool = True, unit: str = "%") -> str:
    if delta is None:
        return ""
    up = delta >= 0
    cls = "up" if up == good_when_up else "down"
    return f'<span class="chip {cls}">{"▲" if up else "▼"} {abs(delta):.1f}{unit}</span>'


def kpi(label: str, value: str, delta: float | None = None, caption: str = "", good_when_up: bool = True,
        unit: str = "%", small: bool = False, extra: str = ""):
    delta_html = f'<div class="delta">{_chip(delta, good_when_up, unit)}{caption}</div>' if (caption or delta is not None) else ""
    html(f'<div class="kpi {"small" if small else ""}"><div class="label">{label}</div>'
         f'<div class="value">{value}</div>{delta_html}{extra}</div>')


INSIGHT_ICON = {"positive": "📈", "negative": "📉", "watch": "👀", "neutral": "ℹ️"}


def insight_card(i: analytics.Insight):
    label = "Measured" if i.basis == "fact" else "Hypothesis — worth checking"
    html(f'<div class="insight {i.kind}"><span class="badge {i.basis}">{label}</span>'
         f'<div class="t">{INSIGHT_ICON.get(i.kind, "")} {escape(i.title)}</div><div class="d">{escape(i.detail)}</div></div>')


def _driver(wk) -> str:
    if abs(wk.ticket_effect) > abs(wk.volume_effect):
        return f"customers spent {'more' if wk.ticket_effect > 0 else 'less'} per bill " \
               f"({format_inr(wk.prev_avg_ticket)} → {format_inr(wk.avg_ticket)})"
    return f"{'more' if wk.volume_effect > 0 else 'fewer'} payments ({wk.prev_txn_count} → {wk.txn_count})"


# --------------------------------------------------------------------------- sidebar

PAGES = ["📊 Overview", "💡 Business insights", "💰 Cash-flow forecast", "🛟 Payment support", "💬 AI chat"]

with st.sidebar:
    html(f'<div class="brand">🛒 Merchant Copilot</div>'
         f'<div class="brand-sub">{escape(profile.get("merchant_name", "Your shop"))} · {escape(profile.get("city", ""))}</div>')
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    ok, reason = ai_assistant.ai_status()
    html(f'<div class="side-pill ai">🤖 <b>AI connected</b><br>{escape(reason)}</div>' if ok else
         f'<div class="side-pill demo">🧪 <b>Demo mode</b> — works offline<br>{escape(reason)}</div>')
    span = f'<br>{format_date(daily["date"].min())} – {format_date(latest_day)}' if not daily.empty else ""
    html(f'<div class="side-pill syn">⚠️ <b>Synthetic data</b> for a fictional shop{span}</div>')
    if st.button("🔄 Regenerate sample data", width="stretch",
                 help="Recreates 90 days of synthetic data ending today (seed 42)."):
        data_generator.write_all()
        st.cache_data.clear()
        st.session_state.pop("planned_editor_data", None)
        st.rerun()


# --------------------------------------------------------------------------- pages

def page_overview():
    t = analytics.today_metrics(txns)
    if not t:
        hero("👋", "Namaste!", "No sales recorded yet. Generate sample data from the sidebar to explore.")
        return
    name = profile.get("owner_name", "ji").split()[0]
    weekday = pd.Timestamp(t["date"]).day_name()
    wk, mo = analytics.period_comparison(daily, 7), analytics.period_comparison(daily, 30)
    change = t["sales_change_pct"]
    tone = "good" if (change or 0) >= 3 else "watch" if (change or 0) <= -3 else ""
    vs = (f"That's <b>{abs(change):.0f}% {'more' if change >= 0 else 'less'}</b> than last {weekday} "
          f"({format_inr(t['prev']['sales'])}).") if change is not None else ""
    week = (f" This week: <b>{format_inr(wk.sales)}</b> ({format_pct(wk.sales_change_pct)} vs last week), "
            f"mainly because {_driver(wk)}.") if wk and wk.complete else ""
    hero("👋", f"Namaste {name} ji — today you sold {format_inr(t['sales'])}",
         f"{format_date_short(latest_day)} · {t['txn_count']} payments received. {vs}{week}", tone)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("💰 Today's sales", format_inr(t["sales"]), change, f"vs last {weekday}")
    with c2:
        kpi("🧾 Payments received", format_number(t["txn_count"]), t["txn_change_pct"], "vs last week")
    with c3:
        kpi("🛍️ Average bill", format_inr(t["avg_ticket"]), t["avg_ticket_change_pct"], "vs last week")
    with c4:
        tiles = (f'<div class="tiles"><div class="tile ok"><div class="n">{t["success_count"]}</div><div class="l">✓ Paid</div></div>'
                 f'<div class="tile pend"><div class="n">{t["pending_count"]}</div><div class="l">◷ Pending</div></div>'
                 f'<div class="tile fail"><div class="n">{t["failed_count"]}</div><div class="l">✕ Failed</div></div></div>')
        html(f'<div class="kpi"><div class="label">📲 Payment status today</div>{tiles}'
             f'<div class="delta">Not yet received: {format_inr(t["pending_amount"] + t["failed_amount"])}</div></div>')

    section("📅 This week vs last week")
    c1, c2, c3, c4 = st.columns(4)
    if wk:
        with c1:
            kpi("Sales · last 7 days", format_inr(wk.sales), wk.sales_change_pct, "", small=True)
        with c2:
            kpi("Average bill · 7 days", format_inr(wk.avg_ticket), wk.avg_ticket_change_pct, "", small=True)
        with c4:
            kpi("Failed payments · 7 days", f"{wk.failed_rate:.1f}%", wk.failed_rate - wk.prev_failed_rate,
                "of attempts", good_when_up=False, unit=" pts", small=True)
    if mo:
        with c3:
            kpi("Sales · last 30 days", format_inr(mo.sales), mo.sales_change_pct if mo.complete else None,
                "" if mo.complete else "not enough history", small=True)

    with st.container(border=True):
        section("📈 Your sales — last 30 days", "Blue bars = each day's sales · orange line = 7-day average (smooths out ups and downs)")
        st.plotly_chart(charts.sales_trend(daily), width="stretch", config=NO_BAR)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        section("📲 Digital payments — last 30 days", "How many UPI/card/wallet payments went through")
        st.plotly_chart(charts.status_bars(analytics.status_breakdown(txns)), width="stretch", config=NO_BAR)
    with c2, st.container(border=True):
        section("💳 How customers pay — last 30 days", "Share of your sales by payment method")
        st.plotly_chart(charts.method_mix(analytics.payment_method_mix(txns)), width="stretch", config=NO_BAR)

    section("💡 What changed", "The most important things in your numbers right now")
    cols = st.columns(3)
    for col, i in zip(cols, analytics.generate_insights(txns, daily)[:3]):
        with col:
            insight_card(i)
    with st.expander("📋 See daily numbers (table)"):
        show = daily.tail(30).sort_values("date", ascending=False)
        st.dataframe(pd.DataFrame({
            "Date": show["date"].dt.strftime("%d-%m-%Y"), "Day": show["weekday"],
            "Sales": show["sales"].map(format_inr), "Payments": show["txn_count"],
            "Avg bill": show["avg_ticket"].map(format_inr), "Failed": show["failed_count"],
        }), hide_index=True, width="stretch")


def _render_answer(question: str, history: list[dict] | None = None) -> str:
    return st.write_stream(ai_assistant.stream_answer(question, ai_context, history))


def page_insights():
    insights = analytics.generate_insights(txns, daily)
    hero("💡", "What your numbers are telling you",
         '<span class="badge fact">Measured</span> calculated from your sales &nbsp;·&nbsp; '
         '<span class="badge hypothesis">Hypothesis</span> a possible reason worth checking')
    left, right = st.columns(2)
    for n, i in enumerate(insights):
        with (left if n % 2 == 0 else right):
            insight_card(i)

    c1, c2 = st.columns(2)
    prof = analytics.weekday_profile(daily, weeks=8)
    with c1, st.container(border=True):
        section("🗓️ Best days of the week", "Average sales per weekday, last 8 weeks")
        if not prof.empty:
            st.plotly_chart(charts.weekday_bars(prof), width="stretch", config=NO_BAR)
    with c2, st.container(border=True):
        section("⏰ Busiest hours", "Average sales per hour, last 30 days")
        st.plotly_chart(charts.hourly_bars(analytics.hourly_profile(txns)), width="stretch", config=NO_BAR)

    with st.container(border=True):
        section("🤖 Ask the AI business analyst", "Tap a question — the AI explains your calculated numbers in simple words")
        questions = ["📉 Why did my sales decline this week?", "🗓️ Which days generate the most sales?",
                     "🧾 Are my transaction counts or average bills declining?", "🚀 What should I investigate to improve sales?"]
        c1, c2 = st.columns(2)
        for n, q in enumerate(questions):
            if (c1 if n % 2 == 0 else c2).button(q, width="stretch"):
                st.session_state["analyst_q"] = q.split(" ", 1)[1]
        c1, c2 = st.columns([5, 1])
        custom = c1.text_input("Or ask your own question", placeholder="e.g. How did the festival week compare?",
                               label_visibility="collapsed")
        if c2.button("Ask", type="primary", width="stretch") and custom.strip():
            st.session_state["analyst_q"] = custom.strip()
        q = st.session_state.get("analyst_q")
        if q:
            st.markdown(f"**You asked:** {q}")
            cached = st.session_state.get("analyst_answer")
            if cached and cached[0] == q:  # don't re-call the API on unrelated reruns
                st.markdown(cached[1])
            else:
                st.session_state["analyst_answer"] = (q, _render_answer(q))
            st.caption("Numbers come from Python calculations on your data; the AI only explains them.")


def _day_strip(table: pd.DataFrame) -> str:
    cls = {"OK": "", "At risk": "risk", "Shortfall": "short"}
    icon = {"OK": "✅ OK", "At risk": "⚠️ Tight", "Shortfall": "🚨 Short"}
    cells = []
    for r in table.itertuples():
        pay = f'<div class="pay">Pay {format_inr(r.planned_payments)}</div>' if r.planned_payments else ""
        cells.append(f'<div class="day {cls[r.flag]}"><div class="d">{r.date:%a %d %b}</div>'
                     f'<div class="b">{format_inr(r.closing_expected)}</div><div class="s">{icon[r.flag]}</div>{pay}</div>')
    return f'<div class="days">{"".join(cells)}</div>'


def page_cashflow():
    if len(daily) < 7:
        hero("💰", "Cash-flow forecast", "At least 7 days of sales history is needed for a forecast.")
        return
    start = latest_day + pd.Timedelta(days=1)
    top = st.container()  # verdict is drawn here after the inputs below are read

    with st.expander("✏️ Change your cash and upcoming payments (supplier, rent, bills…)"):
        with st.form("cashflow_inputs", border=False):
            c1, c2, c3, c4 = st.columns(4)
            opening = c1.number_input("Cash available now (₹)", min_value=0.0, step=1000.0,
                                      value=float(profile.get("opening_cash", 25000)),
                                      help="Cash in drawer + bank balance at close of the latest day. "
                                           "Exclude today's digital payments that haven't settled yet — they are added automatically.")
            buffer = c2.number_input("Minimum cash to keep (₹)", min_value=0.0, step=1000.0,
                                     value=float(profile.get("min_cash_buffer", 5000)))
            op_cost = c3.number_input("Daily running cost (₹)", min_value=0.0, step=100.0,
                                      value=float(profile.get("daily_operating_cost", 600)),
                                      help="Tea, transport, packaging, small daily spends.")
            lag = c4.selectbox("UPI/card money reaches bank", [1, 0, 2],
                               format_func=lambda x: {0: "Same day", 1: "Next day (T+1)", 2: "In 2 days (T+2)"}[x])
            st.markdown(f"**Upcoming payments** · {format_date(start)} – {format_date(start + pd.Timedelta(days=6))} · "
                        "edit, add or delete rows")
            if "planned_editor_data" not in st.session_state:
                p = planned_default.copy()
                p["date"] = pd.to_datetime(p["date"]).dt.date
                st.session_state["planned_editor_data"] = p
            edited = st.data_editor(
                st.session_state["planned_editor_data"], num_rows="dynamic", width="stretch", hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="DD-MM-YYYY", required=True),
                    "category": st.column_config.SelectboxColumn(
                        "Type", options=["Supplier", "Rent", "Salary", "Utilities", "Loan EMI", "Other"], required=True),
                    "description": st.column_config.TextColumn("Description"),
                    "amount": st.column_config.NumberColumn("Amount (₹)", min_value=0, format="₹ %.0f", required=True),
                })
            submitted = st.form_submit_button("🔄 Update forecast", type="primary")
    if submitted:
        st.session_state["planned_editor_data"] = edited

    try:
        res = cashflow.build_cashflow(daily, edited, opening, op_cost, lag, buffer)
    except ValueError as e:
        top.error(str(e))
        return
    s = res.summary
    rng = f"could be anywhere from {format_inr(s['closing_low'])} to {format_inr(s['closing_high'])}"
    with top:
        if s["shortfall_days"]:
            hero("🚨", f"You may run short of cash on {format_date_short(s['first_risk_date'])}",
                 f"Lowest expected balance: <b>{format_inr(s['lowest_expected_balance'])}</b> on "
                 f"{format_date_short(s['lowest_expected_date'])}. Try moving a supplier payment or arranging funds.", "bad")
        elif s["risk_days"]:
            hero("⚠️", f"Cash will be tight around {format_date_short(s['first_risk_date'])}",
                 f"If sales are slow, your balance could drop to <b>{format_inr(s['lowest_low_case_balance'])}</b>. "
                 f"Expected balance after 7 days: {format_inr(s['closing_expected'])} ({rng}).", "watch")
        else:
            hero("✅", "You have enough cash for the next 7 days",
                 f"Expected balance after 7 days: <b>{format_inr(s['closing_expected'])}</b> ({rng}).", "good")
        for w in res.warnings:
            st.warning(w)
        section("🗓️ Your next 7 days", "Expected closing balance each day · ✅ OK · ⚠️ tight if sales are slow · 🚨 below ₹0")
        html(_day_strip(res.table))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("🏦 Cash now", format_inr(opening), caption="what you entered", small=True)
    with c2:
        kpi("⬇️ Money coming in", format_inr(s["total_cash_in"]), caption="expected over 7 days", small=True)
    with c3:
        kpi("⬆️ Money going out", format_inr(s["total_cash_out"]), caption="payments + daily costs", small=True)
    with c4:
        kpi("📉 Lowest point", format_inr(s["lowest_expected_balance"]),
            caption=f"on {format_date_short(s['lowest_expected_date'])}", small=True)

    with st.container(border=True):
        section("📈 Projected balance", "Blue line = expected · shaded band = range if sales are slower or faster than usual")
        st.plotly_chart(charts.balance_forecast(res.table, opening, buffer, latest_day), width="stretch", config=NO_BAR)
    with st.container(border=True):
        section("↕️ Money in vs money out", "Hover over an orange bar to see which payment is due")
        st.plotly_chart(charts.cash_in_out(res.table), width="stretch", config=NO_BAR)

    with st.expander("📋 Day-by-day table"):
        t = res.table
        st.dataframe(pd.DataFrame({
            "Date": t["date"].map(format_date_short), "Forecast sales": t["expected_sales"].map(format_inr),
            "Cash in": t["cash_in"].map(format_inr), "Cash out": t["cash_out"].map(format_inr),
            "Payments": t["payments_detail"].replace("", "—"),
            "Closing (expected)": t["closing_expected"].map(format_inr),
            "Closing (slow sales)": t["closing_low"].map(format_inr),
            "Status": t["flag"].map({"OK": "✅ OK", "At risk": "⚠️ At risk", "Shortfall": "🚨 Shortfall"}),
        }), hide_index=True, width="stretch")
    with st.expander("📐 How this forecast works & how accurate it has been"):
        for a in res.assumptions:
            st.markdown(f"- {a}")
        bt = res.backtest
        if bt.available:
            st.markdown(
                f"**Backtest:** {bt.note} On a typical day, actual sales were **{bt.mape_pct:.0f}%** away from the "
                f"forecast; the weekly total was off by **{bt.weekly_total_error_pct:.0f}%** on average. "
                f"Bias: forecast was {'higher' if bt.bias_pct > 0 else 'lower'} than actual by {abs(bt.bias_pct):.1f}% on average. "
                f"Low case uses {format_pct((bt.low_ratio - 1) * 100)} and high case {format_pct((bt.high_ratio - 1) * 100)} of the forecast.")
        else:
            st.markdown(f"**Backtest:** {bt.note}")
        st.caption("Forecasts are estimates, not guarantees. Festivals, weather and local events can move sales a lot.")


SCENARIO_ICON = {"not_visible": "👀", "pending": "⏳", "failed": "❌", "duplicate": "👥", "refund": "↩️"}
TONE = {"success": ("good", "✅"), "warning": ("watch", "⚠️"), "error": ("bad", "❌"), "info": ("", "ℹ️")}


def page_payments():
    hero("🛟", "Sort out a customer's payment problem",
         "<b>1.</b> Pick the problem &nbsp;→&nbsp; <b>2.</b> Find the payment &nbsp;→&nbsp; <b>3.</b> Follow the steps")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="strip demo">🧪 {escape(payments.DEMO_DISCLAIMER)}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="strip safe">🔒 {escape(payments.SAFETY_NOTE)}</div>', unsafe_allow_html=True)

    section("1. What's the problem?")
    scenario = st.radio("Problem", list(payments.SCENARIOS), horizontal=True, label_visibility="collapsed",
                        format_func=lambda k: f"{SCENARIO_ICON[k]} {payments.SCENARIOS[k]['title']}")
    section("2. Find the payment", payments.SCENARIOS[scenario]["ask"])
    samples = payments.sample_transactions(txns, scenario)
    c1, c2, c3 = st.columns([2, 3, 1.3], vertical_alignment="bottom")
    mode = c1.radio("Find by", ["Sample payment", "Type reference"], horizontal=True)
    if mode == "Sample payment" and not samples.empty:
        options = {f"{r.txn_id} · {format_inr(r.amount)} · {r.status} · {r.timestamp:%d %b %I:%M %p}": r.txn_id
                   for r in samples.itertuples()}
        ref = options[c2.selectbox("Sample payments for this problem", list(options))]
    else:
        ref = c2.text_input("Transaction ID or 12-digit UTR", max_chars=40,
                            placeholder="e.g. T20261010900001 or 412345678901")
    if not c3.button("🔍 Check payment", type="primary", width="stretch"):
        return

    leaked = payments.detect_sensitive_info(ref)
    if leaked:
        st.error(f"That looks like it contains sensitive data ({', '.join(leaked)}). Please don't enter it here. "
                 f"{payments.SAFETY_NOTE}")
        return
    err = payments.validate_reference(ref)
    if err:
        st.error(err)
        return
    txn = payments.lookup_transaction(txns, ref)
    g = payments.diagnose(scenario, txn, txns)

    section("3. What we found")
    tone, emoji = TONE[g.tone]
    hero(emoji, escape(g.headline), " ".join(escape(f) for f in g.findings), tone)
    left, right = st.columns([2, 3])
    with left:
        if txn is not None:
            with st.container(border=True):
                section("🧾 Payment details")
                html("".join(f'<div class="kv"><span class="k">{escape(k)}</span><span class="v">{escape(str(v))}</span></div>'
                             for k, v in payments.merchant_view(txn).items()))
    with right, st.container(border=True):
        section("✅ What to do now")
        html('<ol class="steps">' + "".join(f"<li>{escape(s)}</li>" for s in g.steps) + "</ol>")
        with st.expander("📞 If it isn't resolved"):
            for e in g.escalation:
                st.markdown(f"- {e}")
    if g.related is not None and len(g.related) > 1:
        section("👥 Related payments from the same customer")
        st.dataframe(pd.DataFrame({
            "Transaction ID": g.related["txn_id"], "Time": pd.to_datetime(g.related["timestamp"]).dt.strftime("%d %b %I:%M:%S %p"),
            "Amount": g.related["amount"].map(lambda a: format_inr(a, 2)), "Status": g.related["status"].str.capitalize(),
        }), hide_index=True, width="stretch")
    st.caption("This guidance is general information for a demo, not an action on a real payment.")


CHAT_SUGGESTIONS = ["📉 Why did my sales decline this week?", "🗓️ Which days generate the most sales?",
                    "💰 How much did I sell in the last 30 days?", "🚀 What should I investigate to improve sales?"]


def page_chat():
    ok, reason = ai_assistant.ai_status()
    hero("💬", "Ask anything about your shop",
         "I use the numbers this app calculated from your sales and explain them in simple words."
         + ("" if ok else " <b>Demo mode:</b> answers come from templates — add an Anthropic API key to enable AI."))
    history = st.session_state.setdefault("chat", [])
    q = None
    if not history:
        section("Try one of these")
        c1, c2 = st.columns(2)
        for n, s in enumerate(CHAT_SUGGESTIONS):
            if (c1 if n % 2 == 0 else c2).button(s, width="stretch"):
                q = s.split(" ", 1)[1]
    else:
        _, c2 = st.columns([5, 1])
        if c2.button("🗑️ Clear chat", width="stretch"):
            st.session_state["chat"] = []
            st.rerun()
    for m in history:
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🛒"):
            st.markdown(m["content"])
    q = st.chat_input("Ask about your sales, payments or customers…") or q
    if q:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🛒"):
            reply = _render_answer(q, history)
        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": reply}])


{
    PAGES[0]: page_overview, PAGES[1]: page_insights, PAGES[2]: page_cashflow,
    PAGES[3]: page_payments, PAGES[4]: page_chat,
}[page]()
