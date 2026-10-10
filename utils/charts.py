"""Plotly chart builders with one consistent visual system (light theme)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from utils.formatting import format_inr

# Validated reference palette (categorical slots in fixed order) + reserved status colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND = "rgba(42,120,214,0.14)"
STATUS = {"success": "#0ca30c", "pending": "#fab219", "failed": "#d03b3b"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK_2, size=13),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font_color=INK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_color=INK_2),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont_color=MUTED,
                     tickprefix="₹", separatethousands=True)
    return fig


def sales_trend(daily: pd.DataFrame, days: int = 30) -> go.Figure:
    d = daily.tail(days).copy()
    roll = daily["sales"].rolling(7, min_periods=1).mean().tail(days)
    fig = go.Figure()
    fig.add_bar(x=d["date"], y=d["sales"], name="Daily sales", marker_color=SERIES[0],
                marker_line_width=0, customdata=d["txn_count"],
                hovertemplate="%{x|%a, %d %b}: ₹%{y:,.0f} (%{customdata} txns)<extra></extra>")
    fig.add_scatter(x=d["date"], y=roll, name="7-day average", mode="lines",
                    line=dict(color=SERIES[1], width=2),
                    hovertemplate="7-day avg: ₹%{y:,.0f}<extra></extra>")
    fig.update_xaxes(tickformat="%d %b")
    return _layout(fig)


def status_bars(status_df: pd.DataFrame) -> go.Figure:
    order = ["success", "pending", "failed"]
    s = status_df.set_index("status").reindex(order).fillna(0).reset_index()
    labels = {"success": "✓ Successful", "pending": "◷ Pending", "failed": "✕ Failed"}
    fig = go.Figure(go.Bar(
        x=s["count"], y=[labels[x] for x in s["status"]], orientation="h",
        marker_color=[STATUS[x] for x in s["status"]], text=[f"{int(c):,}" for c in s["count"]],
        textposition="outside", textfont_color=INK_2, customdata=s["amount"],
        hovertemplate="%{y}: %{x:,} payments, ₹%{customdata:,.0f}<extra></extra>"))
    fig = _layout(fig, height=200, legend=False)
    fig.update_xaxes(tickprefix="", visible=False)
    fig.update_yaxes(tickprefix="", gridcolor=SURFACE, autorange="reversed", tickfont_color=INK_2)
    fig.update_layout(hovermode="closest")
    return fig


def method_mix(mix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=mix["share_pct"], y=mix["payment_method"], orientation="h", marker_color=SERIES[0],
        text=[f"{p:.0f}%" for p in mix["share_pct"]], textposition="outside", textfont_color=INK_2,
        customdata=mix["sales"], hovertemplate="%{y}: %{x:.1f}% (₹%{customdata:,.0f})<extra></extra>"))
    fig = _layout(fig, height=200, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(tickprefix="", gridcolor=SURFACE, autorange="reversed", tickfont_color=INK_2)
    fig.update_layout(hovermode="closest")
    return fig


def weekday_bars(prof: pd.DataFrame) -> go.Figure:
    best = prof["avg_sales"].idxmax()
    colors = [SERIES[0]] * len(prof)
    fig = go.Figure(go.Bar(
        x=[w[:3] for w in prof["weekday"]], y=prof["avg_sales"], marker_color=colors,
        text=["Best" if i == best else "" for i in prof.index], textposition="outside",
        textfont_color=INK_2, customdata=prof["avg_txns"],
        hovertemplate="%{x}: ₹%{y:,.0f} avg (%{customdata:.0f} txns)<extra></extra>"))
    fig = _layout(fig, height=280, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def hourly_bars(hours: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in hours["hour"]], y=hours["avg_sales"],
        marker_color=SERIES[0], customdata=hours["avg_txns"],
        hovertemplate="%{x}: ₹%{y:,.0f} avg (%{customdata:.1f} txns)<extra></extra>"))
    fig = _layout(fig, height=280, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def balance_forecast(table: pd.DataFrame, opening: float, buffer: float, start_date) -> go.Figure:
    x = [pd.Timestamp(start_date)] + list(table["date"])
    exp = [opening] + list(table["closing_expected"])
    low = [opening] + list(table["closing_low"])
    high = [opening] + list(table["closing_high"])
    fig = go.Figure()
    fig.add_scatter(x=x, y=high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=low, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BAND,
                    name="Low–high range", hovertemplate="Low case: ₹%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=x, y=exp, mode="lines+markers", name="Expected closing balance",
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                    hovertemplate="Expected: ₹%{y:,.0f}<extra></extra>")
    fig.add_hline(y=0, line_color=STATUS["failed"], line_width=1.5,
                  annotation_text="₹0", annotation_font_color=INK_2, annotation_position="bottom left")
    if buffer > 0:
        fig.add_hline(y=buffer, line_color=MUTED, line_dash="dash", line_width=1,
                      annotation_text=f"Minimum buffer {format_inr(buffer)}", annotation_font_color=INK_2,
                      annotation_position="top left")
    fig.update_xaxes(tickformat="%a %d %b")
    return _layout(fig, height=340)


def cash_in_out(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=table["date"], y=table["cash_in"], name="Cash in (expected)", marker_color=SERIES[2],
                hovertemplate="In: ₹%{y:,.0f}<extra></extra>")
    fig.add_bar(x=table["date"], y=table["cash_out"], name="Cash out", marker_color=SERIES[1],
                customdata=table["payments_detail"],
                hovertemplate="Out: ₹%{y:,.0f}<br>%{customdata}<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(tickformat="%a %d %b")
    return _layout(fig, height=300)
