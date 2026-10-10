"""Claude-powered explanations over verified, pre-computed metrics.

Python computes every number (see services/analytics.py). Claude only explains
them. Without an API key, a deterministic demo mode answers common questions
from the same verified metrics.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

from services.payments import SAFETY_NOTE, detect_sensitive_info
from utils.formatting import format_date, format_inr, format_pct

try:
    import anthropic
except ImportError:  # app still works in demo mode
    anthropic = None

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # low keeps chat replies fast for live demos
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are Bharat Merchant Copilot, a friendly business analyst for a small Indian kirana store owner.
You answer questions about their shop using the VERIFIED_DATA JSON provided below, which was computed by deterministic Python code from their transaction records.

How to answer:
- Use only the numbers in VERIFIED_DATA. Never invent transactions, amounts, dates or totals. Do not add up or compute new financial totals yourself; quote the pre-computed figures (daily, weekly, monthly totals, period comparisons). If the figure needed is not there, say so and offer the closest available figure.
- Keep two clearly labelled parts: "What the data shows" (facts, with numbers and dates) and "Possible reasons to check" (hypotheses). Never claim one thing caused another unless the data directly shows it; use words like "may" or "could".
- "Sales" means gross successful payments. It is not profit and not cash in hand.
- Write for a busy shop owner: plain, simple English, short bullets, rupees with Indian grouping (₹1,23,456), dates like 10 Oct 2026. Aim for under 200 words unless asked for more.
- Finish with 1-3 practical next steps the owner can take this week.
- If the owner writes in Hindi or Hinglish, reply in the same style.
- If data is missing or too short to answer, say so plainly.

Safety:
- Never ask for a UPI PIN, OTP, password, CVV or full bank/card details. If someone shares one, tell them not to.
- You cannot take actions. Never say a payment has been reversed, refunded, settled or escalated. For a specific payment problem, point the owner to the Payment support page.
- The data is synthetic demo data for a fictional merchant."""


def ai_status() -> tuple[bool, str]:
    """(available, human-readable reason)."""
    if os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        return False, "Demo mode is forced on (DEMO_MODE=true)."
    if anthropic is None:
        return False, "The `anthropic` package is not installed."
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False, "No ANTHROPIC_API_KEY found. Add it to a .env file to enable Claude."
    return True, f"Connected to Claude ({MODEL})."


def _client():
    return anthropic.Anthropic(max_retries=2, timeout=90.0)


def stream_answer(question: str, context: dict, history: list[dict] | None = None) -> Iterator[str]:
    """Yield the answer in chunks (Claude when available, otherwise demo mode)."""
    question = (question or "").strip()
    if not question:
        yield "Please type a question about your business."
        return
    leaked = detect_sensitive_info(question)
    if leaked:
        yield (f"⚠️ Your message seems to contain sensitive data ({', '.join(leaked)}). I have not processed it. "
               f"{SAFETY_NOTE} Please rephrase without it.")
        return
    if not context.get("data_available"):
        yield "I don't have any sales data to analyse yet. Add transactions or generate the sample data first."
        return

    available, _ = ai_status()
    if not available:
        yield demo_answer(question, context)
        return

    messages = [m for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
    messages = messages[-10:] + [{"role": "user", "content": question}]
    if messages[0]["role"] != "user":
        messages = messages[1:]
    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": "VERIFIED_DATA:\n" + json.dumps(context, separators=(",", ":"), default=str)},
    ]
    try:
        with _client().beta.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            got_text = False
            for text in stream.text_stream:
                got_text = True
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_I can't help with that request. Try asking about your sales, payments or cash flow._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(Answer cut short — try a more specific question.)_"
        elif not got_text:
            yield demo_answer(question, context)
    except anthropic.AuthenticationError:
        yield "⚠️ Your Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env. Showing demo answer:\n\n"
        yield demo_answer(question, context)
    except anthropic.RateLimitError:
        yield "⚠️ Claude is busy (rate limited). Showing demo answer instead:\n\n"
        yield demo_answer(question, context)
    except anthropic.APIStatusError as e:
        yield f"⚠️ Claude returned an error ({e.status_code}). Showing demo answer instead:\n\n"
        yield demo_answer(question, context)
    except anthropic.APIConnectionError:
        yield "⚠️ Could not reach Claude (network issue). Showing demo answer instead:\n\n"
        yield demo_answer(question, context)


def answer(question: str, context: dict, history: list[dict] | None = None) -> str:
    return "".join(stream_answer(question, context, history))


# --------------------------------------------------------------------------- demo mode

def _has(q: str, *words: str) -> bool:
    return any(w in q for w in words)


def demo_answer(question: str, ctx: dict) -> str:
    """Deterministic, template-based answer built only from verified metrics."""
    q = question.lower()
    wk = ctx.get("last_7_vs_previous_7") or {}
    days = ctx.get("weekday_averages_last_8_weeks") or []
    header = "_Demo mode (no AI): answer generated from your calculated metrics._\n\n"

    if _has(q, "fail", "pending", "upi", "payment"):
        return header + _explain_payments(ctx, wk)
    if _has(q, "which day", "best day", "busiest", "weekday", "days generate", "most sales"):
        return header + _explain_weekdays(days)
    if _has(q, "average", "ticket", "basket", "bill size", "count"):
        return header + _explain_drivers(wk)
    if _has(q, "decline", "drop", "down", "fall", "less", "kam", "why"):
        return header + _explain_week(wk, ctx)
    if _has(q, "cash", "forecast", "balance", "next week"):
        return header + ("Open the **Cash-flow forecast** page: it projects your daily closing balance "
                         "for the next 7 days from your sales history and the payments you enter.")
    if _has(q, "improve", "increase", "investigate", "grow", "badhao", "tips", "should i"):
        return header + _improvement_checklist(wk, days, ctx)
    return header + _overview(ctx, wk)


def _explain_week(wk: dict, ctx: dict) -> str:
    if not wk or wk.get("sales_change_pct") is None:
        return "There isn't enough history to compare this week with last week yet."
    up = wk["sales_change_pct"] >= 0
    lines = ["**What the data shows**",
             f"- Last 7 days ({format_date(wk['current_start'])} – {format_date(wk['current_end'])}): "
             f"{format_inr(wk['sales'])}, vs {format_inr(wk['prev_sales'])} the week before "
             f"({format_pct(wk['sales_change_pct'])}).",
             f"- Transactions: {wk['txn_count']} vs {wk['prev_txn_count']} ({format_pct(wk['txn_change_pct'])}).",
             f"- Average bill: {format_inr(wk['avg_ticket'])} vs {format_inr(wk['prev_avg_ticket'])} "
             f"({format_pct(wk['avg_ticket_change_pct'])}).",
             f"- Of the change, {format_inr(wk['volume_effect'], signed=True)} comes from the number of transactions and "
             f"{format_inr(wk['ticket_effect'], signed=True)} from bill size."]
    unusual = [d for d in ctx.get("unusual_days", []) if d["date"] >= wk["current_start"]]
    for d in unusual:
        lines.append(f"- {format_date(d['date'])} was unusual: {format_inr(d['sales'])} vs a typical "
                     f"{format_inr(d['typical_for_weekday'])}.")
    lines.append("\n**Possible reasons to check** (hypotheses, not proven)")
    if abs(wk["ticket_effect"]) > abs(wk["volume_effect"]):
        lines += ["- Customers are " + ("spending more per visit" if up else "buying smaller baskets") +
                  " — check whether high-value items were out of stock or prices changed.",
                  "- A competitor offer or a change in what regular customers buy."]
    else:
        lines += ["- Fewer customer visits — weather, local events, shop timings, or a competitor nearby.",
                  "- Payment failures putting customers off (see Payment support)."]
    lines += ["\n**Next steps**", "- Check stock of your top-selling items.",
              "- Ask a few regular customers if anything changed."]
    return "\n".join(lines)


def _explain_weekdays(days: list[dict]) -> str:
    if not days:
        return "Not enough data to compare weekdays yet."
    ranked = sorted(days, key=lambda d: d["avg_sales"] or 0, reverse=True)
    lines = ["**What the data shows** (average sales, last 8 weeks)"]
    lines += [f"{i}. {d['weekday']}: {format_inr(d['avg_sales'])} from ~{d['avg_txns']:.0f} transactions"
              for i, d in enumerate(ranked, 1)]
    lines += ["\n**Next steps**",
              f"- Restock fast-moving items before {ranked[0]['weekday']}.",
              f"- Try a small offer on {ranked[-1]['weekday']} to bring in more customers."]
    return "\n".join(lines)


def _explain_drivers(wk: dict) -> str:
    if not wk:
        return "Not enough data to compare periods yet."
    t, a = wk.get("txn_change_pct"), wk.get("avg_ticket_change_pct")
    return "\n".join([
        "**What the data shows** (last 7 days vs previous 7)",
        f"- Number of transactions: {wk['prev_txn_count']} → {wk['txn_count']} ({format_pct(t)})",
        f"- Average transaction value: {format_inr(wk['prev_avg_ticket'])} → {format_inr(wk['avg_ticket'])} ({format_pct(a)})",
        "",
        "So " + ("**average bill size** moved more than footfall." if abs(a or 0) > abs(t or 0)
                 else "**the number of transactions** moved more than bill size."),
    ])


def _explain_payments(ctx: dict, wk: dict) -> str:
    latest = ctx.get("latest_day", {})
    lines = ["**What the data shows**",
             f"- Latest day ({format_date(latest.get('date', ctx['data_range']['latest_day']))}): "
             f"{latest.get('success_count', 0)} successful, {latest.get('pending_count', 0)} pending, "
             f"{latest.get('failed_count', 0)} failed payments."]
    if wk:
        lines.append(f"- Failed share of payment attempts: {wk['failed_rate']:.1f}% this week vs "
                     f"{wk['prev_failed_rate']:.1f}% last week.")
    for d in ctx.get("payment_failure_spikes", []):
        lines.append(f"- {format_date(d['date'])}: {d['failed_payments']} failed payments "
                     f"({d['failed_rate_pct']:.0f}% of attempts vs a usual {d['typical_rate_pct']:.0f}%).")
    lines += ["\n**Next steps**", "- For a specific payment, use the **Payment support** page.",
              f"- {SAFETY_NOTE}"]
    return "\n".join(lines)


def _improvement_checklist(wk: dict, days: list[dict], ctx: dict) -> str:
    lines = ["**Things to investigate this week**"]
    if wk and (wk.get("avg_ticket_change_pct") or 0) < -3:
        lines.append(f"- Average bill fell {format_pct(wk['avg_ticket_change_pct'])}: check stock of higher-value items and try combo packs.")
    if wk and (wk.get("txn_change_pct") or 0) < -3:
        lines.append(f"- Transactions fell {format_pct(wk['txn_change_pct'])}: check timings, footfall and nearby competition.")
    if days:
        worst = min(days, key=lambda d: d["avg_sales"] or 0)
        lines.append(f"- {worst['weekday']} is your slowest day: consider a small offer that day.")
    hours = ctx.get("hourly_avg_last_30_days") or []
    if hours:
        peak = max(hours, key=lambda h: h["avg_sales"] or 0)
        lines.append(f"- Your busiest hour is around {peak['hour']}:00 — keep billing quick and the QR visible then.")
    lines.append("- Review failed payments on the Payment support page; each failure is a sale at risk.")
    return "\n".join(lines)


def _overview(ctx: dict, wk: dict) -> str:
    latest = ctx.get("latest_day", {})
    lines = ["Here's a quick summary:",
             f"- Latest day: {format_inr(latest.get('sales'))} from {latest.get('txn_count', 0)} transactions."]
    if wk:
        lines.append(f"- Last 7 days: {format_inr(wk['sales'])} ({format_pct(wk['sales_change_pct'])} vs previous week).")
    lines.append("\nTry asking: *Why did my sales decline this week?*, *Which days generate the most sales?*, "
                 "or *What should I investigate to improve sales?*")
    return "\n".join(lines)
