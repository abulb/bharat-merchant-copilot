# 🛒 Bharat Merchant Copilot

An AI-powered copilot that helps kirana store owners and small retailers **understand their sales, forecast next week's cash, and troubleshoot digital payment issues** — in plain language.

> ⚠️ **All data is synthetic.** "Sharma General Store" is fictional. The payment-support module is a **demonstration that is not connected to Paytm's payment infrastructure**, any bank or NPCI, and it never reverses, settles, refunds or escalates a real payment.

## Features

| Page | What it does |
|---|---|
| 📊 **Overview** | Today's sales, transactions, average bill, successful/pending/failed payments, comparison with the same day last week and the previous 7/30 days, a 30-day sales trend and the top insights. |
| 💡 **Business insights** | Rule-based insights labelled **Measured** (fact) or **Hypothesis** (possible reason to check), weekday and hourly patterns, plus an **AI business analyst** with one-click questions. |
| 💰 **Cash-flow forecast** | 7-day projected closing balance from sales history + merchant-entered payments (supplier, rent, others), with a low/high range from backtesting and shortfall alerts. |
| 🛟 **Payment support** | Five simulated scenarios (payment not visible, pending, failed, duplicate debit, refund status): look up a transaction and get safe, scenario-specific guidance and escalation steps. |
| 💬 **AI chat** | Ask anything about your business. Python computes every number; Claude only explains them. |

## Quick start

```bash
cd bharat-merchant-copilot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional: add ANTHROPIC_API_KEY to enable Claude
streamlit run app.py
```

Open http://localhost:8501. Sample data is generated automatically on first run (or click **Regenerate sample data** in the sidebar, which creates 90 days ending today).

Run the tests:

```bash
python -m pytest -q
```

### Troubleshooting: Streamlit install hangs on a slow connection

Streamlit depends on **pyarrow (~36 MB)**. On a slow or unstable connection, `pip install` can sit on `Downloading pyarrow…` for a long time, or hang indefinitely if the connection drops. pip restarts a broken download from zero; it never resumes. Fix it by downloading the big wheel with `curl`, which resumes, and then installing offline:

```bash
# 1. Install the small packages first
.venv/bin/pip install --only-binary :all: pandas numpy plotly python-dotenv anthropic pytest

# 2. Download every Streamlit wheel into a local folder (re-run if interrupted)
mkdir -p /tmp/wheels
.venv/bin/pip download --only-binary :all: -d /tmp/wheels streamlit
#    If that stalls on pyarrow, fetch pyarrow with a resumable download instead
#    (copy the wheel URL from https://pypi.org/project/pyarrow/#files for your Python/OS):
#    curl -L -C - --retry 10 -o /tmp/wheels/<pyarrow-wheel-name>.whl <wheel-url>

# 3. Install offline from the local folder (takes seconds, cannot hang on network)
.venv/bin/pip install --no-index --find-links /tmp/wheels streamlit

# 4. Confirm
.venv/bin/pip install --only-binary :all: -r requirements.txt   # should say "already satisfied"
.venv/bin/pip check
```

**Before a live demo:** install everything on good Wi-Fi beforehand, and set `DEMO_MODE=true` in `.env` if the venue network is unreliable. Every page works without internet in demo mode.

Tested with Python 3.11.5, streamlit 1.65.0, pyarrow 25.0.1, pandas 3.0.6, numpy 2.4.6, plotly 7.1.0, anthropic 1.13.0.

Regenerate data from the command line:

```bash
python -m utils.data_generator --end 2026-10-10 --seed 42
```

## Deploy (public link)

**Streamlit Community Cloud (free, always on):**
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the repo, branch `main` and main file `app.py`.
3. Under **Advanced settings** choose Python **3.11**. To enable Claude, add `ANTHROPIC_API_KEY = "..."` in **Secrets**; without it the app runs in demo mode.
4. Click **Deploy**. You get a `https://<name>.streamlit.app` link.

**Quick temporary link from your laptop** (works only while the laptop is on and the app is running):
```bash
.venv/bin/streamlit run app.py          # terminal 1
ssh -R 80:localhost:8501 nokey@localhost.run   # terminal 2 — prints a public https URL
```

> A public link with an API key lets anyone use your Anthropic credits. Set a spend limit in the Anthropic Console, or share the demo-mode version.

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No | — | Enables Claude. Without it the app runs in **demo mode**. |
| `CLAUDE_MODEL` | No | `claude-opus-5-5` | Model used for explanations. |
| `CLAUDE_EFFORT` | No | `low` | `low` / `medium` / `high` — higher is more thorough but slower. |
| `DEMO_MODE` | No | `false` | `true` forces demo mode even when a key is set (useful if the venue Wi-Fi is unreliable). |

Keys live only in `.env`, which is git-ignored. Never commit it.

## Project structure

```
bharat-merchant-copilot/
  app.py                    Streamlit UI (5 pages, sidebar navigation)
  requirements.txt
  .env.example
  .streamlit/config.toml    Light theme
  data/                     Generated CSV/JSON (synthetic)
  services/
    analytics.py            Daily summaries, comparisons, weekday/hour profiles, anomalies, insights, AI context
    cashflow.py             Seasonal-naive forecast, backtest, cash-flow projection, input validation
    payments.py             Transaction lookup, scenario guidance, sensitive-data detection
    ai_assistant.py         Claude integration + deterministic demo mode
  utils/
    data_generator.py       Reproducible synthetic merchant data
    formatting.py           ₹ Indian digit grouping (₹1,23,456), lakh/crore, Indian dates
    charts.py               Plotly charts
  tests/                    pytest suite
```

## How it works

### Numbers come from Python, words come from Claude
`services/analytics.py` computes every metric with pandas. `build_ai_context()` packs the verified figures (latest day, 7- and 30-day comparisons, weekday/hour averages, weekly and monthly totals, a per-day table, unusual days, failure spikes and insights) into JSON. Claude receives that JSON and a system prompt that tells it to:
- use only those numbers and never compute new financial totals;
- separate **"What the data shows"** from **"Possible reasons to check"**, and not claim causation;
- never ask for a PIN or OTP, and never claim a payment was reversed, settled or escalated.

Messages that look like they contain a PIN, OTP, password, CVV or full card number are blocked **before** anything is sent to the API.

The request uses `claude-opus-5-5` with streaming, `effort: low` for fast chat replies, prompt caching on the system and data blocks, and Anthropic's server-side **refusal fallback** (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). If the API is unreachable, rate-limited or the key is wrong, the app shows a notice and falls back to the demo-mode answer.

### Demo mode (no API key)
`demo_answer()` routes common questions (sales decline, best days, ticket size vs. count, payments, improvement ideas) to templates filled with the same verified metrics, so the whole app works offline.

### Sales-change breakdown
Week-over-week sales change is split exactly into:
- **volume effect** = (transactions now − before) × previous average bill
- **ticket effect** = (average bill now − before) × transactions now

These two add up to the total change, so the merchant can see whether fewer customers or smaller baskets drove it.

### Cash-flow forecast
1. **Expected sales** for each of the next 7 days = average sales on the same weekday over the last 4 weeks (falls back to the 7-day average with less history).
2. **Uncertainty**: a rolling-origin backtest over the last 6 weeks forecasts each week using only earlier data. The 10th/90th percentile daily errors set the low/high case. The app shows the average daily error, the weekly-total error and the bias. With under 6 weeks of history it uses a default ±20% band and says so.
3. **Cash in**: the cash share of sales arrives the same day; digital sales (share from the last 28 days) reach the bank after the settlement lag (default T+1). Day 1 includes the latest actual day's digital settlement.
4. **Cash out**: merchant-entered payments on their dates plus a daily running cost.
5. **Flags**: *Shortfall* if the expected balance goes below ₹0; *At risk* if the low case goes below ₹0 or the expected balance falls below the merchant's minimum buffer.

**Sales are not profit.** Restocking costs only appear if the merchant enters supplier payments.

### Synthetic data
`utils/data_generator.py` (seed 42) creates about 7,200 transactions over 90 days for one fictional store: weekend and salary-week peaks, morning/evening hours, a UPI/cash/card/wallet mix, ~4.5% failed digital payments, pending payments on the latest day, T+1 settlements, and some refunds. It also plants these anomalies:
- a festival spike (2 days, ~38 days before the end);
- a UPI outage day with a high failure rate (19 days before the end);
- a drop in average basket size over the last 7 days.

It also plants transactions for each payment-support scenario (IDs ending `9000xx`). Customer identities are masked (`pr****@oksbi`, `Card XXXX-4417`).

## Assumptions
- "Sales" = gross successful payments. Failed and pending payments are excluded; processed refunds are tracked separately.
- The latest day in the data is treated as "today", and today is compared with the same weekday one week earlier.
- Digital payments settle T+1 (configurable to 0–2 days). Cash is assumed to be in hand the same day.
- Payment timelines in the guidance (for example, failed UPI debits auto-reversed by T+1 working day under RBI's turnaround-time framework) are general information, not a guarantee.
- Credit (udhaar), loan EMIs, taxes and owner withdrawals are excluded unless entered as planned payments.

## Known limitations
- Synthetic data only; there is no import from a real POS, bank or payment-provider account yet.
- The forecast is a simple seasonal baseline. It does not know about upcoming festivals, holidays or weather.
- Payment lookups search the local CSV only. Guidance is rule-based and general.
- Chat history lives in the browser session and is lost on refresh.
- Demo-mode answers use keyword routing and only cover common question types.
- English UI only (Claude replies in Hindi or Hinglish if asked that way).
