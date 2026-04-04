"""
S&P 500 Market Regime Dashboard
================================
Sentiment · Momentum · Financial Conditions · Valuation
Composite signal with 2×2 matrix logic and drawdown regime classification.

Data sources: Yahoo Finance, FRED API, AAII Sentiment (historical CSV + manual update), computed indicators.
Architecture supports future expansion to Europe, Japan, EM.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime, timedelta
import requests
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CUSTOM STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    /* Global */
    .stApp {
        background-color: #0a0e17;
        color: #e0e6ed;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Header */
    .dashboard-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f4f8;
        letter-spacing: -0.02em;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1e2a3a;
        margin-bottom: 1rem;
    }
    .dashboard-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #6b7d93;
        margin-top: -0.5rem;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid #1e2a3a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #334155;
    }
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: #6b7d93;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    /* Signal badges */
    .signal-bullish {
        color: #34d399;
        background: rgba(52, 211, 153, 0.1);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
    .signal-bearish {
        color: #f87171;
        background: rgba(248, 113, 113, 0.1);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }
    .signal-neutral {
        color: #fbbf24;
        background: rgba(251, 191, 36, 0.1);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Composite signal box */
    .composite-box {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .composite-bullish {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.08) 100%);
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .composite-bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.08) 100%);
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    .composite-neutral {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.08) 100%);
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .composite-caution {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.15) 0%, rgba(251, 146, 60, 0.08) 100%);
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    .composite-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .composite-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Section headers */
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.6rem 0;
        border-bottom: 1px solid #1e2a3a;
        margin-bottom: 0.8rem;
        margin-top: 1rem;
    }

    /* Matrix table */
    .matrix-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .matrix-table th {
        background: #1e2a3a;
        color: #94a3b8;
        padding: 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
    }
    .matrix-table td {
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        font-weight: 600;
    }
    .matrix-strong-bull { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .matrix-lean-bull { background: rgba(16, 185, 129, 0.1); color: #6ee7b7; }
    .matrix-lean-bear { background: rgba(239, 68, 68, 0.1); color: #fca5a5; }
    .matrix-strong-bear { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Staleness alert */
    .stale-alert {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.15) 0%, rgba(251, 146, 60, 0.08) 100%);
        border: 1px solid rgba(251, 146, 60, 0.4);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        color: #fb923c;
        margin-bottom: 0.8rem;
    }
    .stale-alert-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #111827;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7d93;
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2a3a !important;
        color: #f0f4f8 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MARKET CONFIGURATION (extensible per region)
# ──────────────────────────────────────────────
MARKET_CONFIGS = {
    "S&P 500": {
        "index_ticker": "^GSPC",
        "vix_ticker": "^VIX",
        "dxy_ticker": "DX-Y.NYB",
        "breadth_tickers": {"above_200dma": None, "above_50dma": None},
        "put_call_ticker": None,
        "fred_hy_spread": "BAMLH0A0HYM2",
        "fred_yield_curve": {"2y": "DGS2", "10y": "DGS10"},
        "fred_nfci": "NFCI",
        "label": "S&P 500",
        "currency": "USD",
    },
}

# ──────────────────────────────────────────────
# DATA FETCHING (cached for performance)
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.warning(f"Could not fetch {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, api_key: str, period: str = "2y") -> pd.Series:
    """Fetch a time series from FRED."""
    try:
        fred = Fred(api_key=api_key)
        start = datetime.now() - timedelta(days=730)
        data = fred.get_series(series_id, observation_start=start)
        return data.dropna()
    except Exception as e:
        st.warning(f"Could not fetch FRED series {series_id}: {e}")
        return pd.Series(dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def load_aaii_history() -> pd.DataFrame:
    """Load AAII sentiment history from bundled CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), "aaii_sentiment_history.csv")
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"Could not load AAII history: {e}")
        return pd.DataFrame(columns=["date", "bullish", "neutral", "bearish"])


# ──────────────────────────────────────────────
# AAII SIDEBAR INPUT & STALENESS CHECK
# ──────────────────────────────────────────────

def render_aaii_sidebar(aaii_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar widget for manual AAII data entry.
    Returns the (possibly updated) AAII dataframe.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AAII Sentiment Update")

    if not aaii_df.empty:
        latest_date = aaii_df["date"].max()
        days_old = (datetime.now() - pd.Timestamp(latest_date)).days
        st.sidebar.caption(f"Latest data: **{latest_date.strftime('%Y-%m-%d')}** ({days_old}d ago)")
    else:
        days_old = 999

    st.sidebar.markdown(
        "<span style='font-size:0.78rem; color:#94a3b8;'>"
        "Enter this week's AAII survey results.<br>"
        "Published each Thursday at "
        "<a href='https://www.aaii.com/sentimentsurvey' target='_blank' style='color:#60a5fa;'>aaii.com</a>."
        "</span>",
        unsafe_allow_html=True,
    )

    with st.sidebar.form("aaii_form", clear_on_submit=True):
        new_date = st.date_input(
            "Survey date",
            value=datetime.now().date(),
            help="Use the Thursday date the survey was published",
        )
        col_b, col_n, col_be = st.columns(3)
        with col_b:
            new_bull = st.number_input("Bull %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
        with col_n:
            new_neut = st.number_input("Neut %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
        with col_be:
            new_bear = st.number_input("Bear %", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")

        submitted = st.form_submit_button("Add Entry", use_container_width=True)

        if submitted:
            total = new_bull + new_neut + new_bear
            if total < 95 or total > 105:
                st.sidebar.error(f"Values should sum to ~100%. Currently: {total:.1f}%")
            elif new_bull == 0 and new_neut == 0 and new_bear == 0:
                st.sidebar.error("Please enter the survey values.")
            else:
                # Normalize to decimals
                new_row = pd.DataFrame([{
                    "date": pd.Timestamp(new_date),
                    "bullish": round(new_bull / 100, 6),
                    "neutral": round(new_neut / 100, 6),
                    "bearish": round(new_bear / 100, 6),
                }])

                # Check for duplicate date
                if not aaii_df.empty and pd.Timestamp(new_date) in aaii_df["date"].values:
                    aaii_df.loc[aaii_df["date"] == pd.Timestamp(new_date), ["bullish", "neutral", "bearish"]] = [
                        round(new_bull / 100, 6),
                        round(new_neut / 100, 6),
                        round(new_bear / 100, 6),
                    ]
                    st.sidebar.success(f"Updated entry for {new_date}")
                else:
                    aaii_df = pd.concat([aaii_df, new_row], ignore_index=True)
                    aaii_df = aaii_df.sort_values("date").reset_index(drop=True)
                    st.sidebar.success(f"Added entry for {new_date}")

                # Save back to CSV
                csv_path = os.path.join(os.path.dirname(__file__), "aaii_sentiment_history.csv")
                try:
                    aaii_df_save = aaii_df.copy()
                    aaii_df_save["date"] = aaii_df_save["date"].dt.strftime("%Y-%m-%d")
                    aaii_df_save.to_csv(csv_path, index=False)
                except Exception:
                    st.sidebar.warning("Could not save to CSV. Entry is active for this session only.")

                # Clear the cache so the new data is picked up
                load_aaii_history.clear()

    return aaii_df


def check_aaii_staleness(aaii_df: pd.DataFrame) -> dict:
    """
    Check if AAII data is stale and return status info.
    AAII publishes every Thursday, so data > 9 days old is stale.
    """
    if aaii_df.empty:
        return {"stale": True, "days": 999, "level": "critical", "message": "No AAII sentiment data loaded."}

    latest_date = aaii_df["date"].max()
    days_old = (datetime.now() - pd.Timestamp(latest_date)).days

    if days_old <= 9:
        return {"stale": False, "days": days_old, "level": "ok", "message": ""}
    elif days_old <= 16:
        return {
            "stale": True, "days": days_old, "level": "warning",
            "message": f"⚠️ AAII sentiment data is {days_old} days old. Update via the sidebar with this week's survey (published Thursday at aaii.com).",
        }
    else:
        return {
            "stale": True, "days": days_old, "level": "critical",
            "message": f"🚨 AAII sentiment data is {days_old} days old — significantly stale. Open the sidebar (☰) and enter the latest survey data from aaii.com.",
        }


# ──────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ──────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_price_vs_sma(prices: pd.Series, window: int = 200) -> dict:
    sma = prices.rolling(window).mean()
    latest_price = prices.iloc[-1]
    latest_sma = sma.iloc[-1]
    pct = ((latest_price - latest_sma) / latest_sma) * 100
    return {
        "price": latest_price,
        "sma": latest_sma,
        "pct_above": pct,
        "above": latest_price > latest_sma,
        "sma_series": sma,
    }


def compute_drawdown(prices: pd.Series) -> dict:
    cum_max = prices.cummax()
    drawdown_pct = ((prices - cum_max) / cum_max) * 100
    current_dd = drawdown_pct.iloc[-1]

    if current_dd > -5:
        regime = "Normal"
        color = "#34d399"
    elif current_dd > -10:
        regime = "Pullback"
        color = "#fbbf24"
    elif current_dd > -20:
        regime = "Correction"
        color = "#fb923c"
    else:
        regime = "Bear Market"
        color = "#f87171"

    return {
        "current_dd": current_dd,
        "regime": regime,
        "color": color,
        "series": drawdown_pct,
    }


def classify_signal(value: float, thresholds: dict) -> tuple:
    if value <= thresholds.get("bullish", float("-inf")):
        return "BULLISH", "signal-bullish"
    elif value >= thresholds.get("bearish", float("inf")):
        return "BEARISH", "signal-bearish"
    else:
        return "NEUTRAL", "signal-neutral"


def compute_aaii_sentiment_score(aaii_df: pd.DataFrame) -> dict:
    """
    Compute contrarian sentiment score from AAII data.
    Returns the latest reading and a normalized score from -1 (extreme greed/complacency)
    to +1 (extreme fear/contrarian bullish).

    Historical averages: Bullish 37.5%, Neutral 31.5%, Bearish 31.0%
    """
    if aaii_df.empty:
        return {"available": False, "bullish": 0, "neutral": 0, "bearish": 0, "net": 0, "score": 0, "label": "N/A"}

    latest = aaii_df.iloc[-1]
    bull = latest["bullish"]
    neut = latest["neutral"]
    bear = latest["bearish"]
    net = bull - bear  # positive = net bullish, negative = net bearish

    # Historical averages
    hist_bull_avg = 0.375
    hist_bear_avg = 0.310
    hist_net_avg = hist_bull_avg - hist_bear_avg  # ~0.065

    # Contrarian: extreme bearishness → bullish score, extreme bullishness → bearish score
    # Std dev of net bull-bear spread is roughly 0.16
    deviation = (net - hist_net_avg) / 0.16
    score = max(-1, min(1, -deviation * 0.5))  # invert because contrarian

    if score > 0.3:
        label = "BULLISH"
    elif score < -0.3:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {
        "available": True,
        "bullish": bull,
        "neutral": neut,
        "bearish": bear,
        "net": net,
        "score": score,
        "label": label,
        "date": latest["date"],
    }


def compute_composite_signal(rsi: float, vix: float, sma_above: bool, sentiment_score: float) -> dict:
    """
    2×2 matrix composite signal.
    Axes: Momentum (RSI + SMA) vs Sentiment/Risk (VIX + AAII sentiment).
    """
    # Momentum score: 0 (bearish) to 1 (bullish)
    momentum = 0.5
    if rsi < 30:
        momentum += 0.3
    elif rsi > 70:
        momentum -= 0.3
    elif rsi < 45:
        momentum += 0.15
    elif rsi > 55:
        momentum -= 0.15

    if sma_above:
        momentum += 0.2
    else:
        momentum -= 0.2
    momentum = max(0, min(1, momentum))

    # Sentiment/risk score: 0 (bearish) to 1 (bullish — contrarian)
    risk = 0.5
    if vix > 30:
        risk += 0.25
    elif vix > 25:
        risk += 0.1
    elif vix < 14:
        risk -= 0.2
    elif vix < 18:
        risk -= 0.1

    risk += sentiment_score * 0.25  # AAII contrarian score
    risk = max(0, min(1, risk))

    mom_high = momentum > 0.5
    risk_high = risk > 0.5

    if mom_high and risk_high:
        signal = "STRONG BULLISH"
        css_class = "composite-bullish"
        color = "#34d399"
        desc = "Positive momentum with supportive sentiment — favorable conditions."
    elif mom_high and not risk_high:
        signal = "LEAN BULLISH"
        css_class = "composite-caution"
        color = "#fbbf24"
        desc = "Positive momentum but complacent sentiment — proceed with caution."
    elif not mom_high and risk_high:
        signal = "LEAN BEARISH"
        css_class = "composite-caution"
        color = "#fb923c"
        desc = "Weak momentum but fear elevated — potential contrarian opportunity forming."
    else:
        signal = "STRONG BEARISH"
        css_class = "composite-bearish"
        color = "#f87171"
        desc = "Weak momentum with complacent sentiment — risk-off conditions."

    return {
        "signal": signal,
        "css_class": css_class,
        "color": color,
        "desc": desc,
        "momentum_score": momentum,
        "risk_score": risk,
    }


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,23,1)",
    font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    yaxis=dict(gridcolor="#1e2a3a", zerolinecolor="#1e2a3a"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    height=340,
)


def render_metric_card(label: str, value: str, delta: str = "", signal: str = "", signal_class: str = ""):
    delta_html = ""
    if delta:
        delta_color = "#34d399" if delta.startswith("+") or delta.startswith("▲") else "#f87171"
        delta_html = f'<div class="metric-delta" style="color:{delta_color}">{delta}</div>'

    signal_html = ""
    if signal:
        signal_html = f'<span class="{signal_class}">{signal}</span>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label} {signal_html}</div>
        <div class="metric-value" style="color:#f0f4f8">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="dashboard-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">S&P 500 — Sentiment · Momentum · Financial Conditions · Valuation &nbsp;|&nbsp; Updated at each page load</div>', unsafe_allow_html=True)

    # ── FRED API Key ──
    fred_key = st.secrets.get("FRED_API_KEY", "")
    if not fred_key:
        fred_key = st.sidebar.text_input("FRED API Key", type="password", help="Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")

    if not fred_key:
        st.info("Enter your FRED API key in the sidebar (or add to Streamlit secrets) to load financial conditions data.")

    # ── Load AAII Data ──
    aaii_df = load_aaii_history()

    # ── AAII Sidebar Manual Input ──
    aaii_df = render_aaii_sidebar(aaii_df)

    # ── AAII Staleness Check ──
    aaii_status = check_aaii_staleness(aaii_df)
    if aaii_status["stale"]:
        alert_class = "stale-alert-critical" if aaii_status["level"] == "critical" else ""
        st.markdown(f'<div class="stale-alert {alert_class}">{aaii_status["message"]}</div>', unsafe_allow_html=True)

    market = MARKET_CONFIGS["S&P 500"]

    # ── Fetch Data ──
    with st.spinner("Fetching market data..."):
        spx_data = fetch_price_data(market["index_ticker"])
        vix_data = fetch_price_data(market["vix_ticker"], period="1y")
        dxy_data = fetch_price_data(market["dxy_ticker"], period="1y")

        hy_spread = pd.Series(dtype=float)
        dgs2 = pd.Series(dtype=float)
        dgs10 = pd.Series(dtype=float)
        nfci = pd.Series(dtype=float)

        if fred_key:
            hy_spread = fetch_fred_series(market["fred_hy_spread"], fred_key)
            dgs2 = fetch_fred_series(market["fred_yield_curve"]["2y"], fred_key)
            dgs10 = fetch_fred_series(market["fred_yield_curve"]["10y"], fred_key)
            nfci = fetch_fred_series(market["fred_nfci"], fred_key)

    if spx_data.empty:
        st.error("Could not fetch S&P 500 data. Please try again later.")
        return

    # ── Compute Indicators ──
    close = spx_data["Close"]
    rsi = compute_rsi(close)
    sma_info = compute_price_vs_sma(close)
    dd_info = compute_drawdown(close)

    current_rsi = rsi.iloc[-1]
    current_vix = vix_data["Close"].iloc[-1] if not vix_data.empty else np.nan
    current_dxy = dxy_data["Close"].iloc[-1] if not dxy_data.empty else np.nan

    yield_curve_val = np.nan
    if not dgs10.empty and not dgs2.empty:
        yield_curve_val = dgs10.iloc[-1] - dgs2.iloc[-1]

    hy_val = hy_spread.iloc[-1] if not hy_spread.empty else np.nan
    nfci_val = nfci.iloc[-1] if not nfci.empty else np.nan

    # ── AAII Sentiment Score ──
    aaii_info = compute_aaii_sentiment_score(aaii_df)
    sentiment_score = aaii_info["score"]

    # Fall back to VIX-only proxy if no AAII data
    if not aaii_info["available"]:
        if not np.isnan(current_vix):
            if current_vix > 30:
                sentiment_score = 0.6
            elif current_vix > 25:
                sentiment_score = 0.3
            elif current_vix < 14:
                sentiment_score = -0.5
            elif current_vix < 18:
                sentiment_score = -0.2

    # Composite Signal
    composite = compute_composite_signal(
        rsi=current_rsi,
        vix=current_vix if not np.isnan(current_vix) else 20,
        sma_above=sma_info["above"],
        sentiment_score=sentiment_score,
    )

    # ── Signal classifications ──
    rsi_signal, rsi_class = classify_signal(current_rsi, {"bullish": 35, "bearish": 65})
    vix_signal, vix_class = classify_signal(current_vix, {"bullish": 12, "bearish": 25}) if not np.isnan(current_vix) else ("N/A", "signal-neutral")
    sma_signal = ("BULLISH", "signal-bullish") if sma_info["above"] else ("BEARISH", "signal-bearish")

    # ──────────────────────────────────────────
    # LAYOUT: COMPOSITE SIGNAL (top banner)
    # ──────────────────────────────────────────
    st.markdown(f"""
    <div class="composite-box {composite['css_class']}">
        <div class="composite-label" style="color:{composite['color']}">{composite['signal']}</div>
        <div class="composite-desc">{composite['desc']}</div>
        <div style="margin-top:0.5rem; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7d93;">
            Momentum: {composite['momentum_score']:.2f} &nbsp;|&nbsp; Risk/Sentiment: {composite['risk_score']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # TOP ROW: Key metrics
    # ──────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        daily_chg = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        delta_str = f"{'▲' if daily_chg >= 0 else '▼'} {abs(daily_chg):.2f}%"
        render_metric_card("S&P 500", f"{close.iloc[-1]:,.0f}", delta_str)

    with col2:
        render_metric_card("RSI (14)", f"{current_rsi:.1f}", signal=rsi_signal, signal_class=rsi_class)

    with col3:
        if not np.isnan(current_vix):
            render_metric_card("VIX", f"{current_vix:.1f}", signal=vix_signal, signal_class=vix_class)
        else:
            render_metric_card("VIX", "N/A")

    with col4:
        render_metric_card("vs 200 DMA", f"{sma_info['pct_above']:.1f}%", signal=sma_signal[0], signal_class=sma_signal[1])

    with col5:
        render_metric_card("Drawdown", f"{dd_info['current_dd']:.1f}%",
                          signal=dd_info['regime'],
                          signal_class="signal-bullish" if dd_info['regime'] == "Normal" else "signal-bearish")

    with col6:
        if aaii_info["available"]:
            aaii_net_pct = aaii_info["net"] * 100
            aaii_signal_class = "signal-bullish" if aaii_info["label"] == "BULLISH" else ("signal-bearish" if aaii_info["label"] == "BEARISH" else "signal-neutral")
            render_metric_card("AAII Net", f"{aaii_net_pct:+.1f}%", signal=aaii_info["label"], signal_class=aaii_signal_class)
        else:
            render_metric_card("AAII Net", "N/A")

    # ──────────────────────────────────────────
    # TABS
    # ──────────────────────────────────────────
    tab_momentum, tab_sentiment, tab_conditions, tab_valuation, tab_matrix = st.tabs([
        "📈 Momentum", "🧠 Sentiment", "💰 Financial Conditions", "📐 Valuation", "🔲 2×2 Matrix"
    ])

    # ────── MOMENTUM TAB ──────
    with tab_momentum:
        st.markdown('<div class="section-header">Momentum Indicators</div>', unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)

        with mc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=close.index, y=close.values, name="S&P 500",
                line=dict(color="#60a5fa", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=close.index, y=sma_info["sma_series"].values, name="200 DMA",
                line=dict(color="#fbbf24", width=1.5, dash="dot"),
            ))
            fig.update_layout(title="S&P 500 vs 200-Day Moving Average", **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with mc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rsi.index, y=rsi.values, name="RSI(14)",
                line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
            ))
            fig.add_hline(y=70, line_color="#f87171", line_dash="dash", line_width=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_color="#34d399", line_dash="dash", line_width=1, annotation_text="Oversold")
            fig.add_hline(y=50, line_color="#334155", line_dash="dot", line_width=1)
            fig.update_layout(title="RSI (14)", yaxis_range=[0, 100], **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # Drawdown chart
        fig = go.Figure()
        dd_series = dd_info["series"]
        fig.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values, name="Drawdown",
            line=dict(color="#f87171", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,113,113,0.1)",
        ))
        fig.add_hline(y=-5, line_color="#fbbf24", line_dash="dot", line_width=1, annotation_text="Pullback")
        fig.add_hline(y=-10, line_color="#fb923c", line_dash="dot", line_width=1, annotation_text="Correction")
        fig.add_hline(y=-20, line_color="#f87171", line_dash="dot", line_width=1, annotation_text="Bear Market")
        fig.update_layout(
            title=f"Drawdown from All-Time High — Current Regime: {dd_info['regime']}",
            yaxis_title="Drawdown (%)",
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ────── SENTIMENT TAB ──────
    with tab_sentiment:
        st.markdown('<div class="section-header">Sentiment & Positioning</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)

        with sc1:
            # VIX chart
            if not vix_data.empty:
                vix_close = vix_data["Close"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vix_close.index, y=vix_close.values, name="VIX",
                    line=dict(color="#f59e0b", width=2),
                    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                ))
                fig.add_hline(y=20, line_color="#334155", line_dash="dot", line_width=1, annotation_text="Long-term avg")
                fig.add_hline(y=30, line_color="#f87171", line_dash="dash", line_width=1, annotation_text="Elevated fear")
                fig.update_layout(title="CBOE Volatility Index (VIX)", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with sc2:
            # AAII Bull-Bear Spread chart
            if not aaii_df.empty:
                cutoff = datetime.now() - timedelta(days=730)
                aaii_recent = aaii_df[aaii_df["date"] >= cutoff].copy()

                if not aaii_recent.empty:
                    fig = go.Figure()
                    spread = (aaii_recent["bullish"] - aaii_recent["bearish"]) * 100
                    colors = ["rgba(52,211,153,0.3)" if s > 0 else "rgba(248,113,113,0.3)" for s in spread]

                    fig.add_trace(go.Bar(
                        x=aaii_recent["date"], y=spread, name="Bull-Bear Spread",
                        marker_color=colors,
                    ))

                    spread_ma = spread.rolling(8, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=aaii_recent["date"], y=spread_ma, name="8-wk MA",
                        line=dict(color="#fbbf24", width=2),
                    ))

                    fig.add_hline(y=0, line_color="#f0f4f8", line_width=1)
                    fig.add_hline(y=6.5, line_color="#334155", line_dash="dot", line_width=1, annotation_text="Hist. avg spread")

                    fig.update_layout(
                        title="AAII Bull-Bear Spread (%, contrarian indicator)",
                        yaxis_title="Net Bulls − Bears (%)",
                        **CHART_LAYOUT,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="metric-card" style="min-height:300px;">
                    <div class="metric-label">AAII Net Sentiment</div>
                    <div style="color:#6b7d93; font-family:'DM Sans',sans-serif; font-size:0.85rem; padding-top:1rem;">
                        No AAII data loaded. Upload aaii_sentiment_history.csv to the app directory.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # AAII current reading detail cards
        if aaii_info["available"]:
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                render_metric_card("Bullish", f"{aaii_info['bullish']*100:.1f}%",
                                  delta=f"{'▲' if aaii_info['bullish'] > 0.375 else '▼'} vs 37.5% avg")
            with ac2:
                render_metric_card("Neutral", f"{aaii_info['neutral']*100:.1f}%",
                                  delta=f"{'▲' if aaii_info['neutral'] > 0.315 else '▼'} vs 31.5% avg")
            with ac3:
                render_metric_card("Bearish", f"{aaii_info['bearish']*100:.1f}%",
                                  delta=f"{'▲' if aaii_info['bearish'] > 0.310 else '▼'} vs 31.0% avg")
            with ac4:
                render_metric_card("Contrarian Score", f"{aaii_info['score']:+.2f}",
                                  signal=aaii_info["label"],
                                  signal_class="signal-bullish" if aaii_info["label"] == "BULLISH" else ("signal-bearish" if aaii_info["label"] == "BEARISH" else "signal-neutral"))

            st.markdown(f"""
            <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93; margin-top:0.3rem;">
                Latest survey: {aaii_info['date'].strftime('%Y-%m-%d') if hasattr(aaii_info['date'], 'strftime') else aaii_info['date']}
                &nbsp;|&nbsp; Contrarian interpretation: extreme bearishness = bullish signal, extreme bullishness = bearish signal.
                &nbsp;|&nbsp; Update weekly via sidebar (☰).
            </div>
            """, unsafe_allow_html=True)

        # AAII historical stacked area chart
        if not aaii_df.empty:
            st.markdown('<div class="section-header" style="margin-top:1rem">AAII Sentiment History</div>', unsafe_allow_html=True)

            range_options = {"1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650, "All": 99999}
            selected_range = st.selectbox("Time range", list(range_options.keys()), index=1, label_visibility="collapsed")
            days_back = range_options[selected_range]
            cutoff = datetime.now() - timedelta(days=days_back)
            aaii_plot = aaii_df[aaii_df["date"] >= cutoff].copy()

            if not aaii_plot.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=aaii_plot["date"], y=aaii_plot["bullish"] * 100,
                    name="Bullish", stackgroup="one",
                    line=dict(color="#34d399", width=0),
                    fillcolor="rgba(52,211,153,0.4)",
                ))
                fig.add_trace(go.Scatter(
                    x=aaii_plot["date"], y=aaii_plot["neutral"] * 100,
                    name="Neutral", stackgroup="one",
                    line=dict(color="#fbbf24", width=0),
                    fillcolor="rgba(251,191,36,0.3)",
                ))
                fig.add_trace(go.Scatter(
                    x=aaii_plot["date"], y=aaii_plot["bearish"] * 100,
                    name="Bearish", stackgroup="one",
                    line=dict(color="#f87171", width=0),
                    fillcolor="rgba(248,113,113,0.4)",
                ))
                fig.update_layout(
                    title="AAII Investor Sentiment (stacked %)",
                    yaxis_title="Percentage",
                    yaxis_range=[0, 100],
                    height=380,
                    **{k: v for k, v in CHART_LAYOUT.items() if k != "height"},
                )
                st.plotly_chart(fig, use_container_width=True)

        # Put/Call ratio placeholder
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">CBOE Equity Put/Call Ratio</div>
            <div style="color:#6b7d93; font-family:'DM Sans',sans-serif; font-size:0.85rem; padding:0.5rem 0;">
                Put/Call ratio data requires CBOE data feed. This indicator will be added in a future update.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ────── FINANCIAL CONDITIONS TAB ──────
    with tab_conditions:
        st.markdown('<div class="section-header">Financial Conditions</div>', unsafe_allow_html=True)

        if not fred_key:
            st.warning("Add your FRED API key to view financial conditions data.")
        else:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                if not np.isnan(hy_val):
                    if hy_val > 5:
                        hy_signal, hy_class = "BEARISH", "signal-bearish"
                    elif hy_val < 3.5:
                        hy_signal, hy_class = "BULLISH", "signal-bullish"
                    else:
                        hy_signal, hy_class = "NEUTRAL", "signal-neutral"
                    render_metric_card("HY Credit Spread (OAS)", f"{hy_val:.0f} bps", signal=hy_signal, signal_class=hy_class)
                else:
                    render_metric_card("HY Credit Spread", "N/A")

            with fc2:
                if not np.isnan(yield_curve_val):
                    yc_signal = ("BULLISH", "signal-bullish") if yield_curve_val > 0 else ("BEARISH", "signal-bearish")
                    render_metric_card("2y/10y Yield Curve", f"{yield_curve_val:.2f}%", signal=yc_signal[0], signal_class=yc_signal[1])
                else:
                    render_metric_card("2y/10y Yield Curve", "N/A")

            with fc3:
                if not np.isnan(nfci_val):
                    nfci_signal = ("BULLISH", "signal-bullish") if nfci_val < 0 else ("BEARISH", "signal-bearish")
                    render_metric_card("Chicago Fed NFCI", f"{nfci_val:.2f}", signal=nfci_signal[0], signal_class=nfci_signal[1])
                else:
                    render_metric_card("NFCI", "N/A")

            cc1, cc2 = st.columns(2)

            with cc1:
                if not hy_spread.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hy_spread.index, y=hy_spread.values, name="HY OAS",
                        line=dict(color="#fb923c", width=2),
                        fill="tozeroy", fillcolor="rgba(251,146,60,0.08)",
                    ))
                    fig.update_layout(title="High Yield Credit Spread (OAS, bps)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with cc2:
                if not dgs2.empty and not dgs10.empty:
                    curve = dgs10 - dgs2
                    curve = curve.dropna()
                    fig = go.Figure()
                    colors = ["#34d399" if v > 0 else "#f87171" for v in curve.values]
                    fig.add_trace(go.Bar(
                        x=curve.index, y=curve.values, name="2y/10y Spread",
                        marker_color=colors, opacity=0.7,
                    ))
                    fig.add_hline(y=0, line_color="#f0f4f8", line_width=1)
                    fig.update_layout(title="2y/10y Treasury Yield Curve Spread (%)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            if not nfci.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=nfci.index, y=nfci.values, name="NFCI",
                    line=dict(color="#a78bfa", width=2),
                    fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
                ))
                fig.add_hline(y=0, line_color="#f0f4f8", line_width=1, annotation_text="Avg conditions")
                fig.update_layout(title="Chicago Fed National Financial Conditions Index", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    # ────── VALUATION TAB ──────
    with tab_valuation:
        st.markdown('<div class="section-header">Valuation Indicators</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Forward P/E & Shiller CAPE</div>
            <div style="color:#6b7d93; font-family:'DM Sans',sans-serif; font-size:0.85rem; padding:0.5rem 0;">
                <strong>Forward P/E Ratio</strong> — Requires earnings estimates from FactSet, Refinitiv, or S&P.
                Free proxies: scrape from Yardeni Research or WSJ Markets page.<br><br>
                <strong>Shiller CAPE</strong> — Monthly data available from
                <a href="http://www.econ.yale.edu/~shiller/data.htm" target="_blank" style="color:#60a5fa">Robert Shiller's website</a>
                or multpl.com. Consider adding as a manual entry or scheduled scrape.<br><br>
                These indicators are <em>slow-moving</em> and best used as long-term context
                rather than timing signals. They will be integrated in a future update.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not dxy_data.empty:
            dxy_close = dxy_data["Close"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dxy_close.index, y=dxy_close.values, name="DXY",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
            ))
            fig.update_layout(title="US Dollar Index (DXY)", **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ────── 2×2 MATRIX TAB ──────
    with tab_matrix:
        st.markdown('<div class="section-header">Composite Signal — 2×2 Regime Matrix</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <table class="matrix-table">
            <tr>
                <th></th>
                <th>Sentiment/Risk: RISK-ON<br><span style="font-size:0.65rem;font-weight:400">(Low fear, complacency)</span></th>
                <th>Sentiment/Risk: RISK-OFF<br><span style="font-size:0.65rem;font-weight:400">(High fear, contrarian bullish)</span></th>
            </tr>
            <tr>
                <th>Momentum: BULLISH<br><span style="font-size:0.65rem;font-weight:400">(RSI healthy, above 200DMA)</span></th>
                <td class="matrix-lean-bull">LEAN BULLISH<br><span style="font-size:0.65rem">Trend intact, watch for complacency</span></td>
                <td class="matrix-strong-bull {'outline:3px solid #34d399;' if composite['signal']=='STRONG BULLISH' else ''}">
                    STRONG BULLISH<br><span style="font-size:0.65rem">Trend + fear = buy signal</span>
                </td>
            </tr>
            <tr>
                <th>Momentum: BEARISH<br><span style="font-size:0.65rem;font-weight:400">(RSI weak, below 200DMA)</span></th>
                <td class="matrix-strong-bear {'outline:3px solid #f87171;' if composite['signal']=='STRONG BEARISH' else ''}">
                    STRONG BEARISH<br><span style="font-size:0.65rem">No trend, no fear = avoid</span>
                </td>
                <td class="matrix-lean-bear">LEAN BEARISH<br><span style="font-size:0.65rem">Weak trend, but fear rising — watch for turn</span></td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        sentiment_source = "AAII survey" if aaii_info["available"] else "VIX-only proxy"
        st.markdown(f"""
        <div style="margin-top:1rem; font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#94a3b8;">
            <strong>Current position:</strong> Momentum score = {composite['momentum_score']:.2f}, Risk/Sentiment score = {composite['risk_score']:.2f}<br>
            Sentiment source: {sentiment_source}. The highlighted cell shows the current regime classification.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">How the Signal is Computed</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div style="color:#94a3b8; font-family:'DM Sans',sans-serif; font-size:0.85rem; line-height:1.7;">
                <strong>Momentum Axis (0–1):</strong> RSI(14) position (oversold adds +0.3, overbought subtracts −0.3)
                combined with price vs 200 DMA (above adds +0.2, below subtracts −0.2).<br><br>
                <strong>Risk/Sentiment Axis (0–1):</strong> VIX level (>30 adds +0.25 as contrarian bullish, <14 subtracts −0.2
                as complacency warning) combined with AAII bull-bear spread (contrarian adjusted: extreme bearishness
                among retail investors historically signals market bottoms).<br><br>
                <strong>Drawdown Overlay:</strong> Normal (<5%), Pullback (5-10%), Correction (10-20%), Bear Market (>20%).
                The drawdown regime provides additional context independent of the 2×2 signal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────
    # FOOTER
    # ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#475569; text-align:center; padding:0.5rem 0;">
        Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
        Data: Yahoo Finance, FRED, AAII &nbsp;|&nbsp;
        AAII data: {'current' if not aaii_status['stale'] else f'{aaii_status["days"]}d old'} &nbsp;|&nbsp;
        This dashboard is for informational purposes only — not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
