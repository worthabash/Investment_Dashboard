"""
S&P 500 Market Regime Dashboard
================================
Sentiment · Momentum · Financial Conditions · Valuation
Composite signal with 2×2 matrix logic and drawdown regime classification.

Data sources:
  - Twelve Data  → S&P 500, VIX, DXY, individual stock prices (breadth)
  - FMP          → S&P 500 constituent list, forward P/E / valuation
  - FRED API     → HY spread, yield curve, NFCI
  - Bundled CSVs → AAII sentiment history, Shiller CAPE history
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

    /* Hide Streamlit chrome (keep header for sidebar toggle) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

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
# MARKET CONFIGURATION
# ──────────────────────────────────────────────
MARKET_CONFIGS = {
    "S&P 500": {
        # Twelve Data — used only for equity/ETF price data (indices not supported on Growth plan)
        "td_index_symbol": "SPX",
        # FRED series — free tier, no plan restrictions, used for VIX, DXY, and conditions
        "fred_vix":         "VIXCLS",       # CBOE VIX daily close (1990-present)
        "fred_dxy":         "DTWEXBGS",     # Fed Broad USD Index daily (2006-present)
        "fred_hy_spread":   "BAMLH0A0HYM2", # HY OAS spread
        "fred_yield_curve": {"2y": "DGS2", "10y": "DGS10"},
        "fred_nfci":        "NFCI",
        "label":    "S&P 500",
        "currency": "USD",
    },
}


# ──────────────────────────────────────────────
# TWELVE DATA — PRICE FETCHING
# ──────────────────────────────────────────────

def _td_outputsize(days: int) -> int:
    """Convert calendar days to approximate trading-day output_size."""
    # ~252 trading days per year; add 20% buffer
    return min(int(days * 252 / 365 * 1.2) + 50, 5000)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_td_price(symbol: str, api_key: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Twelve Data for a single symbol.
    Returns a DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
    days: approx calendar days of history to request.

    Symbol notes:
      - S&P 500 index: "SPX"
      - CBOE VIX:      "^VIX"   (caret prefix required)
      - USD Index:     "DX"     (ICE futures, no exchange param needed)
    """
    if not api_key:
        return pd.DataFrame()
    try:
        outputsize = _td_outputsize(days)
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     symbol,
            "interval":   "1day",
            "outputsize": outputsize,
            "order":      "ASC",
            "apikey":     api_key,
            # Note: do NOT pass "type" param — it is invalid for indices and futures
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "error" or "values" not in data:
            st.warning(f"Twelve Data [{symbol}]: {data.get('message', 'unknown error')}")
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.columns = [c.capitalize() for c in df.columns]
        return df.dropna(subset=["Close"])

    except Exception as e:
        st.warning(f"Could not fetch {symbol} from Twelve Data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="Loading S&P 500 breadth data (Twelve Data)...")
def fetch_td_batch_closes(symbols: tuple, api_key: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch daily close prices for a batch of symbols from Twelve Data.
    Uses the /time_series endpoint sequentially (Growth plan = 800 API credits/min).
    Twelve Data batch endpoint supports up to 120 symbols per call — we chunk to stay safe.

    Returns a DataFrame: index=dates, columns=symbols.
    """
    if not api_key or not symbols:
        return pd.DataFrame()

    outputsize = _td_outputsize(days)
    chunk_size = 120  # max symbols per batch call
    all_closes: dict[str, pd.Series] = {}

    for i in range(0, len(symbols), chunk_size):
        chunk = list(symbols[i : i + chunk_size])
        symbol_str = ",".join(chunk)
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol":     symbol_str,
                "interval":   "1day",
                "outputsize": outputsize,
                "order":      "ASC",
                "apikey":     api_key,
            }
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # When multiple symbols, response is a dict keyed by symbol.
            # When a single symbol, response is the series object directly.
            if len(chunk) == 1:
                sym = chunk[0]
                series_data = data
                if series_data.get("status") == "error" or "values" not in series_data:
                    continue
                df_s = pd.DataFrame(series_data["values"])
                df_s["datetime"] = pd.to_datetime(df_s["datetime"])
                df_s = df_s.set_index("datetime").sort_index()
                df_s["close"] = pd.to_numeric(df_s["close"], errors="coerce")
                all_closes[sym] = df_s["close"].rename(sym)
            else:
                for sym in chunk:
                    series_data = data.get(sym, {})
                    if series_data.get("status") == "error" or "values" not in series_data:
                        continue
                    df_s = pd.DataFrame(series_data["values"])
                    df_s["datetime"] = pd.to_datetime(df_s["datetime"])
                    df_s = df_s.set_index("datetime").sort_index()
                    df_s["close"] = pd.to_numeric(df_s["close"], errors="coerce")
                    all_closes[sym] = df_s["close"].rename(sym)

        except Exception:
            continue

    if not all_closes:
        return pd.DataFrame()

    result = pd.concat(all_closes.values(), axis=1)
    result = result.dropna(axis=1, how="all")
    return result


# ──────────────────────────────────────────────
# FMP — CONSTITUENTS & VALUATION
# ──────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_constituents_fmp(fmp_key: str = "") -> list:
    """
    Get S&P 500 constituent tickers.
    Primary: FMP /sp500_constituent endpoint.
    Fallback 1: bundled sp500_tickers.csv.
    Fallback 2: Wikipedia.
    Returns list of Twelve-Data-compatible symbols (dots → dashes).
    """
    # Primary: FMP stable API (new endpoint as of Sept 2025 — /api/v3/ is legacy-only)
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/stable/sp500-constituent?apikey={fmp_key}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 400:
                symbols = [d["symbol"].replace(".", "-") for d in data if "symbol" in d]
                return symbols
        except Exception:
            pass

    # Fallback 1: bundled CSV
    csv_path = os.path.join(os.path.dirname(__file__), "sp500_tickers.csv")
    try:
        df = pd.read_csv(csv_path)
        symbols = df["symbol"].str.replace(".", "-", regex=False).tolist()
        if len(symbols) > 400:
            return symbols
    except Exception:
        pass

    # Fallback 2: Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if tables:
            df = tables[0]
            symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            symbols = [s for s in symbols if isinstance(s, str) and 0 < len(s) < 10]
            if len(symbols) > 400:
                return symbols
    except Exception:
        pass

    return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fmp_valuation(fmp_key: str) -> dict:
    """
    FMP Starter plan ($19/mo) returns 403 on all SPY/ETF financial statement
    and ratio endpoints (/v3/ratios-ttm, /v3/quote, /v3/key-metrics).
    These require Premium or higher.

    Trailing/Forward P/E is therefore sidebar-entry only.
    FMP is still used for the S&P 500 constituent list (fetch_sp500_constituents_fmp),
    which works fine on Starter.
    """
    return {"available": False}


# ──────────────────────────────────────────────
# FRED — FINANCIAL CONDITIONS
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, api_key: str) -> pd.Series:
    """Fetch a time series from FRED (5-year window)."""
    try:
        fred = Fred(api_key=api_key)
        start = datetime.now() - timedelta(days=1825)
        data = fred.get_series(series_id, observation_start=start)
        return data.dropna()
    except Exception as e:
        st.warning(f"Could not fetch FRED series {series_id}: {e}")
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────
# BUNDLED CSVs
# ──────────────────────────────────────────────

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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_shiller_cape() -> pd.DataFrame:
    """Load Shiller CAPE data from bundled CSV file."""
    csv_path = os.path.join(os.path.dirname(__file__), "shiller_cape_history.csv")
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "cape"])


# ──────────────────────────────────────────────
# BREADTH COMPUTATION (unchanged logic)
# ──────────────────────────────────────────────

def compute_breadth_from_prices(closes: pd.DataFrame) -> dict:
    """
    Compute breadth indicators from a DataFrame of S&P 500 close prices.
    Columns = tickers, Index = dates.
    """
    if closes.empty or closes.shape[1] < 5:
        return {}

    n_stocks = closes.shape[1]

    sma200 = closes.rolling(200, min_periods=200).mean()
    above_200 = (closes > sma200).sum(axis=1) / closes.notna().sum(axis=1) * 100

    sma50 = closes.rolling(50, min_periods=50).mean()
    above_50 = (closes > sma50).sum(axis=1) / closes.notna().sum(axis=1) * 100

    daily_ret = closes.pct_change()
    advancing = (daily_ret > 0).sum(axis=1)
    declining = (daily_ret < 0).sum(axis=1)
    total = (advancing + declining).replace(0, np.nan)

    ad_diff = advancing - declining
    ad_line = ad_diff.cumsum()

    adv_ratio = advancing / total
    breadth_thrust = adv_ratio.ewm(span=10, min_periods=5).mean()

    rolling_high = closes.rolling(252, min_periods=50).max()
    rolling_low  = closes.rolling(252, min_periods=50).min()
    at_high = (closes >= rolling_high * 0.99).sum(axis=1)
    at_low  = (closes <= rolling_low  * 1.01).sum(axis=1)
    hi_lo_diff = at_high - at_low

    return {
        "pct_above_200dma": above_200.dropna(),
        "pct_above_50dma":  above_50.dropna(),
        "ad_diff":          ad_diff.dropna(),
        "ad_line":          ad_line.dropna(),
        "breadth_thrust":   breadth_thrust.dropna(),
        "hi_lo_diff":       hi_lo_diff.dropna(),
        "n_stocks":         n_stocks,
    }


# ──────────────────────────────────────────────
# VALUATION SIDEBAR
# ──────────────────────────────────────────────

def render_valuation_sidebar() -> dict:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Valuation Update")
    st.sidebar.markdown(
        "<span style='font-size:0.78rem; color:#94a3b8;'>"
        "Enter latest forward P/E and CAPE if auto-fetch fails.<br>"
        "Sources: <a href='https://www.multpl.com/shiller-pe' target='_blank' style='color:#60a5fa;'>multpl.com</a>, "
        "<a href='https://yardeni.com/charts/sp-500-sectors-forward-p-e-ratios/' target='_blank' style='color:#60a5fa;'>Yardeni</a>"
        "</span>",
        unsafe_allow_html=True,
    )
    manual_fwd_pe = st.sidebar.number_input(
        "Forward P/E (manual)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
        format="%.1f", key="manual_fwd_pe",
    )
    manual_cape = st.sidebar.number_input(
        "Shiller CAPE (manual)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
        format="%.1f", key="manual_cape",
    )
    return {
        "manual_fwd_pe": manual_fwd_pe if manual_fwd_pe > 0 else None,
        "manual_cape":   manual_cape   if manual_cape   > 0 else None,
    }


# ──────────────────────────────────────────────
# GLOBAL DATE RANGE HELPER
# ──────────────────────────────────────────────

DATE_RANGE_OPTIONS = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year":   365,
    "2 Years":  730,
    "5 Years":  1825,
}


def filter_by_date_range(df_or_series, days: int, date_col=None):
    """Filter a DataFrame or Series to the last N days."""
    cutoff = datetime.now() - timedelta(days=days)
    if isinstance(df_or_series, pd.Series):
        if df_or_series.index.dtype == "datetime64[ns]" or hasattr(df_or_series.index, "date"):
            return df_or_series[df_or_series.index >= cutoff]
        return df_or_series
    elif isinstance(df_or_series, pd.DataFrame):
        if date_col and date_col in df_or_series.columns:
            return df_or_series[df_or_series[date_col] >= cutoff]
        elif df_or_series.index.dtype == "datetime64[ns]" or hasattr(df_or_series.index, "date"):
            return df_or_series[df_or_series.index >= cutoff]
    return df_or_series


# ──────────────────────────────────────────────
# AAII SIDEBAR INPUT & STALENESS CHECK
# ──────────────────────────────────────────────

def render_aaii_sidebar(aaii_df: pd.DataFrame) -> pd.DataFrame:
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
        new_date = st.date_input("Survey date", value=datetime.now().date())
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
                new_row = pd.DataFrame([{
                    "date":    pd.Timestamp(new_date),
                    "bullish": round(new_bull / 100, 6),
                    "neutral": round(new_neut / 100, 6),
                    "bearish": round(new_bear / 100, 6),
                }])
                if not aaii_df.empty and pd.Timestamp(new_date) in aaii_df["date"].values:
                    aaii_df.loc[aaii_df["date"] == pd.Timestamp(new_date), ["bullish", "neutral", "bearish"]] = [
                        round(new_bull / 100, 6), round(new_neut / 100, 6), round(new_bear / 100, 6),
                    ]
                    st.sidebar.success(f"Updated entry for {new_date}")
                else:
                    aaii_df = pd.concat([aaii_df, new_row], ignore_index=True)
                    aaii_df = aaii_df.sort_values("date").reset_index(drop=True)
                    st.sidebar.success(f"Added entry for {new_date}")

                csv_path = os.path.join(os.path.dirname(__file__), "aaii_sentiment_history.csv")
                try:
                    aaii_df_save = aaii_df.copy()
                    aaii_df_save["date"] = aaii_df_save["date"].dt.strftime("%Y-%m-%d")
                    aaii_df_save.to_csv(csv_path, index=False)
                except Exception:
                    st.sidebar.warning("Could not save to CSV. Entry is active for this session only.")

                load_aaii_history.clear()

    return aaii_df


def check_aaii_staleness(aaii_df: pd.DataFrame) -> dict:
    if aaii_df.empty:
        return {"stale": True, "days": 999, "level": "critical", "message": "No AAII sentiment data loaded."}
    latest_date = aaii_df["date"].max()
    days_old = (datetime.now() - pd.Timestamp(latest_date)).days
    if days_old <= 9:
        return {"stale": False, "days": days_old, "level": "ok", "message": ""}
    elif days_old <= 16:
        return {"stale": True, "days": days_old, "level": "warning",
                "message": f"⚠️ AAII data is {days_old} days old. Update via sidebar (published Thursday at aaii.com)."}
    else:
        return {"stale": True, "days": days_old, "level": "critical",
                "message": f"🚨 AAII data is {days_old} days old — significantly stale. Open sidebar (☰) to update."}


# ──────────────────────────────────────────────
# INDICATOR CALCULATIONS (unchanged)
# ──────────────────────────────────────────────

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_price_vs_sma(prices: pd.Series, window: int = 200) -> dict:
    sma          = prices.rolling(window).mean()
    latest_price = prices.iloc[-1]
    latest_sma   = sma.iloc[-1]
    pct          = ((latest_price - latest_sma) / latest_sma) * 100
    return {"price": latest_price, "sma": latest_sma, "pct_above": pct,
            "above": latest_price > latest_sma, "sma_series": sma}


def compute_drawdown(prices: pd.Series) -> dict:
    cum_max     = prices.cummax()
    drawdown_pct = ((prices - cum_max) / cum_max) * 100
    current_dd   = drawdown_pct.iloc[-1]
    if current_dd > -5:
        regime, color = "Normal",     "#34d399"
    elif current_dd > -10:
        regime, color = "Pullback",   "#fbbf24"
    elif current_dd > -20:
        regime, color = "Correction", "#fb923c"
    else:
        regime, color = "Bear Market","#f87171"
    return {"current_dd": current_dd, "regime": regime, "color": color, "series": drawdown_pct}


def classify_signal(value: float, thresholds: dict) -> tuple:
    if value <= thresholds.get("bullish", float("-inf")):
        return "BULLISH", "signal-bullish"
    elif value >= thresholds.get("bearish", float("inf")):
        return "BEARISH", "signal-bearish"
    else:
        return "NEUTRAL", "signal-neutral"


def compute_aaii_sentiment_score(aaii_df: pd.DataFrame) -> dict:
    if aaii_df.empty:
        return {"available": False, "bullish": 0, "neutral": 0, "bearish": 0, "net": 0, "score": 0, "label": "N/A"}
    latest = aaii_df.iloc[-1]
    bull = latest["bullish"]
    neut = latest["neutral"]
    bear = latest["bearish"]
    net  = bull - bear
    deviation = (net - 0.065) / 0.16   # hist avg net ~6.5%, std ~16%
    score     = max(-1, min(1, -deviation * 0.5))
    if score > 0.3:   label = "BULLISH"
    elif score < -0.3: label = "BEARISH"
    else:              label = "NEUTRAL"
    return {"available": True, "bullish": bull, "neutral": neut, "bearish": bear,
            "net": net, "score": score, "label": label, "date": latest["date"]}


def compute_composite_signal(rsi: float, vix: float, sma_above: bool, sentiment_score: float) -> dict:
    # Momentum score
    momentum = 0.5
    if rsi < 30:        momentum += 0.3
    elif rsi > 70:      momentum -= 0.3
    elif rsi < 45:      momentum += 0.15
    elif rsi > 55:      momentum -= 0.15
    momentum += 0.2 if sma_above else -0.2
    momentum = max(0, min(1, momentum))

    # Risk/sentiment score (contrarian)
    risk = 0.5
    if vix > 30:    risk += 0.25
    elif vix > 25:  risk += 0.10
    elif vix < 14:  risk -= 0.20
    elif vix < 18:  risk -= 0.10
    risk += sentiment_score * 0.25
    risk = max(0, min(1, risk))

    mom_high  = momentum > 0.5
    risk_high = risk > 0.5

    if mom_high and risk_high:
        signal, css, color = "STRONG BULLISH", "composite-bullish", "#34d399"
        desc = "Positive momentum with supportive sentiment — favorable conditions."
    elif mom_high and not risk_high:
        signal, css, color = "LEAN BULLISH",   "composite-caution", "#fbbf24"
        desc = "Positive momentum but complacent sentiment — proceed with caution."
    elif not mom_high and risk_high:
        signal, css, color = "LEAN BEARISH",   "composite-caution", "#fb923c"
        desc = "Weak momentum but fear elevated — potential contrarian opportunity forming."
    else:
        signal, css, color = "STRONG BEARISH", "composite-bearish", "#f87171"
        desc = "Weak momentum with complacent sentiment — risk-off conditions."

    return {"signal": signal, "css_class": css, "color": color, "desc": desc,
            "momentum_score": momentum, "risk_score": risk}


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
    delta_html  = ""
    if delta:
        delta_color = "#34d399" if delta.startswith("+") or delta.startswith("▲") else "#f87171"
        delta_html  = f'<div class="metric-delta" style="color:{delta_color}">{delta}</div>'
    signal_html = f'<span class="{signal_class}">{signal}</span>' if signal else ""
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
    st.markdown('<div class="dashboard-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">S&P 500 — Sentiment · Momentum · Financial Conditions · Valuation &nbsp;|&nbsp; Updated at each page load</div>', unsafe_allow_html=True)

    # ── API Keys ──
    fred_key    = st.secrets.get("FRED_API_KEY", "")
    fmp_key     = st.secrets.get("FMP_API_KEY", "")
    td_key      = st.secrets.get("TWELVE_DATA_API_KEY", "")

    if not fred_key:
        fred_key = st.sidebar.text_input("FRED API Key", type="password",
            help="Get free at https://fred.stlouisfed.org/docs/api/api_key.html")
    if not td_key:
        td_key = st.sidebar.text_input("Twelve Data API Key", type="password",
            help="Get at https://twelvedata.com — required for price data")

    if not td_key:
        st.error("⚠️ A Twelve Data API key is required. Add `TWELVE_DATA_API_KEY` to Streamlit secrets or enter it in the sidebar.")
        return

    # ── Load static CSVs & sidebar forms ──
    aaii_df  = load_aaii_history()
    aaii_df  = render_aaii_sidebar(aaii_df)
    val_manual = render_valuation_sidebar()
    aaii_status = check_aaii_staleness(aaii_df)
    if aaii_status["stale"]:
        alert_class = "stale-alert-critical" if aaii_status["level"] == "critical" else ""
        st.markdown(f'<div class="stale-alert {alert_class}">{aaii_status["message"]}</div>', unsafe_allow_html=True)

    market = MARKET_CONFIGS["S&P 500"]

    # ── Fetch core price data (5-year history) ──
    with st.spinner("Fetching market data from Twelve Data & FRED..."):
        PRICE_DAYS = 1825  # 5 years

        spx_data = fetch_td_price(market["td_index_symbol"], td_key, days=PRICE_DAYS)

        # VIX and DXY come from FRED — free, reliable, no plan restrictions.
        # (Twelve Data Growth plan does not support CBOE/index-only symbols like VIX.)
        vix_series = pd.Series(dtype=float)
        dxy_series = pd.Series(dtype=float)
        hy_spread  = pd.Series(dtype=float)
        dgs2       = pd.Series(dtype=float)
        dgs10      = pd.Series(dtype=float)
        nfci       = pd.Series(dtype=float)
        if fred_key:
            vix_series = fetch_fred_series(market["fred_vix"],  fred_key)
            dxy_series = fetch_fred_series(market["fred_dxy"],  fred_key)
            hy_spread  = fetch_fred_series(market["fred_hy_spread"], fred_key)
            dgs2       = fetch_fred_series(market["fred_yield_curve"]["2y"], fred_key)
            dgs10      = fetch_fred_series(market["fred_yield_curve"]["10y"], fred_key)
            nfci       = fetch_fred_series(market["fred_nfci"], fred_key)

        # Breadth: FMP for constituents, Twelve Data for prices
        breadth_data  = {}
        sp500_closes  = pd.DataFrame()
        breadth_debug = ""
        constituents  = fetch_sp500_constituents_fmp(fmp_key)
        if constituents:
            breadth_debug = f"Found {len(constituents)} constituents. Downloading prices via Twelve Data..."
            # Twelve Data Growth plan: batch to avoid credit exhaustion
            sp500_closes = fetch_td_batch_closes(tuple(constituents), td_key, days=PRICE_DAYS)
            if not sp500_closes.empty:
                breadth_data  = compute_breadth_from_prices(sp500_closes)
                breadth_debug = f"Loaded {sp500_closes.shape[1]} stocks, {sp500_closes.shape[0]} days of data."
            else:
                breadth_debug = f"Got {len(constituents)} constituents but Twelve Data returned no price data."
        else:
            breadth_debug = f"Could not fetch constituent list. FMP key set: {bool(fmp_key)}."

        # Valuation: FMP for P/E, bundled CSV for CAPE
        shiller_df  = fetch_shiller_cape()
        spy_pe_info = fetch_fmp_valuation(fmp_key) if fmp_key else {"available": False}

    if spx_data.empty:
        st.error("Could not fetch S&P 500 data from Twelve Data. Check your API key and try again.")
        return

    # ── Compute Indicators ──
    close    = spx_data["Close"]
    rsi      = compute_rsi(close)
    sma_info = compute_price_vs_sma(close)
    dd_info  = compute_drawdown(close)

    current_rsi = rsi.iloc[-1]
    current_vix = vix_series.iloc[-1] if not vix_series.empty else np.nan
    current_dxy = dxy_series.iloc[-1] if not dxy_series.empty else np.nan

    yield_curve_val = np.nan
    if not dgs10.empty and not dgs2.empty:
        yield_curve_val = dgs10.iloc[-1] - dgs2.iloc[-1]
    hy_val   = hy_spread.iloc[-1] if not hy_spread.empty else np.nan
    nfci_val = nfci.iloc[-1]     if not nfci.empty     else np.nan

    # AAII
    aaii_info       = compute_aaii_sentiment_score(aaii_df)
    sentiment_score = aaii_info["score"]
    if not aaii_info["available"] and not np.isnan(current_vix):
        if current_vix > 30:   sentiment_score =  0.6
        elif current_vix > 25: sentiment_score =  0.3
        elif current_vix < 14: sentiment_score = -0.5
        elif current_vix < 18: sentiment_score = -0.2

    # Composite Signal
    composite = compute_composite_signal(
        rsi=current_rsi,
        vix=current_vix if not np.isnan(current_vix) else 20,
        sma_above=sma_info["above"],
        sentiment_score=sentiment_score,
    )

    # Signal badges
    rsi_signal, rsi_class = classify_signal(current_rsi, {"bullish": 35, "bearish": 65})
    vix_signal, vix_class = (
        classify_signal(current_vix, {"bullish": 12, "bearish": 25})
        if not np.isnan(current_vix) else ("N/A", "signal-neutral")
    )
    sma_signal = ("BULLISH", "signal-bullish") if sma_info["above"] else ("BEARISH", "signal-bearish")

    # ── Composite Signal Banner ──
    st.markdown(f"""
    <div class="composite-box {composite['css_class']}">
        <div class="composite-label" style="color:{composite['color']}">{composite['signal']}</div>
        <div class="composite-desc">{composite['desc']}</div>
        <div style="margin-top:0.5rem; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6b7d93;">
            Momentum: {composite['momentum_score']:.2f} &nbsp;|&nbsp; Risk/Sentiment: {composite['risk_score']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Metrics Row ──
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        daily_chg = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        render_metric_card("S&P 500", f"{close.iloc[-1]:,.0f}",
                           f"{'▲' if daily_chg >= 0 else '▼'} {abs(daily_chg):.2f}%")
    with col2:
        render_metric_card("RSI (14)", f"{current_rsi:.1f}", signal=rsi_signal, signal_class=rsi_class)
    with col3:
        if not np.isnan(current_vix):
            render_metric_card("VIX", f"{current_vix:.1f}", signal=vix_signal, signal_class=vix_class)
        else:
            render_metric_card("VIX", "N/A")
    with col4:
        render_metric_card("vs 200 DMA", f"{sma_info['pct_above']:.1f}%",
                           signal=sma_signal[0], signal_class=sma_signal[1])
    with col5:
        render_metric_card("Drawdown", f"{dd_info['current_dd']:.1f}%",
                           signal=dd_info['regime'],
                           signal_class="signal-bullish" if dd_info['regime'] == "Normal" else "signal-bearish")
    with col6:
        if aaii_info["available"]:
            aaii_net_pct = aaii_info["net"] * 100
            aaii_sc = ("signal-bullish" if aaii_info["label"] == "BULLISH"
                       else ("signal-bearish" if aaii_info["label"] == "BEARISH" else "signal-neutral"))
            render_metric_card("AAII Net", f"{aaii_net_pct:+.1f}%", signal=aaii_info["label"], signal_class=aaii_sc)
        else:
            render_metric_card("AAII Net", "N/A")

    # ── Global Date Range Selector ──
    dr_cols = st.columns([3, 1])
    with dr_cols[1]:
        selected_range = st.selectbox(
            "Chart range", list(DATE_RANGE_OPTIONS.keys()), index=3, key="global_date_range"
        )
    chart_days = DATE_RANGE_OPTIONS[selected_range]

    # ── Tabs ──
    tab_momentum, tab_sentiment, tab_breadth, tab_conditions, tab_valuation, tab_matrix = st.tabs([
        "📈 Momentum", "🧠 Sentiment", "📊 Breadth", "💰 Financial Conditions", "📐 Valuation", "🔲 2×2 Matrix"
    ])

    # ────── MOMENTUM TAB ──────
    with tab_momentum:
        st.markdown('<div class="section-header">Momentum Indicators</div>', unsafe_allow_html=True)

        close_filtered = filter_by_date_range(close, chart_days)
        rsi_filtered   = filter_by_date_range(rsi, chart_days)
        sma_filtered   = filter_by_date_range(sma_info["sma_series"], chart_days)
        dd_filtered    = filter_by_date_range(dd_info["series"], chart_days)

        mc1, mc2 = st.columns(2)
        with mc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close_filtered.index, y=close_filtered.values,
                                     name="S&P 500", line=dict(color="#60a5fa", width=2)))
            fig.add_trace(go.Scatter(x=sma_filtered.index, y=sma_filtered.values,
                                     name="200 DMA", line=dict(color="#fbbf24", width=1.5, dash="dot")))
            fig.update_layout(title="S&P 500 vs 200-Day Moving Average", **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with mc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rsi_filtered.index, y=rsi_filtered.values,
                                     name="RSI(14)", line=dict(color="#a78bfa", width=2),
                                     fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
            fig.add_hline(y=70, line_color="#f87171", line_dash="dash", line_width=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_color="#34d399", line_dash="dash", line_width=1, annotation_text="Oversold")
            fig.add_hline(y=50, line_color="#334155", line_dash="dot",  line_width=1)
            fig.update_layout(title="RSI (14)", yaxis_range=[0, 100], **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd_filtered.index, y=dd_filtered.values, name="Drawdown",
                                 line=dict(color="#f87171", width=1.5),
                                 fill="tozeroy", fillcolor="rgba(248,113,113,0.1)"))
        fig.add_hline(y=-5,  line_color="#fbbf24", line_dash="dot", line_width=1, annotation_text="Pullback")
        fig.add_hline(y=-10, line_color="#fb923c", line_dash="dot", line_width=1, annotation_text="Correction")
        fig.add_hline(y=-20, line_color="#f87171", line_dash="dot", line_width=1, annotation_text="Bear Market")
        fig.update_layout(
            title=f"Drawdown from All-Time High — Current Regime: {dd_info['regime']}",
            yaxis_title="Drawdown (%)", **CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ────── SENTIMENT TAB ──────
    with tab_sentiment:
        st.markdown('<div class="section-header">Sentiment & Positioning</div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            if not vix_series.empty:
                vix_close = filter_by_date_range(vix_series, chart_days)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vix_close.index, y=vix_close.values, name="VIX",
                                         line=dict(color="#f59e0b", width=2),
                                         fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"))
                fig.add_hline(y=20, line_color="#334155", line_dash="dot",  line_width=1, annotation_text="Long-term avg")
                fig.add_hline(y=30, line_color="#f87171", line_dash="dash", line_width=1, annotation_text="Elevated fear")
                fig.update_layout(title="CBOE Volatility Index (VIX) — Source: FRED/CBOE", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with sc2:
            if not aaii_df.empty:
                cutoff      = datetime.now() - timedelta(days=730)
                aaii_recent = aaii_df[aaii_df["date"] >= cutoff].copy()
                if not aaii_recent.empty:
                    spread  = (aaii_recent["bullish"] - aaii_recent["bearish"]) * 100
                    colors  = ["rgba(52,211,153,0.3)" if s > 0 else "rgba(248,113,113,0.3)" for s in spread]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=aaii_recent["date"], y=spread, name="Bull-Bear Spread",
                                         marker_color=colors))
                    spread_ma = spread.rolling(8, min_periods=1).mean()
                    fig.add_trace(go.Scatter(x=aaii_recent["date"], y=spread_ma, name="8-wk MA",
                                             line=dict(color="#fbbf24", width=2)))
                    fig.add_hline(y=0,   line_color="#f0f4f8", line_width=1)
                    fig.add_hline(y=6.5, line_color="#334155", line_dash="dot", line_width=1,
                                  annotation_text="Hist. avg spread")
                    fig.update_layout(title="AAII Bull-Bear Spread (%, contrarian indicator)",
                                      yaxis_title="Net Bulls − Bears (%)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div class="metric-card" style="min-height:300px;">
                    <div class="metric-label">AAII Net Sentiment</div>
                    <div style="color:#6b7d93; font-family:'DM Sans',sans-serif; font-size:0.85rem; padding-top:1rem;">
                        No AAII data loaded. Upload aaii_sentiment_history.csv to the app directory.
                    </div>
                </div>""", unsafe_allow_html=True)

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
                aaii_sc = ("signal-bullish" if aaii_info["label"] == "BULLISH"
                           else ("signal-bearish" if aaii_info["label"] == "BEARISH" else "signal-neutral"))
                render_metric_card("Contrarian Score", f"{aaii_info['score']:+.2f}",
                                   signal=aaii_info["label"], signal_class=aaii_sc)
            st.markdown(f"""
            <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93; margin-top:0.3rem;">
                Latest survey: {aaii_info['date'].strftime('%Y-%m-%d') if hasattr(aaii_info['date'], 'strftime') else aaii_info['date']}
                &nbsp;|&nbsp; Contrarian interpretation: extreme bearishness = bullish signal.
                &nbsp;|&nbsp; Update weekly via sidebar (☰).
            </div>""", unsafe_allow_html=True)

        if not aaii_df.empty:
            st.markdown('<div class="section-header" style="margin-top:1rem">AAII Sentiment History</div>',
                        unsafe_allow_html=True)
            range_options  = {"1 Year": 365, "2 Years": 730, "5 Years": 1825, "10 Years": 3650, "All": 99999}
            sel_range      = st.selectbox("Time range", list(range_options.keys()), index=1, label_visibility="collapsed")
            aaii_plot      = aaii_df[aaii_df["date"] >= datetime.now() - timedelta(days=range_options[sel_range])].copy()
            if not aaii_plot.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=aaii_plot["date"], y=aaii_plot["bullish"] * 100,
                                         name="Bullish", stackgroup="one",
                                         line=dict(color="#34d399", width=0), fillcolor="rgba(52,211,153,0.4)"))
                fig.add_trace(go.Scatter(x=aaii_plot["date"], y=aaii_plot["neutral"] * 100,
                                         name="Neutral", stackgroup="one",
                                         line=dict(color="#fbbf24", width=0), fillcolor="rgba(251,191,36,0.3)"))
                fig.add_trace(go.Scatter(x=aaii_plot["date"], y=aaii_plot["bearish"] * 100,
                                         name="Bearish", stackgroup="one",
                                         line=dict(color="#f87171", width=0), fillcolor="rgba(248,113,113,0.4)"))
                fig.update_layout(title="AAII Investor Sentiment (stacked %)",
                                  yaxis_title="Percentage", yaxis_range=[0, 100], height=380,
                                  **{k: v for k, v in CHART_LAYOUT.items() if k != "height"})
                st.plotly_chart(fig, use_container_width=True)

    # ────── BREADTH TAB ──────
    with tab_breadth:
        st.markdown('<div class="section-header">S&P 500 Market Breadth Indicators</div>', unsafe_allow_html=True)

        if not breadth_data:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Breadth Data Unavailable</div>
                <div style="color:#6b7d93; font-family:'DM Sans',sans-serif; font-size:0.85rem; padding:0.5rem 0;">
                    Could not load S&P 500 constituent price data from Twelve Data.<br>
                    Try refreshing. This tab requires valid FMP and Twelve Data API keys.<br><br>
                    <strong>Debug:</strong> {breadth_debug}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            n_stocks = breadth_data.get("n_stocks", 500)
            st.markdown(f"""
            <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:#6b7d93; margin-bottom:0.5rem;">
                Computed from {n_stocks} S&P 500 constituents · Source: FMP (constituent list) + Twelve Data (prices)
            </div>""", unsafe_allow_html=True)

            br1, br2 = st.columns(2)
            with br1:
                pct200 = breadth_data.get("pct_above_200dma", pd.Series(dtype=float))
                if not pct200.empty:
                    pct200_f = filter_by_date_range(pct200, chart_days)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pct200_f.index, y=pct200_f.values,
                                             name="% Above 200 DMA", line=dict(color="#60a5fa", width=2),
                                             fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"))
                    fig.add_hline(y=50, line_color="#fbbf24", line_dash="dot", line_width=1, annotation_text="50%")
                    fig.update_layout(title="S&P 500 — % of Stocks Above 200-Day MA", yaxis_title="%", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with br2:
                pct50 = breadth_data.get("pct_above_50dma", pd.Series(dtype=float))
                if not pct50.empty:
                    pct50_f = filter_by_date_range(pct50, chart_days)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=pct50_f.index, y=pct50_f.values,
                                             name="% Above 50 DMA", line=dict(color="#a78bfa", width=2),
                                             fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
                    fig.add_hline(y=50, line_color="#fbbf24", line_dash="dot", line_width=1, annotation_text="50%")
                    fig.update_layout(title="S&P 500 — % of Stocks Above 50-Day MA", yaxis_title="%", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            br3, br4 = st.columns(2)
            with br3:
                ad_line = breadth_data.get("ad_line", pd.Series(dtype=float))
                if not ad_line.empty:
                    ad_f = filter_by_date_range(ad_line, chart_days)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ad_f.index, y=ad_f.values, name="A/D Line",
                                             line=dict(color="#34d399", width=2)))
                    fig.update_layout(title="S&P 500 Advance-Decline Line (cumulative)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            with br4:
                hi_lo = breadth_data.get("hi_lo_diff", pd.Series(dtype=float))
                if not hi_lo.empty:
                    hl_f   = filter_by_date_range(hi_lo, chart_days)
                    colors = ["#34d399" if v > 0 else "#f87171" for v in hl_f.values]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=hl_f.index, y=hl_f.values, name="Highs − Lows",
                                         marker_color=colors, opacity=0.7))
                    hl_ma = hl_f.rolling(10, min_periods=1).mean()
                    fig.add_trace(go.Scatter(x=hl_ma.index, y=hl_ma.values, name="10d MA",
                                             line=dict(color="#fbbf24", width=2)))
                    fig.add_hline(y=0, line_color="#f0f4f8", line_width=1)
                    fig.update_layout(title="S&P 500 — 52-Week New Highs − New Lows", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            bt = breadth_data.get("breadth_thrust", pd.Series(dtype=float))
            if not bt.empty:
                bt_f = filter_by_date_range(bt, chart_days)
                fig  = go.Figure()
                fig.add_trace(go.Scatter(x=bt_f.index, y=bt_f.values * 100, name="Breadth Thrust",
                                         line=dict(color="#38bdf8", width=2),
                                         fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
                fig.add_hline(y=61.5, line_color="#34d399", line_dash="dash", line_width=1,
                              annotation_text="Bullish thrust (>61.5%)")
                fig.add_hline(y=40, line_color="#f87171", line_dash="dash", line_width=1,
                              annotation_text="Weak breadth (<40%)")
                fig.add_hline(y=50, line_color="#334155", line_dash="dot", line_width=1)
                fig.update_layout(title="Zweig Breadth Thrust — 10-Day EMA of Advancing / (Adv + Dec) %",
                                  yaxis_title="%", yaxis_range=[20, 85], **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""
                <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93; margin-top:0.3rem;">
                    Zweig Breadth Thrust: a reading that moves from below 40% to above 61.5% within 10 trading days
                    signals rare but powerful bullish momentum. These signals have historically preceded strong market advances.
                </div>""", unsafe_allow_html=True)

        # Note: put/call ratio removed (yfinance options chain no longer available)
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93; margin-top:1rem;">
            <strong>Note:</strong> SPY put/call ratio (previously from yfinance options chain) has been removed
            in this version. It can be re-added using a dedicated options data provider.
        </div>""", unsafe_allow_html=True)

    # ────── FINANCIAL CONDITIONS TAB ──────
    with tab_conditions:
        st.markdown('<div class="section-header">Financial Conditions</div>', unsafe_allow_html=True)

        if not fred_key:
            st.warning("Add your FRED API key to view financial conditions data.")
        else:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                if not np.isnan(hy_val):
                    hy_signal = "BEARISH" if hy_val > 5 else ("BULLISH" if hy_val < 3.5 else "NEUTRAL")
                    hy_class  = "signal-bullish" if hy_signal == "BULLISH" else ("signal-bearish" if hy_signal == "BEARISH" else "signal-neutral")
                    render_metric_card("HY Credit Spread (OAS)", f"{hy_val:.0f} bps", signal=hy_signal, signal_class=hy_class)
                else:
                    render_metric_card("HY Credit Spread", "N/A")
            with fc2:
                if not np.isnan(yield_curve_val):
                    yc_signal = ("BULLISH", "signal-bullish") if yield_curve_val > 0 else ("BEARISH", "signal-bearish")
                    render_metric_card("2y/10y Yield Curve", f"{yield_curve_val:.2f}%",
                                       signal=yc_signal[0], signal_class=yc_signal[1])
                else:
                    render_metric_card("2y/10y Yield Curve", "N/A")
            with fc3:
                if not np.isnan(nfci_val):
                    nfci_signal = ("BULLISH", "signal-bullish") if nfci_val < 0 else ("BEARISH", "signal-bearish")
                    render_metric_card("Chicago Fed NFCI", f"{nfci_val:.2f}",
                                       signal=nfci_signal[0], signal_class=nfci_signal[1])
                else:
                    render_metric_card("NFCI", "N/A")

            cc1, cc2 = st.columns(2)
            with cc1:
                if not hy_spread.empty:
                    hy_f = filter_by_date_range(hy_spread, chart_days)
                    fig  = go.Figure()
                    fig.add_trace(go.Scatter(x=hy_f.index, y=hy_f.values, name="HY OAS",
                                             line=dict(color="#fb923c", width=2),
                                             fill="tozeroy", fillcolor="rgba(251,146,60,0.08)"))
                    fig.update_layout(title="High Yield Credit Spread (OAS, bps)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
            with cc2:
                if not dgs2.empty and not dgs10.empty:
                    curve   = (dgs10 - dgs2).dropna()
                    curve_f = filter_by_date_range(curve, chart_days)
                    colors  = ["#34d399" if v > 0 else "#f87171" for v in curve_f.values]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=curve_f.index, y=curve_f.values, name="2y/10y Spread",
                                         marker_color=colors, opacity=0.7))
                    fig.add_hline(y=0, line_color="#f0f4f8", line_width=1)
                    fig.update_layout(title="2y/10y Treasury Yield Curve Spread (%)", **CHART_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            if not nfci.empty:
                nfci_f = filter_by_date_range(nfci, chart_days)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=nfci_f.index, y=nfci_f.values, name="NFCI",
                                         line=dict(color="#a78bfa", width=2),
                                         fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
                fig.add_hline(y=0, line_color="#f0f4f8", line_width=1, annotation_text="Avg conditions")
                fig.update_layout(title="Chicago Fed National Financial Conditions Index", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

    # ────── VALUATION TAB ──────
    with tab_valuation:
        st.markdown('<div class="section-header">Valuation Indicators</div>', unsafe_allow_html=True)

        trailing_pe    = spy_pe_info.get("trailing_pe")  if spy_pe_info.get("available") else None
        forward_pe     = val_manual.get("manual_fwd_pe") or (spy_pe_info.get("forward_pe") if spy_pe_info.get("available") else None)
        earnings_yield = spy_pe_info.get("earnings_yield") if spy_pe_info.get("available") else None

        cape_val = None
        if not shiller_df.empty:
            cape_val = shiller_df["cape"].iloc[-1]
        if val_manual.get("manual_cape"):
            cape_val = val_manual["manual_cape"]

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            if trailing_pe:
                pe_signal = "BEARISH" if trailing_pe > 25 else ("BULLISH" if trailing_pe < 18 else "NEUTRAL")
                pe_class  = "signal-bullish" if pe_signal == "BULLISH" else ("signal-bearish" if pe_signal == "BEARISH" else "signal-neutral")
                render_metric_card("Trailing P/E", f"{trailing_pe:.1f}",
                                   delta=f"{'▲' if trailing_pe > 20 else '▼'} vs 20 avg",
                                   signal=pe_signal, signal_class=pe_class)
            else:
                render_metric_card("Trailing P/E", "N/A")
        with vc2:
            if forward_pe:
                fwd_signal = "BEARISH" if forward_pe > 22 else ("BULLISH" if forward_pe < 16 else "NEUTRAL")
                fwd_class  = "signal-bullish" if fwd_signal == "BULLISH" else ("signal-bearish" if fwd_signal == "BEARISH" else "signal-neutral")
                source     = "manual" if val_manual.get("manual_fwd_pe") else "FMP"
                render_metric_card("Forward P/E", f"{forward_pe:.1f}",
                                   delta=f"{'▲' if forward_pe > 18 else '▼'} vs 18 avg · {source}",
                                   signal=fwd_signal, signal_class=fwd_class)
            else:
                render_metric_card("Forward P/E", "N/A — enter in sidebar")
        with vc3:
            if cape_val:
                cape_signal = "BEARISH" if cape_val > 30 else ("BULLISH" if cape_val < 20 else "NEUTRAL")
                cape_class  = "signal-bullish" if cape_signal == "BULLISH" else ("signal-bearish" if cape_signal == "BEARISH" else "signal-neutral")
                source      = "manual" if val_manual.get("manual_cape") else "Shiller"
                render_metric_card("Shiller CAPE", f"{cape_val:.1f}",
                                   delta=f"{'▲' if cape_val > 25 else '▼'} vs 25 median · {source}",
                                   signal=cape_signal, signal_class=cape_class)
            else:
                render_metric_card("Shiller CAPE", "N/A — enter in sidebar")
        with vc4:
            if earnings_yield:
                render_metric_card("Earnings Yield", f"{earnings_yield:.2f}%", delta="inverse of trailing P/E")
            else:
                render_metric_card("Earnings Yield", "N/A")

        # Z-score regime extremity
        st.markdown('<div class="section-header" style="margin-top:0.5rem">Regime Extremity — Z-Scores</div>',
                    unsafe_allow_html=True)
        z_scores = {}
        if not shiller_df.empty and len(shiller_df) > 12:
            cape_mean = shiller_df["cape"].mean()
            cape_std  = shiller_df["cape"].std()
            cape_cur  = cape_val if cape_val else shiller_df["cape"].iloc[-1]
            z_scores["Shiller CAPE"] = (cape_cur - cape_mean) / cape_std if cape_std > 0 else 0
        if len(rsi.dropna()) > 50:
            z_scores["RSI (14)"] = (current_rsi - rsi.mean()) / rsi.std()
        if not vix_series.empty:
            vix_all = vix_series.dropna()
            if len(vix_all) > 50:
                z_scores["VIX"] = (current_vix - vix_all.mean()) / vix_all.std()
        dd_all = dd_info["series"].dropna()
        if len(dd_all) > 50:
            z_scores["Drawdown"] = (dd_info["current_dd"] - dd_all.mean()) / dd_all.std()
        pct_above_series = ((close - close.rolling(200).mean()) / close.rolling(200).mean() * 100).dropna()
        if len(pct_above_series) > 50:
            z_scores["Price vs 200 DMA"] = (sma_info["pct_above"] - pct_above_series.mean()) / pct_above_series.std()
        if aaii_info["available"] and not aaii_df.empty and len(aaii_df) > 50:
            net_series = aaii_df["bullish"] - aaii_df["bearish"]
            z_scores["AAII Net Sentiment"] = (aaii_info["net"] - net_series.mean()) / net_series.std()

        if z_scores:
            z_names  = list(z_scores.keys())
            z_vals   = list(z_scores.values())
            z_colors = ["#f87171" if abs(z) > 2 else ("#fb923c" if abs(z) > 1 else "#34d399") for z in z_vals]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=z_names, x=z_vals, orientation="h", marker_color=z_colors,
                                 text=[f"{z:+.2f}σ" for z in z_vals], textposition="outside",
                                 textfont=dict(size=11, family="JetBrains Mono")))
            fig.add_vline(x=0,  line_color="#f0f4f8", line_width=1)
            fig.add_vline(x=2,  line_color="#f87171", line_dash="dash", line_width=1)
            fig.add_vline(x=-2, line_color="#f87171", line_dash="dash", line_width=1)
            fig.add_vline(x=1,  line_color="#fb923c", line_dash="dot", line_width=1)
            fig.add_vline(x=-1, line_color="#fb923c", line_dash="dot", line_width=1)
            fig.update_layout(title="How Extreme Is the Current Regime? (standard deviations from mean)",
                              xaxis_title="Z-Score (σ)", xaxis_range=[-3.5, 3.5],
                              height=max(250, len(z_scores) * 50 + 100),
                              **{k: v for k, v in CHART_LAYOUT.items() if k != "height"})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93;">
                <strong>Green</strong> = within 1σ (normal) · <strong>Orange</strong> = 1–2σ (elevated) · <strong>Red</strong> = beyond 2σ (extreme).
            </div>""", unsafe_allow_html=True)

        if not shiller_df.empty:
            cape_filtered = filter_by_date_range(shiller_df, chart_days * 3, date_col="date")
            if not cape_filtered.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cape_filtered["date"], y=cape_filtered["cape"],
                                         name="Shiller CAPE", line=dict(color="#f59e0b", width=2),
                                         fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"))
                fig.add_hline(y=25, line_color="#334155", line_dash="dot",  line_width=1, annotation_text="Median (~25)")
                fig.add_hline(y=30, line_color="#fb923c", line_dash="dash", line_width=1, annotation_text="Expensive (>30)")
                fig.add_hline(y=20, line_color="#34d399", line_dash="dash", line_width=1, annotation_text="Fair value (<20)")
                fig.update_layout(title="Shiller CAPE Ratio (Cyclically Adjusted P/E)",
                                  yaxis_title="CAPE", **CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#6b7d93; margin-top:0.3rem;">
            <strong>Trailing P/E</strong> and <strong>Forward P/E</strong> — enter manually via the sidebar (☰).
            Suggested sources: <a href="https://www.multpl.com/s-p-500-pe-ratio" target="_blank" style="color:#60a5fa">multpl.com</a>
            or <a href="https://yardeni.com/charts/sp-500-sectors-forward-p-e-ratios/" target="_blank" style="color:#60a5fa">Yardeni</a>.
            FMP Starter plan does not expose SPY ratio endpoints (requires Premium+).
            <strong>Shiller CAPE</strong> from bundled CSV — update periodically from multpl.com.
            Valuation indicators are slow-moving — best used as long-term context, not timing signals.
        </div>""", unsafe_allow_html=True)

        if not dxy_series.empty:
            st.markdown('<div class="section-header" style="margin-top:1rem">US Dollar Index</div>',
                        unsafe_allow_html=True)
            dxy_close = filter_by_date_range(dxy_series, chart_days)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dxy_close.index, y=dxy_close.values, name="USD Index",
                                     line=dict(color="#38bdf8", width=2),
                                     fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
            fig.update_layout(title="Fed Broad US Dollar Index (DTWEXBGS) — Source: FRED", **CHART_LAYOUT)
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
                <td class="matrix-strong-bull">STRONG BULLISH<br><span style="font-size:0.65rem">Trend + fear = buy signal</span></td>
            </tr>
            <tr>
                <th>Momentum: BEARISH<br><span style="font-size:0.65rem;font-weight:400">(RSI weak, below 200DMA)</span></th>
                <td class="matrix-strong-bear">STRONG BEARISH<br><span style="font-size:0.65rem">No trend, no fear = avoid</span></td>
                <td class="matrix-lean-bear">LEAN BEARISH<br><span style="font-size:0.65rem">Weak trend, but fear rising — watch for turn</span></td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

        sentiment_source = "AAII survey" if aaii_info["available"] else "VIX-only proxy"
        st.markdown(f"""
        <div style="margin-top:1rem; font-family:'DM Sans',sans-serif; font-size:0.85rem; color:#94a3b8;">
            <strong>Current position:</strong> Momentum score = {composite['momentum_score']:.2f},
            Risk/Sentiment score = {composite['risk_score']:.2f}<br>
            Sentiment source: {sentiment_source}.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">How the Signal is Computed</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div style="color:#94a3b8; font-family:'DM Sans',sans-serif; font-size:0.85rem; line-height:1.7;">
                <strong>Momentum Axis (0–1):</strong> RSI(14) position (oversold adds +0.3, overbought subtracts −0.3)
                combined with price vs 200 DMA (above adds +0.2, below subtracts −0.2).<br><br>
                <strong>Risk/Sentiment Axis (0–1):</strong> VIX level (>30 adds +0.25 as contrarian bullish, <14 subtracts −0.2
                as complacency warning) combined with AAII bull-bear spread (contrarian adjusted).<br><br>
                <strong>Drawdown Overlay:</strong> Normal (<5%), Pullback (5-10%), Correction (10-20%), Bear Market (>20%).
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#475569; text-align:center; padding:0.5rem 0;">
        Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
        Price data: Twelve Data &nbsp;|&nbsp; VIX &amp; DXY: FRED &nbsp;|&nbsp; Conditions &amp; Valuation: FRED &nbsp;|&nbsp; Sentiment: AAII &nbsp;|&nbsp;
        AAII data: {'current' if not aaii_status['stale'] else f'{aaii_status["days"]}d old'} &nbsp;|&nbsp;
        This dashboard is for informational purposes only — not financial advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
