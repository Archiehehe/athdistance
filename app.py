import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="S&P 500 Stock Screener",
    page_icon="📈",
    layout="wide",
)


# ------------------------------------------------------------
# Data loading & caching helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_universe() -> pd.DataFrame:
    """
    Load stock universe from a local CSV file (universe.csv).

    The CSV must be in the same folder as app.py with columns:
      - Ticker
      - Company
      - Sector
      - Industry
    """
    try:
        df = pd.read_csv("universe.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "universe.csv not found. Make sure it exists in the same folder as app.py."
        )

    required_cols = ["Ticker", "Company", "Sector", "Industry"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"universe.csv is missing required columns: {missing}")

    # Clean up tickers
    df["Ticker"] = (
        df["Ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )

    return df[required_cols].copy()


@st.cache_data(show_spinner=False)
def fetch_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """
    Fetch daily price history for a ticker using yfinance.

    IMPORTANT: We use unadjusted High/Low/Close (auto_adjust=False) so that
    ATH and 52-week highs match chart-style prices, not dividend-adjusted series.
    """
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,  # <-- critical for correct ATH
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("No historical data returned")

    # We want split-adjusted but not dividend-adjusted High/Low/Close
    needed_cols = ["Close", "High", "Low"]
    for col in needed_cols:
        if col not in data.columns:
            raise ValueError(f"Missing {col} column in price history")

    df = data[needed_cols].copy()
    df = df.dropna(subset=["Close", "High", "Low"])

    if df.empty:
        raise ValueError("No valid OHLC data after dropping NaNs")

    return df


def compute_price_metrics(history: pd.DataFrame) -> Dict[str, float]:
    """
    Given a historical price DataFrame with 'Close', 'High', 'Low',
    compute ATH, 52-week metrics, etc.

    ATH and 52-week high/low are based on the unadjusted High/Low series
    (split-adjusted but NOT dividend-adjusted), matching typical chart behavior.
    """
    if not {"Close", "High", "Low"}.issubset(set(history.columns)):
        raise ValueError("History DataFrame must contain Close, High, Low columns")

    close = history["Close"].dropna()
    high = history["High"].dropna()
    low = history["Low"].dropna()

    if close.empty or high.empty or low.empty:
        raise ValueError("Insufficient OHLC data to compute metrics")

    # All-Time High based on daily High
    ath = float(high.max())
    current_price = float(close.iloc[-1])

    # 52-week window ~ 252 trading days
    window_len = min(252, len(history))
    last_52w = history.tail(window_len)
    high_52w = float(last_52w["High"].max())
    low_52w = float(last_52w["Low"].min())

    # Distance from ATH (as per spec)
    # (Current Price - ATH) / ATH * 100
    if ath > 0:
        distance_from_ath_pct = (current_price - ath) / ath * 100.0
    else:
        distance_from_ath_pct = np.nan

    # 52-Week Range (%)
    # (52W High - 52W Low) / 52W Low * 100
    if low_52w > 0:
        range_52w_pct = (high_52w - low_52w) / low_52w * 100.0
    else:
        range_52w_pct = np.nan

    # Distance from 52W High (%)
    # (Current Price - 52W High) / 52W High * 100
    if high_52w > 0:
        distance_from_52w_high_pct = (current_price - high_52w) / high_52w * 100.0
    else:
        distance_from_52w_high_pct = np.nan

    return {
        "Current Price": current_price,
        "ATH": ath,
        "Distance From ATH (%)": distance_from_ath_pct,
        "52W High": high_52w,
        "52W Low": low_52w,
        "52W Range (%)": range_52w_pct,
        "Distance From 52W High (%)": distance_from_52w_high_pct,
    }


# ------------------------------------------------------------
# Load universe (from CSV) with friendly error message
# ------------------------------------------------------------
try:
    universe_df = load_universe()
except Exception as e:
    st.error(
        "❌ Could not load the stock universe.\n\n"
        "Make sure there is a file called **universe.csv** in the same GitHub repo, "
        "with columns: `Ticker,Company,Sector,Industry`.\n\n"
        f"Technical details: {e}"
    )
    st.stop()


# ------------------------------------------------------------
# Sidebar: universe, filters, and controls
# ------------------------------------------------------------
st.sidebar.title("⚙️ Screener Controls")

with st.sidebar:
    st.markdown("### Stock Universe")
    st.caption("Using S&P 500 tickers defined in **universe.csv** as the default universe.")

# Sector selector (single select)
sector_options = ["All Sectors"] + sorted(universe_df["Sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Sector", sector_options, index=0)

if selected_sector != "All Sectors":
    universe_filtered_sector = universe_df[universe_df["Sector"] == selected_sector]
else:
    universe_filtered_sector = universe_df.copy()

# Industry selector (MULTISELECT, dependent on sector selection)
industry_options = sorted(
    universe_filtered_sector["Industry"].dropna().unique().tolist()
)

selected_industries = st.sidebar.multiselect(
    "Industry (you can pick multiple)",
    options=industry_options,
    default=industry_options,  # all selected by default
    help="Select one or more industries within the chosen sector.",
)

if selected_industries:
    final_universe = universe_filtered_sector[
        universe_filtered_sector["Industry"].isin(selected_industries)
    ].copy()
else:
    # If nothing selected, treat as 'all industries'
    final_universe = universe_filtered_sector.copy()

# Distance from ATH slider
st.sidebar.markdown("### Distance from ATH Filter")
distance_min, distance_max = st.sidebar.slider(
    "Distance from ATH (%) (Current - ATH) / ATH × 100",
    min_value=-100.0,
    max_value=50.0,
    value=(-50.0, 0.0),
    step=1.0,
    help="Filter by how far the stock is from its all-time high.",
)

# Performance: max tickers
st.sidebar.markdown("### Performance")
max_tickers = st.sidebar.slider(
    "Max tickers to process",
    min_value=5,
    max_value=200,
    value=50,
    step=5,
    help="Limit number of tickers (after filters) to keep the screener fast.",
)

run_screener = st.sidebar.button("🔍 Run Screener")


# ------------------------------------------------------------
# Main layout
# ------------------------------------------------------------
st.title("📈 S&P 500 Stock Screener (ATH & 52-Week)")

st.caption(
    "S&P 500 stock screener using Yahoo Finance (`yfinance`) and a predefined S&P 500 universe stored in `universe.csv`."
)

st.markdown(
    f"**S&P 500 universe:** {len(universe_df)} tickers  •  "
    f"After filters: {len(final_universe)} tickers"
)

if not run_screener:
    st.info(
        "Use the **Sector** and **Industry** filters in the sidebar, then click "
        "**'Run Screener'** to compute metrics."
    )
    st.stop()

if final_universe.empty:
    st.warning("No tickers match the selected Sector/Industry filters.")
    st.stop()

# Apply max ticker cap
if len(final_universe) > max_tickers:
    st.warning(
        f"Filtered universe has {len(final_universe)} tickers. "
        f"Processing only the first {max_tickers} for performance. "
        "Increase 'Max tickers to process' in the sidebar if needed."
    )
    final_universe = final_universe.head(max_tickers).copy()


# ------------------------------------------------------------
# Compute metrics for each ticker (with progress bar & error logging)
# ------------------------------------------------------------
results: List[Dict] = []
errors: List[Tuple[str, str]] = []

progress_text = "Downloading data and computing metrics..."
progress_bar = st.progress(0, text=progress_text)

tickers = final_universe["Ticker"].tolist()
n_tickers = len(tickers)

for i, (idx, row) in enumerate(final_universe.iterrows(), start=1):
    ticker = row["Ticker"]
    company = row["Company"]
    sector = row["Sector"]
    industry = row["Industry"]

    try:
        history = fetch_price_history(ticker, period="max")
        metrics = compute_price_metrics(history)

        result_row = {
            "Ticker": ticker,
            "Company Name": company,
            "Sector": sector,
            "Industry": industry,
        }
        result_row.update(metrics)
        results.append(result_row)
    except Exception as e:
        errors.append((ticker, str(e)))

    progress = i / n_tickers
    progress_bar.progress(
        progress,
        text=f"{progress_text} ({i}/{n_tickers} tickers)",
    )

progress_bar.empty()

if not results:
    st.error("No metrics could be computed for the selected tickers.")
    if errors:
        with st.expander("Error log for failed tickers"):
            for t, msg in errors:
                st.write(f"**{t}** → {msg}")
    st.stop()


# ------------------------------------------------------------
# Create DataFrame and apply Distance-from-ATH filter
# ------------------------------------------------------------
df_all = pd.DataFrame(results)

# Apply distance-from-ATH slider filter
df_filtered = df_all[
    (df_all["Distance From ATH (%)"] >= distance_min)
    & (df_all["Distance From ATH (%)"] <= distance_max)
]

# Decide what to show
if df_filtered.empty:
    st.warning(
        "No tickers match the selected **distance from ATH** filter. "
        "Showing all tickers that successfully loaded instead."
    )
    df_show = df_all
else:
    df_show = df_filtered

# Show counts
st.markdown(
    f"**Tickers after distance filter:** {len(df_filtered)} "
    f"(out of {len(df_all)} successfully loaded)"
)

# Sort by Distance From ATH (%) (closest to ATH first)
df_show = df_show.sort_values(by="Distance From ATH (%)", ascending=False)

# Round numeric columns for display
price_cols = ["Current Price", "ATH", "52W High", "52W Low"]
pct_cols = [
    "Distance From ATH (%)",
    "52W Range (%)",
    "Distance From 52W High (%)",
]

for col in price_cols:
    if col in df_show.columns:
        df_show[col] = df_show[col].round(2)

for col in pct_cols:
    if col in df_show.columns:
        df_show[col] = df_show[col].round(2)

# ------------------------------------------------------------
# Display table
# ------------------------------------------------------------
st.subheader("Screened Stocks")

st.dataframe(
    df_show[
        [
            "Ticker",
            "Company Name",
            "Sector",
            "Industry",
            "Current Price",
            "ATH",
            "Distance From ATH (%)",
            "52W High",
            "52W Low",
            "52W Range (%)",
            "Distance From 52W High (%)",
        ]
    ],
    use_container_width=True,
)

# ------------------------------------------------------------
# Error log (if any)
# ------------------------------------------------------------
if errors:
    with st.expander("⚠️ Errors / Skipped Tickers"):
        st.write(
            "Some tickers failed to load or compute. "
            "This can happen due to missing or inconsistent data in Yahoo Finance."
        )
        err_df = pd.DataFrame(errors, columns=["Ticker", "Error"])
        st.dataframe(err_df, use_container_width=True)
