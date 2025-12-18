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
    page_title="US Stock Screener (yfinance)",
    page_icon="📈",
    layout="wide",
)


# ------------------------------------------------------------
# Data loading & caching helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_sp500_universe() -> pd.DataFrame:
    """
    Load S&P 500 constituents from Wikipedia.

    Returns:
        DataFrame with columns: Ticker, Company, Sector, Industry
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    constituents = tables[0]

    constituents = constituents.rename(
        columns={
            "Symbol": "Ticker",
            "Security": "Company",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "Industry",
        }
    )

    # Keep only relevant columns
    df = constituents[["Ticker", "Company", "Sector", "Industry"]].copy()

    # yfinance uses "-" instead of "." for share classes (e.g., BRK.B -> BRK-B)
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)

    return df


@st.cache_data(show_spinner=False)
def fetch_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """
    Fetch daily price history for a ticker using yfinance.

    Args:
        ticker: Ticker symbol.
        period: yfinance period string (e.g., '1y', '5y', 'max').

    Returns:
        DataFrame with at least 'Price' column.

    Raises:
        ValueError if no data is returned.
    """
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,  # Adjust for splits/dividends
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("No historical data returned")

    # Normalize to a generic 'Price' column
    if "Adj Close" in data.columns:
        price_series = data["Adj Close"]
    elif "Close" in data.columns:
        price_series = data["Close"]
    else:
        raise ValueError("No suitable price column found")

    df = data.copy()
    df["Price"] = price_series
    return df


def compute_price_metrics(history: pd.DataFrame) -> Dict[str, float]:
    """
    Given a historical price DataFrame, compute ATH, 52-week metrics, etc.

    Args:
        history: DataFrame returned by fetch_price_history() with 'Price' column.

    Returns:
        Dict of metrics.
    """
    price = history["Price"].dropna()
    if price.empty:
        raise ValueError("Price series is empty")

    # All-Time High
    ath = float(price.max())
    current_price = float(price.iloc[-1])

    # 52-week window ~ 252 trading days
    window_len = min(252, len(price))
    last_52w = price.tail(window_len)
    high_52w = float(last_52w.max())
    low_52w = float(last_52w.min())

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

    return {
        "Current Price": current_price,
        "ATH": ath,
        "Distance From ATH (%)": distance_from_ath_pct,
        "52W High": high_52w,
        "52W Low": low_52w,
        "52W Range (%)": range_52w_pct,
    }


# ------------------------------------------------------------
# Sidebar: universe, filters, and controls
# ------------------------------------------------------------
st.sidebar.title("⚙️ Screener Controls")

with st.sidebar:
    st.markdown("### Stock Universe")
    st.caption("Using **S&P 500** constituents as the default universe.")

universe_df = load_sp500_universe()

# Sector selector (dynamic)
sector_options = ["All Sectors"] + sorted(universe_df["Sector"].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Sector", sector_options, index=0)

if selected_sector != "All Sectors":
    universe_filtered_sector = universe_df[universe_df["Sector"] == selected_sector]
else:
    universe_filtered_sector = universe_df.copy()

# Industry selector (dependent on sector selection)
industry_options = ["All Industries"] + sorted(
    universe_filtered_sector["Industry"].dropna().unique().tolist()
)
selected_industry = st.sidebar.selectbox("Industry", industry_options, index=0)

if selected_industry != "All Industries":
    final_universe = universe_filtered_sector[
        universe_filtered_sector["Industry"] == selected_industry
    ].copy()
else:
    final_universe = universe_filtered_sector.copy()

# Optional: slider to filter by distance from ATH
st.sidebar.markdown("### Distance from ATH Filter")
distance_min, distance_max = st.sidebar.slider(
    "Distance from ATH (%) (Current - ATH) / ATH × 100",
    min_value=-100.0,
    max_value=50.0,
    value=(-50.0, 0.0),
    step=1.0,
    help="Filter by how far the stock is from its all-time high.",
)

# Optional: limit max number of tickers to process (for performance)
st.sidebar.markdown("### Performance")
max_tickers = st.sidebar.slider(
    "Max tickers to process",
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help="Limit number of tickers (after filters) to keep the screener fast.",
)

run_screener = st.sidebar.button("🔍 Run Screener")


# ------------------------------------------------------------
# Main layout
# ------------------------------------------------------------
st.title("📈 US Stock Screener (yfinance)")
st.caption(
    "A simple Streamlit stock screener for S&P 500 stocks using Yahoo Finance (`yfinance`). "
    "Educational use only — **not investment advice**."
)

st.markdown(
    f"**Universe size:** {len(universe_df)} tickers  •  "
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
df = pd.DataFrame(results)

# Apply distance-from-ATH slider filter
df = df[
    (df["Distance From ATH (%)"] >= distance_min)
    & (df["Distance From ATH (%)"] <= distance_max)
]

if df.empty:
    st.warning(
        "No tickers match the selected **distance from ATH** filter. "
        "Try widening the range in the sidebar."
    )
    if errors:
        with st.expander("Error log for failed tickers"):
            for t, msg in errors:
                st.write(f"**{t}** → {msg}")
    st.stop()

# Sort by Distance From ATH (%) (closest to ATH first)
df = df.sort_values(by="Distance From ATH (%)", ascending=False)

# Round numeric columns for display
price_cols = ["Current Price", "ATH", "52W High", "52W Low"]
pct_cols = ["Distance From ATH (%)", "52W Range (%)"]

for col in price_cols:
    if col in df.columns:
        df[col] = df[col].round(2)

for col in pct_cols:
    if col in df.columns:
        df[col] = df[col].round(2)

st.subheader("Screened Stocks")

st.dataframe(
    df[
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
