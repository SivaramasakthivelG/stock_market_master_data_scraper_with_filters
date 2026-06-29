"""
Indian Stock Screener – Quarterly EPS + Price + PE
===================================================
Output CSV columns:
    stock, quarter_end, eps, price, pe

Run:
    pip install yfinance beautifulsoup4 lxml pandas numpy requests
    python quarterly_pipeline.py
"""

import logging
import os
import re
import time
import warnings
from calendar import monthrange
from datetime import date
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# suppress yfinance "possibly delisted" noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NSE_SCAN_LIMIT   = 2500   # how many NSE stocks to consider
MIN_MCAP_CRORE   = 600     # ₹600 Cr minimum market cap
MIN_QUARTERS     = 7       # skip stocks with fewer quarters
BATCH_SIZE       = 25      # tickers per yfinance mcap batch
SLEEP_YFINANCE   = 2       # seconds between mcap batches
SLEEP_SCREENER   = 0.5     # seconds between Screener.in requests
REQUEST_TIMEOUT  = 20

SAVE_DIR   = Path(os.environ.get("STOCK_REPORT_DIR", Path.home() / "stock_reports"))
OUTPUT_CSV = SAVE_DIR / "quarterly_eps_price_pe.csv"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

NSE_SYMBOL_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ── HTTP session ──────────────────────────────────────────────────────────────
_retry = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
session.mount("https://", HTTPAdapter(max_retries=_retry))
session.mount("http://",  HTTPAdapter(max_retries=_retry))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_number(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip().replace("₹", "").replace(",", "").replace("%", "").strip()
    if s in {"", "-", "--", "N/A", "n/a"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def quarter_label_to_end_date(label: str) -> pd.Timestamp:
    if not isinstance(label, str):
        return pd.NaT
    m = re.fullmatch(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})",
        label.strip().lower(),
    )
    if not m:
        return pd.NaT
    month = MONTH_MAP[m.group(1)]
    year  = int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=monthrange(year, month)[1])


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – NSE SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_nse_symbols() -> list[str]:
    print("Stage 1: Fetching NSE symbol list …")
    resp = session.get(NSE_SYMBOL_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    symbols = df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
    symbols = [s + ".NS" for s in symbols[:NSE_SCAN_LIMIT]]
    print(f"  {len(symbols)} symbols loaded.")
    return symbols


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – FILTER BY MARKET CAP
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_market_cap(symbols: list[str]) -> list[str]:
    min_mcap = MIN_MCAP_CRORE * 1e7
    print(f"\nStage 2: Filtering by market cap ≥ ₹{MIN_MCAP_CRORE} Cr …")
    qualified, skipped = [], 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        try:
            tickers = yf.Tickers(" ".join(batch))
            for sym in batch:
                try:
                    mcap = getattr(tickers.tickers[sym].fast_info, "market_cap", None)
                    if mcap and mcap >= min_mcap:
                        qualified.append(sym)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1
        except Exception as e:
            print(f"  ⚠ Batch error: {e}")
        time.sleep(SLEEP_YFINANCE)

    print(f"  ✔ {len(qualified)} passed  |  ✗ {skipped} skipped")
    return qualified


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – SCRAPE EPS PER QUARTER  (Screener.in)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_eps(symbol: str) -> pd.DataFrame:
    slug = symbol.replace(".NS", "")
    url  = f"https://www.screener.in/company/{slug}/consolidated/"

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()

    soup   = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table", class_="data-table")
    if not tables:
        return pd.DataFrame()

    try:
        raw = pd.read_html(StringIO(str(tables[0])))[0]
    except Exception:
        return pd.DataFrame()

    if raw.empty or raw.shape[1] < 2:
        return pd.DataFrame()

    raw = raw.set_index(raw.columns[0])

    eps_row = None
    for idx in raw.index:
        if isinstance(idx, str) and "eps" in idx.lower():
            eps_row = raw.loc[idx]
            break

    if eps_row is None:
        return pd.DataFrame()

    records = []
    for col, val in eps_row.items():
        qdate = quarter_label_to_end_date(str(col))
        eps   = clean_number(val)
        if pd.notna(qdate) and not np.isnan(eps):
            records.append({"quarter_end": qdate.date(), "eps": eps})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – BULK PRICE FETCH FOR ONE STOCK  (yfinance)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices_for_stock(symbol: str, dates: list[date]) -> dict[date, float]:
    """
    Download full price history once per stock, then look up each quarter-end.
    Much faster than one download-per-quarter. Silently returns {} on failure.
    """
    if not dates:
        return {}

    start = pd.Timestamp(min(dates)) - pd.Timedelta(days=10)
    end   = pd.Timestamp(max(dates)) + pd.Timedelta(days=5)

    try:
        hist = yf.download(symbol, start=start, end=end,
                           progress=False, auto_adjust=True)
    except Exception:
        return {}

    if hist.empty:
        return {}

    # flatten MultiIndex if present (yfinance ≥0.2.x)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    close = hist["Close"].dropna()
    if close.empty:
        return {}

    price_map: dict[date, float] = {}
    for d in dates:
        # get last available close on or before d
        ts    = pd.Timestamp(d)
        avail = close[close.index <= ts]
        if not avail.empty:
            price_map[d] = round(float(avail.iloc[-1]), 2)

    return price_map


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – BUILD FINAL TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_output(symbols: list[str]) -> pd.DataFrame:
    print(f"\nStage 3–4: Scraping EPS + price for {len(symbols)} stocks …")
    rows = []

    for sym in symbols:
        eps_df = scrape_eps(sym)

        if eps_df.empty:
            print(f"  ✗ {sym}: no EPS data")
            time.sleep(SLEEP_SCREENER)
            continue

        if len(eps_df) < MIN_QUARTERS:
            print(f"  ⏭ {sym}: only {len(eps_df)} quarters (need {MIN_QUARTERS})")
            time.sleep(SLEEP_SCREENER)
            continue

        # Fetch prices for every available quarter returned by Screener
        quarter_dates = eps_df["quarter_end"].tolist()
        price_map = fetch_prices_for_stock(sym, quarter_dates)

        for _, row in eps_df.iterrows():
            qdate = row["quarter_end"]
            eps   = row["eps"]
            price = price_map.get(qdate, np.nan)
            pe    = round(price / eps, 4) if eps and eps != 0 and not np.isnan(price) else np.nan
            rows.append({
                "stock":       sym,
                "quarter_end": qdate,
                "eps":         eps,
                "price":       price,
                "pe":          pe,
            })

        print(f"  ✔ {sym}: {len(eps_df)} quarters, {len(price_map)} prices fetched")
        time.sleep(SLEEP_SCREENER)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(["stock", "quarter_end"], inplace=True)
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Output → {OUTPUT_CSV}\n")

    symbols   = fetch_nse_symbols()
    qualified = filter_by_market_cap(symbols)

    if not qualified:
        print("⚠ No stocks passed the market cap filter. Exiting.")
        return

    df = build_output(qualified)

    if df.empty:
        print("⚠ No data collected.")
        return

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved → {OUTPUT_CSV}  ({len(df)} rows)")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
