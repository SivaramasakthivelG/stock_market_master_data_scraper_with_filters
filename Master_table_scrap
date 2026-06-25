"""
Indian Stock Screener – Master Financial Pipeline
=================================================

Stage 1 : Fetch NSE symbols
Stage 2 : Process stocks
          • ENABLE_DOUBLING_FILTER = True  → keep only stocks that doubled within WINDOW_DAYS
          • ENABLE_DOUBLING_FILTER = False → process all eligible NSE stocks
Stage 3 : Filter by minimum market capitalization and enrich with Debt/Equity ratio
Stage 4 : Scrape Screener.in master financial tables
          • Quarterly Profit & Loss
          • Annual Profit & Loss
          • Balance Sheet
          • Cash Flow
Stage 5 : Save processed stock list (EVENTS_CSV)
Stage 6 : Save master financial dataset (MASTER_CSV)

Run:
    pip install yfinance beautifulsoup4 lxml pandas numpy requests
    python indian_stock_screener.py
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import os
import re
import time
from calendar import monthrange
from io import StringIO
from pathlib import Path

# ── third-party – fail fast ────────────────────────────────────────────────
_required = {"yfinance": None, "bs4": "beautifulsoup4", "lxml": None}
_missing = []
for _mod, _pkg in _required.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg or _mod)
if _missing:
    raise ImportError(
        f"Missing packages: {_missing}\n"
        f"Install with:  pip install {' '.join(_missing)}"
    )

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── optional Colab support ─────────────────────────────────────────────────
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
ENABLE_DOUBLING_FILTER = False  # set False to skip doubling filter and just collect master tables
PERIOD = "1y"              # yfinance history window
WINDOW_DAYS = 90            # rolling window to detect doubling
THRESHOLD = 2.0             # price must reach 2× start price
BATCH_SIZE = 25             # tickers per yfinance call
SLEEP_TIME = 2              # seconds between batches
MIN_DATA_POINTS = 120       # skip tickers with thin history
MIN_MCAP_CRORE = 600        # market-cap floor in ₹ Cr
MIN_MCAP = MIN_MCAP_CRORE * 10 ** 7
LOOKBACK_DAYS = 730         # keep only financial rows within 2 years before event
REQUEST_TIMEOUT = 20
SCREENER_DELAY = 0.4        # polite delay for Screener.in requests
NSE_SCAN_LIMIT = 3000       # number of NSE stocks to scan first
INCLUDE_SHAREHOLDING = False  # easy extension later

SAVE_DIR = Path(os.environ.get("STOCK_REPORT_DIR", Path.home() / "stock_reports"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_CSV = SAVE_DIR / "doubling_stocks.csv"
MASTER_CSV = SAVE_DIR / "master_financials.csv"

NSE_SYMBOL_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ── HTTP session ────────────────────────────────────────────────────────────
_retry = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
session.mount("https://", HTTPAdapter(max_retries=_retry))
session.mount("http://", HTTPAdapter(max_retries=_retry))

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def clean_number(x) -> float:
    """Parse messy finance strings → float. Returns np.nan on failure."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip().replace("₹", "").replace(",", "").replace("%", "").strip()
    if s in {"", "-", "--", "N/A", "n/a", "NA"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def quarter_label_to_end_date(label: str) -> pd.Timestamp:
    """
    'Mar 2025' → 2025-03-31, 'Feb 2024' → 2024-02-29, etc.
    Works for every month (not just quarter-ends). Returns pd.NaT on bad input.
    """
    if not isinstance(label, str):
        return pd.NaT
    m = re.fullmatch(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})",
        label.strip().lower(),
    )
    if not m:
        return pd.NaT
    month = MONTH_MAP[m.group(1)]
    year = int(m.group(2))
    last_day = monthrange(year, month)[1]
    return pd.Timestamp(year=year, month=month, day=last_day)


def normalize_line_item(name: str) -> str:
    """Stable key for analysis joins/grouping."""
    if not isinstance(name, str):
        return ""
    key = name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def infer_statement_type(title: str) -> str | None:
    t = (title or "").lower()
    if "cash flow" in t:
        return "cashflow"
    if "balance sheet" in t:
        return "balance_sheet"
    if "profit & loss" in t or "profit and loss" in t or "quarterly results" in t:
        return "pnl"
    if "shareholding" in t:
        return "shareholding"
    return None


def infer_period_type(title: str) -> str:
    t = (title or "").lower()
    return "quarterly" if ("quarter" in t or "qtr" in t) else "annual"


def list_has_date_like_columns(columns) -> bool:
    """Heuristic used to decide whether a table is date/period oriented."""
    score = 0
    for col in columns:
        txt = str(col).strip().lower()
        if re.fullmatch(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}", txt):
            score += 1
        elif re.fullmatch(r"\d{4}", txt):
            score += 1
    return score >= max(1, len(list(columns)) // 3)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – NSE SYMBOL LIST
# ══════════════════════════════════════════════════════════════════════════════

def fetch_nse_symbols(limit: int = NSE_SCAN_LIMIT) -> list[str]:
    """Download the NSE equity list and return Yahoo-style tickers."""
    print("Fetching NSE symbol list …")
    try:
        resp = session.get(NSE_SYMBOL_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if "SYMBOL" not in df.columns:
            raise RuntimeError("SYMBOL column not found in NSE csv")
        symbols = df["SYMBOL"].dropna().astype(str).str.strip().unique().tolist()
        symbols = symbols[:limit]
        symbols = [s + ".NS" for s in symbols]
        print(f"  {len(symbols)} symbols loaded (limited to {limit}).")
        return symbols
    except Exception as e:
        raise RuntimeError(f"Could not fetch NSE symbol list: {e}") from e


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – DOUBLING DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _find_doubling_event(prices: pd.Series, symbol: str) -> dict | None:
    """
    Slide a WINDOW_DAYS window over `prices`.
    Return the first window where the price at least doubles, else None.
    """
    series = prices.dropna()
    if len(series) < WINDOW_DAYS:
        return None

    vals = series.values
    dates = series.index

    # quick pre-filter: overall range must be wide enough
    if vals.max() < THRESHOLD * vals.min():
        return None

    for i in range(len(vals) - WINDOW_DAYS + 1):
        start_price = vals[i]
        start_date = dates[i]
        window_vals = vals[i : i + WINDOW_DAYS]
        window_dates = dates[i : i + WINDOW_DAYS]

        max_idx = window_vals.argmax()
        max_price = window_vals[max_idx]
        double_date = window_dates[max_idx]

        if max_price >= THRESHOLD * start_price:
            return {
                "stock": symbol,
                "start_date": start_date,
                "end_date": window_dates[-1],
                "double_date": double_date,
                "start_price": start_price,
                "double_price": max_price,
                "days_to_double": (double_date - start_date).days,
            }

    return None


def screen_for_doubling(symbols: list[str]) -> list[dict]:
    """
    Downloads OHLCV in batches and returns event dicts.
    If ENABLE_DOUBLING_FILTER is True, filters for price-doubling events.
    Always filters by MIN_DATA_POINTS and MIN_MCAP.
    """
    results = []
    total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_no, i in enumerate(range(0, len(symbols), BATCH_SIZE), 1):
        batch = [str(s) for s in symbols[i : i + BATCH_SIZE]]
        print(f"\n[{batch_no}/{total_batches}] Processing batch …")

        try:
            data = yf.download(
                batch,
                period=PERIOD,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as e:
            print(f"  ⚠ yfinance batch failed: {e}")
            continue

        for symbol in batch:
            try:
                # Handle single vs multiple ticker return formats from yfinance
                prices = data[symbol]["Close"] if len(batch) > 1 else data["Close"]

                # 1. Basic Data Integrity Check (Required regardless of filter state)
                prices_clean = prices.dropna()
                if len(prices_clean) < MIN_DATA_POINTS:
                    continue

                # 2. Setup the "event" base object with standard start/end data
                event = {
                    "stock": symbol,
                    "start_date": prices_clean.index[0],
                    "end_date": prices_clean.index[-1],
                    "start_price": prices_clean.iloc[0],
                    "double_price": prices_clean.iloc[-1],
                    "double_date": None,
                    "days_to_double": None,
                }

                # 3. Conditional Doubling Logic
                if ENABLE_DOUBLING_FILTER:
                    doubling_result = _find_doubling_event(prices, symbol)
                    if not doubling_result:
                        continue  # Skip if filter is ON but stock didn't double
                    event.update(doubling_result)  # Apply specific dates/prices

                # 4. Fundamentals Check (Required for all stocks)
                info = yf.Ticker(symbol).info
                shares = info.get("sharesOutstanding")
                if not shares:
                    continue

                ref_mcap = event["start_price"] * shares
                if ref_mcap < MIN_MCAP:
                    continue

                # 5. Append to results
                results.append({
                    "stock": event["stock"],
                    "start_date": pd.to_datetime(event["start_date"]).date(),
                    "end_date": pd.to_datetime(event["end_date"]).date(),
                    "double_date": pd.to_datetime(event["double_date"]).date() if event["double_date"] else None,
                    "start_price": round(event["start_price"], 2),
                    "double_price": round(event["double_price"], 2),
                    "ref_mcap_cr": round(ref_mcap / 10 ** 7, 2),
                    "de_ratio": info.get("debtToEquity"),
                    "days_to_double": event["days_to_double"],
                })
                
                status = "doubled" if ENABLE_DOUBLING_FILTER else "processed"
                print(f"  ✔ {symbol} {status}")

            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
                continue

        time.sleep(SLEEP_TIME)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – SAVE EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def save_events(results: list[dict]) -> pd.DataFrame:
    if not results:
        print("\n⚠ No doubling events found.")
        return pd.DataFrame()

    sort_col = "double_date" if ENABLE_DOUBLING_FILTER else "stock"
    df = pd.DataFrame(results).sort_values(sort_col).reset_index(drop=True)
    df.to_csv(EVENTS_CSV, index=False)
    print(f"\n✅ Events saved → {EVENTS_CSV}  ({len(df)} rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – MASTER TABLE SCRAPING (Screener.in)
# ══════════════════════════════════════════════════════════════════════════════

def _candidate_long_table(raw: pd.DataFrame, nse_symbol: str, statement_type: str,
                          period_type: str, section_title: str, source_url: str) -> pd.DataFrame:
    """
    Candidate A: treat first column as row label and melt columns as periods.
    Output: one row per stock + period + line item.
    """
    if raw is None or raw.empty or raw.shape[1] < 2:
        return pd.DataFrame()

    table = raw.copy().dropna(how="all")
    first_col = table.columns[0]

    long_df = table.melt(
        id_vars=[first_col],
        var_name="period_label",
        value_name="value",
    ).rename(columns={first_col: "line_item"})

    long_df["value"] = long_df["value"].apply(clean_number)
    long_df["period_end"] = long_df["period_label"].apply(quarter_label_to_end_date)
    long_df = long_df.dropna(subset=["period_end", "value"])

    if long_df.empty:
        return pd.DataFrame()

    long_df["stock"] = nse_symbol
    long_df["statement_type"] = statement_type
    long_df["period_type"] = period_type
    long_df["section_title"] = section_title
    long_df["source_url"] = source_url
    long_df["scrape_time"] = pd.Timestamp.utcnow().isoformat()
    long_df["is_consolidated"] = True if "consolidated" in source_url.lower() else False
    long_df["line_item_key"] = long_df["line_item"].apply(normalize_line_item)

    return long_df[[
        "stock", "statement_type", "period_type", "period_label", "period_end",
        "line_item", "line_item_key", "value", "section_title",
        "source_url", "scrape_time", "is_consolidated"
    ]].copy()


def _candidate_transposed_long_table(raw: pd.DataFrame, nse_symbol: str, statement_type: str,
                                    period_type: str, section_title: str, source_url: str) -> pd.DataFrame:
    """
    Candidate B: transpose first, then melt. Used as fallback when table layout changes.
    """
    if raw is None or raw.empty or raw.shape[1] < 2:
        return pd.DataFrame()

    table = raw.copy().dropna(how="all")
    first_col = table.columns[0]

    wide = table.set_index(first_col).T.reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"index": "period_label"})

    value_cols = [c for c in wide.columns if c != "period_label"]
    if not value_cols:
        return pd.DataFrame()

    long_df = wide.melt(
        id_vars=["period_label"],
        value_vars=value_cols,
        var_name="line_item",
        value_name="value",
    )

    long_df["value"] = long_df["value"].apply(clean_number)
    long_df["period_end"] = long_df["period_label"].apply(quarter_label_to_end_date)
    long_df = long_df.dropna(subset=["period_end", "value"])

    if long_df.empty:
        return pd.DataFrame()

    long_df["stock"] = nse_symbol
    long_df["statement_type"] = statement_type
    long_df["period_type"] = period_type
    long_df["section_title"] = section_title
    long_df["source_url"] = source_url
    long_df["scrape_time"] = pd.Timestamp.utcnow().isoformat()
    long_df["is_consolidated"] = True if "consolidated" in source_url.lower() else False
    long_df["line_item_key"] = long_df["line_item"].apply(normalize_line_item)

    return long_df[[
        "stock", "statement_type", "period_type", "period_label", "period_end",
        "line_item", "line_item_key", "value", "section_title",
        "source_url", "scrape_time", "is_consolidated"
    ]].copy()


def _parse_statement_table(raw: pd.DataFrame, nse_symbol: str, statement_type: str,
                           period_type: str, section_title: str, source_url: str) -> pd.DataFrame:
    """
    Parse a Screener table into the master long format.
    Tries the direct layout first, then a transposed fallback.
    """
    cand_a = _candidate_long_table(raw, nse_symbol, statement_type, period_type, section_title, source_url)
    cand_b = _candidate_transposed_long_table(raw, nse_symbol, statement_type, period_type, section_title, source_url)

    # Prefer the candidate that preserves more usable rows.
    if len(cand_a) >= len(cand_b):
        return cand_a
    return cand_b


def scrape_master_financials_for_symbol(nse_symbol: str) -> pd.DataFrame:
    """
    Scrape yearly P&L, quarterly P&L, balance sheet and cash flow into one master table.
    Easy to extend later for shareholding pattern.
    """
    slug = nse_symbol.replace(".NS", "")
    urls = [
        f"https://www.screener.in/company/{slug}/consolidated/",
        f"https://www.screener.in/company/{slug}/",
    ]

    for source_url in urls:
        try:
            resp = session.get(source_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        sections = soup.select("section")
        if not sections:
            continue

        frames = []

        for sec in sections:
            title_tag = sec.find("h2")
            if not title_tag:
                continue

            section_title = title_tag.get_text(" ", strip=True)
            statement_type = infer_statement_type(section_title)
            if statement_type is None:
                continue
            if statement_type == "shareholding" and not INCLUDE_SHAREHOLDING:
                continue

            period_type = infer_period_type(section_title)
            tables = sec.find_all("table", class_="data-table")
            if not tables:
                continue

            for table in tables:
                try:
                    raw = pd.read_html(StringIO(str(table)))[0]
                except Exception:
                    continue

                master = _parse_statement_table(
                    raw=raw,
                    nse_symbol=nse_symbol,
                    statement_type=statement_type,
                    period_type=period_type,
                    section_title=section_title,
                    source_url=source_url,
                )

                if not master.empty:
                    frames.append(master)

        if frames:
            out = pd.concat(frames, ignore_index=True)
            out["period_end"] = pd.to_datetime(out["period_end"])
            return out

    return pd.DataFrame()


def fetch_master_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch Screener master financials.

    ENABLE_DOUBLING_FILTER = True
        -> Fetch only doubling stocks and keep LOOKBACK_DAYS before double_date.

    ENABLE_DOUBLING_FILTER = False
        -> Fetch every stock and keep all financial data.
    """

    if events_df.empty:
        return pd.DataFrame()

    frames = []
    stocks = events_df["stock"].unique().tolist()

    print(f"\nFetching master financials for {len(stocks)} stocks …")

    for symbol in stocks:
        try:
            master = scrape_master_financials_for_symbol(symbol)

            if master.empty:
                print(f"  ✗ {symbol}: no financial tables")
                continue

            master["period_end"] = pd.to_datetime(master["period_end"])

            if ENABLE_DOUBLING_FILTER:
                event_row = events_df.loc[
                    events_df["stock"] == symbol
                ].iloc[0]

                double_date = pd.Timestamp(event_row["double_date"])
                cutoff = double_date - pd.Timedelta(days=LOOKBACK_DAYS)

                master = master[
                    (master["period_end"] >= cutoff) &
                    (master["period_end"] <= double_date)
                ]

            # When ENABLE_DOUBLING_FILTER=False
            # do not filter by dates at all.

            if not master.empty:
                frames.append(master)
                print(f"  ✔ {symbol}: {len(master)} master rows")
            else:
                print(f"  ✗ {symbol}: no usable rows")

        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

        time.sleep(SCREENER_DELAY)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.sort_values(
        ["stock", "statement_type", "period_end", "line_item_key"],
        inplace=True,
    )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – SAVE MASTER TABLE
# ══════════════════════════════════════════════════════════════════════════════

def save_master(df: pd.DataFrame) -> None:
    if df.empty:
        print("⚠ No master financial data to save.")
        return
    df.to_csv(MASTER_CSV, index=False)
    print(f"✅ Master financials saved → {MASTER_CSV}  ({len(df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests():
    # clean_number
    assert clean_number("1,234") == 1234.0
    assert clean_number("12.5%") == 12.5
    assert clean_number("₹1,000") == 1000.0
    assert np.isnan(clean_number("-"))
    assert np.isnan(clean_number("N/A"))
    assert np.isnan(clean_number(None))
    assert clean_number(42) == 42.0

    # period parser
    cases = {
        "Mar 2025": "2025-03-31",
        "Dec 2023": "2023-12-31",
        "Jun 2024": "2024-06-30",
        "Jan 2024": "2024-01-31",
        "Feb 2024": "2024-02-29",
        "Apr 2023": "2023-04-30",
        "Nov 2021": "2021-11-30",
    }
    for label, expected in cases.items():
        result = quarter_label_to_end_date(label)
        assert str(result.date()) == expected, f"{label}: got {result.date()}"

    assert pd.isna(quarter_label_to_end_date("Invalid"))
    assert pd.isna(quarter_label_to_end_date(None))

    # master table candidate parsing (as-is)
    dummy = pd.DataFrame({
        "Particulars": ["Sales", "Other income", "Net profit"],
        "Mar 2024": [100, 10, 30],
        "Jun 2024": [110, 12, 33],
    })
    a = _candidate_long_table(dummy, "TEST.NS", "pnl", "quarterly", "Quarterly Results", "https://example.com")
    b = _candidate_transposed_long_table(dummy, "TEST.NS", "pnl", "quarterly", "Quarterly Results", "https://example.com")
    assert not a.empty
    assert not b.empty
    assert len(a) == 6
    assert set(a["line_item_key"].unique()) == {"sales", "other_income", "net_profit"}

    print("All self-tests passed ✅")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _run_tests()

    print(f"\nSave directory : {SAVE_DIR}")
    print(f"Events CSV     : {EVENTS_CSV}")
    print(f"Master CSV     : {MASTER_CSV}\n")

    # Stage 1
    symbols = fetch_nse_symbols()

    # Stage 2
    events = screen_for_doubling(symbols)

    # Stage 3
    events_df = save_events(events)

    # Stage 4
    master_df = fetch_master_features(events_df)

    # Stage 5
    save_master(master_df)

    print("\n🏁 Pipeline complete.")
    print(f"   Doubling events  : {len(events_df)}")
    print(f"   Master rows      : {len(master_df)}")

    if IN_COLAB:
        from google.colab import files
        files.download(str(EVENTS_CSV))
        files.download(str(MASTER_CSV))


if __name__ == "__main__":
    main()
