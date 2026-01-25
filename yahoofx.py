#!/usr/bin/env python3
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
SYMBOLS: List[str] = [
    "EURUSD=X",
    "JPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "EURJPY=X",
    "GBPJPY=X",
    "EURGBP=X",
    "EURCAD=X",
    "EURSEK=X",
    "EURCHF=X",
    "EURHUF=X",
    "CNY=X",
    "HKD=X",
    "SGD=X",
    "INR=X",
    "MXN=X",
    "PHP=X",
    "IDR=X",
    "THB=X",
    "MYR=X",
    "ZAR=X",
    "RUB=X",
]

PERIOD = "max"
INTERVAL = "1d"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Throttling / retries
SLEEP_BETWEEN_TICKERS_SECS = float(os.getenv("SLEEP_BETWEEN_TICKERS_SECS", "1.5"))
MAX_ATTEMPTS_PER_TICKER = int(os.getenv("MAX_ATTEMPTS_PER_TICKER", "6"))
BACKOFF_BASE_SECS = float(os.getenv("BACKOFF_BASE_SECS", "2.0"))
REQUEST_TIMEOUT_SECS = float(os.getenv("REQUEST_TIMEOUT_SECS", "30"))

# Optional: if you want to pass your browser cookies (e.g. from your curl), set:
# export YAHOO_COOKIE='GUCS=...; A1=...; A1S=...; A3=...; ...'
YAHOO_COOKIE = os.getenv("YAHOO_COOKIE", "").strip()

# Optional: curl_cffi impersonation profile
IMPERSONATE = os.getenv("IMPERSONATE", "chrome").strip()


def make_curl_cffi_session():
    """
    Create a curl_cffi session for yfinance, so yfinance won't reject it.
    If curl_cffi isn't installed, return None and let yfinance create its own session.
    """
    try:
        from curl_cffi import requests as crequests
    except Exception:
        return None

    sess = crequests.Session(impersonate=IMPERSONATE)
    # Browser-like headers (not strictly required, but can help consistency)
    sess.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://finance.yahoo.com/",
            "Origin": "https://finance.yahoo.com",
        }
    )
    if YAHOO_COOKIE:
        sess.headers["Cookie"] = YAHOO_COOKIE
    return sess


def fetch_one(sym: str, session) -> pd.DataFrame:
    """
    Download one symbol. Returns normalized DataFrame with columns:
    date,symbol,open,high,low,close,adjclose,volume(optional)
    """
    df = yf.download(
        sym,
        period=PERIOD,
        interval=INTERVAL,
        actions=False,
        auto_adjust=False,
        threads=False,  # critical: avoid burst parallelism
        progress=False,
        timeout=REQUEST_TIMEOUT_SECS,
        session=session,  # must be curl_cffi session OR None
        multi_level_index=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # normalize date column name
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "date"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjclose",
        "Volume": "volume",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )

    df["symbol"] = sym
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["date"])

    cols = ["date", "symbol", "open", "high", "low", "close", "adjclose"]
    if "volume" in df.columns:
        cols.append("volume")
    return df[cols]


def fetch_with_retry(sym: str, session) -> pd.DataFrame:
    for attempt in range(1, MAX_ATTEMPTS_PER_TICKER + 1):
        try:
            df = fetch_one(sym, session)
            # Treat empty as a retryable condition (common during throttling/rate-limit)
            if not df.empty:
                return df

            sleep_for = BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            logger.warning(
                "%s: empty result, retry %d/%d in %.1fs",
                sym,
                attempt,
                MAX_ATTEMPTS_PER_TICKER,
                sleep_for,
            )
            time.sleep(sleep_for)

        except YFRateLimitError:
            sleep_for = BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            logger.warning(
                "%s: rate limited (YFRateLimitError), retry %d/%d in %.1fs",
                sym,
                attempt,
                MAX_ATTEMPTS_PER_TICKER,
                sleep_for,
            )
            time.sleep(sleep_for)

        except Exception as exc:
            # Some failures are transient; apply backoff and retry
            sleep_for = BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            logger.warning(
                "%s: error (%s), retry %d/%d in %.1fs",
                sym,
                exc,
                attempt,
                MAX_ATTEMPTS_PER_TICKER,
                sleep_for,
            )
            time.sleep(sleep_for)

    return pd.DataFrame()


def main():
    # Prefer curl_cffi session if available; otherwise let yfinance handle session=None
    session = make_curl_cffi_session()

    all_dfs = []
    for i, sym in enumerate(SYMBOLS, 1):
        logger.info(
            "▶ [%d/%d] Fetching %s (period=%s interval=%s)",
            i,
            len(SYMBOLS),
            sym,
            PERIOD,
            INTERVAL,
        )

        df = fetch_with_retry(sym, session)
        logger.info("  %s: %d rows", sym, len(df))
        if not df.empty:
            all_dfs.append(df)

        time.sleep(SLEEP_BETWEEN_TICKERS_SECS)

    if not all_dfs:
        logger.error(
            "No data fetched. If this persists, you may be temporarily rate-limited by Yahoo from your IP."
        )
        sys.exit(1)

    out = pd.concat(all_dfs, ignore_index=True).sort_values(["symbol", "date"])
    out_name = f"yahoo_fx_daily_max_yfinance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = os.path.join(os.getcwd(), out_name)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), out_path)


if __name__ == "__main__":
    main()
