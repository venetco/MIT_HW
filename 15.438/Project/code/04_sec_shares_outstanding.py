#!/usr/bin/env python3
"""Fetch latest shares outstanding from SEC EDGAR XBRL data (cover-page DEI tag)."""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"


def http_get(url: str, user_agent: str, sleep_seconds: float, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for URL: {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for URL: {url} ({exc})") from exc
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        data = gzip.decompress(data)
    time.sleep(sleep_seconds)
    return data


def load_ticker_map(user_agent: str, sleep_seconds: float) -> dict[str, dict]:
    url = f"{SEC_BASE}/files/company_tickers.json"
    data = http_get(url, user_agent, sleep_seconds)
    payload = json.loads(data.decode("utf-8"))
    return {row["ticker"].upper(): row for _, row in payload.items()}


def get_cik(ticker: str, user_agent: str, sleep_seconds: float) -> int:
    mapping = load_ticker_map(user_agent, sleep_seconds)
    if ticker.upper() not in mapping:
        raise ValueError(f"Ticker not found: {ticker}")
    return int(mapping[ticker.upper()]["cik_str"])


def fetch_shares_outstanding(cik: int, user_agent: str, sleep_seconds: float) -> dict:
    cik10 = str(cik).zfill(10)
    url = (
        f"{SEC_DATA_BASE}/api/xbrl/companyconcept/CIK{cik10}/dei/"
        "EntityCommonStockSharesOutstanding.json"
    )
    data = http_get(url, user_agent, sleep_seconds)
    payload = json.loads(data.decode("utf-8"))
    units = payload.get("units", {}).get("shares", [])
    if not units:
        return {"cik": cik, "latest": None, "history_count": 0}
    sorted_units = sorted(units, key=lambda x: (x.get("end", ""), x.get("filed", "")))
    latest = sorted_units[-1]
    return {
        "cik": cik,
        "latest_value": latest.get("val"),
        "latest_end_date": latest.get("end"),
        "latest_filed_date": latest.get("filed"),
        "latest_form": latest.get("form"),
        "history_count": len(units),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest shares outstanding from SEC EDGAR XBRL.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers e.g. EL,COTY,ELF")
    parser.add_argument(
        "--user-agent",
        required=True,
        help="SEC-compliant user agent: 'Your Name your_email@domain.com'",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Delay between SEC requests (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        default="To Upload/sec_shares_outstanding.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided.")

    print(f"Fetching shares outstanding for: {', '.join(tickers)}")
    out: list[dict] = []
    for t in tickers:
        try:
            cik = get_cik(t, args.user_agent, args.sleep_seconds)
            row = fetch_shares_outstanding(cik, args.user_agent, args.sleep_seconds)
            row["ticker"] = t
        except Exception as exc:  # noqa: BLE001
            row = {"ticker": t, "error": str(exc)}
        out.append(row)
        print(json.dumps(row, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
