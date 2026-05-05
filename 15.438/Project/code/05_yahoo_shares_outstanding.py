#!/usr/bin/env python3
"""Fetch shares outstanding for tickers using Yahoo Finance quoteSummary endpoint."""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def make_opener() -> urllib.request.OpenerDirector:
    cj = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def http_get_text(opener: urllib.request.OpenerDirector, url: str, sleep_seconds: float = 1.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with opener.open(req, timeout=30) as resp:
        data = resp.read()
    if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
        data = gzip.decompress(data)
    time.sleep(sleep_seconds)
    return data.decode("utf-8", errors="replace")


def safe_warm_cookie(opener: urllib.request.OpenerDirector, url: str) -> None:
    try:
        http_get_text(opener, url)
    except urllib.error.HTTPError:
        pass
    except urllib.error.URLError:
        pass


def get_crumb(opener: urllib.request.OpenerDirector) -> str:
    safe_warm_cookie(opener, "https://finance.yahoo.com/quote/AAPL")
    safe_warm_cookie(opener, "https://fc.yahoo.com")
    return http_get_text(opener, "https://query2.finance.yahoo.com/v1/test/getcrumb").strip()


def fetch_shares_outstanding(opener, crumb: str, ticker: str) -> dict:
    modules = "defaultKeyStatistics,summaryDetail,price,financialData"
    url = (
        "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{urllib.parse.quote(ticker)}?modules={modules}&crumb={urllib.parse.quote(crumb)}"
    )
    text = http_get_text(opener, url)
    payload = json.loads(text)
    result = payload.get("quoteSummary", {}).get("result") or []
    if not result:
        return {"ticker": ticker, "shares_outstanding": None, "raw": payload}
    item = result[0]
    price = item.get("price", {})
    key_stats = item.get("defaultKeyStatistics", {})

    shares_outstanding = (
        (key_stats.get("sharesOutstanding") or {}).get("raw")
        or (price.get("sharesOutstanding") or {}).get("raw")
    )
    market_cap = (price.get("marketCap") or {}).get("raw")
    regular_market_price = (price.get("regularMarketPrice") or {}).get("raw")
    quote_type = price.get("quoteType")

    return {
        "ticker": ticker,
        "quote_type": quote_type,
        "regular_market_price": regular_market_price,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch shares outstanding from Yahoo Finance.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers e.g. EL,COTY,ELF")
    parser.add_argument(
        "--output",
        default="To Upload/yahoo_shares_outstanding.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opener = make_opener()
    crumb = get_crumb(opener)
    if not crumb:
        raise SystemExit("Could not obtain Yahoo crumb. Try again or use Bloomberg fallback.")

    out: list[dict] = []
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        try:
            row = fetch_shares_outstanding(opener, crumb, ticker)
        except Exception as exc:  # noqa: BLE001
            row = {"ticker": ticker, "error": str(exc)}
        out.append(row)
        print(json.dumps(row, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
