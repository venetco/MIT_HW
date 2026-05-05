#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    by_ticker: dict[str, dict] = {}
    for _, row in payload.items():
        by_ticker[row["ticker"].upper()] = row
    return by_ticker


def get_cik_for_ticker(ticker: str, user_agent: str, sleep_seconds: float) -> int:
    ticker_map = load_ticker_map(user_agent, sleep_seconds)
    t = ticker.upper()
    if t not in ticker_map:
        raise ValueError(f"Ticker not found in SEC mapping: {t}")
    return int(ticker_map[t]["cik_str"])


def load_submissions(cik: int, user_agent: str, sleep_seconds: float) -> dict:
    cik10 = str(cik).zfill(10)
    url = f"{SEC_DATA_BASE}/submissions/CIK{cik10}.json"
    data = http_get(url, user_agent, sleep_seconds)
    return json.loads(data.decode("utf-8"))


def date_in_range(value: str, start_date: str, end_date: str) -> bool:
    return start_date <= value <= end_date


def filter_recent_filings(submissions: dict, forms: set[str], start_date: str, end_date: str) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []
    form_arr = recent.get("form", [])
    acc_arr = recent.get("accessionNumber", [])
    filed_arr = recent.get("filingDate", [])
    primary_arr = recent.get("primaryDocument", [])

    rows: list[dict] = []
    n = min(len(form_arr), len(acc_arr), len(filed_arr), len(primary_arr))
    for i in range(n):
        form = str(form_arr[i]).strip().upper()
        filed = str(filed_arr[i]).strip()
        if form in forms and date_in_range(filed, start_date, end_date):
            rows.append(
                {
                    "form": form,
                    "accession": str(acc_arr[i]).strip(),
                    "filing_date": filed,
                    "primary_document": str(primary_arr[i]).strip(),
                }
            )
    return rows


def save_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def filing_index_json(cik: int, accession: str, user_agent: str, sleep_seconds: float) -> dict:
    accession_nodash = accession.replace("-", "")
    url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession_nodash}/index.json"
    data = http_get(url, user_agent, sleep_seconds)
    return json.loads(data.decode("utf-8"))


def pick_form4_xml(index_json: dict) -> str | None:
    items = index_json.get("directory", {}).get("item", [])
    names = [x.get("name", "") for x in items]
    for name in names:
        lower = name.lower()
        if lower == "form4.xml":
            return name
    for name in names:
        lower = name.lower()
        if lower.endswith(".xml") and "4" in lower:
            return name
    return None


def xml_text(node: ET.Element | None, path: str) -> str:
    if node is None:
        return ""
    found = node.find(path)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def parse_form4_xml(xml_path: Path, filing_meta: dict, ticker: str) -> list[dict]:
    root = ET.fromstring(xml_path.read_bytes())

    issuer_ticker = xml_text(root, "issuer/issuerTradingSymbol") or ticker.upper()
    owner_name = xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    is_director = xml_text(root, "reportingOwner/reportingOwnerRelationship/isDirector")
    is_officer = xml_text(root, "reportingOwner/reportingOwnerRelationship/isOfficer")
    is_ten_percent = xml_text(root, "reportingOwner/reportingOwnerRelationship/isTenPercentOwner")
    officer_title = xml_text(root, "reportingOwner/reportingOwnerRelationship/officerTitle")

    rows: list[dict] = []

    def add_rows(table_tag: str, txn_tag: str, derivative_flag: str) -> None:
        table = root.find(table_tag)
        if table is None:
            return
        for txn in table.findall(txn_tag):
            rows.append(
                {
                    "ticker": issuer_ticker,
                    "form": filing_meta["form"],
                    "filing_date": filing_meta["filing_date"],
                    "accession": filing_meta["accession"],
                    "owner_name": owner_name,
                    "is_director": is_director,
                    "is_officer": is_officer,
                    "is_ten_percent_owner": is_ten_percent,
                    "officer_title": officer_title,
                    "derivative_flag": derivative_flag,
                    "security_title": xml_text(txn, "securityTitle/value"),
                    "transaction_date": xml_text(txn, "transactionDate/value"),
                    "transaction_code": xml_text(txn, "transactionCoding/transactionCode"),
                    "acquired_disposed_code": xml_text(
                        txn, "transactionAmounts/transactionAcquiredDisposedCode/value"
                    ),
                    "transaction_shares": xml_text(txn, "transactionAmounts/transactionShares/value"),
                    "transaction_price": xml_text(txn, "transactionAmounts/transactionPricePerShare/value"),
                    "shares_owned_following_txn": xml_text(txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
                }
            )

    add_rows("nonDerivativeTable", "nonDerivativeTransaction", "N")
    add_rows("derivativeTable", "derivativeTransaction", "Y")

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    ticker: str,
    start_date: str,
    end_date: str,
    outdir: Path,
    user_agent: str,
    sleep_seconds: float,
    forms: set[str],
) -> None:
    ticker_u = ticker.upper()
    cik = get_cik_for_ticker(ticker_u, user_agent, sleep_seconds)
    submissions = load_submissions(cik, user_agent, sleep_seconds)
    filings = filter_recent_filings(submissions, forms, start_date, end_date)
    if not filings:
        print("No filings found for filters.")
        return

    print(f"Ticker: {ticker_u}, CIK: {cik}")
    print(f"Matched filings: {len(filings)}")

    raw_dir = outdir / ticker_u / "raw"
    parsed_dir = outdir / ticker_u / "parsed"
    parsed_rows: list[dict] = []

    for idx, filing in enumerate(filings, start=1):
        accession = filing["accession"]
        accession_nodash = accession.replace("-", "")
        filing_folder = raw_dir / accession
        filing_folder.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(filings)}] {filing['form']} {accession} ({filing['filing_date']})")
        idx_json = filing_index_json(cik, accession, user_agent, sleep_seconds)
        save_bytes(filing_folder / "index.json", json.dumps(idx_json, indent=2).encode("utf-8"))

        primary_doc = filing["primary_document"]
        primary_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession_nodash}/{urllib.parse.quote(primary_doc)}"
        primary_bytes = http_get(primary_url, user_agent, sleep_seconds)
        save_bytes(filing_folder / primary_doc, primary_bytes)

        xml_name = pick_form4_xml(idx_json)
        if xml_name:
            xml_url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{accession_nodash}/{urllib.parse.quote(xml_name)}"
            xml_bytes = http_get(xml_url, user_agent, sleep_seconds)
            xml_path = filing_folder / xml_name
            save_bytes(xml_path, xml_bytes)
            try:
                parsed_rows.extend(parse_form4_xml(xml_path, filing, ticker_u))
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning: failed XML parse for {accession}: {exc}")
        else:
            print(f"  Warning: no XML detected in filing {accession}")

    write_csv(parsed_dir / "form4_transactions.csv", parsed_rows)
    print(f"Saved raw filings under: {raw_dir}")
    print(f"Saved parsed transactions CSV: {parsed_dir / 'form4_transactions.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SEC Form 4 filings and extract transactions.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. EL, COTY, ELF")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--forms",
        default="4,4/A",
        help="Comma-separated forms to include (default: 4,4/A)",
    )
    parser.add_argument(
        "--outdir",
        default="To Upload/sec_form4",
        help="Output directory for downloaded filings and extracted data",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Delay between SEC requests (default: 5.0 seconds)",
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="SEC-compliant user agent, e.g. 'Your Name your_email@domain.com'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forms = {x.strip().upper() for x in args.forms.split(",") if x.strip()}
    run(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        outdir=Path(args.outdir),
        user_agent=args.user_agent,
        sleep_seconds=args.sleep_seconds,
        forms=forms,
    )


if __name__ == "__main__":
    main()
